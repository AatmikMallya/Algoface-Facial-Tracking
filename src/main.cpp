#include <iostream>
#include <filesystem>
#include <vector>

#include "glog/logging.h"

#include <functional>
#include "../include/optimization.h"
#include "../include/tensor.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

#include "ceres/ceres.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/core/mat.hpp>

#include "utility.h"

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>


using std::cout;
using std::endl;
using std::vector;

string WAREHOUSE_PATH = "F:/Capstone/LocalREp/Facial-Tracking2/data/FaceWarehouse/";
string RAW_TENSOR_PATH = "F:/Capstone/LocalREp/Facial-Tracking2/data/raw_tensor.bin";
string SHAPE_TENSOR_PATH = "F:/Capstone/LocalREp/Facial-Tracking2/data/shape_tensor.bin";
//ctl + k + u/c

int main() {

	//********************************** create or load tensors

	// Raw tensor: 150 users X 47 expressions X 11510 vertices
	tensor3 rawTensor(150, 47, 11510);
	tensor3 shapeTensor(150, 47, 73);

	if (std::filesystem::exists(RAW_TENSOR_PATH)) {
		loadRawTensor(RAW_TENSOR_PATH, rawTensor);
	}
	else {
		buildRawTensor(WAREHOUSE_PATH, RAW_TENSOR_PATH, rawTensor);
	}

	if (std::filesystem::exists(SHAPE_TENSOR_PATH)) {
		loadShapeTensor(SHAPE_TENSOR_PATH, shapeTensor);
	}
	else {
		buildShapeTensor(rawTensor, SHAPE_TENSOR_PATH, shapeTensor);
	}

	//**********************************
	//********************************** ceres optimization variable initilization
	//lms = 2D landmarks used to construct x - cx / f
	//pose = 1x6 vector with rotation and translation
	//image = needed to get cx and cy
	//f = focal length
	//w_exp = output

	/*1. Create a vector with length 47 corresponding to 47 different expressions that we have in the Warehouse dataset for each user. */
	int numExpressions = 47;
	Eigen::VectorXf w(numExpressions); // weights in optimization

	/*2.Assign an initial value to each element of that vector. I would say w[0] = 1 and everything else w[i] = 0. That means we would
	like to have the neutral expression as our initial guess because we assign 1 to neutral expression weight and zero to everything else. */
	for (int i = 0; i < numExpressions; i++)
	{
		w[i] = 0;
	}
	w[0] = 1;


	/*3.Then go to identity 137 in your tensor and extract the matrix out of it. The matrix will have dimensions roughly 219 x 47.
	219 represents the xyz face vector and 47 are the different expression blend shapes of the user (in this case user 137).*/
	int n_vectors = 73;
	vector<cv::Point3f> singleExp(n_vectors); // holds 1 expression with 73 vertices used to create multExp
	vector<vector<cv::Point3f>> multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp
	for (int i = 0; i < numExpressions; i++)//numExpresion is 47
	{
		//73 vertices
		for (int j = 0; j < n_vectors; j++) {
			Eigen::Vector3f tens_vec = shapeTensor(137, i, j);
			cv::Point3f cv_vec;
			cv_vec.x = tens_vec.x();
			cv_vec.y = tens_vec.y();
			cv_vec.z = tens_vec.z();
			singleExp[j] = cv_vec;
		}
		multExp[i] = singleExp;
	}

	/*4.Linearly combine 47 expressions by multiplying this matrix by your expression weights that you created at step 1 and 2.
	This will yield a vector of length 33k representing the resultant expression at that point, which will be a neutral
	expression because our initial assigned expression weights enforce it that way. */
	// Copy Eigen vector to OpenCV vector
	vector<cv::Point3f> combinedExp(n_vectors); //holds 1 expression made from 47 expressions (neutral expression)
	//73 vertices
	for (int i = 0; i < n_vectors; i++)
	{
		//47 expressions
		for (int j = 0; j < numExpressions; j++)
		{
			combinedExp[i].x = (combinedExp[i].x + multExp[j][i].x * w[j]);
			combinedExp[i].y = (combinedExp[i].y + multExp[j][i].y * w[j]);
			combinedExp[i].z = (combinedExp[i].z + multExp[j][i].z * w[j]);
		}
	}

	/*5.Now that you have generated a single face out of all the faces that you had in your tensor, you can do pose estimation like before. */
	/** Transform from object coordinates to camera coordinates **/
	// Image vector contains 2d landmark positions
	string img_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.png"; //pose estimation
	string land_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.land"; //pose estimation
	cv::Mat image = cv::imread(img_path, 1);
	std::vector<cv::Point2f> lmsVec = readLandmarksFromFile_2(land_path, image);

	//double fx = 640, fy = 640, cx = 320, cy = 240;
	double f = image.cols;               // ideal camera where fx ~ fy
	double cx = image.cols / 2.0;
	double cy = image.rows / 2.0;
	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
	/* Assuming no distortion
	//cv::Mat distCoeffs(4, 1, CV_64F);
	//distCoeffs.at<double>(0) = 0;
	//distCoeffs.at<double>(1) = 0;
	//distCoeffs.at<double>(2) = 0;
	//distCoeffs.at<double>(3) = 0;*/
	// Get rotation and translation parameters
	cv::Mat rvec(3, 1, CV_64F);
	cv::Mat tvec(3, 1, CV_64F);

	//pose estimation function Object coordinate <---> Camera coordinate
	cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec); //distCoeffs -> cv::Mat

	cout << rvec.t() << endl;
	cout << tvec.t() << endl;
	vector<double> poseVec(6);
	for (int i = 0; i < 3; i++)
	{
		cout << "rvec[" << i << "]: " << rvec.at<double>(i);
		cout << "\ttvec[" << i << "]: " << tvec.at<double>(i) << endl;
		poseVec[i] = rvec.at<double>(i);
		poseVec[i + 3] = tvec.at<double>(i);
	}

	for (float val : poseVec)
		cout << val << " ";
	cout << endl;

	// Convert Euler angles to rotation matrix (1x3 to 3x3)
	cv::Mat R;
	cv::Rodrigues(rvec, R);

	// Combine 3x3 rotation and 3x1 translation into 4x4 transformation matrix
	cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
	T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;
	T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;
	/*T= R R R Tx
		 R R R Ty
		 R R R Tz
		 0 0 0 1 */
		 // Transform object to make cameraVec
	std::vector<cv::Mat> cameraVec;
	for (auto& vec : combinedExp) {
		double data[4] = { vec.x, vec.y, vec.z, 1 };
		cv::Mat vector4d = cv::Mat(4, 1, CV_64F, data);
		cv::Mat result = T * vector4d;
		cameraVec.push_back(result);
	}

	/*6.Then use your resultant 3D along with pose parameters (rotation and translation) in order to
	reproject the 3D vertices (corresponding to each 2D landmark) back onto 2D image. */
	// Project points onto image
	std::vector<cv::Point2f> imageVec;
	for (auto& vec : cameraVec) {
		cv::Point2f result;
		result.x = f * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
		result.y = f * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;
		imageVec.push_back(result);
	}


	/*7. Then use the outputs from step 6 to compare with ground truth 2D
	landmarks inside ceres operator function residual block.*/
	//********************************** ceres optimization
	//bool boolOpt = optimize(multExp, lmsVec, poseVec, image, f, w); //first optimization
	bool boolOpt = optimize(multExp, lmsVec, poseVec, image, f, w);

	/*8. Repeat this process by constantly updating your initial pose and expression parameter, for 3 iterations.
	So for 3 times, you alternately optimize pose and expression and at each iteration you use your estimated outputs of the previous step*/
	for (int opt = 0; opt < 3; opt++) // multiple optimizations
	//while(!(optimize(multExp, lmsVec, poseVec, image, f, w)))
	{
		// create new face based on weights
		for (int i = 0; i < 73; i++)
		{
			for (int j = 0; j < numExpressions; j++)
			{

				combinedExp[i].x = combinedExp[i].x + multExp[j][i].x * w[j];
				combinedExp[i].y = combinedExp[i].y + multExp[j][i].y * w[j];
				combinedExp[i].z = combinedExp[i].z + multExp[j][i].z * w[j];
			}
		}
		//pose estimation
		cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec,true);

		for (int i = 0; i < 3; i++)
		{
			cout << "rvec[" << i << "]: " << rvec.at<double>(i);
			cout << "\ttvec[" << i << "]: " << tvec.at<double>(i) << endl;
			poseVec[i] = rvec.at<double>(i);
			poseVec[i + 3] = tvec.at<double>(i);
		}

		// optimization
		optimize(multExp, lmsVec, poseVec, image, f, w);
	}


	cv::projectPoints(combinedExp, rvec, tvec, cameraMatrix, cv::Mat(), imageVec);//distCoeffs ->cv::Mat()

	cv::Mat visualImage = image.clone();
	double sc = 2; //size of displayed picture
	cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
	for (int i = 0; i < imageVec.size(); i++) {
		//cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
//        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

		cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 0, 255), sc);             // 3d projections (red)
		cv::circle(visualImage, lmsVec[i] * sc, 1, cv::Scalar(0, 255, 0), sc);               // 2d landmarks   (green)

		//cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

	}
	cv::imshow("visualImage", visualImage);

	//**********************************
	//**********************************

	int key = cv::waitKey(0) % 256;
	if (key == 27)                        // Esc button is pressed
		Sleep(1000);

	//**********************************



	/*9. Once done, visualize the final 3D face in Easy3D to see if it resembles
	the open mouth expression. */

	/*apply weights to the raw tensor (the specified identity) to create one dense face for visualization*/
	int n_raw_vectors = 11510;
	vector<cv::Point3f> raw_singleExp(n_raw_vectors); // holds 1 expression with 73 vertices used to create multExp
	vector<vector<cv::Point3f>> raw_multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp
	for (int i = 0; i < numExpressions; i++)//numExpresion is 47
	{
		//11510 vertices
		for (int j = 0; j < n_raw_vectors; j++) {
			Eigen::Vector3f raw_tens_vec = rawTensor(138, i, j);
			cv::Point3f raw_cv_vec;
			raw_cv_vec.x = raw_tens_vec.x();
			raw_cv_vec.y = raw_tens_vec.y();
			raw_cv_vec.z = raw_tens_vec.z();
			raw_singleExp[j] = raw_cv_vec;
		}
		raw_multExp[i] = raw_singleExp;
	}
	vector<cv::Point3f> raw_combinedExp(n_raw_vectors); //holds 1 expression made from 47 expressions (neutral expression)
//73 vertices
	for (int i = 0; i < n_raw_vectors; i++)
	{
		//47 expressions
		for (int j = 0; j < numExpressions; j++)
		{
			raw_combinedExp[i].x = (raw_combinedExp[i].x + raw_multExp[j][i].x * w[j]);
			raw_combinedExp[i].y = (raw_combinedExp[i].y + raw_multExp[j][i].y * w[j]);
			raw_combinedExp[i].z = (raw_combinedExp[i].z + raw_multExp[j][i].z * w[j]);
		}
	}
	//asu::Utility util;
	//vector<vector<uint32_t>> quads = util.readQuadIndicesFromFile("F:/Capstone/LocalREp/Facial-Tracking2/data/faces.obj");
	//easy3d::SurfaceMesh* mesh = new easy3d::SurfaceMesh();
	//vector<easy3d::SurfaceMesh::Vertex> surfaceVertices(faceVerts.size());
	//for (int i = 0; i < faceVerts.size(); i++)
	//	surfaceVertices[i] = mesh->add_vertex(faceVerts[i]);
	//for (int i = 0; i < quads.size(); i++)
	//	mesh->add_quad(surfaceVertices[quads[i][0]], surfaceVertices[quads[i][1]], surfaceVertices[quads[i][2]], surfaceVertices[quads[i][3]]);


	//auto sDrawable = mesh->renderer()->get_triangles_drawable("faces");    // the string must be "faces"

	//********************************** creating 3D face
	vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("F:/Capstone/LocalREp/Facial-Tracking2/data/faces.obj"); //easy3D
	vector<easy3d::vec3> faceVerts = readFace3DFromObj(WAREHOUSE_PATH + "Tester_138/Blendshape/shape_22.obj"); //easy3D
	vector<int> all3dVertices = readVertexIdFromFile("F:/Capstone/LocalREp/Facial-Tracking2/data/lm_vert_internal_73.txt");   // same order as landmarks (Easy3D)

	vector<int> poseIndices = {
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  //face contour
			21, 22, 23, 24, 25, 26,                            //left eyebrow
			18, 17, 16, 15, 20, 19,                            //right eyebrow
			27, 66, 28, 69, 29, 68, 30, 67,                    //left eye
			33, 70, 32, 73, 31, 72, 34, 71,                    //right eye
			35, 36, 37, 38, 44, 39, 45, 40, 41, 42, 43,        //nose contour
			65,												  //nose tip
			46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,    //outer mouth
			63, 62, 61, 60, 59, 58						      //inner mouth
	};
	vector<int> vk(poseIndices.size());
	for (int i = 0; i < vk.size(); i++)
		vk[i] = all3dVertices[poseIndices[i]];

	vector<easy3d::vec3> lmVerts(vk.size());
	for (int i = 0; i < vk.size(); i++)
		lmVerts[i] = faceVerts[vk[i]];

	//--- always initialize viewer first before doing anything else for 3d visualization 
	//========================================================================================
	easy3d::Viewer viewer("internal vertices");

	//------------------------- face surface mesh
	//===========================================================================
	auto surface = new easy3d::TrianglesDrawable("faces");
	surface->update_vertex_buffer(faceVerts);
	surface->update_element_buffer(meshIndices);
	surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

	//------------------- vertices corresponding to landmarks
	//===========================================================================
	auto vertices = new easy3d::PointsDrawable("vertices");
	vertices->update_vertex_buffer(lmVerts);
	vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));
	vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
	vertices->set_point_size(10);

	//---------------------- add drawable objects to viewer
	//===========================================================================
	viewer.add_drawable(surface);
	//viewer.add_model(mesh);        // the model must first be added to the viewer before accessing the drawables
	viewer.add_drawable(vertices);
	// Add the drawable to the viewer
	viewer.add_drawable(surface);
	viewer.add_drawable(vertices);
	// Make sure everything is within the visible region of the viewer.
	viewer.fit_screen();
	// Run the viewer
	viewer.run();

	return 0;

}