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

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>


using std::cout;
using std::endl;
using std::vector;

string WAREHOUSE_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/FaceWarehouse/";
string RAW_TENSOR_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/raw_tensor.bin";
string SHAPE_TENSOR_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/shape_tensor.bin";

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
    int numExpressions = 47;
    Eigen::VectorXf w(numExpressions); // weights in optimization

    for (int i = 0; i < numExpressions; i++)
    {
        w[i] = 0;
    }
    w[0] = 1;

    //Eigen::MatrixXf w_exp = w; // output
    //Eigen::VectorXf w; // weight vector
    //Eigen::MatrixXf prePose; //slice * weight
    //vector<cv::Point2f> lms; // 
    //std::vector<float> pose; //
    //cv::Mat image; //
    //float f; //

    //**********************************
    //********************************** pose estimation

    /** Transform from object coordinates to camera coordinates **/
    // Copy Eigen vector to OpenCV vector
    int n_vectors = 73;
    vector<cv::Point3f> singleExp(n_vectors); // holds 1 expression with 73 vertices used to create multExp
    vector<vector<cv::Point3f>> multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp
    
    //47 expressions
    for (int i = 0; i < numExpressions; i++)
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
        /*
        //73 vertices
        for (int j = 0; j < n_vectors; j++)
        {
            combinedExp[j].x = multExp[i][j].x * w[i];
            combinedExp[j].y = multExp[i][j].y * w[i];
            combinedExp[j].z = multExp[i][j].z * w[i];
        }
        */
    }

    
    vector<cv::Point3f> combinedExp(n_vectors); //holds 1 expression made from 47 expressions 
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
    //cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);

    //// Assuming no distortion
    //cv::Mat distCoeffs(4, 1, CV_64F);
    //distCoeffs.at<double>(0) = 0;
    //distCoeffs.at<double>(1) = 0;
    //distCoeffs.at<double>(2) = 0;
    //distCoeffs.at<double>(3) = 0;

    // Get rotation and translation parameters
    cv::Mat rvec(3, 1, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);


    cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec); //distCoeffs -> cv::Mat
    //
    //********************************** ceres optimization

    cv::Mat poseMat;
    cv::hconcat(rvec, tvec, poseMat);
    /*
    std::vector<float> poseVec(6);
    

    poseVec[0] = poseMat.at<float>(1, 1);
    poseVec[1] = poseMat.at<float>(2, 1);
    poseVec[2] = poseMat.at<float>(3, 1);
    poseVec[3] = poseMat.at<float>(4, 1);
    poseVec[4] = poseMat.at<float>(5, 1);
    poseVec[5] = poseMat.at<float>(6, 1);
    */
    cv::Mat flat = poseMat.reshape(1, poseMat.total() * poseMat.channels());
    std::vector<float>poseVec = poseMat.isContinuous() ? flat : flat.clone();
    
    for (int i = 0; i < 6; i++)
    {
        cout << poseVec[i] << endl;
    }


    bool boolOpt = optimize(multExp, lmsVec, poseVec, image, f, w); //first optimization

    for (int opt = 0; opt < 3; opt++) // multiple optimizations
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
        cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
        cv::Mat poseMat;

        cv::hconcat(tvec, rvec, poseMat);

        cv::Mat flat = poseMat.reshape(1, poseMat.total() * poseMat.channels());
        std::vector<float> poseVec = poseMat.isContinuous() ? flat : flat.clone();

        // optimization
        optimize(multExp, lmsVec, poseVec, image, f, w);


    }


    //if (boolOpt == true) {
    //    cout << "optimization worked" << endl;
    //}
    //else {
    //    cout << "optimization broke" << endl;
    //}

    //**********************************
    // Convert Euler angles to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Combine 3x3 rotation and 3x1 translation into 4x4 transformation matrix
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;
    T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;
    // Transform object
    std::vector<cv::Mat> cameraVec;
    for (auto& vec : combinedExp) {
        double data[4] = { vec.x, vec.y, vec.z, 1 };
        cv::Mat vector4d = cv::Mat(4, 1, CV_64F, data);
        cv::Mat result = T * vector4d;
        cameraVec.push_back(result);
    }

    // Project points onto image
    std::vector<cv::Point2f> imageVec;
    for (auto& vec : cameraVec) {
        cv::Point2f result;
        //result.x = fx * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
        //result.y = fx * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;
        result.x = f * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
        result.y = f * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;
        imageVec.push_back(result);
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
    



    //********************************** creating 3D face

    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/faces.obj"); //easy3D
    vector<easy3d::vec3> faceVerts = readFace3DFromObj(WAREHOUSE_PATH + "Tester_103/Blendshape/shape_22.obj"); //easy3D
    vector<int> all3dVertices = readVertexIdFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/lm_vert_internal_73.txt");   // same order as landmarks (Easy3D)
    
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
    //vector<int> poseIndices = { 27, 31, 35, 39, 54, 55, 61 };

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
    viewer.add_drawable(vertices);
    // Add the drawable to the viewer
    viewer.add_drawable(surface);
    viewer.add_drawable(vertices);
    // Make sure everything is within the visible region of the viewer.
    viewer.fit_screen();
    // Run the viewer
    viewer.run();

    //**********************************
    //**********************************
    /*
    key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);
    */
    //**********************************


    ////**********************************

    //key = cv::waitKey(0) % 256;
    //if (key == 27)                        // Esc button is pressed
    //    exit(1);

    ////**********************************

    return 0;

}
