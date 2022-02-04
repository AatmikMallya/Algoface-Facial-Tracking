#include <iostream>
#include <filesystem>
#include <vector>

#include <random>
#include "ceres/ceres.h"
//#include <glog/logging.h>
#include <functional>

#include "../include/tensor.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

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

//ceres library testing
// I tried changing x and y to vectors of vectors to store the raw tensor values
// I can't tell if I need x and y or what exactly this function does
// I want it to do this: stored face - (storedface * weight) = weighted face
struct Linear {

    Linear(int numObservations, const vector<vector<Eigen::Vector3f>>& x) {//, const vector<vector<Eigen::Vector3f>>& y) {
        _numObservations = numObservations;
        _x.resize(numObservations);
        //_y.resize(numObservations);
        std::copy(x.begin(), x.end(), _x.begin());
        //std::copy(y.begin(), y.end(), _y.begin());
    }

    template <typename T>
    bool operator()(const T* w, T* residual) const {

        for (int i = 0; i < _numObservations; i++)
            residual[i] = ((w[0] * T(_x[i]));

        return true;
    }

private:
    int                 _numObservations = 0;
    vector<vector<Eigen::Vector3f>>      _x;
    //vector<vector<Eigen::Vector3f>>      _y;
};
//

int main() {
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
    } else {
        buildShapeTensor(rawTensor, SHAPE_TENSOR_PATH, shapeTensor);
    }

    /** Transform from object coordinates to camera coordinates **/
    // Copy Eigen vector to OpenCV vector
    int n_vectors = 73;
    std::vector<cv::Point3f> objectVec(n_vectors);
    for (int i = 0; i < n_vectors; ++i) {
        Eigen::Vector3f eigen_vec = shapeTensor(102, 22, i); //tester103, pose1/expression22, vertices
        //cout << "shapeTensor: " << shapeTensor(0, 0, i) << endl;
        cv::Point3f cv_vec;
        cv_vec.x = eigen_vec.x();
        cv_vec.y = eigen_vec.y();
        cv_vec.z = eigen_vec.z();
        objectVec[i] = cv_vec;
    }


    // Image vector contains 2d landmark positions
    string img_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.png";
    string land_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.land";
    cv::Mat image = cv::imread(img_path, 1);
    std::vector<cv::Point2f> lmsVec = readLandmarksFromFile_2(land_path, image);

    double fx = 640, fy = 640, cx = 320, cy = 240;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    // Assuming no distortion
    cv::Mat distCoeffs(4, 1, CV_64F);
    distCoeffs.at<double>(0) = 0;
    distCoeffs.at<double>(1) = 0;
    distCoeffs.at<double>(2) = 0;
    distCoeffs.at<double>(3) = 0;

    // Get rotation and translation parameters
    cv::Mat rvec(3, 1, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);
    cv::solvePnP(objectVec, lmsVec, cameraMatrix, distCoeffs, rvec, tvec);

    // Convert Euler angles to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Combine 3x3 rotation and 3x1 translation into 4x4 transformation matrix
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;
    T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;
    // Transform object
    std::vector<cv::Mat> cameraVec;
    for (auto& vec : objectVec) {
        double data[4] = { vec.x, vec.y, vec.z, 1 };
        cv::Mat vector4d = cv::Mat(4, 1, CV_64F, data);
        cv::Mat result = T * vector4d;
        cameraVec.push_back(result);
    }

    // Project points onto image
    std::vector<cv::Point2f> imageVec;
    for (auto& vec : cameraVec) {
        cv::Point2f result;
        result.x = fx * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
        result.y = fx * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;
        imageVec.push_back(result);
    }
    cv::projectPoints(objectVec, rvec, tvec, cameraMatrix, distCoeffs, imageVec);


    cv::Mat visualImage = image.clone();
    double sc = 2;
    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
    for (int i = 0; i < imageVec.size(); i++) {
        //cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
//        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

        cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 0, 255), sc);             // 3d projections (red)
        cv::circle(visualImage, lmsVec[i] * sc, 1, cv::Scalar(0, 255, 0), sc);               // 2d landmarks   (green)

        //cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

    }
    cv::imshow("visualImage", visualImage);
    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        Sleep(1000);

    //****

    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/faces.obj");
    vector<easy3d::vec3> faceVerts = readFace3DFromObj(WAREHOUSE_PATH + "Tester_103/Blendshape/shape_22.obj");
    vector<int> all3dVertices = readVertexIdFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/lm_vert_internal_73.txt");   // same order as landmarks
    
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

    //****

    key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);

    //-************************************************
    //implement ceres library functions to be later moved to seperate files
    //ctl + k + u = uncomment
    //ctl + k + c = comment

    int numObservations = 47;
    vector<vector<Eigen::Vector3f>> x(numObservations);
    vector<vector<Eigen::Vector3f>> y(numObservations);

    vector<float> w(47); // vector of 3f vector containing 47 expressions from shapetensor
    for (int i = 0; i < 47; i++) {
        w[i] = 0; // = shapeTensor(137, 22, i); //tester138, pose1/expression22, all vertices
    }
    w[0] = 1;

    vector<vector<Eigen::Vector3f>> identity(47); // vector of 3f vector containing 34020 identities from rawtensor
    for (int i = 0; i < 47; i++) {
        for (int j = 0; j < 11510; j++) { // 3836 * 3 = 11510
            Eigen::Vector3f temp = rawTensor(137, i, j); //tester138, all expressions, all vertices
            identity[i][j] = temp;
        }
    }

    /*for (int i = 0; i < 47; i++) {
        //cout << i << " expressions: " << expressions[i] << endl;
        cout << i << " identity: " << identity[i] << endl;
    }*/

    //cout  << "size: " << expressions.size() << endl;
    //cout << "size: " << identity.size() << endl;
    
    for (int i = 0; i < numObservations; i++) {
        for (int j = 0; j < numObservations; j++) {
            x[i][j] = identity[i][j] * w[i];
        }

        //y[i] = 2 * x[i] - 3 + rand() * 1.0 / RAND_MAX;
        //y[i] = identity[i];
    }

    //vector<double> w(47);
    //fill(w.begin(), w.end(), 0);
    //w[0] = 1;

    ceres::Problem problem;
    //Linear* lin = new Linear(numObservations, x, y);
    Linear* lin = new Linear(numObservations, x);

    ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Linear, ceres::DYNAMIC, 1>(lin, numObservations);
    //ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Exponential, ceres::DYNAMIC, 3>(exp, numObservations);
    //ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Linear, 47, 1>(lin);

    problem.AddResidualBlock(costFunction, NULL, &w[0]);

    //problem.SetParameterLowerBound(&w[0], 0, 0);  
    //problem.SetParameterUpperBound(&w[0], 0, 4);

    //problem.SetParameterLowerBound(&w[0], 1, -5); 
    //problem.SetParameterUpperBound(&w[0], 1, -1);

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl << endl;
    for (int i = 0; i < w.size(); i++)
        cout << "i = " << i << ", w = " << w[i] << endl;

    //-************************************************


    //

    key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);

    return 0;

}




//    std::ofstream file ("test.obj");
//    for (int k = 0; k < 73; k++) {
//        Eigen::Vector3f v = shapeTensor(0, 0, k);
//        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
//    }
//    file.close();
