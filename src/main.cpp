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

    //********************************** variable initilization of weights and easy3d
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

    string img_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.png"; //pose estimation
    string land_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.land"; //pose estimation

    /** Transform from object coordinates to camera coordinates **/
    // Copy Eigen vector to OpenCV vector

    //********************************** creating blendshapes
    int n_vectors = 73;
    vector<cv::Point3f> singleExp(n_vectors); // holds 1 expression with 73 vertices used to create multExp
    vector<vector<cv::Point3f>> multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp

    //turn shapeTensor face 137 from eigen to cv vector
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

    }

    //create combined face from vertices and expressions
    vector<cv::Point3f> combinedExp(n_vectors); //holds 1 expression made from 47 expressions 
    //73 vertices
    for (int i = 0; i < n_vectors; i++)
    {
        //47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            combinedExp[i].x += multExp[j][i].x * w[j];
            combinedExp[i].y += multExp[j][i].y * w[j];
            combinedExp[i].z += multExp[j][i].z * w[j];

        }
        //if (i == 1 || i == 2 || i == 3 || i == 4) {
        //    cout << combinedExp[i].x << " " << combinedExp[i].y << " " << combinedExp[i].z << endl;
        //}
    }

    //********************************** pose estimation

    // Image vector contains 2d landmark positions
    cv::Mat image = cv::imread(img_path, 1);
    std::vector<cv::Point2f> lmsVec = readLandmarksFromFile_2(land_path, image);

    double f = image.cols;               // ideal camera where fx ~ fy
    double cx = image.cols / 2.0;
    double cy = image.rows / 2.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);

    // Get rotation and translation parameters
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); //64 = double, 32 = float
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);


    cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec); //distCoeffs -> cv::Mat


    //********************************** ceres optimization w/ pose estimation

    cv::Mat poseMat;
    cv::hconcat(rvec, tvec, poseMat);
    vector<double> poseVec(6, 0);

    poseVec[0] = rvec.at<double>(0);
    poseVec[1] = rvec.at<double>(1);
    poseVec[2] = rvec.at<double>(2);

    poseVec[3] = tvec.at<double>(0);
    poseVec[4] = tvec.at<double>(1);
    poseVec[5] = tvec.at<double>(2);


    // optimization
    optimize(multExp, lmsVec, poseVec, image, f, w); // looped optimization

    // create new combined face based on weights
    for (int i = 0; i < 73; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {

            combinedExp[i].x += multExp[j][i].x * w[j];
            combinedExp[i].y += multExp[j][i].y * w[j];
            combinedExp[i].z += multExp[j][i].z * w[j];
        }
        //if (i == 1 || i == 2 || i == 3 || i == 4) {
        //    cout << combinedExp[i].x << " " << combinedExp[i].y << " " << combinedExp[i].z << endl;
        //}
    }

    //pose estimation
    cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
    //cv::Mat poseMat;

    cv::hconcat(tvec, rvec, poseMat);

    //cv::Mat flat = poseMat.reshape(1, poseMat.total() * poseMat.channels());
    //std::vector<double> poseVec = poseMat.isContinuous() ? flat : flat.clone();


    //pose estimation image
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
        result.x = f * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
        result.y = f * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;
        imageVec.push_back(result);
    }

    /*
    // 2d visualization
    cv::projectPoints(combinedExp, rvec, tvec, cameraMatrix, cv::Mat(), imageVec); //distCoeffs ->cv::Mat()

    cv::Mat visualImage = image.clone();
    double sc = 2; //size of displayed picture
    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
    for (int i = 0; i < imageVec.size(); i++) {
        //cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
        //cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

        cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 0, 255), sc);             // 3d projections (red)
        cv::circle(visualImage, lmsVec[i] * sc, 1, cv::Scalar(0, 255, 0), sc);               // 2d landmarks   (green)

        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

    }
    cv::imshow("visualImage", visualImage);
    */

    /*
    // wait block
    //**********************************

    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        Sleep(1000);

    //**********************************
    */

    /*
    //**********
    //convert cv vector to eigen vector to put in object file to take out as a easy3d vector
    //basic raw tensor data of pose 138 open mouth stored in largeFace
    vector<float> largeFace(11510 * 3);
    for (int i = 0; i < 11510; i++)
    {
        largeFace[(i * 3)] = rawTensor(137, 22, i).x();
        largeFace[(i * 3) + 1] = rawTensor(137, 22, i).y();
        largeFace[(i * 3) + 2] = rawTensor(137, 22, i).z();
    }
    //optimization of data for pose 138 to visualize stored in saveFace
    vector<float> saveFace(n_vectors * 3);
    for (int i = 0; i < n_vectors; i++)
    {
        saveFace[(i * 3)] = combinedExp[i].x;
        saveFace[(i * 3) + 1] = combinedExp[i].y;
        saveFace[(i * 3) + 2] = combinedExp[i].z;
        //if (i == 1 || i == 2 || i == 3 || i == 4) {
        //    cout << combinedExp[i].x << " " << combinedExp[i].y << " " << combinedExp[i].z << endl;
        //}
    }
    for (int i = 0; i < 11510; i++) //store saveFace vars into largeFace
    {
        for (int j = 0; j < n_vectors; j++)
        {
            if (all3dVertices[poseIndices[j]] == i) // modifications of order from observation
            {
                largeFace[i + 1] = saveFace[j];
                largeFace[i + 2] = saveFace[j + 2];
                largeFace[i + 3] = saveFace[j + 1];
            }
        }
    }
    for (int i = 0; i < 11510; i++) //store saveFace vars into largeFace
    {
        for (int j = 0; j < n_vectors; j++)
        {
            if (internal73[j] == i) // modifications of order from observation
            {
                largeFace[i + 1] = saveFace[j];
                largeFace[i + 2] = saveFace[j + 1];
                largeFace[i + 3] = saveFace[j + 2];
            }
        }
    }
    //**********
    */

    //save face to file

    int n_raw_vectors = 11510;
    vector<cv::Point3f> raw_singleExp(n_raw_vectors); // holds 1 expression with 73 vertices used to create multExp
    vector<vector<cv::Point3f>> raw_multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp
    for (int i = 0; i < numExpressions; i++)//numExpresion is 47
    {
        //11510 vertices
        for (int j = 0; j < n_raw_vectors; j++) {
            Eigen::Vector3f raw_tens_vec = rawTensor(137, i, j);
            cv::Point3f raw_cv_vec;
            raw_cv_vec.x = raw_tens_vec.x();
            raw_cv_vec.y = raw_tens_vec.y();
            raw_cv_vec.z = raw_tens_vec.z();
            raw_singleExp[j] = raw_cv_vec;
        }
        raw_multExp[i] = raw_singleExp;
    }
    vector<cv::Point3f> raw_combinedExp(n_raw_vectors); //holds 1 expression made from 47 expressions (neutral expression)
    for (int i = 0; i < n_raw_vectors; i++)
    {
        //47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            raw_combinedExp[i].x += raw_multExp[j][i].x * w[j];
            raw_combinedExp[i].y += raw_multExp[j][i].y * w[j];
            raw_combinedExp[i].z += raw_multExp[j][i].z * w[j];
        }
    }

    createFaceObj(raw_combinedExp, 11510, "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj");



    //********************************** creating 3D face in easy3d

    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFace.obj"); //easy3D
    //vector<easy3d::vec3> faceVerts = readFace3DFromObj(WAREHOUSE_PATH + "Tester_138/Blendshape/shape_22.obj"); //easy3D
    vector<easy3d::vec3> faceVerts = readFace3DFromObj("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj"); //easy3D
    vector<int> all3dVertices = readVertexIdFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/lm_vert_internal_73.txt");   // same order as landmarks (Easy3D)

    int internal73[] = { 3984,10818,499,10543,413,3867,10574,9053,6698,1929,1927,6747,9205,7112,9380,3981,4277,10854,708,
        10742,4159,7135,9413,2138,2127,1986,6969,4437,760,4387,4346,10885,4370,766,4393,7330,7236,7275,9471,7271,7284,2191,
        7256,4227,294,279,3564,10461,8948,6418,6464,6441,6312,9236,8972,3262,3676,182,1596,1607,6575,1633,8864,6644,1790,
        3224,3270,251,1672,1621,6262,6162,10346
    };

    vector<int> c0 = { 3975, 10739, 4223, 10823, 4493, 10254, 5492, 10100, 5196, 10101, 2853, 10104, 5197, 10116 };//14
    vector<int> c1 = { 10757, 4152, 10760, 4215, 10828, 4505, 10960, 5498, 11289, 5169, 11206, 4976 };//13
    vector<int> c2 = { 10646, 3918, 10650, 5547, 1339, 5549, 1340, 5507, 1320, 5509, 1146, 5161, 1053 };//13
    vector<int> c3 = { 393, 3637, 499, 3874, 508, 3844, 493, 5512, 1321, 5510, 1150, 5040 };//12
    vector<int> c4 = { 485, 3701, 424, 3699, 501, 3861, 510, 3879, 497, 3855, 1323, 5178, 1154, 5117, 1126 };//15
    vector<int> c5 = { 10527, 3831, 10536, 3613, 10539, 3865, 10633, 3883, 10614, 3839 };//10
    vector<int> c6 = { 10575, 3812, 10574, 3658, 10573, 3869, 10635, 3884, 10623, 3848 };//10
    vector<int> c7 = { 9137, 3607, 9053, 6703, 9150, 3611, 9045, 3872, 9180, 3887, 9190, 6725 };//12
    vector<int> c8 = { 9079, 6698, 9148, 6541, 9078, 6755, 9178, 6773, 9188, 6733 };//10
    vector<int> c9 = { 9105, 6723, 9159, 6505, 9104, 6754, 9176, 6768, 9186, 6727 };//10
    vector<int> c10 = { 6578, 9100, 6718, 9157, 6577, 9099, 6751, 9174, 6764, 9184, 6738 };//11
    vector<int> c11 = { 1908, 6531, 1818, 6742, 1924, 6744, 1933, 6759, 1918, 8386, 2746, 8388, 2575 };//13
    vector<int> c12 = { 2762, 8419, 2763, 8421, 2764, 8424, 2765, 8426, 2745, 8385, 2571 };//11
    vector<int> c13 = { 2089, 7037, 2071, 7111, 2110, 7113, 2246, 8377, 2741, 8375, 2576, 7920, 2514 };//13
    vector<int> c14 = { 6865, 2112, 7117, 2248, 5942, 1528, 5940, 1425, 5733, 1426, 5731, 1432, 5746 };//13

    vector<int> allcs = { 3975, 10739, 4223, 10823, 4493, 10254, 5492, 10100, 5196, 10101, 2853, 10104, 5197, 10116,
        10757, 4152, 10760, 4215, 10828, 4505, 10960, 5498, 11289, 5169, 11206, 4976,
        10646, 3918, 10650, 5547, 1339, 5549, 1340, 5507, 1320, 5509, 1146, 5161, 1053,
        393, 3637, 499, 3874, 508, 3844, 493, 5512, 1321, 5510, 1150, 5040,
        485, 3701, 424, 3699, 501, 3861, 510, 3879, 497, 3855, 1323, 5178, 1154, 5117, 1126,
        10527, 3831, 10536, 3613, 10539, 3865, 10633, 3883, 10614, 3839,
        10575, 3812, 10574, 3658, 10573, 3869, 10635, 3884, 10623, 3848,
        9137, 3607, 9053, 6703, 9150, 3611, 9045, 3872, 9180, 3887, 9190, 6725,
        9079, 6698, 9148, 6541, 9078, 6755, 9178, 6773, 9188, 6733,
        9105, 6723, 9159, 6505, 9104, 6754, 9176, 6768, 9186, 6727,
        6578, 9100, 6718, 9157, 6577, 9099, 6751, 9174, 6764, 9184, 6738,
        1908, 6531, 1818, 6742, 1924, 6744, 1933, 6759, 1918, 8386, 2746, 8388, 2575,
        2762, 8419, 2763, 8421, 2764, 8424, 2765, 8426, 2745, 8385, 2571,
        2089, 7037, 2071, 7111, 2110, 7113, 2246, 8377, 2741, 8375, 2576, 7920, 2514,
        6865, 2112, 7117, 2248, 5942, 1528, 5940, 1425, 5733, 1426, 5731, 1432, 5746
    };

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
    {
        vk[i] = all3dVertices[poseIndices[i]];
    }

    vector<easy3d::vec3> contourVerts(allcs.size());
    for (int i = 0; i < contourVerts.size(); i++)
    {
        contourVerts[i] = faceVerts[allcs[i]];
    }

    vector<easy3d::vec3> lmVerts(vk.size());
    for (int i = 0; i < vk.size(); i++)
        lmVerts[i] = faceVerts[vk[i]];

    //vk holds internal73 numbers
    //lmVerts holds point3f of internal 73 numbers
    //faceVerts holds point3f

    for (int i = 0; i < vk.size(); i++)
    {
        //cout << lmVerts[i] << ", " << vk[i] << ", " << endl;
        //cout << yawVerts[i] << ", " << allints[i] << ". " << endl;
    }


    //--- always initialize viewer first before doing anything else for 3d visualization 
    //========================================================================================
    //create default East3D viewer, must be created before drawables
    easy3d::Viewer viewer("3d visualization");

    //------------------------- face surface mesh
    //===========================================================================
    auto surface = new easy3d::TrianglesDrawable("faces");
    //upload vertex positions of surface to GPU
    surface->update_vertex_buffer(faceVerts); //face obj of optimized + pose estimated face
    //upload vertex indices of the surface to GPU
    surface->update_element_buffer(meshIndices); //face obj without identifying vertices
    //set color of surface
    surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

    //------------------- vertices corresponding to landmarks
    //===========================================================================
    auto vertices = new easy3d::PointsDrawable("vertices");
    //upload vertex positions to GPU
    vertices->update_vertex_buffer(contourVerts); //lmVerts
    vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));//color of vertices
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE); //shape of vertices
    vertices->set_point_size(10); //size in pixels of vertices

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
  
    ////**********************************

    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1); //Sleep(1000)

    ////**********************************

    return 0;

}
