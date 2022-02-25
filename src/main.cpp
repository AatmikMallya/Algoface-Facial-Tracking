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

string img_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.png"; //pose estimation
string land_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.land"; //pose estimation

//ctl + k + u/c

void createFaceObj(const vector<float>& faceVec, int numVerts, std::string pathToOutputObjFile) {

    // get the suffix
    std::ifstream infile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFace.obj");
    if (infile.fail()) {
        std::cerr << "ERROR: couldn't open the suffix file to read from" << endl;
        exit(-1);
    }
    std::string suffix((std::istreambuf_iterator<char>(infile)), (std::istreambuf_iterator<char>()));
    infile.close();

    std::ofstream outfile(pathToOutputObjFile);   // could be like "testing.obj"
    if (outfile.fail()) {
        std::cerr << "ERROR: couldn't open the sample output obj file to write to" << endl;
        exit(-1);
    }

    for (int i = 0; i < numVerts; i++) {
        size_t idx = i * 3;
        outfile << "v " << std::to_string(faceVec[idx]) << " " << std::to_string(faceVec[idx + 1]) << " " << std::to_string(faceVec[idx + 2]) << endl;
    }
    outfile << suffix;
    outfile.close();
}

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

    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/faces.obj"); //easy3D
    vector<easy3d::vec3> faceVerts = readFace3DFromObj(WAREHOUSE_PATH + "Tester_138/Blendshape/shape_22.obj"); //easy3D
    vector<int> all3dVertices = readVertexIdFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/lm_vert_internal_73.txt");   // same order as landmarks (Easy3D)
    int internal73[] = { 3984,10818,499,10543,413,3867,10574,9053,6698,1929,1927,6747,9205,7112,9380,3981,4277,10854,708,10742,4159,7135,9413,2138,2127,1986,6969,4437,760,4387,4346,10885,4370,766,4393,7330,7236,7275,9471,7271,7284,2191,7256,4227,294,279,3564,10461,8948,6418,6464,6441,6312,9236,8972,3262,3676,182,1596,1607,6575,1633,8864,6644,1790,3224,3270,251,1672,1621,6262,6162,10346,
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
    //**********************************
    //********************************** pose estimation

    /** Transform from object coordinates to camera coordinates **/
    // Copy Eigen vector to OpenCV vector

    //********************************** creating blendshapes
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

    }

    //********************************** 
    
    vector<cv::Point3f> combinedExp(n_vectors); //holds 1 expression made from 47 expressions 
    //73 vertices
    for (int i = 0; i < n_vectors; i++)
    {
        //47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            combinedExp[i].x +=  multExp[j][i].x * w[j];
            combinedExp[i].y +=  multExp[j][i].y * w[j];
            combinedExp[i].z +=  multExp[j][i].z * w[j];

        }
        //if (i == 1 || i == 2 || i == 3 || i == 4) {
        //    cout << combinedExp[i].x << " " << combinedExp[i].y << " " << combinedExp[i].z << endl;
        //}
    }
    
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
    //
    
    //**********


    //**********
    //cout << "creat optimized face" << endl;
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
            if (internal73[j] == i) // modifications of order from observation
            {
                largeFace[i + 1] = saveFace[j];
                largeFace[i + 2] = saveFace[j + 1];
                largeFace[i + 3] = saveFace[j + 2];
            }
        }
    }

    //for (int i = 0; i < 11510; i++)
    //{
    //    cout << largeFace[i] << " " << largeFace[i + 1] << " " << largeFace[i + 2] << endl;
    //}

    createFaceObj(largeFace, 11510, "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj");

    //cout << "reading face" << endl;

    //void createFaceObj(const Eigen::VectorXf & faceVec, int numVerts, std::string pathToOutputObjFile)
    vector<easy3d::vec3> optFace = readFace3DFromObj("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj"); //easy3D

    //cout << "face read" << endl;

    vector<easy3d::vec3> ez3dFace(vk.size());
    for (int i = 0; i < vk.size(); i++)
        ez3dFace[i] = optFace[vk[i]];

    //**********

    //--- always initialize viewer first before doing anything else for 3d visualization 
    //========================================================================================
    easy3d::Viewer viewer3("internal vertices");

    //------------------------- face surface mesh
    //===========================================================================
    auto surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts); //face obj of shape 22
    surface->update_element_buffer(meshIndices); //face obj without identifying vertices
    surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

    //------------------- vertices corresponding to landmarks
    //===========================================================================
    auto vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(ez3dFace); //lmVerts
    vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    vertices->set_point_size(10);

    //---------------------- add drawable objects to viewer
    //===========================================================================
    viewer3.add_drawable(surface);
    viewer3.add_drawable(vertices);
    // Add the drawable to the viewer
    viewer3.add_drawable(surface);
    viewer3.add_drawable(vertices);
    // Make sure everything is within the visible region of the viewer.
    viewer3.fit_screen();
    // Run the viewer
    viewer3.run();

    //**********************************
    //**********************************

    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        Sleep(1000);

    //**********************************
    //********************************** ceres optimization

    cv::Mat poseMat;
    cv::hconcat(rvec, tvec, poseMat);
    vector<double> poseVec(6, 0);

    poseVec[0] = rvec.at<double>(0);
    poseVec[1] = rvec.at<double>(1);
    poseVec[2] = rvec.at<double>(2);

    poseVec[3] = tvec.at<double>(0);
    poseVec[4] = tvec.at<double>(1);
    poseVec[5] = tvec.at<double>(2);


    for (int opt = 0; opt < 3; opt++) // multiple optimizations
    {
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
        cv::Mat poseMat;

        cv::hconcat(tvec, rvec, poseMat);

        cv::Mat flat = poseMat.reshape(1, poseMat.total() * poseMat.channels());
        std::vector<double> poseVec = poseMat.isContinuous() ? flat : flat.clone();

    }


    //for (int i = 0; i < numExpressions; i++)
    //{
    //    cout << w[i] << " ";
    //}
    //cout << endl;

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

    //**********************************
    //**********************************

    key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        Sleep(1000);

    //**********************************
    //********************************** creating 3D face


    //vector<int> vk(poseIndices.size());
    for (int i = 0; i < vk.size(); i++)
        vk[i] = all3dVertices[poseIndices[i]];

    vector<easy3d::vec3> lmVerts(vk.size());
    for (int i = 0; i < vk.size(); i++)
        lmVerts[i] = faceVerts[vk[i]];

    //**********
    //cout << "creat optimized face" << endl;
    //convert cv vector to eigen vector to put in object file to take out as a easy3d vector

    //basic raw tensor data of pose 138 open mouth stored in largeFace
    //vector<float> largeFace(11510 * 3);
    for (int i = 0; i < 11510; i++)
    {
        largeFace[(i * 3)] = rawTensor(137, 22, i).x();
        largeFace[(i * 3) + 1] = rawTensor(137, 22, i).y();
        largeFace[(i * 3) + 2] = rawTensor(137, 22, i).z();
    }

    //optimization of data for pose 138 to visualize stored in saveFace
    //vector<float> saveFace(n_vectors * 3);
    for (int i = 0; i < n_vectors; i++)
    {
        saveFace[(i * 3)] = combinedExp[i].x;
        saveFace[(i * 3) + 1] = combinedExp[i].y;
        saveFace[(i * 3) + 2] = combinedExp[i].z;
        //if (i == 1 || i == 2 || i == 3 || i == 4) {
        //    cout << combinedExp[i].x << " " << combinedExp[i].y << " " << combinedExp[i].z << endl;
        //}
    }

    //for (int i = 0; i < 11510; i++) //store saveFace vars into largeFace
    //{
    //    for (int j = 0; j < n_vectors; j++)
    //    {
    //        if (all3dVertices[poseIndices[j]] == i) // modifications of order from observation
    //        {
    //            largeFace[i + 1] = saveFace[j];
    //            largeFace[i + 2] = saveFace[j + 2];
    //            largeFace[i + 3] = saveFace[j + 1];
    //        }
    //    }
    //}

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

    //for (int i = 0; i < 11510; i++)
    //{
    //    cout << largeFace[i] << " " << largeFace[i + 1] << " " << largeFace[i + 2] << endl;
    //}

    createFaceObj(largeFace, 11510, "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj");

    //cout << "reading face" << endl;

    //void createFaceObj(const Eigen::VectorXf & faceVec, int numVerts, std::string pathToOutputObjFile)
    //vector<easy3d::vec3>
    optFace = readFace3DFromObj("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj"); //easy3D

    //cout << "face read" << endl;

    //vector<easy3d::vec3> ez3dFace(vk.size());
    for (int i = 0; i < vk.size(); i++)
        ez3dFace[i] = optFace[vk[i]];

    //**********
    
    //--- always initialize viewer first before doing anything else for 3d visualization 
    //========================================================================================
    easy3d::Viewer viewer("internal vertices");

    //------------------------- face surface mesh
    //===========================================================================
    //auto
    surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts); //face obj of shape 22
    surface->update_element_buffer(meshIndices); //face obj without identifying vertices
    surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

    //------------------- vertices corresponding to landmarks
    //===========================================================================
    //auto
    vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(ez3dFace); //lmVerts
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

    key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);
    
    //**********************************

    //**********
    
    //--- always initialize viewer first before doing anything else for 3d visualization 
    //========================================================================================
    easy3d::Viewer viewer2("internal vertices 2");

    //------------------------- face surface mesh
    //===========================================================================
    surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts); //face obj of shape 22
    surface->update_element_buffer(meshIndices); //face obj without identifying vertices
    surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

    //------------------- vertices corresponding to landmarks
    //===========================================================================
    vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(lmVerts); //lmVerts
    vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    vertices->set_point_size(10);

    //---------------------- add drawable objects to viewer
    //===========================================================================
    viewer2.add_drawable(surface);
    viewer2.add_drawable(vertices);
    // Add the drawable to the viewer
    viewer2.add_drawable(surface);
    viewer2.add_drawable(vertices);
    // Make sure everything is within the visible region of the viewer.
    viewer2.fit_screen();
    // Run the viewer
    viewer2.run();

    ////**********************************

    //key = cv::waitKey(0) % 256;
    //if (key == 27)                        // Esc button is pressed
    //    exit(1);

    ////**********************************

    /* //what does reserve and push_back do
    
    vector<easy3d::vec3> faceVerts;
    faceVerts.reserve(numDenseVerts); //

    for (int i = 0; i < numDenseVerts; i++)
    {
        float x = denseCombinedExp[i].x;
        float y = denseCombinedExp[i].y;
        float z = denseCombinedExp[i].z;

        faceVerts.push_back(easy3d::vec3(x, y, z)); //
    }
    easy3d::logging::initialize();
    
    */


    return 0;

}
