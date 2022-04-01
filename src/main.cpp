#include <iostream>
#include <filesystem>
#include <vector>

#include "glog/logging.h"

#include <functional>
#include "../include/optimization.h"
#include "../include/identityOptimization.h"
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

//cout << "line: " << __LINE__ << endl;
//save file to git. debug. using "git diff" you can see debug info added
//git stash save "debugging info" will revert file to it's last save and stash the debug info
//git stash apply applies the last stash
//git reset --hard resets file to last save
//git stash drop drops the most recent stash
//git stash list

//if you make a function, you can git commit to keep
//starting new repo:
//git init, git status, git add *, git commit -m "message"

//**************************************
//above
//**************************************

void visualizeFace(tensor3 rawTensor, Eigen::VectorXf w, Eigen::VectorXf identity_w);
void createAllExpressions(tensor3 tensor, Eigen::VectorXf identity_w, int numVerts, std::vector<std::vector<cv::Point3f>>& avgMultExp);
void createAllIdentities(tensor3 tensor, Eigen::VectorXf w, int numVerts, std::vector<std::vector<cv::Point3f>>& allIdnOptExp);
void linearCombination(int numVerts, int numCombinations, std::vector<std::vector<cv::Point3f>> mult, Eigen::VectorXf w, std::vector<cv::Point3f>& linCombo);
void visualization3D(int numVerts, std::vector<cv::Point3f> linCombo);
void getPose(std::vector<double>& poseVec, const cv::Mat& rvec, const cv::Mat& tvec);

int main() {

    //create or load tensors

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

    //variable initilization
    int numIdentities = 150;
    int numExpressions = 47;
    int n_vectors = 73;
    int numDenseVerts = 11510;

    //paths
    string img_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.png"; //pose estimation path
    string land_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.land"; //pose estimation path

    //pose estimation setup
    cv::Mat image = cv::imread(img_path, 1); // Image vector contains 2d landmark positions (pose estimation)
    vector<cv::Point2f> lmsVec = readLandmarksFromFile_2(land_path, image); //ground truth 2d landmarks (pose estimation)

    double f = image.cols;               // ideal camera where fx ~ fy
    double cx = image.cols / 2.0;
    double cy = image.rows / 2.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);

    // Get rotation and translation parameters
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); //64 = double, 32 = float
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    //variable initilization of weights (47)
    Eigen::VectorXf w(numExpressions); // weights in optimization
    for (int i = 0; i < numExpressions; i++)
    {
        w[i] = 0;
    }
    w[0] = 1;

    // vector representing 150 identity weights
    Eigen::VectorXf identity_w(numIdentities);
    for (int i = 0; i < numIdentities; i++)
    {
        identity_w[i] = 1 / 150.0;
    }

    
    //for (int opt = 0; opt < 2; opt++)
    //{
        vector<vector<cv::Point3f>> avgMultExp(numExpressions);
    
        vector<cv::Point3f> singleIdn(n_vectors);
        vector<vector<cv::Point3f>> multIdn(numIdentities);
        vector<cv::Point3f> combinedIdn(n_vectors);

        for (int j = 0; j < numExpressions; j++)
        {
            // 150 identities
            for (int k = 0; k < numIdentities; k++)
            {
                // 73 vertices
                for (int i = 0; i < n_vectors; i++)
                {
                    Eigen::Vector3f tens_vec = shapeTensor(k, j, i); //22 = open mouth
                    cv::Point3f conv_vec;
                    conv_vec.x = tens_vec.x();
                    conv_vec.y = tens_vec.y();
                    conv_vec.z = tens_vec.z();
                    singleIdn[i] = conv_vec;
                }
                multIdn[k] = singleIdn;
            }

            // create an average face
            for (int i = 0; i < n_vectors; i++)
            {
                for (int j = 0; j < numIdentities; j++)
                {
                    combinedIdn[i].x += (multIdn[j][i].x * identity_w[j]);
                    combinedIdn[i].y += (multIdn[j][i].y * identity_w[j]);
                    combinedIdn[i].z += (multIdn[j][i].z * identity_w[j]);
                }
            }
            avgMultExp[j] = combinedIdn;
        }



        //*************************
        //top
        //*************************


        // pose estimation for neutral expression
        cv::solvePnP(avgMultExp[0], lmsVec, cameraMatrix, cv::Mat(), rvec, tvec); //distCoeffs -> cv::Mat

        //cv::Mat poseMat;
        //cv::hconcat(rvec, tvec, poseMat);
        vector<double> poseVec(6, 0);
        getPose(poseVec, rvec, tvec);

        visualizeFace(rawTensor, w, identity_w);

        // optimization expression
        optimize(avgMultExp, lmsVec, poseVec, image, f, w);
        /*
        vector<vector<cv::Point3f>> allIdn(numIdentities);

        vector<cv::Point3f> singleExp(n_vectors); // holds 1 expression with 73 vertices used to create multExp
        vector<vector<cv::Point3f>> multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp
        vector<cv::Point3f> combinedExp(n_vectors); //holds 1 expression made from 47 expressions

        for (int k = 0; k < numIdentities; k++)
        {
            //turn shapeTensor face 137 from eigen to cv vector
            //47 expressions
            for (int j = 0; j < numExpressions; j++)
            {
                //73 vertices
                for (int i = 0; i < n_vectors; i++) {
                    Eigen::Vector3f tens_vec = shapeTensor(k, j, i); //137 = tall face
                    cv::Point3f cv_vec;
                    cv_vec.x = tens_vec.x();
                    cv_vec.y = tens_vec.y();
                    cv_vec.z = tens_vec.z();
                    singleExp[i] = cv_vec;
                }
                multExp[j] = singleExp;
            }


            //create combined face from vertices of expressions based on weights
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
            }

            allIdn[k] = combinedExp;
        }
        */

        std::vector<std::vector<cv::Point3f>> allIdnOptExp(numIdentities);
        createAllIdentities(shapeTensor, w, n_vectors, allIdnOptExp);
        linearCombination(n_vectors, numIdentities, allIdnOptExp, identity_w, combinedIdn);
        cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
        getPose(poseVec, rvec, tvec);

        /* optimize for identity */
        identityOptimize(allIdnOptExp, lmsVec, poseVec, image, f, identity_w);

        visualizeFace(rawTensor, w, identity_w);

        createAllExpressions(shapeTensor, identity_w, n_vectors, avgMultExp);
        linearCombination(n_vectors, numExpressions, avgMultExp, w, combinedIdn);


        cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
        getPose(poseVec, rvec, tvec);

        // expression optimization
        optimize(avgMultExp, lmsVec, poseVec, image, f, w);

        // create new face based on optimized expression weights
        createAllIdentities(shapeTensor, w, n_vectors, allIdnOptExp);
        linearCombination(n_vectors, numIdentities, allIdnOptExp, identity_w, combinedIdn);
        cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
        getPose(poseVec, rvec, tvec);

        /* optimize for identity */
        identityOptimize(allIdnOptExp, lmsVec, poseVec, image, f, identity_w);

    //}

    //printing information on weights
    cout << "identity: " << endl;
    int sumW = 0;
    for (int i = 0; i < numIdentities; i++)
    {
        sumW += identity_w[i];
        cout << i << "__" << identity_w[i] << ",  ";
        if (i % 5 == 0)
            cout << endl;
    }
    cout << endl << "sum__" << sumW << endl;
    cout << "expression: " << endl;
    cout << "sum__" << w[0] << endl;
    for (int i = 1; i < numExpressions; i++)
    {
        cout << i - 1 << "__" << w[i] << ",  ";
        if (i % 5 == 0)
            cout << endl;
    }
    cout << endl;


    // Everything below this point is for generating a face

    //**************************************
    //middle
    //**************************************

    /*

    vector<cv::Point3f> combinedExp(numDenseVerts); //holds 1 expression made from 11510 expressions (neutral expression)
    for (int i = 0; i < numDenseVerts; i++)
    {
        //47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            combinedExp[i].x += multExp[j][i].x * w[j];
            combinedExp[i].y += multExp[j][i].y * w[j];
            combinedExp[i].z += multExp[j][i].z * w[j];
        }
    }
    rawOpt[k] = combinedExp;
    //vector<cv::Point3f> singleFace = rawOpt[137];

    */

    //save identity and expression face to file
    vector<vector<cv::Point3f>> allExpD(numExpressions);
    vector<cv::Point3f> allIdnD(numDenseVerts); // 11510 x 47

    for (int j = 0; j < numExpressions; j++)
    {
        std::vector<cv::Point3f> singleIdn(numDenseVerts);
        std::vector<std::vector<cv::Point3f>> multIdn(numIdentities);
        // 150 identities
        for (int k = 0; k < numIdentities; k++)
        {
            for (int i = 0; i < numDenseVerts; i++)
            {
                Eigen::Vector3f tens_vec = rawTensor(k, j, i);
                cv::Point3f conv_vec;
                conv_vec.x = tens_vec.x();
                conv_vec.y = tens_vec.y();
                conv_vec.z = tens_vec.z();
                singleIdn[i] = conv_vec;
            }
            multIdn[k] = singleIdn;
        }
        // create an average face
        std::vector<cv::Point3f> combinedIdn(numDenseVerts);
        for (int i = 0; i < numDenseVerts; i++)
        {
            for (int j = 0; j < numIdentities; j++)
            {
                combinedIdn[i].x += (multIdn[j][i].x * identity_w[j]);
                combinedIdn[i].y += (multIdn[j][i].y * identity_w[j]);
                combinedIdn[i].z += (multIdn[j][i].z * identity_w[j]);
            }
        }
        allExpD[j] = combinedIdn;
    }
    
    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {
            allIdnD[i].x += (allExpD[j][i].x * w[j]);
            allIdnD[i].y += (allExpD[j][i].y * w[j]);
            allIdnD[i].z += (allExpD[j][i].z * w[j]);
        }
    }


    createFaceObj(allIdnD, 11510, "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj");



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

    //turn into matrix instead of vectors probably
    vector<vector<int>> allcs = {
      { 3975, 10739, 4223, 10823, 4493, 10254, 5492, 10100, 5196, 10101, 2853, 10104, 5197, 10116 }//14
    , { 10757, 4152, 10760, 4215, 10828, 4505, 10960, 5498, 11289, 5169, 11206, 4976 }//13
    , { 10646, 3918, 10650, 5547, 1339, 5549, 1340, 5507, 1320, 5509, 1146, 5161, 1053 }//13
    , { 393, 3637, 499, 3874, 508, 3844, 493, 5512, 1321, 5510, 1150, 5040 }//12
    , { 485, 3701, 424, 3699, 501, 3861, 510, 3879, 497, 3855, 1323, 5178, 1154, 5117, 1126 }//15
    , { 10527, 3831, 10536, 3613, 10539, 3865, 10633, 3883, 10614, 3839 }//10
    , { 10575, 3812, 10574, 3658, 10573, 3869, 10635, 3884, 10623, 3848 }//10
    , { 9137, 3607, 9053, 6703, 9150, 3611, 9045, 3872, 9180, 3887, 9190, 6725 }//12
    , { 9079, 6698, 9148, 6541, 9078, 6755, 9178, 6773, 9188, 6733 }//10
    , { 9105, 6723, 9159, 6505, 9104, 6754, 9176, 6768, 9186, 6727 }//10
    , { 6578, 9100, 6718, 9157, 6577, 9099, 6751, 9174, 6764, 9184, 6738 }//11
    , { 1908, 6531, 1818, 6742, 1924, 6744, 1933, 6759, 1918, 8386, 2746, 8388, 2575 }//13
    , { 2762, 8419, 2763, 8421, 2764, 8424, 2765, 8426, 2745, 8385, 2571 }//11
    , { 2089, 7037, 2071, 7111, 2110, 7113, 2246, 8377, 2741, 8375, 2576, 7920, 2514 }//13
    , { 6865, 2112, 7117, 2248, 5942, 1528, 5940, 1425, 5733, 1426, 5731, 1432, 5746 } };//13

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

    //landmark vertices ordering
    //vk holds internal73 numbers
    vector<int> vk(poseIndices.size());
    for (int i = 0; i < vk.size(); i++)
    {
        vk[i] = all3dVertices[poseIndices[i]];
    }

    //contour vertices
    //faceVerts holds point3f
    vector<easy3d::vec3> contourVerts(allcs.size());
    for (int i = 0; i < contourVerts.size(); i++)
    {
        contourVerts[i] = faceVerts[allcs[i][0]];
    }

    //**************************************
    //bottom
    //**************************************

    //--- always initialize viewer first before doing anything else for 3d visualization 
    //create default East3D viewer, must be created before drawables

    easy3d::Viewer viewer("3d visualization");

    // surface mesh given a buffer of a face
    auto surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts); //try faceVerts
    surface->update_element_buffer(meshIndices);
    surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

    // vertices corresponding to given buffer with given characteristics
    auto vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(contourVerts); //try lmVerts, contourVerts
    vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    vertices->set_point_size(10); //size in pixels of vertices

    // add drawable objects to viewer
    viewer.add_drawable(surface);
    //viewer.add_drawable(vertices); //comment to toggle vertices

    // Add the drawable to the viewer
    viewer.add_drawable(surface);
    //viewer.add_drawable(vertices); //comment to toggle vertices
    
    viewer.fit_screen(); // Make sure everything is within the visible region of the viewer.
    viewer.run(); // Run the viewer

    //**********************************
  
    ////**********************************

    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1); //Sleep(1000)

    ////**********************************

    return 0;

}

void visualizeFace(tensor3 rawTensor, Eigen::VectorXf w, Eigen::VectorXf identity_w)
{
    int numDenseVerts = 11510;
    int numIdentities = 150;
    int numExpressions = 47;

    vector<vector<cv::Point3f>> allExpD(numExpressions);
    vector<cv::Point3f> allIdnD(numDenseVerts); // 11510 x 47

    for (int j = 0; j < numExpressions; j++)
    {
        std::vector<cv::Point3f> singleIdn(numDenseVerts);
        std::vector<std::vector<cv::Point3f>> multIdn(numIdentities);
        // 150 identities
        for (int k = 0; k < numIdentities; k++)
        {
            for (int i = 0; i < numDenseVerts; i++)
            {
                Eigen::Vector3f tens_vec = rawTensor(k, j, i);
                cv::Point3f conv_vec;
                conv_vec.x = tens_vec.x();
                conv_vec.y = tens_vec.y();
                conv_vec.z = tens_vec.z();
                singleIdn[i] = conv_vec;
            }
            multIdn[k] = singleIdn;
        }
        // create an average face
        std::vector<cv::Point3f> combinedIdn(numDenseVerts);
        for (int i = 0; i < numDenseVerts; i++)
        {
            for (int j = 0; j < numIdentities; j++)
            {
                combinedIdn[i].x += (multIdn[j][i].x * identity_w[j]);
                combinedIdn[i].y += (multIdn[j][i].y * identity_w[j]);
                combinedIdn[i].z += (multIdn[j][i].z * identity_w[j]);
            }
        }
        allExpD[j] = combinedIdn;
    }

    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {
            allIdnD[i].x += (allExpD[j][i].x * w[j]);
            allIdnD[i].y += (allExpD[j][i].y * w[j]);
            allIdnD[i].z += (allExpD[j][i].z * w[j]);
        }
    }


    createFaceObj(allIdnD, 11510, "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj");



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

    //turn into matrix instead of vectors probably
    vector<vector<int>> allcs = {
      { 3975, 10739, 4223, 10823, 4493, 10254, 5492, 10100, 5196, 10101, 2853, 10104, 5197, 10116 }//14
    , { 10757, 4152, 10760, 4215, 10828, 4505, 10960, 5498, 11289, 5169, 11206, 4976 }//13
    , { 10646, 3918, 10650, 5547, 1339, 5549, 1340, 5507, 1320, 5509, 1146, 5161, 1053 }//13
    , { 393, 3637, 499, 3874, 508, 3844, 493, 5512, 1321, 5510, 1150, 5040 }//12
    , { 485, 3701, 424, 3699, 501, 3861, 510, 3879, 497, 3855, 1323, 5178, 1154, 5117, 1126 }//15
    , { 10527, 3831, 10536, 3613, 10539, 3865, 10633, 3883, 10614, 3839 }//10
    , { 10575, 3812, 10574, 3658, 10573, 3869, 10635, 3884, 10623, 3848 }//10
    , { 9137, 3607, 9053, 6703, 9150, 3611, 9045, 3872, 9180, 3887, 9190, 6725 }//12
    , { 9079, 6698, 9148, 6541, 9078, 6755, 9178, 6773, 9188, 6733 }//10
    , { 9105, 6723, 9159, 6505, 9104, 6754, 9176, 6768, 9186, 6727 }//10
    , { 6578, 9100, 6718, 9157, 6577, 9099, 6751, 9174, 6764, 9184, 6738 }//11
    , { 1908, 6531, 1818, 6742, 1924, 6744, 1933, 6759, 1918, 8386, 2746, 8388, 2575 }//13
    , { 2762, 8419, 2763, 8421, 2764, 8424, 2765, 8426, 2745, 8385, 2571 }//11
    , { 2089, 7037, 2071, 7111, 2110, 7113, 2246, 8377, 2741, 8375, 2576, 7920, 2514 }//13
    , { 6865, 2112, 7117, 2248, 5942, 1528, 5940, 1425, 5733, 1426, 5731, 1432, 5746 } };//13

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

    //landmark vertices ordering
    //vk holds internal73 numbers
    vector<int> vk(poseIndices.size());
    for (int i = 0; i < vk.size(); i++)
    {
        vk[i] = all3dVertices[poseIndices[i]];
    }

    //contour vertices
    //faceVerts holds point3f
    vector<easy3d::vec3> contourVerts(allcs.size());
    for (int i = 0; i < contourVerts.size(); i++)
    {
        contourVerts[i] = faceVerts[allcs[i][0]];
    }

    //**************************************
    //bottom
    //**************************************

    //--- always initialize viewer first before doing anything else for 3d visualization 
    //create default East3D viewer, must be created before drawables

    easy3d::Viewer viewer("3d visualization");

    // surface mesh given a buffer of a face
    auto surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts); //try faceVerts
    surface->update_element_buffer(meshIndices);
    surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

    // vertices corresponding to given buffer with given characteristics
    auto vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(contourVerts); //try lmVerts, contourVerts
    vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    vertices->set_point_size(10); //size in pixels of vertices

    // add drawable objects to viewer
    viewer.add_drawable(surface);
    //viewer.add_drawable(vertices); //comment to toggle vertices

    // Add the drawable to the viewer
    viewer.add_drawable(surface);
    //viewer.add_drawable(vertices); //comment to toggle vertices

    viewer.fit_screen(); // Make sure everything is within the visible region of the viewer.
    viewer.run(); // Run the viewer
}

void visualization3D(int numVerts, std::vector<cv::Point3f> linCombo)
{
    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFace.obj");

    vector<int> allVerts = {
        5154, 11283, 5157, 11284, 5181, 11295, 5494, 10957, 4499, 10822, 4211, 10803, 4128, 10804,
        5043, 1089, 5172, 1151, 5170, 1316, 5499, 821, 4506, 685, 4151, 646, 4184, 664,
        1054,5164,1147,5162,1318,5502,822, 4220,686,4221,669,4194,668,4193,
        4973,11203,5168,11288,5503,10620,3845,10621,3875,10629,3857,10565,388,10545,
        1126,5117,1154,5178,1323,3855,497,3879,510,3861,501,3699,424,3701,
        10614,3883,10633,3865,10539,3613,10536,3831,10527,3716,10524,3806,10550,3620,
        10623, 3884, 10635, 3869,10573,3658,10574,3812,10575,3660,10576,3808,10578,3651,
        9190,3887,9180,3872,9045,3611,9150,6703,9053,3607,9137,3787,9052,3619,
        9188,6773,9178,6755,9078,6541,9148,6698,9079,6558,9135,6672,9080,6546,
        6739,1923,6741,1937,6767,1928,6752,1854,6720,1912,6719,1855,6693,1899,
        7994,2579,8058,2748,8391,1922,6762,1935,6749,1926,6748,1849,6714,1910,
        7811,2480,7852,2573,8046,2744,6806,1957,6804,1956,6801,1955,6802,6794,
        2461,7819,2484,8048,2574,8050,2472,7392,2244,7390,2108,7109,1990,6878,
        2548,7989,2549,7992,2550,7993,2578,8057,2740,8373,2245,7394,2109,6877,
        6001,8634,5997,8628,5732,8624,5730,8625,5941,8770,7397,9380,7118,9248
    };
    vector<easy3d::vec3> faceVerts;
    faceVerts.reserve(numVerts);

    for (int i = 0; i < numVerts; i++)
    {
        float x = linCombo[i].x;
        float y = linCombo[i].y;
        float z = linCombo[i].z;

        faceVerts.push_back(easy3d::vec3(x, y, z));
    }
    /*vector<easy3d::vec3> contourVerts(allVerts.size());
    for (int i = 0; i < contourVerts.size(); i++)
    {
        contourVerts[i] = faceVerts[allVerts[i]];
    }*/
    easy3d::logging::initialize();

    // Create the default Easy3D viewer.
    // Note: a viewer must be created before creating any drawables.
    easy3d::Viewer viewer("3d visualization");

    auto surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts);
    surface->update_element_buffer(meshIndices);
    viewer.add_drawable(surface);

    auto vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(faceVerts);
    vertices->set_uniform_coloring(easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a

    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    vertices->set_point_size(10);
    //viewer.add_drawable(vertices);

    viewer.fit_screen();
    viewer.run();
}

void linearCombination(int numVerts, int numCombinations, std::vector<std::vector<cv::Point3f>> mult, Eigen::VectorXf w, std::vector<cv::Point3f>& linCombo)
{
    for (int i = 0; i < numVerts; i++)
    {
        for (int j = 0; j < numCombinations; j++)
        {
            linCombo[i].x += (mult[j][i].x * w[j]);
            linCombo[i].y += (mult[j][i].y * w[j]);
            linCombo[i].z += (mult[j][i].z * w[j]);
        }
    }
}

/* creates a matrix of all the expressions for the given identity weights */
void createAllExpressions(tensor3 tensor,
    Eigen::VectorXf identity_w, int numVerts, std::vector<std::vector<cv::Point3f>>& avgMultExp) {
    /* creates a matrix of all the expressions for the average identity */
    int numExpressions = 47;
    int numIdentities = 150;
    for (int j = 0; j < numExpressions; j++)
    {
        std::vector<cv::Point3f> singleIdn(numVerts);
        std::vector<std::vector<cv::Point3f>> multIdn(numIdentities);
        // 150 identities
        for (int k = 0; k < numIdentities; k++)
        {
            for (int i = 0; i < numVerts; i++)
            {
                Eigen::Vector3f tens_vec = tensor(k, j, i);
                cv::Point3f conv_vec;
                conv_vec.x = tens_vec.x();
                conv_vec.y = tens_vec.y();
                conv_vec.z = tens_vec.z();
                singleIdn[i] = conv_vec;
            }
            multIdn[k] = singleIdn;
        }
        // create an average face
        std::vector<cv::Point3f> combinedIdn(numVerts);
        linearCombination(numVerts, numIdentities, multIdn, identity_w, combinedIdn);
        avgMultExp[j] = combinedIdn;
    }
}

/* apply optimized expression weights and create a vector of every identity */
/* vector length is 150 */
void createAllIdentities(tensor3 tensor,
    Eigen::VectorXf w, int numVerts, std::vector<std::vector<cv::Point3f>>& allIdnOptExp) {
    int numExpressions = 47;
    int numIdentities = 150;

    for (int idnNum = 0; idnNum < numIdentities; idnNum++)
    {
        std::vector<cv::Point3f> singleExp(numVerts);
        std::vector<std::vector<cv::Point3f>> multExp(numExpressions);
        // 47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            for (int i = 0; i < numVerts; i++)
            {
                Eigen::Vector3f tens_vec = tensor(idnNum, j, i);
                cv::Point3f conv_vec;
                conv_vec.x = tens_vec.x();
                conv_vec.y = tens_vec.y();
                conv_vec.z = tens_vec.z();
                singleExp[i] = conv_vec;
            }
            multExp[j] = singleExp;
        }

        std::vector<cv::Point3f> combinedExp(numVerts);
        linearCombination(numVerts, numExpressions, multExp, w, combinedExp);
        allIdnOptExp[idnNum] = combinedExp;
    }
}

void getPose(std::vector<double>& poseVec, const cv::Mat& rvec, const cv::Mat& tvec)
{
    poseVec[0] = rvec.at<double>(0);
    poseVec[1] = rvec.at<double>(1);
    poseVec[2] = rvec.at<double>(2);

    poseVec[3] = tvec.at<double>(0);
    poseVec[4] = tvec.at<double>(1);
    poseVec[5] = tvec.at<double>(2);
    //for (auto pose : poseVec)
    //{
    //    cout << pose << endl;
    //}
}







//cleaned up old code with markers throughout program

//was at above

/*
vector<cv::Point3f> applyIdnW(tensor3 shapeTensor, Eigen::VectorXf identity_w, int n_vectors) {

    int numExpressions = 47;
    int numIdentities = 150;

    vector<cv::Point3f> singleIdn(n_vectors);
    vector<vector<cv::Point3f>> multIdn(numIdentities);

    for (int k = 0; k < numIdentities; k++)
    {
        for (int i = 0; i < n_vectors; i++)
        {
            Eigen::Vector3f tens_vec = shapeTensor(k, j, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            singleIdn[i] = conv_vec;
        }
        multIdn[k] = singleIdn;
    }


    vector<cv::Point3f> combinedIdn(n_vectors);
    for (int i = 0; i < n_vectors; i++)
    {
        for (int k = 0; k < numIdentities; k++)
        {
            combinedIdn[i].x = combinedIdn[i].x + (multIdn[k][i].x * identity_w[k]);
            combinedIdn[i].y = combinedIdn[i].y + (multIdn[k][i].y * identity_w[k]);
            combinedIdn[i].z = combinedIdn[i].z + (multIdn[k][i].z * identity_w[k]);
        }
    }
    return combinedIdn;
}


vector<cv::Point3f> applyExpW(tensor3 shapeTensor, Eigen::VectorXf w, int n_vectors) {
    int numExpressions = 47;
    int numIdentities = 150;

    vector<cv::Point3f> singleExp(n_vectors);
    vector<vector<cv::Point3f>> multExp(numExpressions);

    for (int j = 0; j < numExpressions; j++)
    {

        for (int i = 0; i < n_vectors; i++)
        {
            Eigen::Vector3f tens_vec = shapeTensor(k, j, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            singleExp[i] = conv_vec;
        }
        multExp[j] = singleExp;
    }

    vector<cv::Point3f> combinedExp(n_vectors);

    for (int i = 0; i < n_vectors; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {
            combinedExp[i].x = combinedExp[i].x + (multExp[j][i].x * w[j]);
            combinedExp[i].y = combinedExp[i].y + (multExp[j][i].y * w[j]);
            combinedExp[i].z = combinedExp[i].z + (multExp[j][i].z * w[j]);
        }
    }
    return combinedExp;
}
*/


//was at top


//was at top

        /*
        //vector<cv::Point3f> combinedIdn(n_vectors);
        //vector<cv::Point3f> combinedExp(n_vectors);

        //combinedIdn = applyIdnW(shapeTensor, identity_w, n_vectors);
        //combinedExp = applyExpW(shapeTensor, w, n_vectors);


        vector<vector<cv::Point3f>> multIdn(numIdentities);
        vector<vector<cv::Point3f>> multExp(numExpressions);
        vector<cv::Point3f> single(n_vectors);
        // 150 identities
        for (int k = 0; k < numIdentities; k++)
        {
            //47 expressions
            for (int j = 0; j < numExpressions; j++)
            {
                //73 vertices
                for (int i = 0; i < n_vectors; i++) {
                    Eigen::Vector3f tens_vec = shapeTensor(k, j, i);
                    cv::Point3f cv_vec;
                    cv_vec.x = tens_vec.x() * w[i] * identity_w[k];
                    cv_vec.y = tens_vec.y() * w[i] * identity_w[k];
                    cv_vec.z = tens_vec.z() * w[i] * identity_w[k];
                    single[i] = cv_vec;
                }
                multExp[j] = single;
            }
            multIdn[k] = single;
        }
        */


//was in bottom

//landmark vertices
//lmVerts holds point3f of internal 73 numbers
//vector<easy3d::vec3> lmVerts(vk.size());
//for (int i = 0; i < vk.size(); i++)
//{
//    lmVerts[i] = faceVerts[vk[i]];
//}

//vector<easy3d::vec3> contourVertsAlt(allcs.size());
//for (int i = 0; i < contourVertsAlt.size(); i++)
//{
//    contourVertsAlt[i] = faceVertsAlt2[allcs[i]];
//}

//print loop
//for (int i = 0; i < vk.size(); i++)
//{
    //cout << lmVerts[i] << ", " << vk[i] << ", " << endl;
    //cout << yawVerts[i] << ", " << allints[i] << ". " << endl;
//}


//was in middle


    /*
    //********** DENSE EXPRESSION START *********

    int numDenseVerts = 11510;
    std::vector<cv::Point3f> denseSingleExp(numDenseVerts);
    std::vector<std::vector<cv::Point3f>> denseMultExp(numExpressions);
    // 47 expressions
    for (int j = 0; j < numExpressions; j++)
    {

        for (int i = 0; i < numDenseVerts; i++)
        {

            Eigen::Vector3f tens_vec = rawTensor(137, j, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            denseSingleExp[i] = conv_vec;
        }
        denseMultExp[j] = denseSingleExp;
    }

    std::vector<cv::Point3f> denseCombinedExp(numDenseVerts);

    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {
            denseCombinedExp[i].x = denseCombinedExp[i].x + (denseMultExp[j][i].x * w[j]);
            denseCombinedExp[i].y = denseCombinedExp[i].y + (denseMultExp[j][i].y * w[j]);
            denseCombinedExp[i].z = denseCombinedExp[i].z + (denseMultExp[j][i].z * w[j]);
        }
    }

    //********** DENSE EXPRESSION END *********

    //********** DENSE IDENTITY START *********


    std::vector<cv::Point3f> denseSingleIdn(numDenseVerts);
    std::vector<std::vector<cv::Point3f>> denseMultIdn(numIdentities);

    for (int j = 0; j < numIdentities; j++)
    {
        for (int i = 0; i < numDenseVerts; i++)
        {
            Eigen::Vector3f tens_vec = rawTensor(j, 0, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            denseSingleIdn[i] = conv_vec;
        }
        denseMultIdn[j] = denseSingleIdn;
    }

    std::vector<cv::Point3f> denseCombinedIdn(numDenseVerts);

    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numIdentities; j++)
        {
            denseCombinedIdn[i].x = denseCombinedIdn[i].x + (denseMultIdn[j][i].x * identity_w[j]);
            denseCombinedIdn[i].y = denseCombinedIdn[i].y + (denseMultIdn[j][i].y * identity_w[j]);
            denseCombinedIdn[i].z = denseCombinedIdn[i].z + (denseMultIdn[j][i].z * identity_w[j]);
        }
    }

    //********** DENSE IDENTITY END *********

    */


    /*
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
    */


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


    /*
    //save expression face to file
    vector<cv::Point3f> raw_singleExp(numDenseVerts); // holds 1 expression with 73 vertices used to create multExp
    vector<vector<cv::Point3f>> raw_multExp(numExpressions); //holds 47 expressions with 73 vertices used to create combined exp
    for (int i = 0; i < numExpressions; i++)//numExpresion is 47
    {
        //11510 vertices
        for (int j = 0; j < numDenseVerts; j++) {
            Eigen::Vector3f raw_tens_vec = rawTensor(137, i, j);
            cv::Point3f raw_cv_vec;
            raw_cv_vec.x = raw_tens_vec.x();
            raw_cv_vec.y = raw_tens_vec.y();
            raw_cv_vec.z = raw_tens_vec.z();
            raw_singleExp[j] = raw_cv_vec;
        }
        raw_multExp[i] = raw_singleExp;
    }
    vector<cv::Point3f> raw_combinedExp(numDenseVerts); //holds 1 expression made from 47 expressions (neutral expression)
    for (int i = 0; i < numDenseVerts; i++)
    {
        //47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            raw_combinedExp[i].x += raw_multExp[j][i].x * w[j];
            raw_combinedExp[i].y += raw_multExp[j][i].y * w[j];
            raw_combinedExp[i].z += raw_multExp[j][i].z * w[j];
        }
    }

    createFaceObj(raw_combinedExp, numDenseVerts, "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/optFaceNew.obj");
    */