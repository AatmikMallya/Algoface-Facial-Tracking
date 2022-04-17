#include "../include/tensor.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>

#define NUM_OF_VERTICES 11510

void readTensorFaster(string& tensorPath, vector<vector<vector<float>>>& allFaceVerts) { 
    //ex tensorPath = "raw_tensor.bin" or "shape_tensor.bin"
    std::ifstream infile(tensorPath, std::ios::in | std::ios::binary);
    if (infile.fail()) {
        std::cerr << "ERROR" << endl;
        exit(-1);
    }
    int faceVecLen = NUM_OF_VERTICES;

    for (int i = 0; i < 150; i++) {
        for (int e = 0; e < 47; e++) {
            vector<float> face(faceVecLen * 3); //11510 * (x,y,z) vertices
            infile.read((char*)&face[0], faceVecLen * 3 * sizeof(float));
            allFaceVerts[i][e] = face;
        }
    }
    infile.close();
}

// Reads vertices from each expression for each user in the warehouse
void buildRawTensor(string& warehousePath, string& outfile, tensor3& rawTensor) {
    warehousePath += "Tester_";

    // Each of the 150 users corresponds to one "shape.bs" file
    for (int i = 0; i < 150; i++) {
        string fileName = warehousePath + std::to_string(i + 1) + "/Blendshape/shape.bs";

        FILE* fp;
        fp = fopen(fileName.c_str(), "rb");

        int nShapes = 0, nVerts = 0, nFaces = 0;
        fread( &nShapes, sizeof(int), 1, fp );	  // nShape = 46
        fread( &nVerts, sizeof(int), 1, fp );	  // nVerts = 11510
        fread( &nFaces, sizeof(int), 1, fp );	  // nFaces = 11540

        for (int j = 0; j < 47; ++j)
            for (int k = 0; k < 11510; ++k)
                fread(&rawTensor(i, j, k), sizeof(Vector3f), 1, fp);

        fclose(fp);
    }

    writeTensor(outfile, rawTensor);
}

// Saves core tensor to binary file
void writeTensor(const string& filename, tensor3& tensor) {
    std::ofstream file(filename, std::ofstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 11510; k++)
                file.write(reinterpret_cast<const char*>(&tensor(i, j, k)), sizeof(Vector3f));
    file.close();
}

// Loads tensor from binary file
void loadRawTensor(const string& filename, tensor3& tensor) {
    std::ifstream file(filename, std::ifstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 11510; k++)
                file.read(reinterpret_cast<char *>(&tensor(i, j, k)), sizeof(Vector3f));

    file.close();
}

// Prints every vertex in the core tensor (81,145,500 vertices)
void displayEntireTensor(tensor3& tensor) {
    for(int i = 0; i < 150; i++) {
        for(int j = 0; j <47; j++) {
            for(int k = 0; k < 11510; k++) {
                cout << "User " << i << ", Expression " << j << ", Vertex " << k << ": " << tensor(i, j, k) << endl;
            }
        }
    }
}

void loadShapeTensor(string& SHAPE_TENSOR_PATH, tensor3& shapeTensor) {
    std::ifstream file(SHAPE_TENSOR_PATH, std::ifstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 73; k++)
                file.read(reinterpret_cast<char *>(&shapeTensor(i, j, k)), sizeof(Vector3f));

    file.close();
}

void buildShapeTensor(tensor3& rawTensor, string& outfile, tensor3& shapeTensor) {
    int shapeVerts[] = { 3984,10818,499,10543,413,3867,10574,9053,6698,1929,1927,6747,9205,7112,9380,3981,4277,10854,708,10742,4159,7135,9413,2138,2127,1986,6969,4437,760,4387,4346,10885,4370,766,4393,7330,7236,7275,9471,7271,7284,2191,7256,4227,294,279,3564,10461,8948,6418,6464,6441,6312,9236,8972,3262,3676,182,1596,1607,6575,1633,8864,6644,1790,3224,3270,251,1672,1621,6262,6162,10346,
    };
    int len = sizeof(shapeVerts) / sizeof(*shapeVerts);

    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 47; j++) {
            for (int k = 0; k < len; k++) {
                shapeTensor(i, j, k) = rawTensor(i, j, shapeVerts[k]);
            }
        }
    }

    writeShapeTensor(outfile, shapeTensor);
}

void writeShapeTensor(const string& filename, tensor3& tensor) {
    std::ofstream file(filename, std::ofstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 73; k++)
                file.write(reinterpret_cast<const char*>(&tensor(i, j, k)), sizeof(Vector3f));
    file.close();
}


vector<cv::Point2f> readLandmarksFromFile_2(const std::string& path, const cv::Mat& image) {
    vector<int> orderedIndices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  //face contour
                                   21, 22, 23, 24, 25, 26,                            //left eyebrow
                                   18, 17, 16, 15, 20, 19,                            //right eyebrow
                                   27, 66, 28, 69, 29, 68, 30, 67,                    //left eye
                                   33, 70, 32, 73, 31, 72, 34, 71,                    //right eye
                                   35, 36, 37, 38, 44, 39, 45, 40, 41, 42, 43,        //nose contour
                                   65,												  //nose tip
                                   46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,    //outer mouth
                                   63, 62, 61, 60, 59, 58						      //inner mouth
    };

    std::ifstream infile(path);
    if (infile.fail()) {
        std::cerr << "ERROR: unable to open the ladndmarks file, refer to file " << __FILE__ << ", line " << __LINE__ << endl;
        exit(-1);
    }
    std::string hay;
    std::getline(infile, hay);
    int nOrigPoints = std::stof(hay);
    vector<float> origLms(nOrigPoints * 2);
    for (int i = 0; i < nOrigPoints; i++) {
        std::string temp;
        std::getline(infile, temp, ' ');
        origLms[i] = std::stof(temp) * image.cols;
        std::getline(infile, temp);
        origLms[i + nOrigPoints] = image.rows - (std::stof(temp) * image.rows);
    }
    infile.close();

    int nPoints = orderedIndices.size();
    vector<cv::Point2f> lms(nPoints);
    for (int i = 0; i < nPoints; i++) {
        lms[i].x = origLms[orderedIndices[i]];
        lms[i].y = origLms[orderedIndices[i] + nOrigPoints];
    }

    return lms;

}

vector<uint32_t> readMeshTriangleIndicesFromFile(const std::string& path) {

    FILE* file = fopen(path.c_str(), "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        exit(-1);
    }

    vector<uint32_t> indices;
    indices.reserve(50000);

    while (true) {

        char lineHeader[128];
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        if (strcmp(lineHeader, "f") == 0) {

            unsigned int vertexIndex[4], uvIndex[4], normalIndex[4];

            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n",
                &vertexIndex[0], &uvIndex[0], &normalIndex[0],
                &vertexIndex[1], &uvIndex[1], &normalIndex[1],
                &vertexIndex[2], &uvIndex[2], &normalIndex[2],
                &vertexIndex[3], &uvIndex[3], &normalIndex[3]
            );

            for (int i = 0; i < 4; i++) {
                vertexIndex[i] -= 1;     // obj file indices start from 1
                uvIndex[i] -= 1;
                normalIndex[i] -= 1;
            }

            //====== change from quads to triangle
            indices.push_back(vertexIndex[0]);
            indices.push_back(vertexIndex[1]);
            indices.push_back(vertexIndex[2]);
            indices.push_back(vertexIndex[2]);
            indices.push_back(vertexIndex[3]);
            indices.push_back(vertexIndex[0]);


            if (matches != 12) {
                cout << "ERROR: couldn't read the faces! number of quads didn't match" << endl;
                exit(-1);
            }

        }

    }

    return indices;
}

vector<easy3d::vec3> readFace3DFromObj(std::string path) {

    std::ifstream infile(path);
    if (infile.fail()) {
        std::cerr << "ERROR: couldn't open the Obj file to read the face from" << endl;
        exit(-1);
    }

    vector<easy3d::vec3> faceVerts;
    faceVerts.reserve(NUM_OF_VERTICES);

    for (int i = 0; i < NUM_OF_VERTICES; i++) {
        std::string hay;
        std::getline(infile, hay, ' ');
        std::getline(infile, hay, ' ');
        float x = std::stof(hay);
        std::getline(infile, hay, ' ');
        float y = std::stof(hay);
        std::getline(infile, hay);
        float z = std::stof(hay);

        faceVerts.push_back(easy3d::vec3(x, y, z));
    }

    infile.close();

    return faceVerts;
}

vector<int> readVertexIdFromFile(std::string path) {

    std::ifstream infile(path);
    if (infile.fail()) {
        std::cerr << "ERROR: couldn't open the landmark vertex file " << endl;
        exit(-1);
    }
    std::string temp;
    std::getline(infile, temp);
    int numLms = std::stoi(temp);
    vector<int> vk(numLms);
    for (int i = 0; i < numLms; i++) {
        std::getline(infile, temp, ',');
        vk[i] = std::stoi(temp);
        std::getline(infile, temp);
    }
    infile.close();

    return vk;
}

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

void createFaceObj(const vector<cv::Point3f> faceVec, int numVerts, std::string pathToOutputObjFile)
{

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
        outfile << "v " << std::to_string(faceVec[i].x) << " " << std::to_string(faceVec[i].y) << " " << std::to_string(faceVec[i].z) << endl;
    }
    outfile << suffix;
    outfile.close();
}