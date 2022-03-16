#ifndef DEMO_CORE_TENSOR_H
#define DEMO_CORE_TENSOR_H

#include "utility.h"
#include <iostream>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using Eigen::Vector3f;

namespace asu {

    class Tensor : public Utility {

    public:
        Tensor() {}
        ~Tensor() {}

    private:

    };

}

typedef Eigen::Tensor<Eigen::Vector3f, 3> tensor3;

void buildRawTensor(string& warehousePath, string& outfile, tensor3& rawTensor);
void writeTensor(const string& filename, tensor3& tensor);
void loadRawTensor(const string& filename, tensor3& tensor);
void displayEntireTensor(tensor3& rawTensor);

void loadShapeTensor(string& SHAPE_TENSOR_PATH, tensor3& shapeTensor);
void buildShapeTensor(tensor3& rawTensor, string& outfile, tensor3& shapeTensor);
void writeShapeTensor(const string& filename, tensor3& tensor);

vector<cv::Point2f> readLandmarksFromFile_2(const std::string& path, const cv::Mat& image);

vector<uint32_t> readMeshTriangleIndicesFromFile(const std::string& path);
vector<easy3d::vec3> readFace3DFromObj(std::string path);
vector<int> readVertexIdFromFile(std::string path);

void createFaceObj(const vector<float>& faceVec, int numVerts, std::string pathToOutputObjFile);
void createFaceObj(const vector<cv::Point3f> faceVec, int numVerts, std::string pathToOutputObjFile);

#endif //DEMO_CORE_TENSOR_H

