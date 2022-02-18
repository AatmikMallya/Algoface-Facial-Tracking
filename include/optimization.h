#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <iostream>
#include <vector>
#include <random>

#include "../include/tensor.h"

#include "ceres/ceres.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;


//vector optimize(const Eigen::MatrixXf& w_exp, //const Eigen::VectorXf& w,
//    const vector<cv::Point2f>& lms, const Eigen::VectorXf& pose, const cv::Mat& image, float f, Eigen::Tensor<Eigen::Vector3f, 3>& shapeTensor);
bool optimize(std::vector<std::vector<cv::Point3f>> multExp, const vector<cv::Point2f>& lms, const std::vector<float>& pose, const cv::Mat& image,
	float f, Eigen::VectorXf& w_exp);

#endif //OPTIMIZATION_H