#ifndef IDENTITYOPTIMIZATION_H
#define IDENTITYOPTIMIZATION_H

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

bool identityOptimize(std::vector<std::vector<cv::Point3f>> multIdn, const vector<cv::Point2f>& lms, const std::vector<double>& pose, const cv::Mat& image,
	double f, Eigen::VectorXf& w_idn);



#endif //IDENTITYOPTIMIZATION_H