#ifndef IDENTITYOPTIMIZATION_H
#define IDENTITYOPTIMIZATION_H

#include <iostream>
#include <vector>
#include <random>

#include "ceres/ceres.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;

bool identityOptimize(const vector<cv::Point2f>& lms,
    const std::vector<float>& pose, const cv::Mat& image, float f, Eigen::VectorXf& w_exp,
    const std::vector<std::vector<cv::Point3f>>& multIdn);

#endif //IDENTITYOPTIMIZATION_H