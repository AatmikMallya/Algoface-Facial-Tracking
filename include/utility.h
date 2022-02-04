#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <vector>
#include <fstream>


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;

namespace asu {

    class Utility {

    public:

        Utility() {}
        ~Utility() {}

        vector<cv::Point2f> readLandmarksFromFile(const std::string& path, const cv::Mat& image);

    private:

    };

}

#endif