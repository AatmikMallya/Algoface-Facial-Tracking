#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <vector>
#include <fstream>


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "easy3d/viewer/viewer.h"
#include "easy3d/renderer/drawable_lines.h"
#include "easy3d/renderer/drawable_points.h"
#include "easy3d/renderer/drawable_triangles.h"
#include "easy3d/renderer/renderer.h"
#include "easy3d/renderer/camera.h"
#include "easy3d/renderer/manipulated_frame.h"
#include "easy3d/renderer/manipulator.h"
#include "easy3d/renderer/frame.h"

using std::cout;
using std::endl;
using std::vector;

namespace asu {

    class Utility {

    public:

        Utility() {}
        ~Utility() {}

        vector<cv::Point2f> readLandmarksFromFile(const std::string& path, const cv::Mat& image);
        vector<vector<uint32_t>> readQuadIndicesFromFile(const std::string& path);
        

    private:

    };

}

#endif