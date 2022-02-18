#include "utility.h"

using namespace asu;

vector<cv::Point2f> Utility::readLandmarksFromFile(const std::string& path, const cv::Mat& image) {

    vector<int> orderedIndices = {
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

    std::ifstream infile(path);
    if (infile.fail()) {
        std::cerr << "ERROR: unable to open the landmarks file, refer to file " << __FILE__ << ", line " << __LINE__ << endl;
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
    vector<cv::Point2f> lms(nPoints * 2);
    for (int i = 0; i < nPoints; i++) {
        lms[i].x = origLms[orderedIndices[i]];
        lms[i].y = origLms[orderedIndices[i] + nOrigPoints];
    }

    return lms;

}
vector<vector<uint32_t>> Utility::readQuadIndicesFromFile(const std::string& path) {

    FILE* file = fopen(path.c_str(), "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        exit(-1);
    }

    vector<vector<uint32_t>> quads;
    quads.reserve(12000);

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


            quads.push_back({ vertexIndex[0], vertexIndex[1], vertexIndex[2], vertexIndex[3] });


            if (matches != 12) {
                cout << "ERROR: couldn't read the faces! number of quads didn't match" << endl;
                exit(-1);
            }

        }

    }

    return quads;
}

