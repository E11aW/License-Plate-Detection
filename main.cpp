#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

#include "PlateDetector.h"

using namespace cv;

int main()
{
    Mat image = imread("images/test-car3.jpg");

    if (image.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    PlateDetector detector;

    std::vector<RotatedRect> candidates = detector.detectPlates(image);

    detector.drawCandidates(image, candidates);

    imshow("Detected License Plate Candidates", image);
    waitKey(0);

    return 0;
}