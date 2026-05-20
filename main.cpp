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

    RotatedRect bestPlate;
    Mat extractedPlate;

    bool foundPlate = detector.detectBestPlate(
        image,
        bestPlate,
        extractedPlate
    );

    if (!foundPlate)
    {
        std::cout << "No license plate candidate found." << std::endl;

        imshow("Original Image", image);
        waitKey(0);

        return 0;
    }

    detector.drawBestPlate(image, bestPlate);

    imshow("Best License Plate Candidate", image);

    if (!extractedPlate.empty())
    {
        imshow("Extracted License Plate", extractedPlate);

        imwrite("images/extracted-plate.jpg", extractedPlate);
    }

    waitKey(0);

    return 0;
}