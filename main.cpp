#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "PlateDetector.h"
#include "TextDetection.h"
#include "TextRecognition.h"

using namespace cv;

cv::Mat normalizeCharacter(const cv::Mat &character)
{
    /*
        Step 1:
        Find tight bounding box around white pixels
    */

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        character.clone(),
        contours,
        cv::RETR_LIST,
        cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return cv::Mat();
    }

    cv::Rect boundingBox =
        cv::boundingRect(contours[0]);

    for (size_t i = 1; i < contours.size(); i++)
    {
        boundingBox |= cv::boundingRect(contours[i]);
    }

    cv::Mat cropped =
        character(boundingBox).clone();

    /*
        Step 2:
        Pad to square
    */

    int size =
        std::max(cropped.cols, cropped.rows);

    cv::Mat square =
        cv::Mat::zeros(size, size, CV_8UC1);

    int x =
        (size - cropped.cols) / 2;

    int y =
        (size - cropped.rows) / 2;

    cropped.copyTo(
        square(cv::Rect(
            x,
            y,
            cropped.cols,
            cropped.rows)));

    /*
        Step 3:
        Resize to fixed OCR size
    */

    cv::Mat normalized;

    cv::resize(square, normalized, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

    cv::threshold(
        normalized,
        normalized,
        128,
        255,
        cv::THRESH_BINARY);

    return normalized;
}

int main()
{
    Mat image = imread("images/diffFont.png");

    if (image.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    /*
        -----------------------------
        Main Text Region Detection
        -----------------------------
    */

    TextDetection textDetector;

    Mat textRegion = textDetector.extractMainTextRegion(image);

    if (textRegion.empty())
    {
        std::cout << "Could not extract main text region." << std::endl;
        return 0;
    }

    imshow("Main Text Region", textRegion);

    imwrite("images/text-region.jpg", textRegion);

    /*
        -----------------------------
        Binary Text Region
        -----------------------------
    */

    Mat binaryPlate = textDetector.preprocessPlate(textRegion);

    /*
        Slightly thicken thin character strokes.

        This helps prevent:
        - fragmented contours
        - broken letters
        - weak template shapes
    */
    cv::Mat kernel =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(3, 3));

    imshow("Binary Text Region", binaryPlate);

    imwrite("images/binary-text-region.jpg", binaryPlate);

    /*
        -----------------------------
        Character Detection
        -----------------------------
    */

    std::vector<Rect> characterRegions =
        textDetector.detectCharacterRegions(binaryPlate);

    std::vector<Mat> characterImages =
        textDetector.segmentCharacters(
            binaryPlate,
            characterRegions);

    /*
        Draw character boxes for visualization
    */

    Mat characterDisplay = textRegion.clone();

    for (const auto &rect : characterRegions)
    {
        rectangle(
            characterDisplay,
            rect,
            Scalar(0, 255, 0),
            2);
    }

    imshow("Detected Characters", characterDisplay);

    imwrite("images/detected-characters.jpg", characterDisplay);

    /*
        -----------------------------
        Show Segmented Characters
        -----------------------------
    */

    std::string labels =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (size_t i = 0; i < characterImages.size(); i++)
    {
        std::string windowName =
            "Character " + std::to_string(i);

        Mat normalized =
            normalizeCharacter(characterImages[i]);

        std::string outputPath =
            "templates/" +
            std::string(1, labels[i]) +
            ".png";

        imwrite(outputPath, normalized);
    }

    waitKey(0);

    return 0;
}