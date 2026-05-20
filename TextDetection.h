#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

class TextDetection
{
public:
    TextDetection();

    // Preprocess plate image into binary image
    cv::Mat preprocessPlate(const cv::Mat &plateImage);

    // Crop the main large-character region from a license plate
    cv::Mat extractMainTextRegion(const cv::Mat &plateImage);

    // Extract character regions from a binary plate/text image
    std::vector<cv::Rect> detectCharacterRegions(const cv::Mat &binaryImage);

    // Crop characters from bounding boxes
    std::vector<cv::Mat> segmentCharacters(
        const cv::Mat &binaryImage,
        const std::vector<cv::Rect> &regions);

private:
    bool isValidCharacterRegion(
        const cv::Rect &rect,
        int imageWidth,
        int imageHeight);

    cv::Rect clampRect(
        const cv::Rect &rect,
        int imageWidth,
        int imageHeight);
};

#endif