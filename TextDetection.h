#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

class TextDetection
{
public:
    TextDetection();

    // preprocess plate image
    cv::Mat preprocessPlate(const cv::Mat &plateImage);

    // extract character regions
    std::vector<cv::Rect> detectCharacterRegions(const cv::Mat &binaryImage);

    // crop characters from bounding boxes
    std::vector<cv::Mat> segmentCharacters(
        const cv::Mat &binaryImage,
        const std::vector<cv::Rect> &regions);

private:
    bool isValidCharacterRegion(const cv::Rect &rect);
};

#endif // TEXTDETECTION_H