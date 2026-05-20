#include "TextDetection.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

TextDetection::TextDetection() {}

/*
    Preprocess license plate image

    Steps:
    1. Convert to grayscale
    2. Blur to reduce noise
    3. Adaptive threshold
    4. Morphological cleanup
*/
cv::Mat TextDetection::preprocessPlate(const cv::Mat &plateImage)
{
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat binary;

    // Convert to grayscale if needed
    if (plateImage.channels() == 3)
    {
        cv::cvtColor(plateImage, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = plateImage.clone();
    }

    // Reduce noise
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Adaptive threshold
    cv::adaptiveThreshold(
        blurred,
        binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        11,
        2);

    // Morphological cleanup
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(3, 3));

    cv::morphologyEx(
        binary,
        binary,
        cv::MORPH_CLOSE,
        kernel);

    return binary;
}

/*
    Detect possible character regions using contours
*/
std::vector<cv::Rect> TextDetection::detectCharacterRegions(
    const cv::Mat &binaryImage)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect> characterRegions;

    cv::findContours(
        binaryImage,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours)
    {
        cv::Rect boundingBox = cv::boundingRect(contour);

        if (isValidCharacterRegion(boundingBox))
        {
            characterRegions.push_back(boundingBox);
        }
    }

    // Sort characters from left to right
    std::sort(
        characterRegions.begin(),
        characterRegions.end(),
        [](const cv::Rect &a, const cv::Rect &b)
        {
            return a.x < b.x;
        });

    return characterRegions;
}

/*
    Crop segmented character images
*/
std::vector<cv::Mat> TextDetection::segmentCharacters(
    const cv::Mat &binaryImage,
    const std::vector<cv::Rect> &regions)
{
    std::vector<cv::Mat> characters;

    for (const auto &rect : regions)
    {
        cv::Mat character = binaryImage(rect).clone();
        characters.push_back(character);
    }

    return characters;
}

/*
    Filter invalid contours

    Typical license plate characters:
    - Taller than wide
    - Minimum area
    - Reasonable aspect ratio
*/
bool TextDetection::isValidCharacterRegion(const cv::Rect &rect)
{
    const int MIN_WIDTH = 5;
    const int MIN_HEIGHT = 15;
    const int MIN_AREA = 100;

    if (rect.width < MIN_WIDTH)
    {
        return false;
    }

    if (rect.height < MIN_HEIGHT)
    {
        return false;
    }

    if ((rect.width * rect.height) < MIN_AREA)
    {
        return false;
    }

    float aspectRatio =
        static_cast<float>(rect.width) /
        static_cast<float>(rect.height);

    // Characters are usually taller than wide
    if (aspectRatio < 0.15f || aspectRatio > 1.0f)
    {
        return false;
    }

    return true;
}