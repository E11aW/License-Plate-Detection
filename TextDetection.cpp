#include "TextDetection.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>

TextDetection::TextDetection() {}

/*
    Preprocess license plate image.

    Steps:
    1. Convert to grayscale
    2. Blur to reduce noise
    3. Threshold so dark characters become white
    4. Small morphology cleanup

    Output:
    - White foreground = likely characters / dark objects
    - Black background = plate background
*/
cv::Mat TextDetection::preprocessPlate(const cv::Mat &plateImage)
{
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat binary;

    if (plateImage.channels() == 3)
    {
        cv::cvtColor(plateImage, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = plateImage.clone();
    }

    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);

    /*
        Otsu is useful here because most US plates have
        dark characters on a lighter background.
    */
    cv::threshold(
        blurred,
        binary,
        0,
        255,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    /*
        Small cleanup. Do not use a large kernel here or the
        characters may merge together too much.
    */
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2, 2));

    cv::morphologyEx(
        binary,
        binary,
        cv::MORPH_OPEN,
        kernel);

    return binary;
}

/*
    Extract only the main text band from a US license plate.

    This intentionally ignores:
    - outer border
    - state name at the bottom
    - stickers
    - screws
    - small decorative graphics

    It looks for tall character-like contours, unions them,
    adds padding, and crops that region from the original plate.
*/
cv::Mat TextDetection::extractMainTextRegion(const cv::Mat &plateImage)
{
    if (plateImage.empty())
    {
        return cv::Mat();
    }

    cv::Mat binary = preprocessPlate(plateImage);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        binary,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> largeCharacterCandidates;

    int plateWidth = binary.cols;
    int plateHeight = binary.rows;

    for (const auto &contour : contours)
    {
        cv::Rect box = cv::boundingRect(contour);

        double area = box.area();
        double plateArea = plateWidth * plateHeight;

        float aspectRatio =
            static_cast<float>(box.width) /
            static_cast<float>(box.height);

        /*
            Main plate characters are usually tall relative to the plate.
            This rejects the state text such as "COLORADO", screws, dots,
            and small decoration.
        */
        bool tallEnough = box.height >= plateHeight * 0.35;
        bool notTooTall = box.height <= plateHeight * 0.95;

        bool wideEnough = box.width >= plateWidth * 0.015;
        bool notTooWide = box.width <= plateWidth * 0.25;

        bool goodAspect = aspectRatio >= 0.15f && aspectRatio <= 1.25f;

        bool goodArea =
            area >= plateArea * 0.002 &&
            area <= plateArea * 0.20;

        // Ignore contours that are mostly on the extreme border.
        bool notBorder =
            box.x > 1 &&
            box.y > 1 &&
            box.x + box.width < plateWidth - 1 &&
            box.y + box.height < plateHeight - 1;

        if (tallEnough &&
            notTooTall &&
            wideEnough &&
            notTooWide &&
            goodAspect &&
            goodArea &&
            notBorder)
        {
            largeCharacterCandidates.push_back(box);
        }
    }

    if (largeCharacterCandidates.empty())
    {
        /*
            Fallback:
            US plate numbers are usually in the middle/upper-middle.
            This keeps the program from failing if contour filtering
            misses the characters.
        */
        int y = static_cast<int>(plateHeight * 0.20);
        int h = static_cast<int>(plateHeight * 0.60);

        cv::Rect fallback(
            0,
            y,
            plateWidth,
            h);

        fallback = clampRect(fallback, plateWidth, plateHeight);

        return plateImage(fallback).clone();
    }

    // Merge all main character boxes into one text-band box
    cv::Rect textBox = largeCharacterCandidates[0];

    for (size_t i = 1; i < largeCharacterCandidates.size(); i++)
    {
        textBox = textBox | largeCharacterCandidates[i];
    }

    // Add padding so the crop does not cut off character edges
    int padX = static_cast<int>(plateWidth * 0.04);
    int padY = static_cast<int>(plateHeight * 0.08);

    textBox.x -= padX;
    textBox.y -= padY;
    textBox.width += 2 * padX;
    textBox.height += 2 * padY;

    textBox = clampRect(textBox, plateWidth, plateHeight);

    return plateImage(textBox).clone();
}

// Detect possible character regions using contours
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

        if (isValidCharacterRegion(
                boundingBox,
                binaryImage.cols,
                binaryImage.rows))
        {
            characterRegions.push_back(boundingBox);
        }
    }

    std::sort(
        characterRegions.begin(),
        characterRegions.end(),
        [](const cv::Rect &a, const cv::Rect &b)
        {
            return a.x < b.x;
        });

    return characterRegions;
}

// Crop segmented character images
std::vector<cv::Mat> TextDetection::segmentCharacters(
    const cv::Mat &binaryImage,
    const std::vector<cv::Rect> &regions)
{
    std::vector<cv::Mat> characters;

    for (const auto &rect : regions)
    {
        cv::Rect safeRect = clampRect(
            rect,
            binaryImage.cols,
            binaryImage.rows);

        if (safeRect.width > 0 && safeRect.height > 0)
        {
            cv::Mat character = binaryImage(safeRect).clone();
            characters.push_back(character);
        }
    }

    return characters;
}

/*
    Filter invalid character contours.

    This is now relative to the image size, so it works better
    after cropping the main text region.
*/
bool TextDetection::isValidCharacterRegion(
    const cv::Rect &rect,
    int imageWidth,
    int imageHeight)
{
    if (rect.width <= 0 || rect.height <= 0)
    {
        return false;
    }

    double imageArea = imageWidth * imageHeight;
    double rectArea = rect.area();

    float aspectRatio =
        static_cast<float>(rect.width) /
        static_cast<float>(rect.height);

    bool goodHeight =
        rect.height >= imageHeight * 0.35 &&
        rect.height <= imageHeight * 0.95;

    bool goodWidth =
        rect.width >= imageWidth * 0.015 &&
        rect.width <= imageWidth * 0.25;

    bool goodAspect =
        aspectRatio >= 0.12f &&
        aspectRatio <= 1.25f;

    bool goodArea =
        rectArea >= imageArea * 0.005 &&
        rectArea <= imageArea * 0.30;

    return goodHeight &&
           goodWidth &&
           goodAspect &&
           goodArea;
}

cv::Rect TextDetection::clampRect(
    const cv::Rect &rect,
    int imageWidth,
    int imageHeight)
{
    cv::Rect imageBounds(0, 0, imageWidth, imageHeight);

    return rect & imageBounds;
}