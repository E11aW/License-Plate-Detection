/*
    File contents:
    This file defines the TextDetection class.
    The class converts an extracted license plate into a binary image,
    isolates the main text band, finds possible character regions, and
    crops those characters for OCR.

    Assumptions:
    The input plate image has already been extracted from the vehicle image.
    The main license plate number is larger than state text and decoration.
    Character regions can be separated using contour based filtering.
*/

#include "TextDetection.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>

/*
    Function purpose:
    Construct a TextDetection object.

    Preconditions:
    No input is required.

    Postconditions:
    The object is ready to process plate images.
*/
TextDetection::TextDetection() {}

/*
    Function purpose:
    Convert a plate image into a binary image for text segmentation.

    Preconditions:
    plateImage must contain a valid plate or text region image.

    Postconditions:
    Returns a binary image where likely characters are white and the
    background is black.
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
    Function purpose:
    Extract the main large text band from a license plate image.

    Preconditions:
    plateImage must contain a valid extracted license plate image.

    Postconditions:
    Returns a cropped text band image. If contour filtering fails, a safe
    fallback crop from the middle of the plate is returned.
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

        /* Ignore contours that are mostly on the extreme border. */
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
            US plate numbers are usually in the middle or upper middle.
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

    /* Merge all main character boxes into one text band box. */
    cv::Rect textBox = largeCharacterCandidates[0];

    for (size_t i = 1; i < largeCharacterCandidates.size(); i++)
    {
        textBox = textBox | largeCharacterCandidates[i];
    }

    /* Add padding so the crop does not cut off character edges. */
    int padX = static_cast<int>(plateWidth * 0.04);
    int padY = static_cast<int>(plateHeight * 0.08);

    textBox.x -= padX;
    textBox.y -= padY;
    textBox.width += 2 * padX;
    textBox.height += 2 * padY;

    textBox = clampRect(textBox, plateWidth, plateHeight);

    return plateImage(textBox).clone();
}

/*
    Function purpose:
    Detect possible character regions in a binary text image.

    Preconditions:
    binaryImage must contain a valid binary text image.

    Postconditions:
    Returns character bounding boxes sorted from left to right.
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

/*
    Function purpose:
    Crop each segmented character image from the binary text image.

    Preconditions:
    binaryImage must contain a valid binary image.
    regions must contain rectangles for that same image.

    Postconditions:
    Returns a vector of cropped character images.
*/
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
    Function purpose:
    Decide whether a contour rectangle is shaped like a plate character.

    Preconditions:
    rect must contain a contour bounding box.
    imageWidth and imageHeight must be positive.

    Postconditions:
    Returns true when the rectangle passes character size and shape checks.
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

/*
    Function purpose:
    Keep a rectangle inside the image boundaries.

    Preconditions:
    imageWidth and imageHeight must be positive.

    Postconditions:
    Returns the portion of rect that is inside the image.
*/
cv::Rect TextDetection::clampRect(
    const cv::Rect &rect,
    int imageWidth,
    int imageHeight)
{
    cv::Rect imageBounds(0, 0, imageWidth, imageHeight);

    return rect & imageBounds;
}