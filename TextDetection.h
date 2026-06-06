/*
    File contents:
    This header declares the TextDetection class.
    The class prepares an extracted plate image for OCR by finding the main
    text region, creating a binary image, locating character boxes, and
    cropping individual character images.

    Assumptions:
    The plate image has already been cropped from the vehicle image.
    Plate characters are darker than much of the plate background.
    The main plate number is larger than small state text and decorations.
*/

#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

class TextDetection
{
public:
    /*
        Function purpose:
        Construct a TextDetection object.

        Preconditions:
        No input is required.

        Postconditions:
        A TextDetection object is ready to process plate images.
    */
    TextDetection();

    /*
        Function purpose:
        Convert a plate or text image into a binary image for contour detection.

        Preconditions:
        plateImage must contain a valid plate or text region image.

        Postconditions:
        Returns a binary image where likely characters are white.
    */
    cv::Mat preprocessPlate(const cv::Mat& plateImage);

    /*
        Function purpose:
        Crop the main large character region from a license plate image.

        Preconditions:
        plateImage must contain a valid extracted plate image.

        Postconditions:
        Returns the main plate text region or an empty Mat if the input is empty.
    */
    cv::Mat extractMainTextRegion(const cv::Mat& plateImage);

    /*
        Function purpose:
        Find bounding boxes for possible characters in a binary text image.

        Preconditions:
        binaryImage must contain a valid binary image.

        Postconditions:
        Returns character rectangles sorted from left to right.
    */
    std::vector<cv::Rect> detectCharacterRegions(const cv::Mat& binaryImage);

    /*
        Function purpose:
        Crop each character image from the binary text image.

        Preconditions:
        binaryImage must contain the binary text image.
        regions must contain character rectangles for that image.

        Postconditions:
        Returns cropped character images in the same order as regions.
    */
    std::vector<cv::Mat> segmentCharacters(
        const cv::Mat& binaryImage,
        const std::vector<cv::Rect>& regions);

private:
    /*
        Function purpose:
        Check whether a contour rectangle has character like dimensions.

        Preconditions:
        rect must describe a contour bounding box.
        imageWidth and imageHeight must be positive.

        Postconditions:
        Returns true when the rectangle is a reasonable character candidate.
    */
    bool isValidCharacterRegion(
        const cv::Rect& rect,
        int imageWidth,
        int imageHeight);

    /*
        Function purpose:
        Clamp a rectangle so it stays inside image boundaries.

        Preconditions:
        imageWidth and imageHeight must be positive.

        Postconditions:
        Returns the intersection of rect and the image bounds.
    */
    cv::Rect clampRect(
        const cv::Rect& rect,
        int imageWidth,
        int imageHeight);
};

#endif