/*
    File contents:
    This header declares the PlateDetector class.
    The class is responsible for locating the best license plate candidate
    in a vehicle image and extracting a perspective corrected plate image.

    Assumptions:
    Input images are valid OpenCV Mat objects.
    Vehicle plates are roughly rectangular and wider than they are tall.
    The detected plate may be angled, so perspective correction is needed.
*/

#ifndef PLATE_DETECTOR_H
#define PLATE_DETECTOR_H

#include <opencv2/core.hpp>
#include <vector>

class PlateDetector
{
public:
    /*
        Function purpose:
        Find the best license plate candidate in an input image.

        Preconditions:
        input must contain a valid color image.
        bestPlate and extractedPlate must be valid output variables.

        Postconditions:
        Returns true when a plate candidate is found.
        bestPlate stores the best rotated rectangle candidate.
        extractedPlate stores the cropped and corrected plate image.
    */
    bool detectBestPlate(
        const cv::Mat& input,
        cv::RotatedRect& bestPlate,
        cv::Mat& extractedPlate);

    /*
        Function purpose:
        Draw the detected plate rectangle on an image.

        Preconditions:
        image must contain a valid display image.
        plate must contain a valid rotated rectangle.

        Postconditions:
        The plate outline is drawn directly onto image.
    */
    void drawBestPlate(cv::Mat& image, const cv::RotatedRect& plate);

private:
    /*
        Function purpose:
        Convert an input image into a smoothed grayscale image.

        Preconditions:
        input must contain a valid color image.

        Postconditions:
        Returns a grayscale image prepared for edge detection.
    */
    cv::Mat preprocessImage(const cv::Mat& input);

    /*
        Function purpose:
        Detect normal edges in a grayscale image.

        Preconditions:
        gray must contain a valid grayscale image.

        Postconditions:
        Returns a binary edge image.
    */
    cv::Mat detectEdges(const cv::Mat& gray);

    /*
        Function purpose:
        Detect edges after emphasizing dark text on a light plate background.

        Preconditions:
        gray must contain a valid grayscale image.

        Postconditions:
        Returns a binary edge image that emphasizes plate text and borders.
    */
    cv::Mat detectBlackhatEdges(const cv::Mat& gray);

    /*
        Function purpose:
        Connect nearby edge pixels so plate shaped regions become stronger.

        Preconditions:
        edges must contain a valid binary edge image.
        kernelSize must contain positive width and height values.

        Postconditions:
        Returns a strengthened binary image for contour detection.
    */
    cv::Mat strengthenPlateRegions(const cv::Mat& edges, const cv::Size& kernelSize);

    /*
        Function purpose:
        Decide whether a contour is a possible license plate candidate.

        Preconditions:
        contour must contain points from contour detection.
        candidate must be the rotated rectangle for that contour.
        image must be the original input image.

        Postconditions:
        Returns true when the contour passes basic size and shape checks.
    */
    bool isLicensePlateCandidate(
        const std::vector<cv::Point>& contour,
        const cv::RotatedRect& candidate,
        const cv::Mat& image);

    /*
        Function purpose:
        Give a numeric score to a possible plate candidate.

        Preconditions:
        contour, candidate, edges, and image must refer to the same image area.

        Postconditions:
        Returns a higher score for candidates that look more like plates.
    */
    double scoreCandidate(
        const std::vector<cv::Point>& contour,
        const cv::RotatedRect& candidate,
        const cv::Mat& edges,
        const cv::Mat& image);

    /*
        Function purpose:
        Score whether a candidate contains character like edge components.

        Preconditions:
        candidate must be inside or partly inside the image.
        edges must correspond to the input image.

        Postconditions:
        Returns a value between zero and one based on character like content.
    */
    double scoreCharacterLikeContent(
        const cv::RotatedRect& candidate,
        const cv::Mat& edges,
        const cv::Mat& image);

    /*
        Function purpose:
        Penalize candidates that touch the image border.

        Preconditions:
        boundingBox must describe a candidate region.
        image must be the original input image.

        Postconditions:
        Returns a score multiplier for the candidate.
    */
    double scoreBorderPenalty(
        const cv::Rect& boundingBox,
        const cv::Mat& image);

    /*
        Function purpose:
        Score a candidate based on likely plate location in the image.

        Preconditions:
        candidate must describe a possible plate region.
        image must be the original input image.

        Postconditions:
        Returns a soft location score for the candidate.
    */
    double scorePlateLocation(
        const cv::RotatedRect& candidate,
        const cv::Mat& image);

    /*
        Function purpose:
        Crop and perspective correct the detected plate region.

        Preconditions:
        input must contain the original image.
        plate must describe the detected plate rectangle.

        Postconditions:
        Returns a cropped plate image, or an empty Mat if extraction fails.
    */
    cv::Mat extractPlateRegion(
        const cv::Mat& input,
        const cv::RotatedRect& plate);

    /*
        Function purpose:
        Arrange the four rotated rectangle points into a consistent order.

        Preconditions:
        points must contain exactly four corner points.
        ordered must have room for exactly four corner points.

        Postconditions:
        ordered contains top left, top right, bottom right, and bottom left.
    */
    void orderPoints(
        cv::Point2f points[4],
        cv::Point2f ordered[4]);
};

#endif
