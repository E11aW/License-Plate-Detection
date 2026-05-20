#ifndef PLATE_DETECTOR_H
#define PLATE_DETECTOR_H

#include <opencv2/core.hpp>
#include <vector>

class PlateDetector
{
public:
    bool detectBestPlate(
        const cv::Mat& input,
        cv::RotatedRect& bestPlate,
        cv::Mat& extractedPlate
    );

    void drawBestPlate(cv::Mat& image, const cv::RotatedRect& plate);

private:
    cv::Mat preprocessImage(const cv::Mat& input);
    cv::Mat detectEdges(const cv::Mat& gray);
    cv::Mat strengthenPlateRegions(const cv::Mat& edges);

    bool isLicensePlateCandidate(
        const std::vector<cv::Point>& contour,
        const cv::Mat& image
    );

    double scoreCandidate(
        const cv::RotatedRect& candidate,
        const cv::Mat& edges,
        const cv::Mat& image
    );

    cv::Mat extractPlateRegion(
        const cv::Mat& input,
        const cv::RotatedRect& plate
    );

    void orderPoints(
        cv::Point2f points[4],
        cv::Point2f ordered[4]
    );
};

#endif