#ifndef PLATE_DETECTOR_H
#define PLATE_DETECTOR_H

#include <opencv2/core.hpp>
#include <vector>

class PlateDetector
{
public:
    std::vector<cv::RotatedRect> detectPlates(const cv::Mat& input);
    void drawCandidates(cv::Mat& image, const std::vector<cv::RotatedRect>& candidates);

private:
    cv::Mat preprocessImage(const cv::Mat& input);
    cv::Mat detectEdges(const cv::Mat& gray);
    cv::Mat strengthenPlateRegions(const cv::Mat& edges);

    bool isLicensePlateCandidate(const std::vector<cv::Point>& contour);
    double scoreCandidate(const cv::RotatedRect& candidate, const cv::Mat& edges);
};

#endif