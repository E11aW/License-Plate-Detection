#include "PlateDetector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

using namespace cv;

std::vector<RotatedRect> PlateDetector::detectPlates(const Mat& input)
{
    Mat gray = preprocessImage(input);
    Mat edges = detectEdges(gray);
    Mat strengthened = strengthenPlateRegions(edges);

    std::vector<std::vector<Point>> contours;
    findContours(strengthened, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<RotatedRect> candidates;

    for (const auto& contour : contours)
    {
        if (isLicensePlateCandidate(contour))
        {
            candidates.push_back(minAreaRect(contour));
        }
    }

    return candidates;
}

Mat PlateDetector::preprocessImage(const Mat& input)
{
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, gray, Size(5, 5), 1.5);

    return gray;
}

Mat PlateDetector::detectEdges(const Mat& gray)
{
    Mat edges;

    Canny(gray, edges, 75, 200);

    return edges;
}

Mat PlateDetector::strengthenPlateRegions(const Mat& edges)
{
    Mat strengthened;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(17, 5));

    morphologyEx(edges, strengthened, MORPH_CLOSE, kernel);

    return strengthened;
}

bool PlateDetector::isLicensePlateCandidate(const std::vector<Point>& contour)
{
    double contourAreaValue = contourArea(contour);

    if (contourAreaValue < 500)
    {
        return false;
    }

    RotatedRect box = minAreaRect(contour);

    float width = box.size.width;
    float height = box.size.height;

    if (width <= 0 || height <= 0)
    {
        return false;
    }

    if (height > width)
    {
        std::swap(width, height);
    }

    float aspectRatio = width / height;

    if (aspectRatio < 2.0 || aspectRatio > 6.0)
    {
        return false;
    }

    if (contourAreaValue < 1000 || contourAreaValue > 50000)
    {
        return false;
    }

    return true;
}

double PlateDetector::scoreCandidate(const RotatedRect& candidate, const Mat& edges)
{
    float width = candidate.size.width;
    float height = candidate.size.height;

    if (width <= 0 || height <= 0)
    {
        return 0.0;
    }

    if (height > width)
    {
        std::swap(width, height);
    }

    double aspectRatio = width / height;
    double idealAspectRatio = 4.0;

    double aspectScore = 1.0 / (1.0 + std::abs(aspectRatio - idealAspectRatio));

    Rect boundingBox = candidate.boundingRect();
    boundingBox &= Rect(0, 0, edges.cols, edges.rows);

    if (boundingBox.width <= 0 || boundingBox.height <= 0)
    {
        return 0.0;
    }

    Mat region = edges(boundingBox);

    double edgeDensity = static_cast<double>(countNonZero(region)) / boundingBox.area();

    return aspectScore + edgeDensity;
}

void PlateDetector::drawCandidates(Mat& image, const std::vector<RotatedRect>& candidates)
{
    for (const auto& candidate : candidates)
    {
        Point2f vertices[4];
        candidate.points(vertices);

        for (int i = 0; i < 4; i++)
        {
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }
}