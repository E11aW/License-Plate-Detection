#include "PlateDetector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cv;

bool PlateDetector::detectBestPlate(
    const Mat& input,
    RotatedRect& bestPlate,
    Mat& extractedPlate
)
{
    Mat gray = preprocessImage(input);
    Mat edges = detectEdges(gray);
    Mat strengthened = strengthenPlateRegions(edges);

    std::vector<std::vector<Point>> contours;
    findContours(strengthened, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double bestScore = 0.0;
    bool foundPlate = false;

    for (const auto& contour : contours)
    {
        if (!isLicensePlateCandidate(contour, input))
        {
            continue;
        }

        RotatedRect candidate = minAreaRect(contour);
        double score = scoreCandidate(candidate, edges, input);

        if (score > bestScore)
        {
            bestScore = score;
            bestPlate = candidate;
            foundPlate = true;
        }
    }

    if (foundPlate)
    {
        extractedPlate = extractPlateRegion(input, bestPlate);
    }

    return foundPlate;
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

bool PlateDetector::isLicensePlateCandidate(
    const std::vector<Point>& contour,
    const Mat& image
)
{
    double area = contourArea(contour);

    double imageArea = image.rows * image.cols;

    if (area < imageArea * 0.0005)
    {
        return false;
    }

    if (area > imageArea * 0.25)
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

    // Most license plates are wide rectangles.
    if (aspectRatio < 2.0 || aspectRatio > 6.5)
    {
        return false;
    }

    Rect boundingBox = box.boundingRect();
    boundingBox &= Rect(0, 0, image.cols, image.rows);

    if (boundingBox.width <= 0 || boundingBox.height <= 0)
    {
        return false;
    }

    return true;
}

double PlateDetector::scoreCandidate(
    const RotatedRect& candidate,
    const Mat& edges,
    const Mat& image
)
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

    // A common license plate ratio is around 4:1.
    double idealAspectRatio = 4.0;
    double aspectScore = 1.0 / (1.0 + std::abs(aspectRatio - idealAspectRatio));

    Rect boundingBox = candidate.boundingRect();
    boundingBox &= Rect(0, 0, image.cols, image.rows);

    if (boundingBox.width <= 0 || boundingBox.height <= 0)
    {
        return 0.0;
    }

    Mat edgeRegion = edges(boundingBox);

    double edgeDensity =
        static_cast<double>(countNonZero(edgeRegion)) /
        static_cast<double>(boundingBox.area());

    double area = boundingBox.area();
    double imageArea = image.rows * image.cols;

    double relativeArea = area / imageArea;

    // Prefer regions that are not tiny but also not huge.
    double areaScore = 1.0;

    if (relativeArea < 0.005)
    {
        areaScore = relativeArea / 0.005;
    }
    else if (relativeArea > 0.12)
    {
        areaScore = 0.12 / relativeArea;
    }

    // Penalize steep angles. Most plates are close to horizontal.
    double angle = std::abs(candidate.angle);

    if (angle > 45.0)
    {
        angle = 90.0 - angle;
    }

    double angleScore = 1.0 / (1.0 + angle / 20.0);

    double finalScore =
        (2.0 * aspectScore) +
        (3.0 * edgeDensity) +
        (1.5 * areaScore) +
        (1.0 * angleScore);

    return finalScore;
}

void PlateDetector::drawBestPlate(Mat& image, const RotatedRect& plate)
{
    Point2f vertices[4];
    plate.points(vertices);

    for (int i = 0; i < 4; i++)
    {
        line(
            image,
            vertices[i],
            vertices[(i + 1) % 4],
            Scalar(0, 255, 0),
            3
        );
    }
}

Mat PlateDetector::extractPlateRegion(
    const Mat& input,
    const RotatedRect& plate
)
{
    Point2f points[4];
    Point2f ordered[4];

    plate.points(points);
    orderPoints(points, ordered);

    float widthA = static_cast<float>(norm(ordered[2] - ordered[3]));
    float widthB = static_cast<float>(norm(ordered[1] - ordered[0]));
    float maxWidth = std::max(widthA, widthB);

    float heightA = static_cast<float>(norm(ordered[1] - ordered[2]));
    float heightB = static_cast<float>(norm(ordered[0] - ordered[3]));
    float maxHeight = std::max(heightA, heightB);

    if (maxWidth <= 0 || maxHeight <= 0)
    {
        return Mat();
    }

    Point2f destination[4];

    destination[0] = Point2f(0, 0);
    destination[1] = Point2f(maxWidth - 1, 0);
    destination[2] = Point2f(maxWidth - 1, maxHeight - 1);
    destination[3] = Point2f(0, maxHeight - 1);

    Mat transform = getPerspectiveTransform(ordered, destination);

    Mat warped;
    warpPerspective(
        input,
        warped,
        transform,
        Size(static_cast<int>(maxWidth), static_cast<int>(maxHeight))
    );

    return warped;
}

void PlateDetector::orderPoints(
    Point2f points[4],
    Point2f ordered[4]
)
{
    // ordered[0] = top-left
    // ordered[1] = top-right
    // ordered[2] = bottom-right
    // ordered[3] = bottom-left

    Point2f topLeft;
    Point2f topRight;
    Point2f bottomRight;
    Point2f bottomLeft;

    double minSum = points[0].x + points[0].y;
    double maxSum = minSum;

    double minDiff = points[0].y - points[0].x;
    double maxDiff = minDiff;

    topLeft = points[0];
    bottomRight = points[0];
    topRight = points[0];
    bottomLeft = points[0];

    for (int i = 1; i < 4; i++)
    {
        double sum = points[i].x + points[i].y;
        double diff = points[i].y - points[i].x;

        if (sum < minSum)
        {
            minSum = sum;
            topLeft = points[i];
        }

        if (sum > maxSum)
        {
            maxSum = sum;
            bottomRight = points[i];
        }

        if (diff < minDiff)
        {
            minDiff = diff;
            topRight = points[i];
        }

        if (diff > maxDiff)
        {
            maxDiff = diff;
            bottomLeft = points[i];
        }
    }

    ordered[0] = topLeft;
    ordered[1] = topRight;
    ordered[2] = bottomRight;
    ordered[3] = bottomLeft;
}