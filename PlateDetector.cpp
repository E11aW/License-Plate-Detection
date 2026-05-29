#include "PlateDetector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cv;

bool PlateDetector::detectBestPlate(
    const Mat &input,
    RotatedRect &bestPlate,
    Mat &extractedPlate)
{
    Mat gray = preprocessImage(input);

    /*
        Use multiple edge-generation strategies.

        Normal Canny:
        - Good for plate borders and car body edges.

        Blackhat Canny:
        - Good for dark plate characters on a lighter plate background.
    */
    std::vector<Mat> edgeImages;
    edgeImages.push_back(detectEdges(gray));
    edgeImages.push_back(detectBlackhatEdges(gray));

    /*
        Try multiple morphology kernels.

        This helps with different image sizes:
        - small/far plates need smaller kernels
        - close-up plates can use larger kernels
    */
    std::vector<Size> kernelSizes = {
        Size(13, 3),
        Size(17, 5),
        Size(25, 7),
        Size(35, 9)
    };

    double bestScore = 0.0;
    bool foundPlate = false;

    for (const auto& contour : contours)
    {
        if (!isLicensePlateCandidate(contour, input))
        {
            continue;
        }

                double score = scoreCandidate(
                    contour,
                    candidate,
                    edges,
                    input
                );

                if (score > bestScore)
                {
                    bestScore = score;
                    bestPlate = candidate;
                    foundPlate = true;
                }
            }
        }
    }

    if (foundPlate)
    {
        extractedPlate = extractPlateRegion(input, bestPlate);

        std::cout << "Best plate score: "
                  << bestScore
                  << std::endl;
    }

    return foundPlate;
}

Mat PlateDetector::preprocessImage(const Mat &input)
{
    Mat gray;
    Mat equalized;
    Mat blurred;

    cvtColor(input, gray, COLOR_BGR2GRAY);

    /*
        Equalization helps with hard images that have shadows,
        glare, or low contrast.
    */
    equalizeHist(gray, equalized);

    GaussianBlur(equalized, blurred, Size(5, 5), 1.5);

    return blurred;
}

Mat PlateDetector::detectEdges(const Mat &gray)
{
    Mat edges;

    Canny(gray, edges, 60, 180);

    return edges;
}

Mat PlateDetector::detectBlackhatEdges(const Mat& gray)
{
    Mat blackhat;
    Mat normalized;
    Mat edges;

    /*
        Blackhat emphasizes dark objects on light backgrounds.
        This is useful because many US plates have dark letters
        on a lighter plate surface.
    */
    Mat kernel = getStructuringElement(
        MORPH_RECT,
        Size(17, 5)
    );

    morphologyEx(
        gray,
        blackhat,
        MORPH_BLACKHAT,
        kernel
    );

    normalize(
        blackhat,
        normalized,
        0,
        255,
        NORM_MINMAX
    );

    GaussianBlur(
        normalized,
        normalized,
        Size(3, 3),
        0
    );

    Canny(
        normalized,
        edges,
        30,
        120
    );

    return edges;
}

Mat PlateDetector::strengthenPlateRegions(const Mat& edges)
{
    Mat strengthened;

    Mat kernel = getStructuringElement(
        MORPH_RECT,
        kernelSize
    );

    morphologyEx(
        edges,
        strengthened,
        MORPH_CLOSE,
        kernel
    );

    /*
        Small dilation helps connect broken plate borders
        and character edges without completely filling the image.
    */
    Mat dilateKernel = getStructuringElement(
        MORPH_RECT,
        Size(3, 3)
    );

    dilate(
        strengthened,
        strengthened,
        dilateKernel,
        Point(-1, -1),
        1
    );

    return strengthened;
}

bool PlateDetector::isLicensePlateCandidate(
    const std::vector<Point>& contour,
    const Mat& image
)
{
    double contourAreaValue = contourArea(contour);
    double imageArea = static_cast<double>(image.rows * image.cols);

    if (contourAreaValue < imageArea * 0.0003)
    {
        return false;
    }

    if (contourAreaValue > imageArea * 0.20)
    {
        return false;
    }

    float width = candidate.size.width;
    float height = candidate.size.height;

    if (width <= 0 || height <= 0)
    {
        return false;
    }

    if (height > width)
    {
        std::swap(width, height);
    }

    double aspectRatio = width / height;

    /*
        US plates are usually wide rectangles.
        This range is intentionally a little loose so angled
        or partially detected plates can still survive.
    */
    if (aspectRatio < 1.8 || aspectRatio > 7.0)
    {
        return false;
    }

    Rect boundingBox = candidate.boundingRect();
    boundingBox &= Rect(0, 0, image.cols, image.rows);

    if (boundingBox.width <= 0 || boundingBox.height <= 0)
    {
        return false;
    }

    /*
        Relative dimension filters help reject tiny logos,
        headlights, bumpers, and very large car-body regions.
    */
    double relativeWidth =
        static_cast<double>(boundingBox.width) /
        static_cast<double>(image.cols);

    double relativeHeight =
        static_cast<double>(boundingBox.height) /
        static_cast<double>(image.rows);

    if (relativeWidth < 0.05 || relativeWidth > 0.85)
    {
        return false;
    }

    if (relativeHeight < 0.015 || relativeHeight > 0.35)
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

    /*
        A common US plate shape is roughly 4:1 to 5:1.
        This is a score, not a hard requirement.
    */
    double idealAspectRatio = 4.5;
    double aspectScore =
        1.0 / (1.0 + std::abs(aspectRatio - idealAspectRatio));

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

    /*
        Area score:
        - tiny candidates are suspicious
        - huge candidates are suspicious
    */
    double area = static_cast<double>(boundingBox.area());
    double imageArea = static_cast<double>(image.rows * image.cols);
    double relativeArea = area / imageArea;

    double areaScore = 1.0;

    if (relativeArea < 0.003)
    {
        areaScore = relativeArea / 0.003;
    }
    else if (relativeArea > 0.12)
    {
        areaScore = 0.12 / relativeArea;
    }

    areaScore = std::max(0.0, std::min(1.0, areaScore));

    /*
        Angle score:
        - most plates are close to horizontal
        - angled plates are still allowed, just ranked lower
    */
    double angle = std::abs(candidate.angle);

    if (angle > 45.0)
    {
        angle = 90.0 - angle;
    }

    double angleScore =
        1.0 / (1.0 + angle / 20.0);

    /*
        Rectangularity:
        - compares actual contour area to rotated rectangle area
        - helps reject jagged or sparse false positives
    */
    double rotatedBoxArea =
        static_cast<double>(candidate.size.width) *
        static_cast<double>(candidate.size.height);

    double rectangularityScore = 0.0;

    if (rotatedBoxArea > 0.0)
    {
        double rectangularity =
            contourArea(contour) / rotatedBoxArea;

        rectangularityScore =
            std::max(0.0, std::min(1.0, rectangularity / 0.75));
    }

    /*
        Character-like score:
        - checks whether the candidate contains multiple tall,
          narrow components similar to plate letters/numbers.
    */
    double characterScore =
        scoreCharacterLikeContent(candidate, edges, image);

    double borderScore =
        scoreBorderPenalty(boundingBox, image);

    double locationScore =
        scorePlateLocation(candidate, image);

    double finalScore =
        (2.0 * aspectScore) +
        (2.0 * edgeDensity) +
        (1.5 * areaScore) +
        (1.0 * angleScore) +
        (1.5 * rectangularityScore) +
        (3.0 * characterScore) +
        (0.75 * locationScore);

    finalScore *= borderScore;

    return finalScore;
}

double PlateDetector::scoreCharacterLikeContent(
    const RotatedRect& candidate,
    const Mat& edges,
    const Mat& image
)
{
    Rect boundingBox = candidate.boundingRect();
    boundingBox &= Rect(0, 0, image.cols, image.rows);

    if (boundingBox.width <= 0 || boundingBox.height <= 0)
    {
        return 0.0;
    }

    Mat roi = edges(boundingBox).clone();

    std::vector<std::vector<Point>> contours;

    findContours(
        roi,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE
    );

    int characterLikeCount = 0;

    for (const auto& contour : contours)
    {
        Rect r = boundingRect(contour);

        if (r.width <= 0 || r.height <= 0)
        {
            continue;
        }

        double aspect =
            static_cast<double>(r.width) /
            static_cast<double>(r.height);

        bool tallEnough =
            r.height >= boundingBox.height * 0.20;

        bool notTooTall =
            r.height <= boundingBox.height * 0.95;

        bool reasonableWidth =
            r.width >= boundingBox.width * 0.01 &&
            r.width <= boundingBox.width * 0.25;

        bool characterAspect =
            aspect >= 0.08 &&
            aspect <= 1.30;

        bool reasonableArea =
            r.area() >= boundingBox.area() * 0.002 &&
            r.area() <= boundingBox.area() * 0.25;

        if (tallEnough &&
            notTooTall &&
            reasonableWidth &&
            characterAspect &&
            reasonableArea)
        {
            characterLikeCount++;
        }
    }

    /*
        US plates commonly have around 5-8 large characters.
        We do not require exactly that because some contours merge
        and some states have separators.
    */
    if (characterLikeCount < 2)
    {
        return 0.0;
    }

    if (characterLikeCount >= 5 && characterLikeCount <= 9)
    {
        return 1.0;
    }

    if (characterLikeCount > 9)
    {
        return 0.6;
    }

    return static_cast<double>(characterLikeCount) / 5.0;
}

double PlateDetector::scoreBorderPenalty(
    const Rect& boundingBox,
    const Mat& image
)
{
    int marginX = static_cast<int>(image.cols * 0.02);
    int marginY = static_cast<int>(image.rows * 0.02);

    bool touchesBorder =
        boundingBox.x <= marginX ||
        boundingBox.y <= marginY ||
        boundingBox.x + boundingBox.width >= image.cols - marginX ||
        boundingBox.y + boundingBox.height >= image.rows - marginY;

    if (touchesBorder)
    {
        return 0.40;
    }

    return 1.0;
}

double PlateDetector::scorePlateLocation(
    const RotatedRect& candidate,
    const Mat& image
)
{
    double centerX =
        candidate.center.x /
        static_cast<double>(image.cols);

    double centerY =
        candidate.center.y /
        static_cast<double>(image.rows);

    /*
        Plates are often near the horizontal center,
        but this should only be a soft score because
        side-angle photos can place plates off-center.
    */
    double centerXScore =
        1.0 - std::abs(centerX - 0.5);

    centerXScore =
        std::max(0.0, std::min(1.0, centerXScore));

    /*
        Front/rear license plates are usually not at the very
        top of the image. This is intentionally weak.
    */
    double yScore = 1.0;

    if (centerY < 0.20)
    {
        yScore = 0.45;
    }
    else if (centerY > 0.95)
    {
        yScore = 0.60;
    }

    return centerXScore * yScore;
}

void PlateDetector::drawBestPlate(Mat &image, const RotatedRect &plate)
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
            3);
    }
}

Mat PlateDetector::extractPlateRegion(
    const Mat &input,
    const RotatedRect &plate)
{
    Point2f points[4];
    Point2f ordered[4];

    plate.points(points);
    orderPoints(points, ordered);

    float widthA =
        static_cast<float>(norm(ordered[2] - ordered[3]));

    float widthB =
        static_cast<float>(norm(ordered[1] - ordered[0]));

    float maxWidth =
        std::max(widthA, widthB);

    float heightA =
        static_cast<float>(norm(ordered[1] - ordered[2]));

    float heightB =
        static_cast<float>(norm(ordered[0] - ordered[3]));

    float maxHeight =
        std::max(heightA, heightB);

    if (maxWidth <= 0 || maxHeight <= 0)
    {
        return Mat();
    }

    Point2f destination[4];

    destination[0] = Point2f(0, 0);
    destination[1] = Point2f(maxWidth - 1, 0);
    destination[2] = Point2f(maxWidth - 1, maxHeight - 1);
    destination[3] = Point2f(0, maxHeight - 1);

    Mat transform =
        getPerspectiveTransform(ordered, destination);

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
    Point2f ordered[4])
{
    /*
        ordered[0] = top-left
        ordered[1] = top-right
        ordered[2] = bottom-right
        ordered[3] = bottom-left
    */

    Point2f topLeft = points[0];
    Point2f topRight = points[0];
    Point2f bottomRight = points[0];
    Point2f bottomLeft = points[0];

    double minSum = points[0].x + points[0].y;
    double maxSum = minSum;

    double minDiff = points[0].y - points[0].x;
    double maxDiff = minDiff;

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