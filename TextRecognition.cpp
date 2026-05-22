#include "TextRecognition.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <iostream>

TextRecognition::TextRecognition()
{
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    hog = cv::HOGDescriptor(
        cv::Size(32, 32),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9);

    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setGamma(0.5);
    svm->setC(12.5);
    svm->setC(1.0);
    svm->setTermCriteria(cv::TermCriteria(
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
        1000,
        1e-6));

    trainSVMFromTemplates();
}

cv::Mat TextRecognition::computeHOG(const cv::Mat &img)
{
    if (img.empty())
        return cv::Mat();

    cv::Mat normalized = normalizeCharacter(img);

    std::vector<float> descriptors;

    hog.compute(normalized, descriptors);

    cv::Mat feature(descriptors);

    feature = feature.reshape(1, 1);

    feature.convertTo(feature, CV_32F);

    return feature;
}

cv::Mat TextRecognition::normalizeCharacter(const cv::Mat &src)
{
    cv::Mat gray;

    if (src.channels() == 3)
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src.clone();

    // Ensure binary
    cv::threshold(
        gray,
        gray,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Ensure WHITE foreground on BLACK background
    int whitePixels = cv::countNonZero(gray);

    if (whitePixels > gray.total() / 2)
    {
        cv::bitwise_not(gray, gray);
    }

    // Find tight bounding box
    std::vector<cv::Point> points;
    cv::findNonZero(gray, points);

    if (points.empty())
        return cv::Mat();

    cv::Rect box = cv::boundingRect(points);

    cv::Mat roi = gray(box);

    // Preserve aspect ratio
    int target = 24;

    float scale = std::min(
        target / (float)roi.cols,
        target / (float)roi.rows);

    int newW = (int)(roi.cols * scale);
    int newH = (int)(roi.rows * scale);

    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(newW, newH));

    // Center into 32x32
    cv::Mat output = cv::Mat::zeros(32, 32, CV_8U);

    int x = (32 - newW) / 2;
    int y = (32 - newH) / 2;

    resized.copyTo(output(cv::Rect(x, y, newW, newH)));

    return output;
}

void TextRecognition::augmentImage(const cv::Mat &src, std::vector<cv::Mat> &out)
{
    out.push_back(src);

    for (int dx = -2; dx <= 2; dx++)
    {
        for (int dy = -2; dy <= 2; dy++)
        {
            cv::Mat shifted =
                cv::Mat::zeros(src.size(), src.type());

            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx,
                         0, 1, dy);

            cv::warpAffine(
                src,
                shifted,
                M,
                src.size());

            out.push_back(shifted);
        }
    }

    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(3, 3), 0);
    out.push_back(blur);

    cv::Mat eroded;
    cv::erode(
        src,
        eroded,
        cv::getStructuringElement(cv::MORPH_RECT, {2, 2}));
    out.push_back(eroded);

    cv::Mat dilated;
    cv::dilate(
        src,
        dilated,
        cv::getStructuringElement(cv::MORPH_RECT, {2, 2}));
    out.push_back(dilated);
}

void TextRecognition::trainSVMFromTemplates()
{
    std::vector<cv::Mat> trainingData;
    std::vector<int> labelsVec;

    std::string basePath = "templates/";

    for (size_t i = 0; i < labels.size(); i++)
    {
        std::string path = basePath + labels[i] + ".png";

        cv::Mat base = cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (base.empty())
            continue;

        std::vector<cv::Mat> augmented;
        augmentImage(base, augmented);

        for (const auto &img : augmented)
        {
            cv::Mat feature = computeHOG(img);

            trainingData.push_back(feature);
            labelsVec.push_back((int)i);
        }
    }

    cv::Mat trainMat(
        (int)trainingData.size(),
        trainingData[0].cols,
        CV_32F);

    for (size_t i = 0; i < trainingData.size(); i++)
    {
        trainingData[i].copyTo(trainMat.row((int)i));
    }

    cv::Mat labelMat(labelsVec);

    svm->train(trainMat, cv::ml::ROW_SAMPLE, labelMat);
}

char TextRecognition::predictCharacter(const cv::Mat &character)
{
    cv::Mat feature = computeHOG(character);

    float response = svm->predict(feature);

    int idx = static_cast<int>(response);

    if (idx >= 0 && idx < (int)labels.size())
        return labels[idx];

    return '?';
}

std::string TextRecognition::recognizePlate(
    const std::vector<cv::Mat> &characterImages)
{
    std::string result;

    for (const auto &img : characterImages)
    {
        if (img.empty())
            continue;

        char c = predictCharacter(img);
        result += c;
    }

    return result;
}