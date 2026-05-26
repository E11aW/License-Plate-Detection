#include "TextRecognition.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <filesystem>
#include <algorithm>

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

    // Initialize SVM
    svm = cv::ml::SVM::create();

    svm->setType(cv::ml::SVM::C_SVC);

    svm->setKernel(cv::ml::SVM::RBF);

    svm->setGamma(0.5);

    svm->setC(12.5);

    svm->setTermCriteria(
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER +
                cv::TermCriteria::EPS,
            2000,
            1e-6));

    trainSVMFromTemplates();
}

cv::Mat TextRecognition::computeHOG(
    const cv::Mat &img)
{
    if (img.empty())
        return cv::Mat();

    cv::Mat normalized =
        normalizeCharacter(img);

    if (normalized.empty())
        return cv::Mat();

    std::vector<float> descriptors;

    hog.compute(normalized, descriptors);

    cv::Mat feature(descriptors);

    feature = feature.reshape(1, 1);

    feature.convertTo(feature, CV_32F);

    return feature.clone();
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

    /*
    Determine polarity using border pixels
    */

    int borderWhite = 0;

    for (int x = 0; x < gray.cols; x++)
    {
        if (gray.at<uchar>(0, x) > 0)
            borderWhite++;

        if (gray.at<uchar>(gray.rows - 1, x) > 0)
            borderWhite++;
    }

    for (int y = 0; y < gray.rows; y++)
    {
        if (gray.at<uchar>(y, 0) > 0)
            borderWhite++;

        if (gray.at<uchar>(y, gray.cols - 1) > 0)
            borderWhite++;
    }

    /*
        If border is mostly white, invert image
    */

    int borderPixels =
        gray.cols * 2 +
        gray.rows * 2;

    if (borderWhite > borderPixels / 2)
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

    for (size_t labelIdx = 0; labelIdx < labels.size(); labelIdx++)
    {
        char labelChar = labels[labelIdx];

        std::string folder =
            basePath + std::string(1, labelChar);

        if (!std::filesystem::exists(folder))
        {
            std::cout << "Missing folder: "
                      << folder
                      << std::endl;

            continue;
        }

        std::vector<std::string> imagePaths;

        for (const auto &entry :
             std::filesystem::directory_iterator(folder))
        {
            if (!entry.is_regular_file())
                continue;

            imagePaths.push_back(entry.path().string());
        }

        std::sort(imagePaths.begin(), imagePaths.end());

        std::cout << "Loading "
                  << imagePaths.size()
                  << " templates for "
                  << labelChar
                  << std::endl;

        for (const auto &path : imagePaths)
        {
            cv::Mat img =
                cv::imread(path, cv::IMREAD_GRAYSCALE);

            if (img.empty())
                continue;

            cv::Mat feature = computeHOG(img);

            if (feature.empty())
                continue;

            trainingData.push_back(feature);
            labelsVec.push_back((int)labelIdx);
        }
    }

    if (trainingData.empty())
    {
        std::cout << "ERROR: No training data loaded."
                  << std::endl;
        return;
    }

    int featureSize = trainingData[0].cols;

    cv::Mat trainMat(
        (int)trainingData.size(),
        featureSize,
        CV_32F);

    for (size_t i = 0; i < trainingData.size(); i++)
    {
        trainingData[i].copyTo(
            trainMat.row((int)i));
    }

    cv::Mat labelMat(labelsVec);

    std::cout << std::endl;
    std::cout << "Training SVM..." << std::endl;
    std::cout << "Samples: "
              << trainMat.rows
              << std::endl;
    std::cout << "Features: "
              << trainMat.cols
              << std::endl;

    svm->train(
        trainMat,
        cv::ml::ROW_SAMPLE,
        labelMat);

    std::cout << "SVM training complete."
              << std::endl;
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

        static int counter = 0;

        cv::imwrite(
            "realchars/" +
                std::to_string(counter++) +
                "_" +
                c +
                ".png",
            normalizeCharacter(img));
    }

    return result;
}