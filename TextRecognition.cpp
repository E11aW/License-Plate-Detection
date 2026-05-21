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
    svm->setKernel(cv::ml::SVM::LINEAR);
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

    cv::Mat gray;

    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    // normalize contrast (IMPORTANT)
    cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX);

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(32, 32));

    // enforce correct type
    resized.convertTo(resized, CV_8U);

    std::vector<float> descriptors;
    hog.compute(resized, descriptors);

    return cv::Mat(descriptors).reshape(1, 1);
}

void augmentImage(const cv::Mat &src, std::vector<cv::Mat> &out)
{
    out.push_back(src);

    cv::Mat tmp;

    // 1. slight blur
    cv::GaussianBlur(src, tmp, cv::Size(3, 3), 0);
    out.push_back(tmp);

    // 2. dilation
    cv::dilate(src, tmp, cv::getStructuringElement(cv::MORPH_RECT, {2, 2}));
    out.push_back(tmp);

    // 3. erosion
    cv::erode(src, tmp, cv::getStructuringElement(cv::MORPH_RECT, {2, 2}));
    out.push_back(tmp);

    // 4. slight threshold variation
    cv::threshold(src, tmp, 100, 255, cv::THRESH_BINARY);
    out.push_back(tmp);
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
        {
            continue;
        }

        std::vector<cv::Mat> augmented;
        augmentImage(base, augmented);

        for (const auto &img : augmented)
        {
            cv::Mat feature = computeHOG(img);

            trainingData.push_back(feature);
            labelsVec.push_back(i);
        }

        cv::Mat feature = computeHOG(img);

        trainingData.push_back(feature);
        labelsVec.push_back(i);
    }

    cv::Mat trainMat(trainingData.size(), trainingData[0].cols, CV_32F);

    for (size_t i = 0; i < trainingData.size(); i++)
    {
        trainingData[i].copyTo(trainMat.row(i));
    }

    cv::Mat labelMat(labelsVec, true);

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