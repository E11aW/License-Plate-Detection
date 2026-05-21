#ifndef TEXT_RECOGNITION_H
#define TEXT_RECOGNITION_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include <iostream>

class TextRecognition
{
public:
    TextRecognition();

    std::string recognizePlate(const std::vector<cv::Mat> &characterImages);

private:
    cv::Ptr<cv::ml::SVM> svm;

    cv::HOGDescriptor hog;

    std::string labels;

    void trainSVMFromTemplates();

    cv::Mat computeHOG(const cv::Mat &img);

    char predictCharacter(const cv::Mat &character);
};

#endif