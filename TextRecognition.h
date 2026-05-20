#ifndef TEXTRECOGNITION_H
#define TEXTRECOGNITION_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

class TextRecognition
{
public:
    TextRecognition();

    // train classifier
    void train(
        const cv::Mat &trainingData,
        const cv::Mat &labels);

    // recognize one character
    char recognizeCharacter(const cv::Mat &characterImage);

    // recognize full plate
    std::string recognizePlate(
        const std::vector<cv::Mat> &characters);

private:
    cv::Ptr<cv::ml::KNearest> knn;

    cv::Mat preprocessCharacter(const cv::Mat &character);
};

#endif // TEXTRECOGNITION_H