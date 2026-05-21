#ifndef TEXTRECOGNITION_H
#define TEXTRECOGNITION_H

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <map>

class TextRecognition
{
public:
    TextRecognition();

    std::string recognizePlate(
        const std::vector<cv::Mat> &characters);

private:
    std::map<char, cv::Mat> templates;

    void loadTemplates();

    cv::Mat preprocessCharacter(
        const cv::Mat &character);

    char recognizeCharacter(
        const cv::Mat &character);
};

#endif