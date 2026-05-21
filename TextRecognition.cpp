#include "TextRecognition.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

TextRecognition::TextRecognition()
{
    loadTemplates();
}

void TextRecognition::loadTemplates()
{
    std::string path = "templates/";

    std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (char c : chars)
    {
        std::string file = path + std::string(1, c) + ".png";

        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);

        if (img.empty())
        {
            std::cerr << "Missing template: " << file << std::endl;
            continue;
        }

        // Normalize to fixed size for matching
        cv::resize(img, img, cv::Size(32, 32));

        templates[c] = img;
    }
}

cv::Mat TextRecognition::preprocessCharacter(const cv::Mat &character)
{
    cv::Mat gray;

    if (character.channels() == 3)
    {
        cv::cvtColor(character, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = character.clone();
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(32, 32));

    cv::GaussianBlur(resized, resized, cv::Size(3, 3), 0);

    cv::threshold(
        resized,
        resized,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU
    );

    // Ensure same polarity as templates (important!)
    int whitePixels = cv::countNonZero(resized);

    if (whitePixels > (32 * 32) / 2)
    {
        cv::bitwise_not(resized, resized);
    }

    return resized;
}

char TextRecognition::recognizeCharacter(const cv::Mat &character)
{
    cv::Mat input = preprocessCharacter(character);

    char bestChar = '?';
    double bestScore = -1.0;

    for (const auto &pair : templates)
    {
        char templateChar = pair.first;
        cv::Mat templ = pair.second;

        cv::Mat result;

        cv::matchTemplate(input, templ, result, cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::minMaxLoc(result, &minVal, &maxVal);

        if (maxVal > bestScore)
        {
            bestScore = maxVal;
            bestChar = templateChar;
        }
    }

    return bestChar;
}

std::string TextRecognition::recognizePlate(
    const std::vector<cv::Mat> &characters)
{
    std::string result;

    for (const auto &ch : characters)
    {
        char recognized = recognizeCharacter(ch);
        result.push_back(recognized);
    }

    return result;
}