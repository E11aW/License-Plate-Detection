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
            continue;
        }

        templates[c] = preprocessCharacter(img);
    }
}

cv::Mat TextRecognition::preprocessCharacter(const cv::Mat &character)
{
    cv::Mat gray;
    if (character.channels() == 3)
        cv::cvtColor(character, gray, cv::COLOR_BGR2GRAY);
    else
        gray = character.clone();

    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::bitwise_not(bin, bin);

    // Crop to bounding box of ink
    cv::Mat points;
    if (points.total() == 0)
    {
        return cv::Mat::zeros(32, 32, CV_8UC1);
    }
    cv::Rect bbox = cv::boundingRect(points);

    cv::Mat cropped = bin(bbox);

    // Pad to square
    int size = std::max(cropped.cols, cropped.rows);
    cv::Mat square = cv::Mat::zeros(size, size, CV_8UC1);

    cropped.copyTo(square(cv::Rect(
        (size - cropped.cols) / 2,
        (size - cropped.rows) / 2,
        cropped.cols,
        cropped.rows)));

    cv::Mat resized;
    cv::resize(square, resized, cv::Size(32, 32));

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

        // Normalize to [0,1] for better comparison
        input.convertTo(input, CV_32F, 1.0 / 255.0);
        templ.convertTo(templ, CV_32F, 1.0 / 255.0);

        cv::Mat result;

        cv::Mat diff;
        cv::absdiff(input, templ, diff);
        double score = cv::sum(diff)[0];

        if (bestScore < 0 || score < bestScore)
        {
            bestScore = score;
            bestChar = templateChar;
        }
    }

    return bestChar;
}

std::string TextRecognition::recognizePlate(const std::vector<cv::Mat> &characters)
{
    // Sort characters left to right based on bounding box x-coordinate
    std::vector<size_t> indices(characters.size());

    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b)
              {
                  return cv::boundingRect(characters[a]).x <
                         cv::boundingRect(characters[b]).x;
              });
    std::string result;

    for (size_t i : indices)
    {
        char recognized = recognizeCharacter(characters[i]);
        result.push_back(recognized);
    }

    return result;
}