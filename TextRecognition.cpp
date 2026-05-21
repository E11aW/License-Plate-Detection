#include "TextRecognition.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

TextRecognition::TextRecognition()
{
    loadTemplates();
}

/*
    Load template images from disk.
*/
void TextRecognition::loadTemplates()
{
    std::string characters =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (char c : characters)
    {
        std::string path =
            "templates/" + std::string(1, c) + ".png";

        cv::Mat image =
            cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (image.empty())
        {
            std::cout << "Could not load template: "
                      << path << std::endl;
            continue;
        }

        /*
            Ensure template is binary.
        */
        cv::threshold(
            image,
            image,
            128,
            255,
            cv::THRESH_BINARY);

        templates[c] = image;
    }
}

/*
    Normalize character image.
*/
cv::Mat TextRecognition::preprocessCharacter(
    const cv::Mat &character)
{
    cv::Mat resized;

    cv::resize(
        character,
        resized,
        cv::Size(32, 32));

    cv::threshold(
        resized,
        resized,
        128,
        255,
        cv::THRESH_BINARY);

    return resized;
}

/*
    Recognize a single character using template matching.
*/
char TextRecognition::recognizeCharacter(
    const cv::Mat &character)
{
    cv::Mat normalized =
        preprocessCharacter(character);

    char bestMatch = '?';

    double bestScore = 1e12;

    for (const auto &pair : templates)
    {
        char label = pair.first;

        cv::Mat templ =
            preprocessCharacter(pair.second);

        /*
            Compute pixel difference.
        */
        cv::Mat diff;

        cv::absdiff(normalized, templ, diff);

        double score =
            cv::sum(diff)[0];

        if (score < bestScore)
        {
            bestScore = score;
            bestMatch = label;
        }
    }

    return bestMatch;
}

/*
    Recognize full license plate.
*/
std::string TextRecognition::recognizePlate(
    const std::vector<cv::Mat> &characters)
{
    std::string result;

    for (const auto &character : characters)
    {
        char recognized =
            recognizeCharacter(character);

        result += recognized;
    }

    return result;
}