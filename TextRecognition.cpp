#include "TextRecognition.h"

TextRecognition::TextRecognition()
{
}

std::string TextRecognition::recognizePlate(
    const std::vector<cv::Mat> &characters)
{
    std::string result;

    // Placeholder implementation
    for (size_t i = 0; i < characters.size(); i++)
    {
        result += '?';
    }

    return result;
}