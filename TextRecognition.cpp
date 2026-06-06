/*
    File contents:
    This file defines the TextRecognition class.
    The class trains an SVM classifier from template images and recognizes
    segmented license plate characters using HOG features.

    Assumptions:
    The templates folder exists in the working directory.
    Template subfolders are named with the character they represent.
    Segmented input characters are readable binary or grayscale images.
*/

#include "TextRecognition.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <filesystem>
#include <algorithm>

#include <iostream>

/*
    Function purpose:
    Construct the OCR recognizer, configure HOG and SVM settings, and train the model.

    Preconditions:
    The templates folder must exist in the working directory.

    Postconditions:
    The SVM is trained when template data is loaded successfully.
*/
TextRecognition::TextRecognition()
{
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    hog = cv::HOGDescriptor(
        cv::Size(48, 48),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9);

    /* Initialize the SVM classifier used for OCR prediction. */
    svm = cv::ml::SVM::create();

    svm->setType(cv::ml::SVM::C_SVC);

    svm->setKernel(cv::ml::SVM::LINEAR);

    svm->setC(2.0);

    svm->setTermCriteria(
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER +
                cv::TermCriteria::EPS,
            2000,
            1e-6));

    trainSVMFromTemplates();
}

/*
    Function purpose:
    Compute one row of HOG features for a character image.

    Preconditions:
    img must contain a readable character image.

    Postconditions:
    Returns a floating point feature row or an empty Mat if processing fails.
*/
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

/*
    Function purpose:
    Normalize a character crop into a centered 48 by 48 image.

    Preconditions:
    src must contain one character crop.

    Postconditions:
    Returns a normalized binary character image or an empty Mat if no foreground exists.
*/
cv::Mat TextRecognition::normalizeCharacter(const cv::Mat &src)
{
    cv::Mat gray;

    if (src.channels() == 3)
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src.clone();

    /* Convert to binary so HOG sees a consistent character shape. */
    cv::threshold(
        gray,
        gray,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU);

    /* Stabilize character strokes before feature extraction. */
    cv::morphologyEx(
        gray,
        gray,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(3, 3)));

    /* Add a border so character strokes are not clipped during normalization. */
    cv::copyMakeBorder(
        gray,
        gray,
        4,
        4,
        4,
        4,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));

    /* Determine foreground polarity by checking border pixels. */
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

    /* Invert the image when the background appears mostly white. */
    int borderPixels =
        gray.cols * 2 +
        gray.rows * 2;

    if (borderWhite > borderPixels / 2)
    {
        cv::bitwise_not(gray, gray);
    }

    /* Deskew the character using image moments. */
    cv::Moments m = cv::moments(gray, true);

    if (std::abs(m.mu02) > 1e-2)
    {
        double skew = m.mu11 / m.mu02;

        cv::Mat warpMat =
            (cv::Mat_<float>(2, 3)
                 << 1,
             skew, -0.5f * gray.rows * skew,
             0, 1, 0);

        cv::warpAffine(
            gray,
            gray,
            warpMat,
            gray.size(),
            cv::WARP_INVERSE_MAP | cv::INTER_LINEAR);
    }

    /* Find the tight foreground bounding box. */
    std::vector<cv::Point> points;
    cv::findNonZero(gray, points);

    if (points.empty())
        return cv::Mat();

    cv::Rect box = cv::boundingRect(points);

    cv::Mat roi = gray(box);

    /* Preserve the aspect ratio while fitting into the target area. */
    int target = 36;

    float scale = std::min(
        target / (float)roi.cols,
        target / (float)roi.rows);

    int newW = (int)(roi.cols * scale);
    int newH = (int)(roi.rows * scale);

    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(newW, newH));

    /* Center the resized character inside a 48 by 48 image. */
    cv::Mat output = cv::Mat::zeros(48, 48, CV_8U);

    int x = (48 - newW) / 2;
    int y = (48 - newH) / 2;

    resized.copyTo(output(cv::Rect(x, y, newW, newH)));

    return output;
}

/*
    Function purpose:
    Create shifted and filtered copies of a character template.

    Preconditions:
    src must contain a valid character image.
    out must be a valid vector for storing augmented images.

    Postconditions:
    Augmented images are appended to out.
*/
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

/*
    Function purpose:
    Load template images from disk and train the SVM classifier.

    Preconditions:
    The templates folder must contain character subfolders.

    Postconditions:
    The SVM is trained when enough valid template images are found.
*/
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

/*
    Function purpose:
    Predict one segmented character with the trained SVM.

    Preconditions:
    character must contain one segmented character image.
    The SVM should already have been trained.

    Postconditions:
    Returns a predicted character or question mark when prediction fails.
*/
char TextRecognition::predictCharacter(const cv::Mat &character)
{
    cv::Mat feature = computeHOG(character);

    if (feature.empty())
    {
        return '?';
    }

    float response =
        svm->predict(
            feature,
            cv::noArray(),
            cv::ml::StatModel::RAW_OUTPUT);

    int idx = static_cast<int>(response);

    if (idx >= 0 && idx < (int)labels.size())
        return labels[idx];

    return '?';
}

/*
    Function purpose:
    Recognize all segmented characters in left to right order.

    Preconditions:
    characterImages should contain segmented character crops.
    The SVM should already have been trained.

    Postconditions:
    Returns the recognized license plate text.
*/
std::string TextRecognition::recognizePlate(
    const std::vector<cv::Mat> &characterImages)
{
    std::string result;

    std::filesystem::create_directories("realchars");

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