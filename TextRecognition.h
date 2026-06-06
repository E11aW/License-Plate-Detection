/*
    File contents:
    This header declares the TextRecognition class.
    The class trains an OpenCV SVM from character template images and uses
    HOG features to recognize segmented license plate characters.

    Assumptions:
    The templates folder exists in the program working directory.
    Each character has its own folder inside templates.
    Character images are already segmented before recognition is called.
*/

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
    /*
        Function purpose:
        Construct the OCR recognizer and train the SVM from templates.

        Preconditions:
        The templates folder must be available in the working directory.

        Postconditions:
        The SVM is trained if template images are found.
    */
    TextRecognition();

    /*
        Function purpose:
        Recognize a sequence of segmented character images.

        Preconditions:
        characterImages should contain binary or grayscale character crops.
        The SVM should already be trained.

        Postconditions:
        Returns the recognized plate text as a string.
    */
    std::string recognizePlate(const std::vector<cv::Mat>& characterImages);

private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::HOGDescriptor hog;
    std::string labels;

    /*
        Function purpose:
        Load template images and train the SVM classifier.

        Preconditions:
        The templates folder must contain subfolders for letters and numbers.

        Postconditions:
        The SVM contains a trained model when data is loaded successfully.
    */
    void trainSVMFromTemplates();

    /*
        Function purpose:
        Compute HOG feature data for one character image.

        Preconditions:
        img must contain a readable character image.

        Postconditions:
        Returns one row of floating point feature values.
    */
    cv::Mat computeHOG(const cv::Mat& img);

    /*
        Function purpose:
        Normalize a character crop before HOG feature extraction.

        Preconditions:
        src must contain one character image.

        Postconditions:
        Returns a centered 48 by 48 binary image.
    */
    cv::Mat normalizeCharacter(const cv::Mat& src);

    /*
        Function purpose:
        Predict one character using the trained SVM.

        Preconditions:
        character must contain one segmented character image.
        The SVM should already be trained.

        Postconditions:
        Returns the predicted character or question mark when prediction fails.
    */
    char predictCharacter(const cv::Mat& character);

    /*
        Function purpose:
        Create augmented versions of a template image for training.

        Preconditions:
        src must contain a valid character template image.
        out must be a valid vector for storing generated images.

        Postconditions:
        Additional shifted and filtered images are added to out.
    */
    void augmentImage(const cv::Mat& src, std::vector<cv::Mat>& out);
};

#endif
