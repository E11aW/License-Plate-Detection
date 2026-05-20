#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "PlateDetector.h"
#include "TextDetection.h"
#include "TextRecognition.h"

using namespace cv;

int main()
{
    Mat image = imread("images/test-car.jpg");

    if (image.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    /*
        -----------------------------
        Plate Detection
        -----------------------------
    */

    PlateDetector plateDetector;

    RotatedRect bestPlate;
    Mat extractedPlate;

    bool foundPlate = plateDetector.detectBestPlate(
        image,
        bestPlate,
        extractedPlate);

    if (!foundPlate)
    {
        std::cout << "No license plate candidate found." << std::endl;

        imshow("Original Image", image);
        waitKey(0);

        return 0;
    }

    plateDetector.drawBestPlate(image, bestPlate);

    imshow("Detected License Plate", image);

    /*
        -----------------------------
        Save / Show Extracted Plate
        -----------------------------
    */

    if (extractedPlate.empty())
    {
        std::cout << "Extracted plate image is empty." << std::endl;
        return 0;
    }

    imshow("Extracted Plate", extractedPlate);

    imwrite("images/extracted-plate.jpg", extractedPlate);

    /*
        -----------------------------
        Text Detection
        -----------------------------
    */

    TextDetection textDetector;

    // Convert plate to binary image
    Mat binaryPlate = textDetector.preprocessPlate(extractedPlate);

    imshow("Binary Plate", binaryPlate);

    // Find character bounding boxes
    std::vector<Rect> characterRegions =
        textDetector.detectCharacterRegions(binaryPlate);

    // Segment character images
    std::vector<Mat> characterImages =
        textDetector.segmentCharacters(
            binaryPlate,
            characterRegions);

    /*
        Draw character boxes for visualization
    */

    Mat characterDisplay = extractedPlate.clone();

    for (const auto &rect : characterRegions)
    {
        rectangle(
            characterDisplay,
            rect,
            Scalar(0, 255, 0),
            2);
    }

    imshow("Detected Characters", characterDisplay);

    /*
        -----------------------------
        Text Recognition
        -----------------------------
    */

    TextRecognition recognizer;

    /*
        NOTE:
        You must train your classifier before
        recognition will work correctly.

        Example future usage:

        recognizer.train(trainingData, labels);
    */

    std::string recognizedPlate =
        recognizer.recognizePlate(characterImages);

    std::cout << "Recognized Plate: "
              << recognizedPlate
              << std::endl;

    /*
        -----------------------------
        Show Segmented Characters
        -----------------------------
    */

    for (size_t i = 0; i < characterImages.size(); i++)
    {
        std::string windowName =
            "Character " + std::to_string(i);

        imshow(windowName, characterImages[i]);
    }

    waitKey(0);

    return 0;
}