/*
    File contents:
    This file contains the main driver for the license plate detection program.
    The program loads every JPG image from a chosen image folder, detects the
    best license plate candidate, extracts the plate, segments the characters,
    and recognizes the plate text using the OCR system.

    Assumptions:
    The images folder exists in the project working directory.
    The templates folder exists in the project working directory.
    The templates folder contains character examples used to train the SVM.
    Input images are JPG files.
*/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "PlateDetector.h"
#include "TextDetection.h"
#include "TextRecognition.h"

using namespace cv;

/*
    Function purpose:
    Return true when a file path has the .jpg or .jpeg extension.

    Preconditions:
    The path must refer to a file system path.

    Postconditions:
    Returns true for JPG image paths and false for all other paths.
*/
bool isJpgImage(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();

    for (char& character : extension)
    {
        character = static_cast<char>(std::tolower(character));
    }

    return extension == ".jpg" || extension == ".jpeg";
}

/*
    Function purpose:
    Process one vehicle image through the full license plate pipeline.

    Preconditions:
    imagePath must point to a readable JPG image.
    plateDetector, textDetector, and recognizer must already be constructed.
    The templates folder must be available before the recognizer is constructed.

    Postconditions:
    The function displays intermediate results when a plate is found.
    The function writes extracted result images into the images folder.
    The function prints the recognized plate text to the console.
*/
void processImage(
    const std::filesystem::path& imagePath,
    PlateDetector& plateDetector,
    TextDetection& textDetector,
    TextRecognition& recognizer)
{
    Mat image = imread(imagePath.string());

    if (image.empty())
    {
        std::cerr << "Error: Could not load image: "
                  << imagePath.string()
                  << std::endl;
        return;
    }

    std::cout << std::endl;
    std::cout << "Processing image: "
              << imagePath.filename().string()
              << std::endl;

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
        destroyAllWindows();
        return;
    }

    plateDetector.drawBestPlate(image, bestPlate);
    imshow("Detected License Plate", image);

    if (extractedPlate.empty())
    {
        std::cout << "Extracted plate image is empty." << std::endl;
        destroyAllWindows();
        return;
    }

    std::filesystem::create_directories("results");

    imshow("Extracted Plate", extractedPlate);
    imwrite("results/extracted_plate_" + imagePath.stem().string() + ".jpg", extractedPlate);

    Mat textRegion = textDetector.extractMainTextRegion(extractedPlate);

    if (textRegion.empty())
    {
        std::cout << "Could not extract main text region." << std::endl;
        destroyAllWindows();
        return;
    }

    imshow("Main Text Region", textRegion);
    imwrite("results/text_region_" + imagePath.stem().string() + ".jpg", textRegion);

    Mat binaryPlate = textDetector.preprocessPlate(textRegion);

    imshow("Binary Text Region", binaryPlate);
    imwrite("results/binary_text_region_" + imagePath.stem().string() + ".jpg", binaryPlate);

    std::vector<Rect> characterRegions =
        textDetector.detectCharacterRegions(binaryPlate);

    std::vector<Mat> characterImages =
        textDetector.segmentCharacters(
            binaryPlate,
            characterRegions);

    Mat characterDisplay = textRegion.clone();

    for (const auto& rect : characterRegions)
    {
        rectangle(
            characterDisplay,
            rect,
            Scalar(0, 255, 0),
            2);
    }

    imshow("Detected Characters", characterDisplay);
    imwrite("results/detected_characters_" + imagePath.stem().string() + ".jpg", characterDisplay);

    std::string recognizedPlate = recognizer.recognizePlate(characterImages);

    std::cout << "Recognized Plate: "
              << recognizedPlate
              << std::endl;

    for (size_t i = 0; i < characterImages.size(); i++)
    {
        std::string windowName =
            "Character " + std::to_string(i);

        imshow(windowName, characterImages[i]);
    }

    waitKey(0);
    destroyAllWindows();
}

/*
    Function purpose:
    Start the program, locate JPG test images, train the OCR system,
    and process each image one at a time.

    Preconditions:
    OpenCV must be installed and linked correctly.
    The image folder path below must point to the folder containing test images.
    The templates folder must be available in the working directory.

    Postconditions:
    Every JPG image in the image folder is processed once.
    Intermediate result images are saved in the results folder.
    Recognized plate text is printed to the console.
*/
int main()
{
    /*
        Put the test image folder path here.
        Use "images" when the images folder is next to the Visual Studio project files.
        Example: std::filesystem::path imageFolder = "images";
    */
    std::filesystem::path imageFolder = "images";

    if (!std::filesystem::exists(imageFolder))
    {
        std::cerr << "Error: Image folder not found: "
                  << imageFolder.string()
                  << std::endl;
        return 1;
    }

    PlateDetector plateDetector;
    TextDetection textDetector;
    TextRecognition recognizer;

    int imagesProcessed = 0;

    for (const auto& entry : std::filesystem::directory_iterator(imageFolder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (!isJpgImage(entry.path()))
        {
            continue;
        }

        processImage(
            entry.path(),
            plateDetector,
            textDetector,
            recognizer);

        imagesProcessed++;
    }

    if (imagesProcessed == 0)
    {
        std::cout << "No JPG images were found in the image folder." << std::endl;
    }

    return 0;
}
