// main.cpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;

// Function to detect edges in an image using Canny edge detection
// Preconditions: input is a valid image matrix
// Postconditions: returns a matrix containing the detected edges
Mat detectEdges(const Mat &input)
{
    Mat gray, edges;

    // Convert the image to grayscale
    cvtColor(input, gray, COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    GaussianBlur(gray, gray, Size(5, 5), 1.5);

    // Use Canny edge detection
    Canny(gray, edges, 100, 200);
    return edges;
}

// Main function that .....
// Preconditions:
// Postconditions:
int main()
{
    // Load the test image and ensure it is valid
    Mat test = imread("test-car.jpg");
    if (test.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Detect edges in the image and display the result
    Mat output = detectEdges(test);
    imshow("Detected Edges", output);
    waitKey(0);

    return 0;
}
