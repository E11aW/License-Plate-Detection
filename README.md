License Plate Detection

Requirements:
Visual Studio 2022
OpenCV 4
Windows x64

Running the Program:
Extract the zip file.
Open in Visual Studio project with OpenCV.
Build the project using the x64 configuration.
Click Local Windows Debugger to run the program.
Before running, make sure the following folders are located in the project directory beside the solution file:
CMakeLists.txt
main.cpp
PlateDetector.cpp
PlateDetector.h
TextDetection.cpp
TextDetection.h
TextRecognition.cpp
TextRecognition.h
images/
templates/

The program automatically processes every JPG image located in the images folder.

Input Files
images

The images folder contains the test vehicle images that will be processed by the program.

Any JPG image placed in this folder will be processed when the program runs.

templates

The templates folder contains the character template images used to train the OCR classifier. This folder is required for character recognition and must not be removed.

Program Output

For each image in the images folder the program:

Detects the license plate.
Extracts the license plate region.
Preprocesses the plate image.
Segments the individual characters.
Trains the OCR classifier using the template images.
Recognizes the license plate text.

The recognized license plate text is displayed in the console window.

OpenCV windows display the original image, detected license plate, preprocessing results, segmented characters, and the final recognition result.

The program also saves intermediate character segmentation results generated during processing. These files can be used to inspect the OCR pipeline and verify recognition performance.

Included Files
CMakeLists.txt
main.cpp
PlateDetector.cpp
PlateDetector.h
TextDetection.cpp
TextDetection.h
TextRecognition.cpp
TextRecognition.h
images/
templates/

Notes:

The project was developed and tested using Visual Studio 2022 and OpenCV 4.

The templates folder must remain in the project directory because it is used to train the OCR classifier during execution.