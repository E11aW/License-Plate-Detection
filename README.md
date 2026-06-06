License Plate Detection Project

Requirements:
Visual Studio 2022
OpenCV 4.x
CMake

How to Build:
1. Extract the zip file.
2. Open the folder in Visual Studio, or run build.bat.
3. Make sure OpenCV 4 is installed and available to CMake.
4. Build using x64 Debug.

How to Run:
1. Run run.bat after building.
2. The program uses the images folder for test images.
3. The templates folder is required for training the SVM/OCR system.

Important Folders:
images/
    Contains test car images.

templates/
    Contains character template images used to train the SVM.

Do not delete the templates folder.