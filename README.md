# License Plate Detection

## Overview

This project is a C++ and OpenCV-based license plate detection system. The program takes an input vehicle image, detects the most likely license plate region, extracts the plate, corrects its perspective, and prepares it for character detection.

The current system focuses on two major stages:

1. License plate detection
2. Character/text region detection

The plate detection stage uses image preprocessing, edge detection, morphology, contour detection, and a custom ranking system to select the most likely license plate region.

---

## Project Goals

The goal of this project is to detect a U.S. license plate from a vehicle image and prepare the detected plate for later character recognition.

The system attempts to:

- Detect rectangular plate-like regions in a vehicle image
- Rank candidate regions based on how likely they are to be a license plate
- Highlight the best-scoring plate candidate
- Extract the selected plate as a new image
- Apply perspective correction so the plate is front-facing
- Locate the main character/text region on the extracted plate

---

## Detection Pipeline

The general image-processing pipeline is:

```text
Input Image
→ Grayscale Conversion
→ Edge Detection
→ Morphological Closing
→ Contour Detection
→ Candidate Filtering
→ Candidate Ranking
→ Best Plate Selection
→ Perspective-Corrected Plate Extraction
→ Text Region Detection
```

The program starts by loading an image from the `images/` folder. The main image path is currently set in `main.cpp` using:

```cpp
Mat image = imread("images/test-car.jpg");
```

The program then uses `PlateDetector` to find and extract the best plate candidate before passing the extracted plate to `TextDetection`.

---

## Project Structure

```text
License-Plate-Detection/
│
├── CMakeLists.txt
├── main.cpp
├── PlateDetector.h
├── PlateDetector.cpp
├── TextDetection.h
├── TextDetection.cpp
├── TextRecognition.h
├── TextRecognition.cpp
│
└── images/
    ├── test-car.jpg
    ├── extracted-plate.jpg
    ├── text-region.jpg
    ├── binary-text-region.jpg
    └── detected-characters.jpg
```

### Main Files

| File | Purpose |
|---|---|
| `main.cpp` | Runs the full program pipeline |
| `PlateDetector.h/.cpp` | Handles license plate detection, scoring, drawing, extraction, and perspective correction |
| `TextDetection.h/.cpp` | Handles plate thresholding, text-region extraction, and character segmentation |
| `TextRecognition.h/.cpp` | Placeholder/future OCR recognition logic |
| `images/` | Stores input test images and generated output images |

---

## License Plate Detection

The `PlateDetector` class is responsible for locating the best license plate candidate.

It performs:

- Grayscale conversion
- Blur/noise reduction
- Canny edge detection
- Morphological closing
- Contour extraction
- Candidate filtering
- Candidate ranking
- Perspective correction

The detector selects the best plate candidate and extracts it using a perspective transform. The extracted plate is saved as:

```text
images/extracted-plate.jpg
```

---

## Candidate Ranking System

The plate detector does not simply choose the first rectangle it finds. Instead, each candidate receives a score.

Candidates are ranked using factors such as:

- **Aspect ratio**: favors wide U.S. plate-like rectangles
- **Area / size**: rejects regions that are too small or too large
- **Edge density**: rewards regions with strong internal detail
- **Angle**: gives preference to mostly horizontal candidates
- **Rectangularity**: favors clean rectangular regions
- **Character-like content**: looks for repeated vertical shapes that resemble letters or numbers
- **Location and border penalty**: penalizes unlikely regions such as image borders, bumpers, logos, or headlights

The highest-scoring region is selected as the best plate candidate.

---

## Text Detection

After the plate is extracted, the `TextDetection` class processes the plate image.

This stage attempts to:

- Convert the plate to grayscale
- Threshold the image into a binary image
- Detect character-like contours
- Segment each possible character
- Save intermediate output images

Generated text-related images include:

```text
images/text-region.jpg
images/binary-text-region.jpg
images/detected-characters.jpg
images/character-0.jpg
images/character-1.jpg
...
```

---

## Requirements

This project requires:

- C++
- CMake
- OpenCV 4
- A C++ compiler

On macOS, you can install CMake and OpenCV using Homebrew:

```bash
brew install cmake
brew install opencv
```

On Windows, install:

- Visual Studio
- OpenCV 4
- CMake

Make sure OpenCV is correctly added to your system path or configured through CMake.

---

## How to Build and Run

### macOS / Linux

From the project root folder:

```bash
rm -rf build
cmake -S . -B build
cmake --build build
./build/FinalProject
```

### Windows / Visual Studio

From the project root folder:

```bash
cmake -S . -B build
cmake --build build --config Debug
```

Then run the executable from Visual Studio or from the build output folder.

Depending on your generator, the executable may be located somewhere like:

```text
build/Debug/FinalProject.exe
```

---

## How to Use a Different Test Image

To test a different image, place the image inside the `images/` folder.

For example:

```text
images/car1.jpg
images/car2.jpg
images/hard-example.jpg
```

Then open `main.cpp` and change this line:

```cpp
Mat image = imread("images/test-car.jpg");
```

to whichever image you want to test:

```cpp
Mat image = imread("images/car1.jpg");
```

or:

```cpp
Mat image = imread("images/hard-example.jpg");
```

Then rebuild and run:

```bash
cmake --build build
./build/FinalProject
```

---

## Recommended Testing Workflow

For testing multiple examples, use this process:

1. Add a new image to the `images/` folder.
2. Change the image path in `main.cpp`.
3. Rebuild the project.
4. Run the program.
5. Check the output windows and generated images.

Example:

```cpp
Mat image = imread("images/test-car.jpg");
```

Change to:

```cpp
Mat image = imread("images/test-car-2.jpg");
```

Then run:

```bash
cmake --build build
./build/FinalProject
```

---

## Output Images

The program saves several output images into the `images/` folder:

| Output File | Description |
|---|---|
| `extracted-plate.jpg` | Perspective-corrected crop of the detected plate |
| `text-region.jpg` | Cropped region containing the main plate characters |
| `binary-text-region.jpg` | Binary thresholded text region |
| `detected-characters.jpg` | Text region with character boxes drawn |
| `character-0.jpg`, `character-1.jpg`, etc. | Individual segmented character images |

These outputs are useful for debugging and for showing results in the final project write-up.

---

## Notes

The current `TextRecognition` class is a placeholder. It currently returns question marks for detected characters until a classifier or recognition method is trained and implemented.

The main focus of the current implementation is reliable license plate detection, ranking, extraction, and preparation for character detection.

---

## Future Improvements

Possible future improvements include:

- Better handling of low-light images
- More robust detection for angled or partially blocked plates
- Better filtering of false positives such as headlights and grilles
- Improved text-region extraction
- Character recognition using a trained classifier
- Batch testing on multiple images
- Accuracy reporting with success/failure tables

---

## Team Write-Up Notes

For the final write-up, useful screenshots include:

- Original input image
- Detected license plate box
- Extracted plate image
- Text region image
- Binary text image
- Detected character boxes
- Segmented character images

These images help demonstrate each stage of the pipeline and show the value added beyond basic OpenCV image loading and edge detection.
