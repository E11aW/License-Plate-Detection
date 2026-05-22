#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <random>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

/*
    --------------------------------------------------
    CONFIG
    --------------------------------------------------
*/

static const string LABELS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

static const int OUTPUT_SIZE = 32;

static const int SAMPLES_PER_CHARACTER = 200;

static const string OUTPUT_FOLDER = "templates";

/*
    --------------------------------------------------
    RANDOM HELPERS
    --------------------------------------------------
*/

std::random_device rd;
std::mt19937 rng(rd());

float randFloat(float minVal, float maxVal)
{
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    return dist(rng);
}

int randInt(int minVal, int maxVal)
{
    std::uniform_int_distribution<int> dist(minVal, maxVal);
    return dist(rng);
}

/*
    --------------------------------------------------
    NORMALIZATION
    --------------------------------------------------
*/

Mat normalizeCharacter(const Mat &src)
{
    if (src.empty())
        return Mat();

    Mat gray;

    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else
        gray = src.clone();

    threshold(
        gray,
        gray,
        0,
        255,
        THRESH_BINARY | THRESH_OTSU);

    int whitePixels = countNonZero(gray);

    if (whitePixels > gray.total() / 2)
    {
        bitwise_not(gray, gray);
    }

    vector<Point> points;
    findNonZero(gray, points);

    if (points.empty())
        return Mat();

    Rect box = boundingRect(points);

    Mat roi = gray(box);

    int target = 24;

    float scale = min(
        target / (float)roi.cols,
        target / (float)roi.rows);

    int newW = max(1, (int)(roi.cols * scale));
    int newH = max(1, (int)(roi.rows * scale));

    Mat resized;

    resize(
        roi,
        resized,
        Size(newW, newH),
        0,
        0,
        INTER_AREA);

    Mat output = Mat::zeros(
        OUTPUT_SIZE,
        OUTPUT_SIZE,
        CV_8U);

    int x = (OUTPUT_SIZE - newW) / 2;
    int y = (OUTPUT_SIZE - newH) / 2;

    resized.copyTo(output(Rect(x, y, newW, newH)));

    return output;
}

/*
    --------------------------------------------------
    SYNTHETIC CHARACTER GENERATION
    --------------------------------------------------
*/

Mat generateSyntheticCharacter(char c)
{
    Mat img = Mat::zeros(80, 80, CV_8U);

    int fontFace = FONT_HERSHEY_DUPLEX;

    double fontScale = randFloat(1.6f, 2.4f);

    int thickness = randInt(2, 5);

    string text(1, c);

    int baseline = 0;

    Size textSize = getTextSize(
        text,
        fontFace,
        fontScale,
        thickness,
        &baseline);

    int x = (img.cols - textSize.width) / 2 + randInt(-4, 4);
    int y = (img.rows + textSize.height) / 2 + randInt(-4, 4);

    putText(
        img,
        text,
        Point(x, y),
        fontFace,
        fontScale,
        Scalar(255),
        thickness,
        LINE_AA);

    /*
        Slight rotation
    */

    float angle = randFloat(-6.0f, 6.0f);

    Mat rot = getRotationMatrix2D(
        Point2f(img.cols / 2.0f, img.rows / 2.0f),
        angle,
        1.0);

    warpAffine(
        img,
        img,
        rot,
        img.size(),
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar(0));

    /*
        Random blur
    */

    if (randInt(0, 1))
    {
        GaussianBlur(
            img,
            img,
            Size(3, 3),
            randFloat(0.2f, 1.2f));
    }

    /*
        Random morphology
    */

    int morphChoice = randInt(0, 2);

    if (morphChoice == 1)
    {
        erode(
            img,
            img,
            getStructuringElement(MORPH_RECT, Size(2, 2)));
    }
    else if (morphChoice == 2)
    {
        dilate(
            img,
            img,
            getStructuringElement(MORPH_RECT, Size(2, 2)));
    }

    /*
     Add Gaussian noise
 */

    Mat noise(img.size(), CV_16S);

    randn(noise, 0, 12);

    Mat noisy;

    img.convertTo(noisy, CV_16S);

    add(noisy, noise, noisy);

    noisy.convertTo(img, CV_8U);

    /*
        Final threshold
    */

    threshold(
        img,
        img,
        0,
        255,
        THRESH_BINARY | THRESH_OTSU);

    return normalizeCharacter(img);
}

/*
    --------------------------------------------------
    MAIN
    --------------------------------------------------
*/

int main()
{
    cout << "Generating templates..." << endl;

    fs::create_directories(OUTPUT_FOLDER);

    for (char c : LABELS)
    {
        string folder = OUTPUT_FOLDER + string("/") + c;

        fs::create_directories(folder);

        cout << "Generating class: " << c << endl;

        for (int i = 0; i < SAMPLES_PER_CHARACTER; i++)
        {
            Mat character = generateSyntheticCharacter(c);

            if (character.empty())
                continue;

            string filename =
                folder + "/" + to_string(i) + ".png";

            imwrite(filename, character);
        }
    }

    cout << "Done." << endl;

    return 0;
}
