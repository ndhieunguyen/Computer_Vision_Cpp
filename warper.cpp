#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat imgOriginal, imgGray, imgBlur, imgCanny, imgThre, imgDil, imgErode, imgWarp, imgCrop;
vector<Point> initialPoints, docPoints;
float w = 420, h = 596;

/**
 * The function performs pre-processing on an input image by converting it to grayscale, applying
 * Gaussian blur, detecting edges using Canny algorithm, and dilating the edges using a morphological
 * operation.
 *
 * Args:
 *   img (Mat): The input image that needs to be preprocessed.
 *
 * Returns:
 *   a processed image after applying various pre-processing techniques such as converting the image to
 * grayscale, applying Gaussian blur, detecting edges using Canny edge detection, and dilating the
 * edges using a morphological operation. The returned image is the dilated version of the Canny
 * edge-detected image.
 */
Mat preProcessing(Mat img)
{
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);
    return imgDil;
}

/**
 * The function returns the largest contour with four points in a given image.
 *
 * Args:
 *   image (Mat): A grayscale or binary image on which the contours are to be found.
 *
 * Returns:
 *   a vector of Points representing the coordinates of the four corners of the largest contour with
 * four sides in the input image.
 */
vector<Point> getContours(Mat image)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point> biggest;
    int maxArea = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);
        string objectType;
        float peri = arcLength(contours[i], true);
        approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

        if (area > maxArea && conPoly[i].size() == 4)
        {
            biggest = {conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
            maxArea = area;
        }
    }
    return biggest;
}

/**
 * The function draws circles and text on an image for each point in a vector of points.
 *
 * Args:
 *   points (vector<Point>): A vector of Point objects representing the coordinates of the points to be
 * drawn on the image.
 *   color (Scalar): The color of the circle and text that will be drawn on the image. It is of type
 * Scalar, which is a 4-element vector representing the color in the order of blue, green, red, and
 * alpha (transparency) values. For example, Scalar(255, 0,
 */
void drawPoints(vector<Point> points, Scalar color)
{
    for (int i = 0; i < points.size(); i++)
    {
        circle(imgOriginal, points[i], 10, color, FILLED);
        putText(imgOriginal, to_string(i), points[i], FONT_HERSHEY_PLAIN, 4, color, 4);
    }
}

/**
 * The function reorders a vector of points based on their x and y coordinates.
 *
 * Args:
 *   points (vector<Point>): a vector of Point objects representing the coordinates of four points in a
 * 2D plane.
 *
 * Returns:
 *   The function `reorder` returns a vector of `Point` objects, which contains the same points as the
 * input vector `points`, but in a different order. The order is determined by the following criteria:
 */
vector<Point> reorder(vector<Point> points)
{
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints;

    for (int i = 0; i < 4; i++)
    {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }

    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 1
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 2
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 3

    return newPoints;
}

/**
 * The function takes an image and four points, and returns a warped version of the image based on the
 * perspective transformation defined by the four points.
 *
 * Args:
 *   img (Mat): The input image that needs to be warped.
 *   points (vector<Point>): A vector of 4 points representing the corners of a quadrilateral in the
 * input image that needs to be warped.
 *   w (float): The width of the output image after perspective transformation.
 *   h (float): The height of the output image after perspective transformation.
 *
 * Returns:
 *   a warped version of the input image (img) based on the four points provided (points) and the
 * desired output width (w) and height (h). The warped image is stored in a Mat object called imgWarp,
 * which is not shown in the code snippet.
 */
Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
    Point2f src[4] = {points[0], points[1], points[2], points[3]};
    Point2f dst[4] = {{0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h}};
    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));

    return imgWarp;
}

int main()
{
    string path = "";
    imgOriginal = imread(path);
    resize(imgOriginal, imgOriginal, Size(), 0.5, 0.5);
    imgThre = preProcessing(imgOriginal);

    initialPoints = getContours(imgThre);
    cout << initialPoints;
    docPoints = reorder(initialPoints);

    imgWarp = getWarp(imgOriginal, docPoints, w, h);
    int cropVal = 5;
    Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
    imgCrop = imgWarp(roi);

    imshow("Image", imgOriginal);
    imshow("Image Crop", imgCrop);
    waitKey(0);

    return 0;
}