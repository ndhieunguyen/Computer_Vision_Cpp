
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void main()
{
    Mat img;
    VideoCapture cap(0);
    vector<Rect> plates;
    CascadeClassifier plateCascade;
    plateCascade.load("haarcascade_russian_plate_number.xml");

    if (plateCascade.empty())
    {
        cout << "XML file not loaded" << endl;
    }

    while (true)
    {

        cap.read(img);
        plateCascade.detectMultiScale(img, plates, 1.1, 10);

        for (int i = 0; i < plates.size(); i++)
        {
            Mat imgCrop = img(plates[i]);
            imwrite("plate_" + to_string(i) + ".png", imgCrop);
            rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);
        }

        imshow("Image", img);
        waitKey(1);
    }
}