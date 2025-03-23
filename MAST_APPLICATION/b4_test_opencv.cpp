#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // RTSP stream URL
    string rtsp_url = "rtsp://192.168.88.50:8554/mystream";
    
    // Create a VideoCapture object
    VideoCapture cap;
    
    // Open the RTSP stream
    cap.open(rtsp_url);
    
    // Check if stream is opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening RTSP stream" << endl;
        return -1;
    }
    
    // Create a window to display the stream
    namedWindow("RTSP Stream", WINDOW_AUTOSIZE);
    
    Mat frame;
    while (true) {
        // Read a new frame from the stream
        if (!cap.read(frame)) {
            cout << "Failed to grab frame" << endl;
            break;
        }
        
        // Check if frame is empty
        if (frame.empty()) {
            cout << "Empty frame received" << endl;
            break;
        }
        
        // Show the frame
        imshow("RTSP Stream", frame);
        
        // Break the loop if 'q' is pressed
        char c = (char)waitKey(1);
        if (c == 'q') {
            break;
        }
    }
    
    // Release the VideoCapture object and destroy windows
    cap.release();
    destroyAllWindows();
    
    return 0;
}