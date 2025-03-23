#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <string>
// #include <cuda_runtime.h> 

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
//#include "opencv2/cuda"



//lib
// #pragma comment(lib, "opencv_core470.lib")
// #pragma comment(lib, "opencv_highgui470.lib")
// #pragma comment(lib, "opencv_videoio470.lib")
// #pragma comment(lib, "opencv_imgproc470.lib")
// #pragma comment(lib, "cudart.lib")


//MAST_S2
#include "MareArtsStitcher.h"
// #pragma comment(lib, "MareArtsStitcher.lib")
#include "SN.h"
//std::string MY_SN = "your_serial_code";
//std::string MY_EMAIL = "your_email";


bool MAST_calibration_N_saveParam();


int main()
{
	MAST_calibration_N_saveParam();
    return 0;
}


bool MAST_calibration_N_saveParam() {

    //video source
    std::vector<cv::VideoCapture> streams_vector;

    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_L1.mp4"));
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_C1.mp4"));
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_R1.mp4"));

    std::vector< cv::Mat > frames(streams_vector.size());

    char string_buffer[50];
    int video_cnt = 0;
    bool stitching_done = false;

    MareArtsStitcher MAST;
    if (MAST.check_SNcertification(MY_EMAIL, MY_SN) == false) {
        std::cout << "certification fail" << std::endl;
        return false;
    }
    MAST.DisplayCertificationInfo();
    
    MAST.m_use_GPU = true;  //true, false anything is ok, you can decide gpu usage on stithcer phase
    MAST.m_FeatureMatchConfidence = 0.5;
    MAST.m_FeatureFindAlgorithm = "akaze";
    MAST.m_PairingConfidence = 0.5;
    MAST.m_ExposureCompensatorType = "gain";
    MAST.m_ImageBlendingAlgorithm = "multiband"; // "no";// "multiband"; //"feather"
    MAST.m_ImageWarppingWay = "plane";
    MAST.m_MatchingEstType = "homography";
    MAST.m_BA_EstCOSTFuncType = "reproj";

    //calibration window init
    cv::namedWindow("st calibration result", cv::WINDOW_NORMAL);
    cv::resizeWindow("st calibration result", 500, 500);

    //get frame
    printf("============== get frames\n");
    for (int i = 0; i < streams_vector.size(); ++i)
    {
        streams_vector[i].read(frames[i]);
    }

    //frame window init
    for (int i = 0; i < frames.size(); ++i) {
        //show frames    
        sprintf(string_buffer, "img%d", i);
        cv::namedWindow(string_buffer, cv::WINDOW_NORMAL);
        cv::resizeWindow(string_buffer, 500, 500);
    }


    while (1)
    {

        //count frames
        video_cnt++;
        printf("--- frame %d --- \n", video_cnt);
        //skip first 10fps
        if (video_cnt < 10)
            continue;

        //get frame
        printf("============== get frames\n");
        for (int i = 0; i < streams_vector.size(); ++i)
        {
            streams_vector[i].read(frames[i]);
        }

        //show frames
        for (int i = 0; i < frames.size(); ++i) {
            //frame window
            sprintf(string_buffer, "img%d", i);
            cv::imshow(string_buffer, frames[i]);
        }

        //calibration stitching
        cv::Mat result;
        if (MAST.calibrateFrame(frames, result) == false)
        {
            std::cout << "calibration error" << std::endl;
            return false;
        }
        stitching_done = true;
        //stitching calibration window
        cv::imshow("st calibration result", result);
        if (MAST.saveCameraParams("./params/save_param") == false)
            return false;

        if (cv::waitKey(0) > 10)
            break;
    }

    return true;
}
