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

bool MAST_calibration_saveParam_N_loadParam_stitching();


int main()
{
    MAST_calibration_saveParam_N_loadParam_stitching();
    return 0;
}


bool MAST_calibration_saveParam_N_loadParam_stitching() {

    //video source
    std::vector<cv::VideoCapture> streams_vector;
    
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_L1.mp4"));
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_C1.mp4"));
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_R1.mp4"));

    std::vector< cv::Mat > frames(streams_vector.size());
    std::vector< cv::cuda::GpuMat > frames_gpu(streams_vector.size());

    char string_buffer[50];
    int video_cnt = 0;
    bool stitching_done = false;

    MareArtsStitcher MAST;
    if (MAST.check_SNcertification(MY_EMAIL, MY_SN) == false) {
        std::cout << "certification fail" << std::endl;
        return false;
    }
    MAST.DisplayCertificationInfo();
    
    MAST.m_use_GPU = true; //false;  //false;
    MAST.m_FeatureMatchConfidence = 0.5;
    MAST.m_FeatureFindAlgorithm = "akaze";
    MAST.m_PairingConfidence = 0.5;
    MAST.m_ExposureCompensatorType = "gain";
    MAST.m_ImageBlendingAlgorithm = "multiband"; // "no";// "multiband"; //"feather"
    MAST.m_ImageWarppingWay = "plane";
    MAST.m_MatchingEstType = "homography";
    MAST.m_BA_EstCOSTFuncType = "reproj";

    //st window init
    cv::namedWindow("st result", cv::WINDOW_NORMAL);
    cv::resizeWindow("st result", 500, 500);
    //calibration window init
    cv::namedWindow("st calibration result", cv::WINDOW_NORMAL);
    cv::resizeWindow("st calibration result", 500, 500);

    //get frame
    printf("============== get frames\n");
    for (int i = 0; i < streams_vector.size(); ++i)
    {
        streams_vector[i].read(frames[i]);
        frames_gpu[i].upload(frames[i]);
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
            frames_gpu[i].upload(frames[i]);
        }

        //empty break
        bool stop_loop = false;
        for (int i = 0; i < frames.size(); ++i)
        {
            //std::cout << frames[i].empty() << std::endl;
            if (frames[i].empty())
            {
                stop_loop = true;
                break;
            }
        }
        if (stop_loop)
            break;

        //show frames
        for (int i = 0; i < frames.size(); ++i) {
            //frame window
            sprintf(string_buffer, "img%d", i);
            cv::imshow(string_buffer, frames[i]);
        }

        if (!stitching_done) {

            //calibration stitching
            cv::Mat result;
            if (MAST.calibrateFrame(frames, result) == false)
            {
                std::cout << "calibration error" << std::endl;
                return false;
            }
            MAST.saveCameraParams("./params/save_param_01");
            MAST.loadCameraParams("./params/save_param_01");
            stitching_done = true;
            //stitching calibration window
            cv::imshow("st calibration result", result);

            //
            if (cv::waitKey(10) > 10) {
                break;
            }
        }
        else {


            cv::Mat result_st;
            cv::cuda::GpuMat result_st_gpu;
            if (MAST.m_use_GPU == false) {
                //cpu mode
                MAST.stitcher(frames, result_st);
            }
            else if (MAST.m_use_GPU == true) {
                //gpu mode
                MAST.stitcher(frames_gpu, result_st_gpu);
                result_st_gpu.download(result_st);
            }

            //st window
            cv::imshow("st result", result_st);

            if (cv::waitKey(10) > 10) {
                break;
            }
        }
    }

    return true;
}
