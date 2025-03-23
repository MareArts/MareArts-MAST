#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h> 

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




void LOG_procTime(int64 inAtime, std::string msg) {
    int64 inBtime = cv::getTickCount(); //check processing time 
    float proc_sec = (inBtime - inAtime) / cv::getTickFrequency();
    printf("%s : %.6lf sec %f fps \n", msg.c_str(), proc_sec, 1 / proc_sec);
}

void setLabel(cv::Mat& im, const std::string label, const cv::Point& OR)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = im.cols / 1500.0;
    scale = scale < 0 ? 1 : scale;
    int thickness = int(im.cols / 1000.0) + 2;
    int baseline = 100;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, OR + cv::Point(0, baseline), OR + cv::Point(text.width, -text.height), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(im, label, OR, fontface, scale, cv::Scalar(0, 191, 255), thickness, cv::LINE_AA, false);
}


//-------------------------------------------------------
//thread setting
#include <thread>
// #include <windows.h>
// receive mat Thread
// thread function for video getting
void StreamThread(std::vector< cv::Mat >& frames, std::vector< cv::cuda::GpuMat >& frames_gpu, std::vector<cv::VideoCapture>& streams_vector, bool& isRun)
{
    while (isRun) {


        for (int i = 0; i < streams_vector.size(); ++i)
        {
            streams_vector[i].read(frames[i]);
            frames_gpu[i].upload(frames[i]);

            if (frames[i].empty())
            {
                isRun = false;
                break;
            }
        }

    }
}

void viewLoopCode(cv::Mat& frameCV, bool& isRun) {

    //st window init
    cv::namedWindow("MAST result", cv::WINDOW_NORMAL);
    cv::moveWindow("MAST result", 40, 40);
    cv::resizeWindow("MAST result", 500, 500);


    while (isRun) {
        std::cout << "output size:" << frameCV.size() << std::endl;
        imshow("MAST result", frameCV);
        if (cv::waitKey(1) > 0)
            isRun = false;
    }
}
//-------------------------------------------------------



bool best_practice_thread_gpuOnly_MAST_loadParam_N_stitching();


int main()
{
    best_practice_thread_gpuOnly_MAST_loadParam_N_stitching();
    return 0;
}





bool best_practice_thread_gpuOnly_MAST_loadParam_N_stitching() {
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //input Mat, gpuMat
    std::vector< cv::Mat > frames;
    std::vector< cv::cuda::GpuMat > frames_gpu;
    std::vector<cv::VideoCapture> streams_vector;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //output Mat, gpuMat
    cv::Mat result_st;
    cv::cuda::GpuMat result_st_gpu;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //get video stream & initiate image size
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_L1.mp4"));
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_C1.mp4"));
    streams_vector.push_back(cv::VideoCapture("../ASSETS/soccer_R1.mp4"));


    // frames = std::vector< cv::Mat >(streams_vector.size());
    // frames_gpu = std::vector< cv::cuda::GpuMat >(streams_vector.size());

    // First get a frame from each stream to know the sizes
    for (int i = 0; i < streams_vector.size(); i++) {
        cv::Mat temp;
        streams_vector[i].read(temp);
        if (temp.empty()) {
            std::cerr << "Failed to read initial frame from stream " << i << std::endl;
            return false;
        }
        frames.push_back(temp);
        // Create properly allocated GpuMat
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.create(temp.rows, temp.cols, temp.type());
        gpu_frame.upload(temp);
        frames_gpu.push_back(gpu_frame);
    }
    

    //frame window init
    //other setting
    char string_buffer[50];
    for (int i = 0; i < frames.size(); ++i) {
        //show frames    
        sprintf(string_buffer, "img%d", i);
        cv::namedWindow(string_buffer, cv::WINDOW_NORMAL);
        cv::resizeWindow(string_buffer, 500, 500);
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    //MAST class
    MareArtsStitcher MAST;
    if (MAST.check_SNcertification(MY_EMAIL, MY_SN) == false) {
        std::cout << "certification fail" << std::endl;
        return false;
    }
    MAST.DisplayCertificationInfo();
    
    //load param
    MAST.loadCameraParams("./params/bale"); //load calibration result!!
    bool stitching_done = false;
    ///////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // get video frame thread run
    bool isRun = true;
    //reset video frame 0
    // thread function for video getting
    std::thread StreamVideothread(StreamThread, std::ref(frames), std::ref(frames_gpu), std::ref(streams_vector), std::ref(isRun));
    StreamVideothread.detach();


    ///////////////////////////////////////////////////////////////////////////////////////////////
    //wait get first img and make sure all frames are ready to stitching
    while (1) {
        bool ready_img = true;
        for (int i = 0; i < frames.size(); ++i) {
            if (frames[i].empty())
            {
                ready_img = false;
                break;
            }

            sprintf(string_buffer, "[%d x %d]", frames[i].cols, frames[i].rows);
            setLabel(frames[i], string_buffer, cv::Point(100, 200));

            sprintf(string_buffer, "img%d", i);
            cv::imshow(string_buffer, frames[i]);

            cv::waitKey(10);
            
        }

        

        if (ready_img)
            break;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////



    ///////////////////////////////////////////////////////////////////////////////////////////////
    //connect mat <-> gpumat
    //https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf
    //https://forums.developer.nvidia.com/t/unified-memory-vs-pinned-host-memory-vs-gpu-global-memory/34640
    std::cout << "Using pinned memory" << std::endl;
    bool cpu_gpu_pin_memory = false;
    void* device_ptr, * host_ptr;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    //first stitching to get image size
    MAST.stitcher(frames_gpu, result_st_gpu);
    //momory pinning
    result_st_gpu.download(result_st);
    size_t frameByteSize = result_st.step[0] * result_st.rows;
    cudaHostAlloc((void**)&host_ptr, frameByteSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&device_ptr, (void*)host_ptr, 0);
    result_st = cv::Mat(result_st.size().height, result_st.size().width, CV_8UC3, host_ptr);
    result_st_gpu = cv::cuda::GpuMat(result_st.size().height, result_st.size().width, CV_8UC3, device_ptr);
    ///////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //view thread run
    std::thread viewLoop(viewLoopCode, std::ref(result_st), std::ref(isRun));
    viewLoop.detach();

    //realtime stching loop start
    while (isRun)
    {
        int64 total_time = cv::getTickCount();
        MAST.stitcher(frames_gpu, result_st_gpu);
        LOG_procTime(total_time, "--[loop] total proc: ");
        std::cout << std::endl;
    }
    return true;
}