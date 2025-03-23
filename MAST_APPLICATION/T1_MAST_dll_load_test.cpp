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

//MAST_S2
#include "MareArtsStitcher.h"
// #pragma comment(lib, "MareArtsStitcher.lib")
#include "SN.h"


//cpu, gpu calibration N stitching
bool MAST_calibration_N_stitching();

int main()
{
	MAST_calibration_N_stitching();
    return 0;
}

bool MAST_calibration_N_stitching() {

    MareArtsStitcher MAST;    
    if (MAST.check_SNcertification(MY_EMAIL, MY_SN) == false) {
        std::cout << "certification fail" << std::endl;
        return false;
    }
    MAST.DisplayCertificationInfo();
    MAST.DisplayCertificationInfo();
    return true;
}