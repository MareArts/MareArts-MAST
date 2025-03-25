
#pragma once
#include "opencv2/core/core.hpp"
#include <vector>


#ifdef _WIN32
    #ifdef MYLIBRARY_EXPORTS
        #define DllExport __declspec(dllexport)
    #else
        #define DllExport __declspec(dllimport)
    #endif
#else
    #define DllExport
#endif

class DllExport MareArtsStitcher
{
public:
    //--- general setting
    //use CPU or GPU(cuda) - true:gpu, false:cpu
    bool m_use_GPU = false; //flase or true

    //--- Feature Find and Matching
    //Feature Find algorithm 
    std::string m_FeatureFindAlgorithm = "akaze"; //"orb", "akaze", sift
    //Feature Matching confidence value 
    float m_FeatureMatchConfidence = 0.3; //ex:0.3 orb, 0.5 others
    //Image Pairing confidence value
    float m_PairingConfidence = 1.0; //default 1.0
    //Exposure Compensator
    std::string m_ExposureCompensatorType = "gain"; //"no", "gain", "gain_blocks", "channels", "channels_blocks"
    
    //--- estimator
    //feature mathing & estimate Type
    std::string m_MatchingEstType = "homography"; //"affine", "homography"
    //estimation cost function type
    std::string m_BA_EstCOSTFuncType = "reproj"; //"ray", "reproj", "affine", "no"
    
    //--- warping type
    //CPU mode:
    //"plane", "affine", "cylindrical", "spherical", "fisheye", "stereographic", "compressedPlaneA2B1", "compressedPlaneA1.5B1", "compressedPlanePortraitA2B1"
    //"compressedPlanePortraitA1.5B1", "paniniA2B1", "paniniA1.5B1", "paniniPortraitA2B1", "paniniPortraitA1.5B1", "mercator", "transverseMercator"
    //GPU mode: 
    //"plane", "cylindrical", "spherical"
    std::string m_ImageWarppingWay = "cylindrical"; //"plane", "cylindrical" , "spherical"

    //--- blending
    //Color Blending Algorithm
    //CPU mode: "no", "multiband", "feather"
    //GPU mode: "no", "multiband"
    std::string m_ImageBlendingAlgorithm = "multiband"; 
    //blend_strength, Blending strength from [0,100] range. The default is 5, multi band only"
    float m_blend_strength = 5;
    //log_out
    bool log_out = false;

private:
    bool m_calibration_done = false;

public:
    //constructor
    MareArtsStitcher();
    ~MareArtsStitcher();
    ////////////////
    //Main Functions
    //calibration cameras, find relation parameters between cameras
    bool calibrateFrame(const std::vector<cv::Mat>& cameras, cv::Mat& result);
    //realtime stitching based on calibrated parameters
    bool stitcher(const std::vector<cv::Mat>& input, cv::Mat& output);
    bool stitcher(const std::vector<cv::cuda::GpuMat>& input, cv::cuda::GpuMat& output);
    
    ////////////////
    //sn certification
    bool check_SNcertification(std::string customer_email, std::string USN);
    ////////////////
    //Utils
    //check cuda is avaiable or not
    bool checkCudaEnable();
    void getConfig();
    bool saveCameraParams(std::string fileName);
    bool loadCameraParams(std::string fileName);
    void DisplayCertificationInfo();
    ////////////////
};