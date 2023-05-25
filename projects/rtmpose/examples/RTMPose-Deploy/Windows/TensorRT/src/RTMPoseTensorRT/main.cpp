#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "rtmdet.h"
#include "rtmpose.h"
#include "utils.h"
#include "inference.h"


using namespace std;

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


int main()
{
    // set engine file path
    string detEngineFile = "./model/rtmdet.engine";
    string poseEngineFile = "./model/rtmpose_m.engine";

    // init model
    RTMDet det_model(detEngineFile, logger);
    RTMPose pose_model(poseEngineFile, logger);

    // open cap
    cv::VideoCapture cap(0);

    while (cap.isOpened())
    {
        cv::Mat frame;
        cv::Mat show_frame;
        cap >> frame;

        if (frame.empty())
            break;

        frame.copyTo(show_frame);
        auto result = inference(frame, det_model, pose_model);
        draw_pose(show_frame, result);

        cv::imshow("result", show_frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
    cv::destroyAllWindows();
    cap.release();

    return 0;
}
