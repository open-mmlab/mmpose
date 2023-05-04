#include <iostream>

#include "opencv2/opencv.hpp"

#include "rtmpose_utils.h"
#include "rtmpose_onnxruntime.h"
#include "rtmdet_onnxruntime.h"
#include "rtmpose_tracker_onnxruntime.h"

std::vector<std::pair<int, int>> coco_17_joint_links = {
	{0,1},{0,2},{1,3},{2,4},{5,7},{7,9},{6,8},{8,10},{5,6},{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};

int main()
{
	std::string rtm_detnano_onnx_path = "";
	std::string rtm_pose_onnx_path = "";
#ifdef _DEBUG
	rtm_detnano_onnx_path = "../../resource/model/rtmpose-cpu/rtmpose-ort/rtmdet-nano/end2end.onnx";
	rtm_pose_onnx_path = "../../resource/model/rtmpose-cpu/rtmpose-ort/rtmpose-m/end2end.onnx";
#else
	rtm_detnano_onnx_path = "./resource/model/rtmpose-cpu/rtmpose-ort/rtmdet-nano/end2end.onnx";
	rtm_pose_onnx_path = "./resource/model/rtmpose-cpu/rtmpose-ort/rtmpose-m/end2end.onnx";
#endif

	RTMPoseTrackerOnnxruntime rtmpose_tracker_onnxruntime(rtm_detnano_onnx_path, rtm_pose_onnx_path);

	cv::VideoCapture video_reader(0);
	int frame_num = 0;
	DetectBox detect_box;
	while (video_reader.isOpened())
	{
		cv::Mat frame;
		video_reader >> frame;

		if (frame.empty())
			break;

		std::pair<DetectBox, std::vector<PosePoint>> inference_box= rtmpose_tracker_onnxruntime.Inference(frame);
		DetectBox detect_box = inference_box.first;
		std::vector<PosePoint> pose_result = inference_box.second;

		cv::rectangle(
			frame,
			cv::Point(detect_box.left, detect_box.top),
			cv::Point(detect_box.right, detect_box.bottom),
			cv::Scalar{ 255, 0, 0 },
			2);

		for (int i = 0; i < pose_result.size(); ++i)
		{
			cv::circle(frame, cv::Point(pose_result[i].x, pose_result[i].y), 1, cv::Scalar{ 0, 0, 255 }, 5, cv::LINE_AA);
		}

		for (int i = 0; i < coco_17_joint_links.size(); ++i)
		{
			std::pair<int, int> joint_links = coco_17_joint_links[i];
			cv::line(
				frame,
				cv::Point(pose_result[joint_links.first].x, pose_result[joint_links.first].y),
				cv::Point(pose_result[joint_links.second].x, pose_result[joint_links.second].y),
				cv::Scalar{ 0, 255, 0 },
				2,
				cv::LINE_AA);
		}

		imshow("RTMPose", frame);
		cv::waitKey(1);
	}

	video_reader.release();
	cv::destroyAllWindows();

	return 0;
}
