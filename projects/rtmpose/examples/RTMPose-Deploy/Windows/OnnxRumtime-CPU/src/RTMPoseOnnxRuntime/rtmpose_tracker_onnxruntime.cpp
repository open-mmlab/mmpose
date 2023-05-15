#include "rtmpose_tracker_onnxruntime.h"

RTMPoseTrackerOnnxruntime::RTMPoseTrackerOnnxruntime(const std::string& det_model_path, const std::string& pose_model_path, int dectect_interval)
	:m_rtm_det_ptr(nullptr),
	m_rtm_pose_ptr(nullptr),
	m_frame_num(0),
	m_dectect_interval(dectect_interval)
{
	m_rtm_det_ptr = std::make_unique<RTMDetOnnxruntime>(det_model_path);
	m_rtm_pose_ptr = std::make_unique<RTMPoseOnnxruntime>(pose_model_path);
}

RTMPoseTrackerOnnxruntime::~RTMPoseTrackerOnnxruntime()
{
}

std::pair<DetectBox, std::vector<PosePoint>> RTMPoseTrackerOnnxruntime::Inference(const cv::Mat& input_mat)
{
	std::pair<DetectBox, std::vector<PosePoint>> result;

	if (m_rtm_det_ptr == nullptr || m_rtm_pose_ptr == nullptr)
		return result;

	if (m_frame_num % m_dectect_interval == 0)
	{
		m_detect_box = m_rtm_det_ptr->Inference(input_mat);
	}

	std::vector<PosePoint> pose_result = m_rtm_pose_ptr->Inference(input_mat, m_detect_box);

	m_frame_num += 1;

	return std::make_pair(m_detect_box, pose_result);
}
