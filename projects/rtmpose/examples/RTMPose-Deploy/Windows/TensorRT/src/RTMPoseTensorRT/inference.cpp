#include "inference.h"


/**
 * @brief Inference network
 * @param image Input image
 * @param detect_model RTMDet model
 * @param pose_model RTMPose model
 * @return Inference result
*/
std::vector<std::vector<PosePoint>> inference(cv::Mat& image, RTMDet& detect_model, RTMPose& pose_model)
{
	cv::Mat im0;
	image.copyTo(im0);

	// inference detection model
	std::vector<Box> det_result = detect_model.predict(image);
	std::vector<std::vector<PosePoint>> result;
	for (int i = 0; i < det_result.size(); i++)
	{
		// Select the detection box labeled as human
		if (!isEqual(det_result[i].cls, 0.0))
			continue;

		// cut image to input the pose model
		cv::Mat person_image = img_cut(im0, det_result[i].x1, det_result[i].y1, det_result[i].x2, det_result[i].y2);
		std::vector<PosePoint> pose_result = pose_model.predict(person_image);

		// Restore points to original image
		for (int j = 0; j < pose_result.size(); j++)
		{
			pose_result[j].x += det_result[i].x1;
			pose_result[j].y += det_result[i].y1;
		}
		result.push_back(pose_result);
	}
	return result;
}
