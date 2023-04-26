#ifndef _RTM_POSE_UTILS_H_
#define _RTM_POSE_UTILS_H_

#include "opencv2/opencv.hpp"

const std::vector<float> IMAGE_MEAN{ 123.675, 116.28, 103.53 };
const std::vector<float> IMAGE_STD{ 58.395, 57.12, 57.375 };

struct DetectBox
{
	int left;
	int top;
	int right;
	int bottom;
	float score;
	int label;

	DetectBox()
	{
		left = -1;
		top = -1;
		right = -1;
		bottom = -1;
		score = -1.0;
		label = -1;
	}

	bool IsValid() const
	{
		return left != -1 && top != -1 && right != -1 && bottom != -1 && score != -1.0 && label != -1;
	}
};

static bool BoxCompare(
	const DetectBox& a,
	const DetectBox& b) {
	return a.score > b.score;
}

struct PosePoint
{
	int x;
	int y;
	float score;

	PosePoint()
	{
		x = 0;
		y = 0;
		score = 0.0;
	}
};

typedef PosePoint Vector2D;


static cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, int output_image_width, int output_image_height, bool inverse = false)
{
	// solve the affine transformation matrix

	// get the three points corresponding to the source picture and the target picture
	cv::Point2f src_point_1;
	src_point_1.x = center_x;
	src_point_1.y = center_y;

	cv::Point2f src_point_2;
	src_point_2.x = center_x;
	src_point_2.y = center_y - scale_width * 0.5;

	cv::Point2f src_point_3;
	src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
	src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);


	float alphapose_image_center_x = output_image_width / 2;
	float alphapose_image_center_y = output_image_height / 2;

	cv::Point2f dst_point_1;
	dst_point_1.x = alphapose_image_center_x;
	dst_point_1.y = alphapose_image_center_y;

	cv::Point2f dst_point_2;
	dst_point_2.x = alphapose_image_center_x;
	dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

	cv::Point2f dst_point_3;
	dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
	dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);


	cv::Point2f srcPoints[3];
	srcPoints[0] = src_point_1;
	srcPoints[1] = src_point_2;
	srcPoints[2] = src_point_3;

	cv::Point2f dstPoints[3];
	dstPoints[0] = dst_point_1;
	dstPoints[1] = dst_point_2;
	dstPoints[2] = dst_point_3;

	// get affine matrix
	cv::Mat affineTransform;
	if (inverse)
	{
		affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
	}
	else
	{
		affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
	}

	return affineTransform;
}

#endif // !_RTM_POSE_UTILS_H_
