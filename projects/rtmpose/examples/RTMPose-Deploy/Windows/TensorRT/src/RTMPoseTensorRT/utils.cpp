#include "utils.h"


// set points links
std::vector<std::pair<int, int>> coco_17_joint_links = {
	{0,1},{0,2},{1,3},{2,4},{5,7},{7,9},{6,8},{8,10},{5,6},
	{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};


/**
 * @brief Mix two images
 * @param srcImage Original image
 * @param mixImage Past image
 * @param startPoint Start point
 * @return Success or not
*/
bool MixImage(cv::Mat& srcImage, cv::Mat mixImage, cv::Point startPoint)
{

	if (!srcImage.data || !mixImage.data)
	{
		return false;
	}

	int addCols = startPoint.x + mixImage.cols > srcImage.cols ? 0 : mixImage.cols;
	int addRows = startPoint.y + mixImage.rows > srcImage.rows ? 0 : mixImage.rows;
	if (addCols == 0 || addRows == 0)
	{
		return false;
	}

	cv::Mat roiImage = srcImage(cv::Rect(startPoint.x, startPoint.y, addCols, addRows));

	mixImage.copyTo(roiImage, mixImage);
	return true;
}


/**
 * @brief Resize image
 * @param img Input image
 * @param w Resized width
 * @param h Resized height
 * @return Resized image and offset
*/
std::tuple<cv::Mat, int, int> resize(cv::Mat& img, int w, int h)
{
	cv::Mat result;

	int ih = img.rows;
	int iw = img.cols;

	float scale = MIN(float(w) / float(iw), float(h) / float(ih));
	int nw = iw * scale;
	int nh = ih * scale;

	cv::resize(img, img, cv::Size(nw, nh));
	result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
	cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	bool ifg = MixImage(result, img, cv::Point((w - nw) / 2, (h - nh) / 2));
	if (!ifg)
	{
		std::cerr << "MixImage failed" << std::endl;
		abort();
	}

	std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(result, (w - nw) / 2, (h - nh) / 2);

	return res_tuple;
}


/**
 * @brief Compare two boxes
 * @param b1 Box1
 * @param b2 Box2
 * @return Compare result
*/
bool compare_boxes(const Box& b1, const Box& b2)
{
	return b1.conf < b2.conf;
}


/**
 * @brief Iou function
 * @param b1 Box1
 * @param b2 Box2
 * @return Iou
*/
float intersection_over_union(const Box& b1, const Box& b2)
{
	float x1 = std::max(b1.x1, b2.x1);
	float y1 = std::max(b1.y1, b2.y1);
	float x2 = std::min(b1.x2, b2.x2);
	float y2 = std::min(b1.y2, b2.y2);

	// get intersection
	float box_intersection = std::max((float)0, x2 - x1) * std::max((float)0, y2 - y1);

	// get union
	float area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
	float area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
	float box_union = area1 + area2 - box_intersection;

	// To prevent the denominator from being zero, add a very small numerical value to the denominator
	float iou = box_intersection / (box_union + 0.0001);

	return iou;
}


/**
 * @brief Non-Maximum Suppression function
 * @param boxes Input boxes
 * @param iou_thre Iou threshold
 * @return Boxes after nms
*/
std::vector<Box> non_maximum_suppression(std::vector<Box> boxes, float iou_thre)
{
	// Sort boxes based on confidence
	std::sort(boxes.begin(), boxes.end(), compare_boxes);

	std::vector<Box> result;
	std::vector<Box> temp;
	while (!boxes.empty())
	{
		temp.clear();

		Box chosen_box = boxes.back();
		boxes.pop_back();
		for (int i = 0; i < boxes.size(); i++)
		{
			if (boxes[i].cls != chosen_box.cls || intersection_over_union(boxes[i], chosen_box) < iou_thre)
				temp.push_back(boxes[i]);
		}

		boxes = temp;
		result.push_back(chosen_box);
	}
	return result;
}


/**
 * @brief Cut image
 * @param image Input image
 * @param x1 The left coordinate of cut box
 * @param y1 The top coordinate of cut box
 * @param x2 The right coordinate of cut box
 * @param y2 The bottom coordinate of cut box
 * @return Cut image
*/
cv::Mat img_cut(cv::Mat& image, int x1, int y1, int x2, int y2)
{
	cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
	cv::Mat croppedImage = image(roi);
	return croppedImage;
}


/**
 * @brief Judge whether two floating point numbers are equal
 * @param a Number a
 * @param b Number b
 * @return Result
*/
bool isEqual(float a, float b)
{
	return std::fabs(a - b) < 1e-5;
}


/**
 * @brief Draw detection result to image
 * @param image Input image
 * @param points Detection result
*/
void draw_pose(cv::Mat& image, std::vector<std::vector<PosePoint>> points)
{
	for (int p = 0; p < points.size(); p++)
	{
		// draw points links
		for (int i = 0; i < coco_17_joint_links.size(); i++)
		{
			std::pair<int, int> joint_link = coco_17_joint_links[i];
			cv::line(
				image,
				cv::Point(points[p][joint_link.first].x, points[p][joint_link.first].y),
				cv::Point(points[p][joint_link.second].x, points[p][joint_link.second].y),
				cv::Scalar{ 0, 255, 0 },
				2,
				cv::LINE_AA
			);
		}
		//draw points
		for (int i = 0; i < points[p].size(); i++)
		{
			cv::circle(
				image,
				cv::Point(points[p][i].x, points[p][i].y),
				1,
				cv::Scalar{ 0, 0, 255 },
				5,
				cv::LINE_AA
			);
		}
	}
}
