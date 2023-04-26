#include "rtmdet_onnxruntime.h"

#include <iostream>
#include <thread>

#include "characterset_convert.h"


RTMDetOnnxruntime::RTMDetOnnxruntime(const std::string& onnx_model_path)
	:m_session(nullptr),
	m_env(nullptr)
{
	std::wstring onnx_model_path_wstr = stubbornhuang::CharactersetConvert::string_to_wstring(onnx_model_path);

	m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "rtmdet_onnxruntime_cpu");

	int cpu_processor_num = std::thread::hardware_concurrency();
	cpu_processor_num /= 2;

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(cpu_processor_num);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetLogSeverityLevel(4);

	OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
	m_session = Ort::Session(m_env, onnx_model_path_wstr.c_str(), session_options);

	PrintModelInfo(m_session);
}

RTMDetOnnxruntime::~RTMDetOnnxruntime()
{
}

DetectBox RTMDetOnnxruntime::Inference(const cv::Mat& input_mat)
{
	// Deep copy
	cv::Mat input_mat_copy;
	input_mat.copyTo(input_mat_copy);

	// BGR to RGB
	cv::Mat input_mat_copy_rgb;
	cv::cvtColor(input_mat_copy, input_mat_copy_rgb, CV_BGR2RGB);

	// image data, HWC->CHW, image_data - mean / std normalize
	int image_height = input_mat_copy_rgb.rows;
	int image_width = input_mat_copy_rgb.cols;
	int image_channels = input_mat_copy_rgb.channels();

	std::vector<float> input_image_array;
	input_image_array.resize(1 * image_channels * image_height * image_width);

	float* input_image = input_image_array.data();
	for (int h = 0; h < image_height; ++h)
	{
		for (int w = 0; w < image_width; ++w)
		{
			for (int c = 0; c < image_channels; ++c)
			{
				int chw_index = c * image_height * image_width + h * image_width + w;

				float tmp = input_mat_copy_rgb.ptr<uchar>(h)[w * 3 + c];

				input_image[chw_index] = (tmp - IMAGE_MEAN[c]) / IMAGE_STD[c];
			}
		}
	}

	// inference
	std::vector<const char*> m_onnx_input_names{ "input" };
	std::vector<const char*> m_onnx_output_names{ "dets","labels"};
	std::array<int64_t, 4> input_shape{ 1, image_channels, image_height, image_width };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		input_image_array.data(),
		input_image_array.size(),
		input_shape.data(),
		input_shape.size()
	);

	assert(input_tensor.IsTensor());

	auto output_tensors = m_session.Run(
		Ort::RunOptions{ nullptr },
		m_onnx_input_names.data(),
		&input_tensor,
		1,
		m_onnx_output_names.data(),
		m_onnx_output_names.size()
	);

	// pose process
	std::vector<int64_t> det_result_dims = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> label_result_dims = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

	assert(det_result_dims.size() == 3 && label_result_dims.size() == 2);

	int batch_size = det_result_dims[0] == label_result_dims[0] ? det_result_dims[0] : 0;
	int num_dets = det_result_dims[1] == label_result_dims[1] ? det_result_dims[1] : 0;
	int reshap_dims = det_result_dims[2];

	float* det_result = output_tensors[0].GetTensorMutableData<float>();
	int* label_result = output_tensors[1].GetTensorMutableData<int>();

	std::vector<DetectBox> all_box;
	for (int i = 0; i < num_dets; ++i)
	{
		int classes = label_result[i];
		if (classes != 0)
			continue;

		DetectBox temp_box;
		temp_box.left = int(det_result[i * reshap_dims]);
		temp_box.top = int(det_result[i * reshap_dims + 1]);
		temp_box.right = int(det_result[i * reshap_dims + 2]);
		temp_box.bottom = int(det_result[i * reshap_dims + 3]);
		temp_box.score = det_result[i * reshap_dims + 4];
		temp_box.label = label_result[i];

		all_box.emplace_back(temp_box);
	}

	// descending sort
	std::sort(all_box.begin(), all_box.end(), BoxCompare);

	//cv::rectangle(input_mat_copy, cv::Point{ all_box[0].left, all_box[0].top }, cv::Point{ all_box[0].right, all_box[0].bottom }, cv::Scalar{ 0, 255, 0 });

	//cv::imwrite("detect.jpg", input_mat_copy);

	DetectBox result_box;

	if (!all_box.empty())
	{
		result_box = all_box[0];
	}

	return result_box;
}

void RTMDetOnnxruntime::PrintModelInfo(Ort::Session& session)
{
	// print the number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	size_t num_output_nodes = session.GetOutputCount();
	std::cout << "Number of input node is:" << num_input_nodes << std::endl;
	std::cout << "Number of output node is:" << num_output_nodes << std::endl;

	// print node name
	Ort::AllocatorWithDefaultOptions allocator;
	std::cout << std::endl;
	for (auto i = 0; i < num_input_nodes; i++)
		std::cout << "The input op-name " << i << " is:" << session.GetInputNameAllocated(i, allocator) << std::endl;
	for (auto i = 0; i < num_output_nodes; i++)
		std::cout << "The output op-name " << i << " is:" << session.GetOutputNameAllocated(i, allocator) << std::endl;


	// print input and output dims
	for (auto i = 0; i < num_input_nodes; i++)
	{
		std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "input " << i << " dim is: ";
		for (auto j = 0; j < input_dims.size(); j++)
			std::cout << input_dims[j] << " ";
	}
	for (auto i = 0; i < num_output_nodes; i++)
	{
		std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "output " << i << " dim is: ";
		for (auto j = 0; j < output_dims.size(); j++)
			std::cout << output_dims[j] << " ";
	}
}
