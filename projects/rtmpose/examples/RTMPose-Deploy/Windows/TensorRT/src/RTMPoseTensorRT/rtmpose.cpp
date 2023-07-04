#include "rtmpose.h"


// set network params
float RTMPose::input_h = 256;
float RTMPose::input_w = 192;
int RTMPose::extend_width = 384;
int RTMPose::extend_height = 512;
int RTMPose::num_points = 17;
float RTMPose::mean[3] = { 123.675, 116.28, 103.53 };
float RTMPose::std[3] = { 58.395, 57.12, 57.375 };

/**
 * @brief RTMPose`s constructor
 * @param model_path RTMPose engine file path
 * @param logger Nvinfer ILogger
*/
RTMPose::RTMPose(std::string model_path, nvinfer1::ILogger& logger)
{
    // read the engine file
    std::ifstream engineStream(model_path, std::ios::binary);
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // create tensorrt model
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Define input dimensions
    context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, input_h, input_w));

    // create CUDA stream
    cudaStreamCreate(&stream);

    // Initialize offset
    offset.push_back(0);
    offset.push_back(0);
}

/**
 * @brief RTMPose`s destructor
*/
RTMPose::~RTMPose()
{
    cudaFree(stream);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    cudaFree(buffer[2]);
}


/**
 * @brief Display network input and output parameters
*/
void RTMPose::show()
{
    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        std::cout << "node: " << engine->getBindingName(i) << ", ";
        if (engine->bindingIsInput(i))
        {
            std::cout << "type: input" << ", ";
        }
        else
        {
            std::cout << "type: output" << ", ";
        }
        nvinfer1::Dims dim = engine->getBindingDimensions(i);
        std::cout << "dimensions: ";
        for (int d = 0; d < dim.nbDims; d++)
        {
            std::cout << dim.d[d] << " ";
        }
        std::cout << "\n";
    }
}


/**
 * @brief Network preprocessing function
 * @param image Input image
 * @return Processed Tensor
*/
std::vector<float> RTMPose::preprocess(cv::Mat& image)
{
    // resize image
    std::tuple<cv::Mat, int, int> resized = resize(image, input_w, input_h);
    cv::Mat resized_image = std::get<0>(resized);
    offset[0] = std::get<1>(resized);
    offset[1] = std::get<2>(resized);

    // BGR2RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // subtract mean and divide variance
    std::vector<float> input_tensor;
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < resized_image.rows; i++)
        {
            for (int j = 0; j < resized_image.cols; j++)
            {
                input_tensor.emplace_back(((float)resized_image.at<cv::Vec3b>(i, j)[k] - mean[k]) / std[k]);
            }
        }
    }

    return input_tensor;
}


/**
 * @brief Network post-processing function
 * @param simcc_x_result SimCC x dimension output
 * @param simcc_y_result SimCC y dimension output
 * @param img_w The width of input image
 * @param img_h The height of input image
 * @return
*/
std::vector<PosePoint> RTMPose::postprocess(std::vector<float> simcc_x_result, std::vector<float> simcc_y_result, int img_w, int img_h)
{
    std::vector<PosePoint> pose_result;
    for (int i = 0; i < num_points; ++i)
    {
        // find the maximum and maximum indexes in the value of each Extend_width length
        auto x_biggest_iter = std::max_element(simcc_x_result.begin() + i * extend_width, simcc_x_result.begin() + i * extend_width + extend_width);
        int max_x_pos = std::distance(simcc_x_result.begin() + i * extend_width, x_biggest_iter);
        int pose_x = max_x_pos / 2;
        float score_x = *x_biggest_iter;

        // find the maximum and maximum indexes in the value of each exten_height length
        auto y_biggest_iter = std::max_element(simcc_y_result.begin() + i * extend_height, simcc_y_result.begin() + i * extend_height + extend_height);
        int max_y_pos = std::distance(simcc_y_result.begin() + i * extend_height, y_biggest_iter);
        int pose_y = max_y_pos / 2;
        float score_y = *y_biggest_iter;

        // get point confidence
        float score = MAX(score_x, score_y);

        PosePoint temp_point;
        temp_point.x = (pose_x - offset[0]) * img_w / (input_w - 2 * offset[0]);
        temp_point.y = (pose_y - offset[1]) * img_h / (input_h - 2 * offset[1]);
        temp_point.score = score;
        pose_result.emplace_back(temp_point);
    }

    return pose_result;
}


/**
 * @brief Predict function
 * @param image Input image
 * @return Predict results
*/
std::vector<PosePoint> RTMPose::predict(cv::Mat& image)
{
    // get input image size
    int img_w = image.cols;
    int img_h = image.rows;
    std::vector<float> input = preprocess(image);

    // apply for GPU space
    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], num_points * extend_width * sizeof(float));
    cudaMalloc(&buffer[2], num_points * extend_height * sizeof(float));

    // copy data to GPU
    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

    // network inference
    context->enqueueV2(buffer, stream, nullptr);
    cudaStreamSynchronize(stream);

    // get result from GPU
    std::vector<float> simcc_x_result(num_points * extend_width);
    std::vector<float> simcc_y_result(num_points * extend_height);
    cudaMemcpyAsync(simcc_x_result.data(), buffer[1], num_points * extend_width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(simcc_y_result.data(), buffer[2], num_points * extend_height * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<PosePoint> pose_result = postprocess(simcc_x_result, simcc_y_result, img_w, img_h);

    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    cudaFree(buffer[2]);

    return pose_result;
}
