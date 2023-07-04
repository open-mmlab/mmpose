#include "rtmdet.h"


// set network params
float RTMDet::input_h = 640;
float RTMDet::input_w = 640;
float RTMDet::mean[3] = { 123.675, 116.28, 103.53 };
float RTMDet::std[3] = { 58.395, 57.12, 57.375 };

/**
 * @brief RTMDet`s constructor
 * @param model_path RTMDet engine file path
 * @param logger Nvinfer ILogger
 * @param conf_thre The confidence threshold
 * @param iou_thre The iou threshold of nms
*/
RTMDet::RTMDet(std::string model_path, nvinfer1::ILogger& logger, float conf_thre, float iou_thre) : conf_thre(conf_thre), iou_thre(iou_thre)
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
 * @brief RTMDet`s destructor
*/
RTMDet::~RTMDet()
{
    cudaFree(stream);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
}


/**
 * @brief Display network input and output parameters
*/
void RTMDet::show()
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
std::vector<float> RTMDet::preprocess(cv::Mat& image)
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
 * @param boxes_result The result of rtmdet
 * @param img_w The width of input image
 * @param img_h The height of input image
 * @return Detect boxes
*/
std::vector<Box> RTMDet::postprocess(std::vector<float> boxes_result, int img_w, int img_h)
{
    std::vector<Box> result;
    std::vector<float> buff;
    for (int i = 0; i < 8400; i++)
    {
        // x1, y1, x2, y2, class, confidence
        buff.insert(buff.end(), boxes_result.begin() + i * 6, boxes_result.begin() + i * 6 + 6);
        // drop the box which confidence less than threshold
        if (buff[5] < conf_thre)
        {
            buff.clear();
            continue;
        }

        Box box;
        box.x1 = buff[0];
        box.y1 = buff[1];
        box.x2 = buff[2];
        box.y2 = buff[3];
        box.cls = buff[4];
        box.conf = buff[5];
        result.emplace_back(box);
        buff.clear();
    }

    // nms
    result = non_maximum_suppression(result, iou_thre);

    // return the box to real image
    for (int i = 0; i < result.size(); i++)
    {
        result[i].x1 = MAX((result[i].x1 - offset[0]) * img_w / (input_w - 2 * offset[0]), 0);
        result[i].y1 = MAX((result[i].y1 - offset[1]) * img_h / (input_h - 2 * offset[1]), 0);
        result[i].x2 = MIN((result[i].x2 - offset[0]) * img_w / (input_w - 2 * offset[0]), img_w);
        result[i].y2 = MIN((result[i].y2 - offset[1]) * img_h / (input_h - 2 * offset[1]), img_h);
    }

    return result;
}


/**
 * @brief Predict function
 * @param image Input image
 * @return Predict results
*/
std::vector<Box> RTMDet::predict(cv::Mat& image)
{
    // get input image size
    int img_w = image.cols;
    int img_h = image.rows;
    std::vector<float> input = preprocess(image);

    // apply for GPU space
    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], 8400 * 6 * sizeof(float));

    // copy data to GPU
    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

    // network inference
    context->enqueueV2(buffer, stream, nullptr);
    cudaStreamSynchronize(stream);

    // get result from GPU
    std::vector<float> boxes_result(8400 * 6);
    cudaMemcpyAsync(boxes_result.data(), buffer[1], 8400 * 6 * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<Box> result = postprocess(boxes_result, img_w, img_h);

    cudaFree(buffer[0]);
    cudaFree(buffer[1]);

    return result;
}
