#include "ReID.h"

#include "INIReader.h"
#include "TRT_InferenceEngine/TensorRT_InferenceEngine.h"

///
class ReIDModel::InferenceImpl
{
public:
    InferenceImpl() = default;
    virtual ~InferenceImpl() = default;

    virtual bool Init(const ReIDParams &params, const std::string &onnx_model_path) = 0;

    virtual const std::string &get_distance_metric() const noexcept = 0;
    virtual FeatureVector extract_features(cv::Mat &image_patch) = 0;
};

///
class ReIDInferenceImpl final : public ReIDModel::InferenceImpl
{
public:
    bool Init(const ReIDParams& params, const std::string& onnx_model_path) override
    {
        _onnx_model_path = onnx_model_path;

        _load_params_from_config(params);

        _trt_inference_engine =
                std::make_unique<inference_backend::TensorRTInferenceEngine>(
                        _model_optimization_params, _trt_logging_level);

        auto res = _trt_inference_engine->load_model(onnx_model_path);
        return res;
    }

    FeatureVector extract_features(cv::Mat &image_patch) override
    {
        pre_process(image_patch);
        std::vector<std::vector<float>> output =
                _trt_inference_engine->forward(image_patch);

        // TODO: Clean this up
        FeatureVector feature_vector = FeatureVector::Zero(1, FEATURE_DIM);
        for (int i = 0; i < FEATURE_DIM; i++)
        {
            feature_vector(0, i) = output[0][i];
        }
        return feature_vector;
    }

    const std::string &get_distance_metric() const noexcept override
    {
        return _distance_metric;
    }
    
private:
    inference_backend::TRTOptimizerParams _model_optimization_params;
    std::unique_ptr<inference_backend::TensorRTInferenceEngine> _trt_inference_engine;

    int8_t _trt_logging_level;
    cv::Size _input_size;

    std::string _onnx_model_path;
    std::string _distance_metric;


    void _load_params_from_config(const ReIDParams &params)
    {
        _distance_metric = params.distance_metric;
        _trt_logging_level = params.trt_logging_level;
        _model_optimization_params.batch_size =
                static_cast<int>(params.batch_size);
        _model_optimization_params.fp16 = params.enable_fp16;
        _model_optimization_params.tf32 = params.enable_tf32;
        _model_optimization_params.input_layer_name = params.input_layer_name;

        std::cout << "Trying to get input dims" << std::endl;
        const auto &input_dims = params.input_layer_dimensions;
        _input_size = cv::Size(input_dims[3], input_dims[2]);

        std::cout << "Read input dims" << std::endl;
        std::cout << "Input dims: " << input_dims[0] << " " << input_dims[1]
                  << " " << input_dims[2] << " " << input_dims[3] << std::endl;

        _model_optimization_params.input_dims = nvinfer1::Dims4{
                input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
        _model_optimization_params.swapRB = params.swap_rb;

        _model_optimization_params.output_layer_names =
                params.output_layer_names;
    }

    void pre_process(cv::Mat &image)
    {
        cv::resize(image, image, _input_size);
        if (_model_optimization_params.swapRB)
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
};

///
ReIDModel::ReIDModel(const ReIDParams &params,
                     const std::string &onnx_model_path)
{
    std::cout << "Initializing ReID model" << std::endl;

    _inference_impl = new ReIDInferenceImpl();

    bool net_initialized = _inference_impl->Init(params, onnx_model_path);
    if (!net_initialized)
    {
        std::cout << "Failed to initialize ReID model" << std::endl;
        exit(1);
    }
}

///
ReIDModel::~ReIDModel()
{
    if (_inference_impl)
        delete _inference_impl;
}

///
FeatureVector ReIDModel::extract_features(cv::Mat &image_patch)
{
    return _inference_impl->extract_features(image_patch);
}

///
const std::string& ReIDModel::get_distance_metric() const noexcept
{
    return _inference_impl->get_distance_metric();
}
