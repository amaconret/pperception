#ifndef INCLUDE_PPERCEPTION_MODEL_MOBILENETV1SSD_H_
#define INCLUDE_PPERCEPTION_MODEL_MOBILENETV1SSD_H_

#include "pperception_postprocessing_nms.h"

#pragma once

namespace pperception {

/**
 * model output parameter
 */
struct PPerceptionModelPostprocessParamMobilenetv1Ssd {
    float confidence;            //!< threshold to limit the output to those have bigger conf value
};

struct PPerceptionModelInputMobilenetv1Ssd {
    uint8_t * image;   //!< pointer to image data. Use this instead of vector to avoid copy
    size_t    width;   //!< image width
    size_t    height;  //!< image height
    size_t    channel; //!< number of channels
    size_t    pixelbyte; //!< how many bytes per pixel
};

struct PPerceptionModelBoundingBox {
    float x;  //!< x-coordinate of top-left corner, presumably in [0, 1] scale
    float y;  //!< y-coordinate of top-left corner, presumably in [0, 1] scale
    float w;  //!< width (along x-axis) of the bounding box, presumably in [0, 1] scale
    float h;  //!< height (along y-axis) of the bounding box, presumably in [0, 1] scale

    /**
     * Default Constructor
     */
    PPerceptionModelBoundingBox(): x(0.f), y(0.f), w(0.f), h(0.f){};

    /**
     * Constructor
     *
     * @param x x-coordinate of top-left corner
     * @param y y-coordinate of top-left corner
     * @param w width (along x-axis) of the bounding box
     * @param h height (along y-axis) of the bounding box
     */
    PPerceptionModelBoundingBox(float x, float y, float w, float h):
                 x(x), y(y), w(w), h(h){};

    // For comparing the bounding boxes.
    bool operator==(const PPerceptionModelBoundingBox& other) const {
#define CMP(a, b)                          \
    {                                      \
        if (std::abs((a) - (b)) >= 1e-3) { \
            return false;                  \
        }                                  \
    }
        CMP(x, other.x);
        CMP(y, other.y);
        CMP(w, other.w);
        CMP(h, other.h);
        return true;
#undef CMP
    }

    bool operator!=(const PPerceptionModelBoundingBox& other) const {
        return !(*this == other);
    }
};

/**
 * derived class for model output
 */
struct PPerceptionModelOutputMobilenetv1Ssd {
    uint32_t box_class;       //!< class
    float    box_score;       //!< score, e.g. confidence
    PPerceptionModelBoundingBox box_coord;       //!< bounding box location
};

/**
 * detection plus HKD (human keypoints detection) model output
 */

#define DETECTION_HKD_NUM_KP 17
struct PPerceptionModelAuxOutputDetectionHKD {
    struct {
        float y;
        float x;
    } kp[DETECTION_HKD_NUM_KP];
    float visibility[DETECTION_HKD_NUM_KP];
};

// used by the anchor boxes and model raw output box
struct CenterSizeEncoding {
    float y;
    float x;
    float h;
    float w;
};


struct PPerceptionModelOutputDetectionHKD {
    struct CenterSizeEncoding   coord;
    PPerceptionModelAuxOutputDetectionHKD auxdata;
};

/**
 * model preprocessing parameters
 * can contain std, mean, etc. params
 */
struct PPerceptionModelPreprocessParamMobilenetv1Ssd {
};

struct PPerceptionModelNodeMapping {
    std::string name_in_model;
    std::string name_mapped;
};

typedef std::vector<struct PPerceptionModelNodeMapping> ModelNodeMap;

/**
 * The quantization parameters for input/output nodes
 * these are currently set from engine. Later on can support config file assignment
 * This structure mimics TfLiteQuantizationParams
 */
struct PPerceptionQuantizationParameters {
    float scale = 1.0;
    int   zero_point = 0;

    PPerceptionQuantizationParameters (float s, int z) : scale(s), zero_point(z) {}
};

typedef std::vector<struct PPerceptionQuantizationParameters> QuantParams;

struct PPerceptionModelConfig {
    std::string              model_name_;       //!< mandatory, model name
    std::string              model_path_;       //!< mandatory, weight file
    std::string              model_graph_path_; //!< optional, graph only
    std::string              model_postp_path_; //!< optional, post processing json config file
    std::string              model_version_;    //!< mandatory
    int                      model_split_node_index_; //!< optional, split the graph by node index
    int                      model_number_of_threads = 4;  //!< optional, number of threads used in runtime (default: 4)
    int                      runtime_;          //!< mandatory
    ModelNodeMap             input_node_maps_;  //!< optional, input nodes and their mapped name
    ModelNodeMap             output_node_maps_; //!< optional, output nodes and their mapped name

    //!< optional, whether the input data is float and we need to quantize it to uint8 to input to the engine
    bool input_needs_quantization_;
    //!< optional, whether the output data is uint8 and we need to dequantize it to float to return to the client
    bool output_needs_dequantization_;
};

/**
 * Represent a bounding box with score
 */
struct PPerceptionModelBoundingBoxWithScore {
    float    box_score;                  //!< score, e.g. confidence
    PPerceptionModelBoundingBox box_coord;  //!< bounding box location

    /**
     * Default Constructor
     */
    PPerceptionModelBoundingBoxWithScore(): box_score(0.f), box_coord(){};

    /**
     * Constructor
     *
     * @param score confidence score
     * @param coord box coordinates
     */
    PPerceptionModelBoundingBoxWithScore(float score, PPerceptionModelBoundingBox coord):
                 box_score(score), box_coord(coord.x, coord.y, coord.w, coord.h){};

    // For comparing the bounding boxes and scores.
    bool operator==(const PPerceptionModelBoundingBoxWithScore& other) const {
        if (std::abs(box_score - other.box_score) >= 1e-3) {
            return false;
        }
        return box_coord == other.box_coord;
    }

    bool operator!=(const PPerceptionModelBoundingBoxWithScore& other) const {
        return !(*this == other);
    }
};


struct PPerceptionModelHumanKeyPoint {
    float x;  //!< x-coordinate of top-left corner in [0, 1] scale
    float y;  //!< y-coordinate of top-left corner in [0, 1] scale
};

 

class PPerceptionEngine {
    public:
    /**
     * virtual destructor
     */
    virtual ~PPerceptionEngine() = default;

    /**
     * @brief Set up the engine.
     * @param configs Configurations of the model.
     * @return true when success, false otherwise
     */
    virtual bool PPerceptionEngineSetup(const pperception::PPerceptionModelConfig& configs) = 0;

    /**
     * @brief Run the inference engine.
     * Must be called after `EdgeflowEngineSetInputs`.
     * @return true when success, false otherwise
     */
    virtual bool PPerceptionEngineExecute() = 0;

    virtual bool PPerceptionEngineSetInputs(const std::vector< pperception::PPerceptionTensorSharedPtr >& input)   = 0;
    virtual bool PPerceptionEngineGetOutputs(std::vector< pperception::PPerceptionTensorSharedPtr >& output) const = 0;

    // The size of the input tensors in bytes.
    virtual std::vector< size_t > PPerceptionEngineGetInputByteSize() const  = 0;
    // The size of the output tensors in bytes.
    virtual std::vector< size_t > PPerceptionEngineGetOutputByteSize() const = 0;

    virtual const IOInfo& io_info() const = 0;
};
  
/**
 * Main class for model, it defined 4 APIs for upper layer to use
 */
class PPerceptionModel {
public:
    /**
     * virtual destructor
     */
    virtual ~PPerceptionModel();

    /**
     * constructor
     */
    PPerceptionModel();

//   /**
//    * Setup: needs to be called first
//    *
//    * This API will use customized model and config path
//    * model_config_dir and model_dir shoud be passed at the same time
//    * users cannot pass the standalone model_config_dir or model_dir argument
//    * @param model the model type that will be used.
//    * @param model_config_dir Path where the model configuration is stored
//    * @param model_dir Path where the model is stored
//    */
virtual bool PPerceptionModelSetup(
            const std::string&                     model_config_dir,
            const std::string&                     model_dir);

    /**
     * set model inputs
     * @param input  input vector, each one contains one shared pointer to tensor.
     */
    bool PPerceptionModelSetInputs(const std::vector< PPerceptionTensorSharedPtr >& input);

    /**
     * Execute model
     * @param callback        callback function pointer. If set will be async mode
     * @param max_allowed_ms  max allowed time for execution. If set will be sync mode.
     *                        0 means endless wait
     */
    bool PPerceptionModelExecute(PPerceptionModelExecuteFinishCallback callback = nullptr, uint64_t max_allowed_ms = 0);

    /**
     * get model outputs
     * @param output  output result, each one contains one shared pointer to tensor.
     */
    bool PPerceptionModelGetOutputs(std::vector< PPerceptionTensorSharedPtr >& output) const;

    /**
     * get model configs
     * @param none
     */
    const pperception::PPerceptionModelConfig& PPerceptionModelGetConfig() const;


    /**
     * get model input tensor dimensions
     */
    inline std::vector< PPerceptionTensorShape > PPerceptionModelGetInputTensorShape() const {
        return pperception_core_engine_->io_info().input_tensor_shapes;
    }

    /**
     * get model output tensor dimensions
     */
    inline std::vector< PPerceptionTensorShape > PPerceptionModelGetOutputTensorShape() const {
        return pperception_core_engine_->io_info().output_tensor_shapes;
    }

    /**
     * get model input tensor size in bytes
     */
    std::vector< size_t > PPerceptionModelGetInputTensorByteSize() const;

    /**
     * get model output tensor size in bytes
     */
    std::vector< size_t > PPerceptionModelGetOutputTensorByteSize() const;


/**
 * main class for mobilenetv1ssd model
 */
struct PPerceptionModelMobilenetv1Ssd : public PPerceptionModel {
    /**
     * Setup: needs to be called first.
     * This will call parent model's setup and then deal with its own postprocessing configs
     * @param model_config_dir Path where the model configuration is stored
     * @param model_dir Path where the model is stored
     */
    bool PPerceptionModelSetup(
            const std::string&                     model_config_dir = "/opt/edgeflow/etc/",
            const std::string&                     model_dir        = "/opt/edgeflow/model/");

    /**
     * set model inputs with Mobilenetv1Ssd model specific preprocess
     * @param input:  input data
     * @param param:  preprocessing parameters
     */
    bool PPerceptionModelSetInputs(
            const PPerceptionModelInputMobilenetv1Ssd& input,
            const PPerceptionModelPreprocessParamMobilenetv1Ssd& param);


    /**
     * Get model output with mobilenetv1ssd specific data structures
     * @param output: output data, a vector of EdgeflowModelOutputMobilenetv1Ssd
     * @param param:  parameters to control what results to output
     */
    bool PPerceptionModelGetOutputs(
            std::vector<PPerceptionModelOutputMobilenetv1Ssd>& output,
            const PPerceptionModelPostprocessParamMobilenetv1Ssd& param);

    /**
    * Get model output with mobilenetv1ssd specific data structures
    * This function gets raw output from the derived EdgeflowModelGetOutputs() and then do post processing
    * @param output: output data, a vector of EdgeflowModelOutputMobilenetv1Ssd
    * @param output_auxdata: some models output other data together with the coordinates
    * @param param:  parameters to control what results to output
    */
    bool PPerceptionModelGetOutputs(
            std::vector<PPerceptionModelOutputMobilenetv1Ssd>& output,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& output_auxdata,
            const PPerceptionModelPostprocessParamMobilenetv1Ssd& param);



    typedef void (*PPerceptionModelExecuteFinishCallback)(bool status);
    typedef std::vector<size_t> PPerceptionTensorShape;
        
    struct IOInfo {
        std::vector< PPerceptionTensorShape > input_tensor_shapes;
        std::vector< PPerceptionTensorShape > output_tensor_shapes;
    
        // Data types of the input tensors.
        std::vector< pperception::PPerceptionTensorDataType > input_tensor_data_types;
        // Data types of the output tensors.
        std::vector< PPerceptionTensorDataType > output_tensor_data_types;
    
        // Names of the output tensors.
        std::vector< std::string > output_tensor_names;
    
        // Quantization parameters for the input tensors.
        // Only used by engines compatible with quantization, such as TFLite and MTK NeuroPilot DLA.
        // For engines incompatible with quantization, this should be an empty vector.
        QuantParams input_tensor_quantization_params;
        // Quantization parameters for the output tensors.
        // Only used by engines compatible with quantization, such as TFLite and MTK NeuroPilot DLA.
        // For engines incompatible with quantization, this should be an empty vector.
        QuantParams output_tensor_quantization_params;
    };
    
    template<typename T> bool loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector);
    
    enum class PPerceptionTensorType
    {
        TENSOR_IMAGE,           //!< image, already preprocessed
        TENSOR_BBOX_CLASS,      //!< bounding box class
        TENSOR_BBOX_SCORE,      //!< bounding box score (confidence)
        TENSOR_BBOX_COORD,      //!< bounding box coordinates (uint, in the format of x, y, w, h)
        TENSOR_FACE_LANDMARK,   //!< face landmarks
        TENSOR_FEATURE_VECTOR,  //!< feature vector
        TENSOR_KEYPOINTS,       //!< human body key points
        TENSOR_HOE,             //!< human orientation
        TENSOR_HEATMAP,         //!< human keypoints heatmap (64X48X17)
    
        TENSOR_OTHER            //!< any generic binary tensor
    };
    
    /**
     * Enumeration of tensor data type
     */
    enum class PPerceptionTensorDataType {
        TENSOR_FLOAT,
        TENSOR_UINT8,
        TENSOR_INT64,
        TENSOR_BOOL,
        TENSOR_INT32,
        TENSOR_FLOAT16,
        TENSOR_FLOAT64,
    };
    
    enum class PPerceptionNMSApproaches : uint32_t {
        NMS_REGULAR = 0,        //!< Regular NMS
        NMS_HYBRID_HEAD,        //!< NMS with hybrid heads (first head for detection, and other heads for classification)
        NMS_FAST,               //!< Fast NMS
        NMS_CROSS_CLASS,        //!< Cross-class NMS
    };
    
    
    struct PPerceptionTensor
    {
        void *                      data;  //!< data pointer, we don't distinguish which data type it is
        PPerceptionTensorType       type;  //!< type of tensor
        std::string                 name;  //!< the mapped node name
        uint32_t                    batch; //!< batch size
        std::vector<size_t>         shape; //!< for image it's HWC
        PPerceptionTensorDataType data_type;  //!< for image data type, default to EDGEFLOW_TENSOR_FLOAT
    
        PPerceptionTensor(void* const data,
                        const PPerceptionTensorType type,
                        const std::string name,
                        const unsigned int batch,
                        const std::vector<size_t> shape,
                        const PPerceptionTensorDataType data_type = PPerceptionTensorDataType::TENSOR_FLOAT);
    
        template < typename T >
        std::vector< T > toVector() const {
            return std::vector< T >(reinterpret_cast< T* >(data), reinterpret_cast< T* >(data) + GetTensorSize());
        }
    
        size_t GetTensorSize() const;
    };
        


  private:

    /* This function does post processing for mobilenetssd, which includes converting score/coordinates and NMS.
     * @param input_vec: tensors have "confidence" and "boundingbox"
     * @param postprocessed_class: output box class
     * @param postprocessed_score: output score
     * @param postprocessed_coord: output coordinates
     * @param postprocessed_auxdata: output optional auxdata, contains human key points data
     */
    bool PPerceptionPostProcess(
            const std::vector<PPerceptionTensorSharedPtr>& input_vec,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata);

    PPerceptionPostProcessingNMS nms_;

    // only for preprocessing, hopefully not needed later when model and input image are same dimension
    // using std::malloc() might be better since memory will only be actually allocated when used
    std::vector<uint8_t> preprocess_buf_;

    bool postprocess_initialized_ = false;
};



protected:
    pperception::PPerceptionModelConfig               pperception_model_config_;
    std::shared_ptr< pperception::PPerceptionEngine > pperception_core_engine_;
    uint8_t                                           pperception_engine_type_;
    bool                                              is_pperception_model_initialized_;

    // Store model input size in bytes
    std::vector< size_t > input_tensor_byte_size_;

    // Store model output size in bytes
    std::vector< size_t > output_tensor_byte_size_;
};





}

#endif /* INCLUDE_PPERCEPTION_MODEL_MOBILENETV1SSD_H_ */



