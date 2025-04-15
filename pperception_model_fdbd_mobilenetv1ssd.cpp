#include "pperception_model_fdbd_mobilenetv1ssd.h"

#include <getopt.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "pperception_postprocessing_nms.h"

using namespace pperception;
using namespace std;

pperception::PPerceptionModelConfig getPPerceptionModelConfig(
            const std::string& model_config_dir,
            const std::string& model_dir) {
    pperception::PPerceptionModelConfig pperception_model_config{};

    pperception_model_config.model_name_ = "FDBD_dsp";
    pperception_model_config.model_path_ = model_dir + "/mobilenetssdv1/hdfd_hkd.tflite";
    pperception_model_config.model_postp_path_ = model_config_dir + "/detection_nms.json";
    pperception_model_config.model_split_node_index_ = -1;
    pperception_model_config.model_number_of_threads = 4;
    pperception_model_config.model_version_ = 6.9;
#ifdef FORCE_CPU
    pperception_model_config.runtime_ = 0;
#else
    pperception_model_config.runtime_ = 2;
#endif

    pperception_model_config.output_node_maps_.clear();
    
    struct PPerceptionModelNodeMapping node_map;
    node_map.name_in_model = "concat";
    node_map.name_mapped   = "confidence";
    pperception_model_config.output_node_maps_.push_back(node_map);
    struct PPerceptionModelNodeMapping node_map_1;
    node_map_1.name_in_model = "concat_1";
    node_map_1.name_mapped   = "boundingbox";
    pperception_model_config.output_node_maps_.push_back(node_map_1);
    
    pperception_model_config.input_node_maps_.clear();
    struct PPerceptionModelNodeMapping node_map;
    node_map.name_in_model = "Preprocessor/sub";
    node_map.name_mapped   = "input";
    pperception_model_config.input_node_maps_.push_back(node_map);
    
    pperception_model_config.input_needs_quantization_ = false;
    pperception_model_config.output_needs_dequantization_ = true;

    return pperception_model_config;
}

bool pperception::PPerceptionModelMobilenetv1Ssd::PPerceptionModelSetup(
                                                            const std::string& model_config_dir,
                                                            const std::string& model_dir) {
    //EDGEFLOW_LOG_VERBOSE("PPerceptionModdelMobilenetv1Ssd::PPerceptionModelSetup");

    try {

        pperception::PPerceptionModelConfig model_configuration = getPPerceptionModelConfig(
                model_config_dir, model_dir);

        if (pperception::PPerceptionModel::PPerceptionModelSetup(model_config_dir, model_dir) == false) 
            return false;
        else {
            //EDGEFLOW_LOG_INFO("EdgeflowModel setup ok");
        }

        nms_.setNMSConfig();
        
    } catch (const std::length_error& le) {
        //EDGEFLOW_LOG_ERROR("EdgeflowModel setup failed, %s", le.what());
        return false;
    }

    return true;
}


//     /* if there's post processing json then handle it */
//     if (!edgeflow_model_config_.model_postp_path_.empty()) {
//         try {
//             postprocess_initialized_ = nms_.ReadNMSConfigs(edgeflow_model_config_.model_postp_path_, edgeflow_model_config_.model_path_);
//         } catch (const std::length_error& le) {
//             EDGEFLOW_LOG_ERROR("ReadNMSConfigs Failed, %s", le.what());
//             return false;
//         }
//         return postprocess_initialized_;
//     }

//     return true;
// }




/**
 * Set model input with mobilenetv1ssd specific data structures
 * This function does preprocessing (if needed) and then
 * calls to the derived EdgeflowModelSetOutputs() function to set raw input
 * @param input  a reference of input structure
 * @param param  configs input parameters (like mean, std etc.)
 * @return true when success, false otherwise
 */
bool pperception::PPerceptionModelMobilenetv1Ssd::PPerceptionModelSetInputs(
                 const PPerceptionModelInputMobilenetv1Ssd& input,
                 const PPerceptionModelPreprocessParamMobilenetv1Ssd& param) {
    // EDGEFLOW_LOG_VERBOSE("PPerceptionModelSetInputs with preprocess. Channel=%lu, width=%lu, height=%lu, pixelbyte=%lu",
    //                      input.channel,
    //                      input.width,
    //                      input.height,
    //                      input.pixelbyte);

    size_t image_size = input.width * input.height * input.channel * input.pixelbyte;
    uint8_t * datap = nullptr;

    /* Note this resize should only happen once. The vector element is float */
    preprocess_buf_.resize(input.width * input.height * 3);

    if (input.channel == 1) {
        for (size_t i = 0; i < image_size; i++) {
            preprocess_buf_[i*3+0] = input.image[i];
            preprocess_buf_[i*3+1] = input.image[i];
            preprocess_buf_[i*3+2] = input.image[i];
        }
        datap = &preprocess_buf_[0];
    }
    else {
        datap = input.image;
    }

    std::vector<size_t> shape {(size_t)input.height, (size_t)input.width, 3};
    auto input_tensor = std::make_shared<PPerceptionTensor>(datap,
                                                         pperception::PPerceptionTensorType::TENSOR_IMAGE,
                                                         "input",
                                                         1,
                                                         shape,
                                                         PPerceptionTensorDataType::TENSOR_UINT8);
    std::vector<PPerceptionTensorSharedPtr> input_vec {input_tensor};

    return pperception::PPerceptionModel::PPerceptionModelSetInputs(input_vec);
}


/* This function does post processing for mobilenetssd, which includes converting score/coordinates and NMS.
 * @param input_vec: tensors have "confidence" and "boundingbox"
 * @param postprocessed_class: output box class
 * @param postprocessed_score: output score
 * @param postprocessed_coord: output coordinates
 * @param postprocessed_auxdata: output optional auxdata, contains human key points data
 */
bool pperception::PPerceptionModelMobilenetv1Ssd::PPerceptionPostProcess(
            const std::vector<PPerceptionTensorSharedPtr>& input_vec,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata)
{
    PPerceptionTensorSharedPtr score_tensor;
    PPerceptionTensorSharedPtr coord_tensor;

    if (postprocess_initialized_ == false) return false;

    for (auto ptr: input_vec){
        if (ptr->name.compare("confidence") == 0)
            score_tensor = ptr;
        if (ptr->name.compare("boundingbox") == 0)
            coord_tensor = ptr;
    }

    return nms_.NonMaxSuppressionMultiClassHelper(score_tensor, coord_tensor,
             postprocessed_class, postprocessed_score, postprocessed_coord, postprocessed_auxdata);
}

/**
 * Get model output with mobilenetv1ssd specific data structures
 * @param output: output data, a vector of EdgeflowModelOutputMobilenetv1Ssd
 * @param param:  parameters to control what results to output
 */
bool pperception::PPerceptionModelMobilenetv1Ssd::PPerceptionModelGetOutputs(
     std::vector<PPerceptionModelOutputMobilenetv1Ssd>& output,
     const PPerceptionModelPostprocessParamMobilenetv1Ssd& param) {

    std::vector<PPerceptionModelAuxOutputDetectionHKD> output_auxdata;
    return PPerceptionModelGetOutputs(output, output_auxdata, param);
}

/**
 * Get model output with mobilenetv1ssd specific data structures
 * This function gets raw output from the derived EdgeflowModelGetOutputs() and then do post processing
 * @param output: output data, a vector of EdgeflowModelOutputMobilenetv1Ssd
 * @param output_auxdata: some models output other data together with the coordinates
 * @param param:  parameters to control what results to output
*/
bool pperception::PPerceptionModelMobilenetv1Ssd::PPerceptionModelGetOutputs(
     std::vector<PPerceptionModelOutputMobilenetv1Ssd>& output,
     std::vector<PPerceptionModelAuxOutputDetectionHKD>& output_auxdata,
     const PPerceptionModelPostprocessParamMobilenetv1Ssd& param) {

    //EDGEFLOW_LOG_VERBOSE("EdgeflowModelGetOutputs with postprocessing");

    std::vector<PPerceptionTensorSharedPtr> output_vec; // model output, can be either NMSed or not, depends on model
    std::vector<float> postprocessed_class;
    std::vector<float> postprocessed_score;
    std::vector<BoxCornerEncoding> postprocessed_coord;

    float * ci = nullptr;
    float * si = nullptr;
    float * bi = nullptr;
    int num_detections = 0;

    pperception::PPerceptionModel::PPerceptionModelGetOutputs(output_vec);

    bool postprocess_needed = false;
    for (const auto& op: output_vec){
        /* if there's 'confidence', means we need to do post processing */
        if (op->name.compare("confidence") == 0)
            postprocess_needed = true;
    }
    if (postprocess_needed) {
        //EDGEFLOW_CHECK(output_vec.size() == 2, "Expecting 2 output tensors but got %lu!", output_vec.size());

        //EDGEFLOW_LOG_VERBOSE("Got concat output, need to do post processing");

        PPerceptionPostProcess(output_vec, postprocessed_class, postprocessed_score, postprocessed_coord, output_auxdata);

        ci = (float *)postprocessed_class.data();
        si = (float *)postprocessed_score.data();
        bi = (float *)postprocessed_coord.data();
        num_detections = postprocessed_class.size();

        // EDGEFLOW_LOG_VERBOSE("aux data size: %lu, HKD data x=%f, y=%f", output_auxdata.size(),
        //                    (output_auxdata.size() > 0) ? output_auxdata[0].kp[0].x : 0,
        //                    (output_auxdata.size() > 1) ? output_auxdata[0].kp[0].y : 0);
    }
    else {
        /* model did post processing.
           get each output, note the compare strings need to be same as in loadconfig.cpp
         */
        PPerceptionTensorSharedPtr class_tensor;
        PPerceptionTensorSharedPtr coord_tensor;
        PPerceptionTensorSharedPtr score_tensor;
        for (auto ptr: output_vec){
            if (ptr->name.compare("nms_class") == 0){
                class_tensor = ptr; /* shape: 1 1 100 */
            }
            else if (ptr->name.compare("nms_boxes") == 0){
                coord_tensor = ptr; /* shape: 1 100 4 */
            }
            else if (ptr->name.compare("nms_score") == 0){
                score_tensor = ptr; /* shape: 1 1 100 */
            }
        }
        if (class_tensor && class_tensor->data && score_tensor && score_tensor->data && coord_tensor &&
            coord_tensor->data) {
            ci = (float*)class_tensor->data;
            si = (float*)score_tensor->data;
            bi = (float*)coord_tensor->data;
        } else {
            //EDGEFLOW_LOG_ERROR("class_tensor, score_tensor or coord_tensor data is null");
            return false;
        }
        num_detections = class_tensor->shape[2];
    }

    for (int i = 0; i < num_detections; i++) {
         pperception::PPerceptionModelOutputMobilenetv1Ssd nnOut;

         nnOut.box_class = (uint32_t)(ci[0]) + 1;  // class needs to add 1 to adapt with tensorflow.
         ci ++;

         nnOut.box_score = si[0];
         si ++;
         if (nnOut.box_score < param.confidence)
            break;

         nnOut.box_coord.y = bi[0];
         nnOut.box_coord.x = bi[1];
         nnOut.box_coord.h = bi[2] - nnOut.box_coord.y;
         nnOut.box_coord.w = bi[3] - nnOut.box_coord.x;
         bi += 4;

        //  EDGEFLOW_LOG_VERBOSE("BBOX Class: %d Score: %.4f x(%.4f), y(%.4f), w(%.4f), h(%.4f)",
        //                nnOut.box_class, nnOut.box_score, nnOut.box_coord.x, nnOut.box_coord.y,
        //                nnOut.box_coord.w, nnOut.box_coord.h);

         output.push_back(nnOut);
      }

    return true;
}
