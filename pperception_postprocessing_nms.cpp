#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <numeric>
#include <iostream>
#include <iterator>
#include <cstring>
#include <string>
#include <vector>

#include "pperception_postprocessing_nms.h"
#include "pperception_model_fdbd_mobilenetv1ssd.h"

using namespace pperception;
using namespace std;

#define CLAMP_0_1(x)  ((x) < 0 ? 0 : ((x) > 1.0 ? 1.0 : (x)))


std::vector<float> loadFloatDataFile(const std::string& inputFile)
{
    std::vector<float> vec;
    loadByteDataFile(inputFile, vec);
    return vec;
}

template<typename T>
bool loadBytDaetaFile(const std::string& inputFile, std::vector<T>& loadVector)
{
   std::ifstream in(inputFile, std::ifstream::binary);

   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
      return false;
   }

   in.seekg(0, in.end);
   size_t length = in.tellg();
   in.seekg(0, in.beg);

   if (loadVector.size() == 0) {
      try {
         loadVector.resize(length / sizeof(T));
      } catch (const std::length_error& le) {
      std::cerr << "loadByteDataFile(), Length error: " << le.what() << "\n";
   }
   } else if (loadVector.size()*sizeof(T) < length) {
      std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
      return false;
   }

   if (!in.read(reinterpret_cast<char*>(&loadVector[0]), length))
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
      return false;
   }
   return true;
}

class Dequantizer {
    public:
        Dequantizer(int zero_point, float scale)
            : zero_point_(zero_point), scale_(scale) {}
        float operator()(uint8_t x) {
            return (static_cast<float>(x) - zero_point_) * scale_;
        }

    private:
        int zero_point_;
        float scale_;
};

bool ReadAnchorBoxEncodings(
        const std::string& anchor_path,
        const int   anchor_quant_zero_point,
        const float anchor_quant_scale,
        std::vector<CenterSizeEncoding>& anchors) {


    static constexpr char kNumpyHeader[]     = "\x93NUMPY";
    static constexpr int  kNumpyHeaderLenPos = 8;
    static constexpr int  kNumpyHeaderLenPrevious = 10; // the length before and including the length byte
    static constexpr char kNumpyDescr[]      = "'descr': '";
    static constexpr char kNumpyLittleEndian = '<';
    static constexpr char kNumpyFloat    = 'f';
    static constexpr char kNumpyUnsigned = 'u';
    
    std::vector<unsigned char> anchor_vec;
    if (loadByteDataFile(anchor_path, anchor_vec) == false) {
        //EDGEFLOW_LOG_ERROR("Failed to read quantized anchor data file %s", anchor_path.c_str());
        return false;
    }

    if (!std::strncmp(reinterpret_cast<char *>(anchor_vec.data()), kNumpyHeader, std::strlen(kNumpyHeader))) {

        /* get the numpy data type and data element size */
        const char *numpy_descr = kNumpyDescr;
        auto it = std::search(anchor_vec.begin(), anchor_vec.end(),
                            numpy_descr, numpy_descr + strlen(numpy_descr));
        if (it == anchor_vec.end()) {
            //EDGEFLOW_LOG_ERROR("Anchor data file does not contain needed descriptor.");
            return false;
        }

        it += strlen(numpy_descr);
        char numpy_endian = *it++;
        if (numpy_endian != kNumpyLittleEndian) {
            //EDGEFLOW_LOG_ERROR("Anchor data file data is not little endian. Got %c.", numpy_endian);
            return false;
        }

        char numpy_dtype  = *it++;
        if (numpy_dtype != kNumpyUnsigned && numpy_dtype != kNumpyFloat) {
            //EDGEFLOW_LOG_ERROR("Anchor data file data type is not float or unsigned int. Got %c.", numpy_dtype);
            return false;
        }

        int  numpy_dlen   = *it++;
        numpy_dlen -= '0';
        if (numpy_dlen != 1 && numpy_dlen != 4) {
            //EDGEFLOW_LOG_ERROR("Anchor data file data element size not correct. Got %i.", numpy_dlen);
            return false;
        }
        //EDGEFLOW_LOG_VERBOSE("Anchor data file %s dtype %c dlen %d", anchor_path.c_str(), numpy_dtype, numpy_dlen);

        /* get header length, then read the data after it */
        int header_len = *reinterpret_cast<short *>(anchor_vec.data() + kNumpyHeaderLenPos);
        header_len += kNumpyHeaderLenPrevious;
        if ((anchor_vec.size() - header_len) % 4 != 0) {
            //EDGEFLOW_LOG_ERROR("Anchor data file %s size %ld not correct.", anchor_path.c_str(), anchor_vec.size() - header_len);
            return false;
        }

        if (numpy_dlen == 1) { /* quantized */
            Dequantizer dequantize(anchor_quant_zero_point, anchor_quant_scale);

            for (size_t idx = header_len; idx < anchor_vec.size();) {
                CenterSizeEncoding box;
                box.y = dequantize(anchor_vec[idx]); idx += numpy_dlen;
                box.x = dequantize(anchor_vec[idx]); idx += numpy_dlen;
                box.h = dequantize(anchor_vec[idx]); idx += numpy_dlen;
                box.w = dequantize(anchor_vec[idx]); idx += numpy_dlen;
                // EDGEFLOW_LOG_INFO("Anchor data %f %f %f %f", box.y, box.x, box.h, box.w);
                anchors.push_back(box);
            }
        }
        else {
            float *float_anchor = reinterpret_cast<float *>(anchor_vec.data() + header_len);

            for (size_t idx = header_len; idx < anchor_vec.size();) {
                CenterSizeEncoding box;
                box.y = *float_anchor++; idx += numpy_dlen;
                box.x = *float_anchor++; idx += numpy_dlen;
                box.h = *float_anchor++; idx += numpy_dlen;
                box.w = *float_anchor++; idx += numpy_dlen;
                //EDGEFLOW_LOG_INFO("Anchor data %f %f %f %f", box.y, box.x, box.h, box.w);
                anchors.push_back(box);
            }
        }
    }
    else {
        //EDGEFLOW_LOG_ERROR("Invalid anchor data file %s, it needs to start with x93NUMPY", anchor_path.c_str());
        return false;
    }

    return true;
}


bool PPerceptionPostProcessingNMS::setNMSConfig() {
    //nms_.ReadNMSConfigs(pperception_model_config_.model_postp_path_, pperception_model_config_.model_path_);
    
    //EDGEFLOW_LOG_VERBOSE("Reading NMS Anchor config file");
    
    const std::string& anchor_file = "/opt/edgeflow/model/mobilenetssdv1/hdfd_mv1ssd_quantize_trained.anchor.npy";

    int anchor_quant_zero_point = 0;
    float anchor_quant_scale = 0.006113426294177771;

    /* Now read numpy file and dequantize */
    ReadAnchorBoxEncodings(anchor_file, anchor_quant_zero_point, anchor_quant_scale, anchors_);

    nms_param_.anchor_valid = (anchors_.size() != 0);
    anchor_exist_ = (anchors_.size() != 0);

    nms_param_.anchor_scale.x = 10;
    nms_param_.anchor_scale.y = 10;
    nms_param_.anchor_scale.h = 5;
    nms_param_.anchor_scale.w = 5;

    nms_param_.max_detections = 100;
    nms_param_.return_sigmoid_score = true;
    nms_param_.score_threshold = 0.1;
    nms_param_.score_threshold = INVERSE_SIGMOID(nms_param_.score_threshold);
    
    nms_param_.iou_threshold = 0.3;
    nms_param_.cross_class_iou_threshold = 0.3;
    nms_param_.detections_per_class = 50;
    
    nms_param_.max_categories_per_anchor = 1;
    
    nms_param_.num_background_class = 1;

    //EDGEFLOW_LOG_INFO("GetEdgeFlowModelConfig NMS params: approach: %d detections_per_class %d max_detections %d, score_threshold %f iou_threshold %f class_score_bias.size %d",
    //        static_cast<uint32_t>(nms_param_.approach),
    //        nms_param_.detections_per_class, nms_param_.max_detections,
    //        nms_param_.score_threshold, nms_param_.iou_threshold,
    //        static_cast< int >(nms_param_.class_score_bias.size()));

    return true;
}


void PPerceptionPostProcessingNMS::DecreasingPartialArgSort(
        const float* values, int num_values,
        int num_to_sort, int* indices) {
    std::iota(indices, indices + num_values, 0);
    std::partial_sort(
        indices, indices + num_to_sort, indices + num_values,
        [&values](const int i, const int j) { return values[i] > values[j]; });
}

void PPerceptionPostProcessingNMS::SelectDetectionsAboveScoreThreshold(
        const std::vector<float>& values,
        const float threshold,
        std::vector<float>* keep_values,
        std::vector<int>* keep_indices) {
    for (size_t i = 0; i < values.size(); i++) {
        if (values[i] >= threshold) {
            keep_values->emplace_back(values[i]);
            keep_indices->emplace_back(i);
        }
    }
}

float PPerceptionPostProcessingNMS::ComputeIntersectionOverUnion(
        const BoxCornerEncoding* decoded_boxes,
        const int i, const int j) {
    auto& box_i = decoded_boxes[i];
    auto& box_j = decoded_boxes[j];
    const float area_i = (box_i.ymax - box_i.ymin) * (box_i.xmax - box_i.xmin);
    const float area_j = (box_j.ymax - box_j.ymin) * (box_j.xmax - box_j.xmin);
    if (area_i <= 0 || area_j <= 0) return 0.0;
    const float intersection_ymin = std::max<float>(box_i.ymin, box_j.ymin);
    const float intersection_xmin = std::max<float>(box_i.xmin, box_j.xmin);
    const float intersection_ymax = std::min<float>(box_i.ymax, box_j.ymax);
    const float intersection_xmax = std::min<float>(box_i.xmax, box_j.xmax);
    const float intersection_area =
        std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
        std::max<float>(intersection_xmax - intersection_xmin, 0.0);

   return intersection_area / (area_i + area_j - intersection_area);
}

// original TF version, added anchor_valid flag
// decode box from raw_boxes to decoded_boxes, calculated with anchar box
// calculation refer to BoxCornerEncoding comments in header file
void PPerceptionPostProcessingNMS::DecodeBox(
        BoxCornerEncoding* decoded_boxes,
        const CenterSizeEncoding *raw_boxes,
        const int i,
        const CenterSizeEncoding& scale_values,
        bool anchor_valid) {
    if (!decoded_boxes || !raw_boxes || !anchor_exist_) {
        //EDGEFLOW_LOG_ERROR("DecodeBox called without initialization, or there's no anchor boxes.");
    }
    if (decoded_boxes != nullptr && raw_boxes != nullptr) {
        auto& box = decoded_boxes[i];
        if ((box.xmin == 0) && (box.xmax == 0) && (box.ymin == 0) && (box.ymax == 0)) {
            // this decoded_box need to be calculated
            // The calculation is based on tflite's DecodeCenterSizeBoxes()
            auto& box_centersize = raw_boxes[i];
            float ycenter, xcenter, half_h, half_w;
            if (anchor_valid) {
                auto& anchor = anchors_[i];
                ycenter = box_centersize.y / scale_values.y * anchor.h + anchor.y;
                xcenter = box_centersize.x / scale_values.x * anchor.w + anchor.x;
                half_h  = 0.5f * static_cast<float>(std::exp(box_centersize.h / scale_values.h)) * anchor.h;
                half_w  = 0.5f * static_cast<float>(std::exp(box_centersize.w / scale_values.w)) * anchor.w;
            }
            else {
                ycenter = box_centersize.y;
                xcenter = box_centersize.x;
                half_h  = 0.5f * box_centersize.h;
                half_w  = 0.5f * box_centersize.w;
            }

            box.ymin = CLAMP_0_1(ycenter - half_h);
            box.xmin = CLAMP_0_1(xcenter - half_w);
            box.ymax = CLAMP_0_1(ycenter + half_h);
            box.xmax = CLAMP_0_1(xcenter + half_w);

            //EDGEFLOW_LOG_VERBOSE("DecodeBox %d, raw xyhw %f %f %f %f decoded xxyy %f %f %f %f", i,
            //    box_centersize.x, box_centersize.y, box_centersize.h, box_centersize.w,
            //    box.xmin, box.xmax, box.ymin, box.ymax);
        }
    }
}

// input contains generic auxdata after the coordinates
// decode box from raw_boxes to decoded_boxes, calculated with anchar box
// calculation refer to BoxCornerEncoding comments in header file
// use EdgeflowTensorSharedPtr instead of CenterSizeEncoding, because input data might have aux data afterwards
void PPerceptionPostProcessingNMS::DecodeBox(
        BoxCornerEncoding* decoded_boxes,
        const PPerceptionTensorSharedPtr& coord_tensor,
        const int i,
        const CenterSizeEncoding& scale_values) {
    if (decoded_boxes != nullptr) {
        auto& box = decoded_boxes[i];
        if ((box.xmin == 0) && (box.xmax == 0) && (box.ymin == 0) && (box.ymax == 0)) {
            // this decoded_box need to be calculated
            // The calculation is based on tflite's DecodeCenterSizeBoxes()
            auto& anchor = anchors_[i];
            // the coord_tensor is supposed to have shape like 1,6048,1,4 where shape[3] is each box #of floats
            // shape[3] might be larger than 4 (for example 55), in this case, the remaining floats are the aux data
            // We use below calculation instead of casting to array of CenterSizeEncoding because of this
            auto next_box_coord = static_cast<float *>(coord_tensor->data) + i*coord_tensor->shape[3];
            auto& box_centersize = *reinterpret_cast<CenterSizeEncoding *>(next_box_coord);

            float ycenter = box_centersize.y / scale_values.y * anchor.h + anchor.y;
            float xcenter = box_centersize.x / scale_values.x * anchor.w + anchor.x;
            float half_h =
                0.5f * static_cast<float>(std::exp(box_centersize.h / scale_values.h)) *
                anchor.h;
            float half_w =
                0.5f * static_cast<float>(std::exp(box_centersize.w / scale_values.w)) *
                anchor.w;

            box.ymin = CLAMP_0_1(ycenter - half_h);
            box.xmin = CLAMP_0_1(xcenter - half_w);
            box.ymax = CLAMP_0_1(ycenter + half_h);
            box.xmax = CLAMP_0_1(xcenter + half_w);

            //EDGEFLOW_LOG_VERBOSE("DecodeBox %d, raw xyhw %f %f %f %f decoded xxyy %f %f %f %f", i,
            //    box_centersize.x, box_centersize.y, box_centersize.h, box_centersize.w,
            //    box.xmin, box.xmax, box.ymin, box.ymax);
        }
    } else {
        //EDGEFLOW_LOG_ERROR("DecodeBox called without initialization");
    }
}

// NonMaxSuppressionSingleClass() prunes out the box locations with high overlap
// before selecting the highest scoring boxes (max_detections in number)
// It assumes all boxes are good in beginning and sorts based on the scores.
// If lower-scoring box has too much overlap with a higher-scoring box,
// we get rid of the lower-scoring box.
// Complexity is O(N^2) pairwise comparison between boxes
//
// If anchors_ is empty, means that there's no anchor box, so no need to decode box
bool PPerceptionPostProcessingNMS::NonMaxSuppressionSingleClassHelper(
    const PPerceptionParamNMS& param,
    const std::vector<float>& scores,
    const PPerceptionTensorSharedPtr& coord_tensor,
    std::vector<int>* selected,
    int max_detections) {

    const float non_max_suppression_score_threshold = param.score_threshold;
    const float intersection_over_union_threshold   = param.iou_threshold;

    // threshold scores
    std::vector<int> keep_indices;
    // TODO (chowdhery): Remove the dynamic allocation and replace it
    // with temporaries, esp for std::vector<float>
    std::vector<float> keep_scores;
    SelectDetectionsAboveScoreThreshold(
        scores, non_max_suppression_score_threshold, &keep_scores, &keep_indices);

    int num_scores_kept = keep_scores.size();
    //EDGEFLOW_LOG_VERBOSE("num_scores_kept before NMS: %d", num_scores_kept);
    std::vector<int> sorted_indices;
    sorted_indices.resize(num_scores_kept);
    DecreasingPartialArgSort(keep_scores.data(), num_scores_kept, num_scores_kept,
                             sorted_indices.data());

    const int num_boxes_kept = num_scores_kept;
    const int output_size = std::min(num_boxes_kept, max_detections);

    selected->clear();

    int num_active_candidate = num_boxes_kept;
    std::vector<uint8_t> active_box_candidate(num_boxes_kept);
    for (int row = 0; row < num_boxes_kept; row++) {
        active_box_candidate[row] = 1;
    }

    for (int i = 0; i < num_boxes_kept; ++i) {
        if (num_active_candidate == 0 || selected->size() >= (size_t)output_size) break;

        if (active_box_candidate[i] == 1) {
            // Need to decode box i here, not in the DIFF2 below, otherwise if box i is the only box then it can't got decoded
            // We only do it when there's anchor boxes
            if (anchor_exist_) {
                DecodeBox(decoded_boxes_.data(), coord_tensor, keep_indices[sorted_indices[i]], nms_param_.anchor_scale);
            }
            selected->push_back(keep_indices[sorted_indices[i]]);
            active_box_candidate[i] = 0;
            num_active_candidate--;
        } else {
            continue;
        }

        for (int j = i + 1; j < num_boxes_kept; ++j) {
            if (active_box_candidate[j] == 1) {

                // DIFF2: decode box
                if (anchor_exist_) {
                    DecodeBox(decoded_boxes_.data(), coord_tensor, keep_indices[sorted_indices[j]], nms_param_.anchor_scale);
                }

                float intersection_over_union = ComputeIntersectionOverUnion(
                    (anchor_exist_) ? decoded_boxes_.data() : (const pperception::BoxCornerEncoding*)coord_tensor->data,
                    keep_indices[sorted_indices[i]],
                    keep_indices[sorted_indices[j]]);

                if (intersection_over_union > intersection_over_union_threshold) {
                    active_box_candidate[j] = 0;
                    num_active_candidate--;
                }
            }
        }
    }
    return true;
}


// This function implements a regular version of Non Maximal Suppression (NMS)
// for multiple classes where
// 1) we do NMS separately for each class across all anchors and
// 2) keep only the highest anchor scores across all classes
// 3) The worst runtime of the regular NMS is O(K*N^2)
// where N is the number of anchors and K the number of
// classes.
bool PPerceptionPostProcessingNMS::NonMaxSuppressionMultiClassRegularHelper(
        const PPerceptionTensorSharedPtr& score_tensor,
        const PPerceptionTensorSharedPtr& coord_tensor,
        const int                      num_classes_to_consider,
        std::vector<float>&             postprocessed_class,
        std::vector<float>&             postprocessed_score,
        std::vector<BoxCornerEncoding>& postprocessed_coord,
        std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata,
        std::vector<int>&               anchor_indices) {

    const PPerceptionParamNMS& param = nms_param_;
    const int num_background = param.num_background_class; /* # background class at the begining */
    const int num_boxes = coord_tensor->shape[1];
    const int num_classes_with_background = score_tensor->shape[2];
    const int num_classes = num_classes_with_background - num_background;
    const int num_detections_per_class = param.detections_per_class;
    const int max_detections = param.max_detections;

    if (coord_tensor->shape[3] != sizeof(struct CenterSizeEncoding)/sizeof(float) &&
        coord_tensor->shape[3] != sizeof(struct PPerceptionModelOutputDetectionHKD)/sizeof(float)) {
        //EDGEFLOW_LOG_ERROR("coord_tensor->shape[3] = %lu, not expected", coord_tensor->shape[3]);
        return false;
    }
    //EDGEFLOW_LOG_VERBOSE("num_classes %d, num_background %d, num_boxes %d", num_classes, num_background, num_boxes);

    float *scores = static_cast<float *>(score_tensor->data);
    std::vector<float> class_scores(num_boxes);
    std::vector<int>   box_indices_after_regular_non_max_suppression(num_boxes + max_detections);
    std::vector<float> scores_after_regular_non_max_suppression(num_boxes + max_detections);

    int size_of_sorted_indices = 0;
    std::vector<int>   sorted_indices;
    sorted_indices.resize(num_boxes + max_detections);
    std::vector<float> sorted_values;
    sorted_values.resize(max_detections);

    // determine the final number of classes we consider to run NMS and output
    int final_num_classes_to_consider = num_classes;
    if (num_classes_to_consider > 0) {
        if (num_classes_to_consider > num_classes) {
            //EDGEFLOW_LOG_ERROR("num_classes_to_consider %d > num_classes %d", num_classes_to_consider, num_classes);
            return false;
        } else {
            final_num_classes_to_consider = num_classes_to_consider;
        }
    }

    for (int col = 0; col < final_num_classes_to_consider; col++) {
        for (int row = 0; row < num_boxes; row++) {
            // Get scores of boxes corresponding to all anchors for single class
            float raw_score = *(scores + row * num_classes_with_background + col + num_background);
            // We take raw scores (e.g. without sigmoid) to save time
            class_scores[row] = raw_score;
        }
        // Perform non-maximal suppression on single class
        std::vector<int> selected;
        NonMaxSuppressionSingleClassHelper(param, class_scores, coord_tensor, &selected, num_detections_per_class);
        //EDGEFLOW_LOG_VERBOSE("class %d number of selected boxes after NMS: %ld", col, selected.size());
        // Add selected indices from non-max suppression of boxes in this class
        int output_index = size_of_sorted_indices;
        for (int selected_index : selected) {
            box_indices_after_regular_non_max_suppression[output_index] =
                (selected_index * num_classes_with_background + col + num_background);
            scores_after_regular_non_max_suppression[output_index] =
                class_scores[selected_index];
            output_index++;
        }

        // Sort the max scores among the selected indices
        // Get the indices for top scores
        int num_indices_to_sort = std::min(output_index, max_detections);
        DecreasingPartialArgSort(scores_after_regular_non_max_suppression.data(),
                                 output_index, num_indices_to_sort,
                                 sorted_indices.data());

        // Copy values to temporary vectors
        for (int row = 0; row < num_indices_to_sort; row++) {
            int temp = sorted_indices[row];
            sorted_indices[row] = box_indices_after_regular_non_max_suppression[temp];
            sorted_values[row] = scores_after_regular_non_max_suppression[temp];
        }
        // Copy scores and indices from temporary vectors
        for (int row = 0; row < num_indices_to_sort; row++) {
            box_indices_after_regular_non_max_suppression[row] = sorted_indices[row];
            scores_after_regular_non_max_suppression[row] = sorted_values[row];
        }
        size_of_sorted_indices = num_indices_to_sort;
    }

    // Allocate output tensors
    BoxCornerEncoding * input_box = (BoxCornerEncoding *)coord_tensor->data;
    for (int output_box_index = 0; output_box_index < max_detections; output_box_index++) {
        if (output_box_index < size_of_sorted_indices) {
            const int anchor_index = floor(
                box_indices_after_regular_non_max_suppression[output_box_index] /
                num_classes_with_background);
            const int class_index =
                box_indices_after_regular_non_max_suppression[output_box_index] -
                anchor_index * num_classes_with_background - num_background;
            const float selected_score = param.return_sigmoid_score ?
                SIGMOID(scores_after_regular_non_max_suppression[output_box_index]) : // DIFF1: convert to sigmoid score
                scores_after_regular_non_max_suppression[output_box_index];
            // detection_boxes
            postprocessed_coord.push_back(anchor_exist_ ? decoded_boxes_[anchor_index] : input_box[anchor_index]);
            // detection_classes
            postprocessed_class.push_back(static_cast<float>(class_index));
            // detection_scores
            postprocessed_score.push_back(selected_score);
            // anchor indices
            anchor_indices.push_back(anchor_index);
            // aux data (those data that's after the original model generated coordinates
            if (coord_tensor->shape[3] == sizeof(struct PPerceptionModelOutputDetectionHKD)/sizeof(float)) {
                auto model_outputs = static_cast<PPerceptionModelOutputDetectionHKD *>(coord_tensor->data);
                // now decodes the key points coordinates in place and push to output vector
                // the decoding of x,y is exactly same as the bounding box coordinates
                for (int i = 0; i < DETECTION_HKD_NUM_KP; i++) {
                    float x = model_outputs[anchor_index].auxdata.kp[i].x;
                    float y = model_outputs[anchor_index].auxdata.kp[i].y;

                    auto& anchor = anchors_[anchor_index];
                    auto& scale_values = nms_param_.anchor_scale;

                    float ycenter = y / scale_values.y * anchor.h + anchor.y;
                    float xcenter = x / scale_values.x * anchor.w + anchor.x;

                    model_outputs[anchor_index].auxdata.kp[i].x = xcenter;
                    model_outputs[anchor_index].auxdata.kp[i].y = ycenter;

                    //EDGEFLOW_LOG_VERBOSE("decodes kp x from %f to %f, y from %f to %f", x, xcenter, y, ycenter);
                    //EDGEFLOW_LOG_VERBOSE("vis %d = %f", i, model_outputs[anchor_index].auxdata.visibility[i]);
                }
                postprocessed_auxdata.push_back(model_outputs[anchor_index].auxdata);
            }

            //EDGEFLOW_LOG_VERBOSE("RegularNMS Final box %d, box_index %d anchor_index %d class %d score %f ymin/xmin/ymax/xmax %f %f %f %f",
                // output_box_index, box_indices_after_regular_non_max_suppression[output_box_index],
                // anchor_index, class_index, selected_score,
                // anchor_exist_ ? decoded_boxes_[anchor_index].ymin : input_box[anchor_index].ymin,
                // anchor_exist_ ? decoded_boxes_[anchor_index].xmin : input_box[anchor_index].xmin,
                // anchor_exist_ ? decoded_boxes_[anchor_index].ymax : input_box[anchor_index].ymax,
                // anchor_exist_ ? decoded_boxes_[anchor_index].xmax : input_box[anchor_index].xmax);
        } else {
            break; // DIFF3: do not fill in 0s for the rest
        }
    }
    box_indices_after_regular_non_max_suppression.clear();
    scores_after_regular_non_max_suppression.clear();

    return true;
}


// This function implements Non Maximal Suppression (NMS) for hybrid heads where
// 1) The first head is for detection, and the other heads are for classification
// 2) We do NMS only on the detections from the first head
// 3) The classes of detections are determined by the other heads after NMS
// 4) Keep only the highest anchor detection scores across all classes
// 5) The worst runtime of the regular NMS is O(N^2 + K*N)
// where N is the number of anchors and K the number of classes.
bool PPerceptionPostProcessingNMS::NonMaxSuppressionMultiClassWithHybridHeadsHelper(
        const PPerceptionTensorSharedPtr&                     score_tensor,
        const PPerceptionTensorSharedPtr&                     coord_tensor,
        std::vector< float >&                              postprocessed_class,
        std::vector< float >&                              postprocessed_score,
        std::vector< BoxCornerEncoding >&                  postprocessed_coord,
        std::vector< PPerceptionModelAuxOutputDetectionHKD >& postprocessed_auxdata) {
    const PPerceptionParamNMS& param          = nms_param_;
    const int               num_background = param.num_background_class;     // # of background class at the begining
    const int               num_heads      = score_tensor->shape[2];         // # of heads
    const int               num_classes    = num_heads - num_background - 1; // # of classification heads

    if (num_classes <= 0) {
        //EDGEFLOW_LOG_ERROR("num_classes <= 0 for the hybrid head nms");
        return false;
    }

    if (param.class_score_bias.size() != static_cast< std::size_t >(num_classes)) {
        //EDGEFLOW_LOG_ERROR("param.class_score_bias.size() != num_classes");
        return false;
    }

    // get the NMS results from the first detection head
    std::vector<int> anchor_indices;
    if (!NonMaxSuppressionMultiClassRegularHelper(score_tensor, coord_tensor, 1,
            postprocessed_class, postprocessed_score, postprocessed_coord, postprocessed_auxdata, anchor_indices)) {
        // NMS failed, return false
        return false;
    }

    // update `postprocessed_class` based on other classification heads
    const float* scores = static_cast< float* >(score_tensor->data);
    for (std::size_t output_box_index = 0; output_box_index < anchor_indices.size(); ++output_box_index) {
        const int anchor_index = anchor_indices[output_box_index];

        // determine the class index (0 ~ (num_classes - 1)) based on the scores of classification heads
        const float* class_score_ptr = scores + anchor_index * num_heads + num_background + 1;  // the starting pointer to
                                                                                                // the first class score
        int max_score_class_idx = 0;
        float max_score = class_score_ptr[0] + param.class_score_bias[0];
        for (int i = 1; i < num_classes; ++i) {
            float current_class_score = class_score_ptr[i] + param.class_score_bias[i];
            if (max_score < current_class_score) {
                max_score = current_class_score;
                max_score_class_idx = i;
            }
        }

        postprocessed_class[output_box_index] = max_score_class_idx;
    }
    return true;
}


// This function implements a fast version of Non Maximal Suppression for
// multiple classes where
// 1) we keep the top-k scores for each anchor and
// 2) during NMS, each anchor only uses the highest class score for sorting.
// 3) Compared to standard NMS, the worst runtime of this version is O(N^2)
// instead of O(KN^2) where N is the number of anchors and K the number of
// classes.
bool PPerceptionPostProcessingNMS::NonMaxSuppressionMultiClassFastHelper(
        const PPerceptionTensorSharedPtr& score_tensor,
        const PPerceptionTensorSharedPtr& coord_tensor,
        std::vector<float>&             postprocessed_class,
        std::vector<float>&             postprocessed_score,
        std::vector<BoxCornerEncoding>& postprocessed_coord) {

    const PPerceptionParamNMS& param = nms_param_;
    const int num_background = param.num_background_class; /* usually 1, one background class at the begining */
    const int num_boxes = coord_tensor->shape[1];
    const int num_classes_with_background = score_tensor->shape[3];
    const int num_classes = num_classes_with_background - num_background;
    const int max_categories_per_anchor = 1; //op_data->max_classes_per_detection;
    const int num_categories_per_anchor = std::min(max_categories_per_anchor, num_classes);
    const int max_detections = param.max_detections;

    float *scores = static_cast<float *>(score_tensor->data);

    std::vector<float> max_scores;
    max_scores.resize(num_boxes);
    std::vector<int> sorted_class_indices;
    sorted_class_indices.resize(static_cast<size_t>(num_boxes) * static_cast<size_t>(num_classes));
    for (int row = 0; row < num_boxes; row++) {
        const float* box_scores = scores + row * num_classes_with_background + num_background;
        int* class_indices = sorted_class_indices.data() + row * num_classes;
        DecreasingPartialArgSort(box_scores, num_classes, num_categories_per_anchor, class_indices);
        max_scores[row] = box_scores[class_indices[0]];
    }

    // Perform non-maximal suppression on max scores
    std::vector<int> selected;
    NonMaxSuppressionSingleClassHelper(param, max_scores, coord_tensor, &selected, max_detections);

    // Allocate output tensors
    int output_box_index = 0;
    for (const auto& selected_index : selected) {
        const float* box_scores = scores + selected_index * num_classes_with_background + num_background;
        const int* class_indices = sorted_class_indices.data() + selected_index * num_classes;

        for (int col = 0; col < num_categories_per_anchor; ++col) {
            // detection_boxes
            postprocessed_coord.push_back(decoded_boxes_[selected_index]);
            // detection_classes
            postprocessed_class.push_back(static_cast<float>(class_indices[col]));
            // detection_scores
            postprocessed_score.push_back(SIGMOID(box_scores[class_indices[col]]));

            output_box_index++;

            //EDGEFLOW_LOG_VERBOSE("FastNMS Final box %d, class %d score %f ymin/xmin/ymax/xmax %f %f %f %f", output_box_index,
                // class_indices[col], box_scores[class_indices[col]],
                // decoded_boxes_[selected_index].ymin, decoded_boxes_[selected_index].xmin,
                // decoded_boxes_[selected_index].ymax, decoded_boxes_[selected_index].xmax);
        }
    }

    return true;
}


/* This function implements a cross_class version of Non Maximal Suppression for
* multiple classes where
* 1) we run multi class regular NMS first
* 2) then combine detections from all classes, and run NMS.
* 3) the worst runtime of this version is O(N^2)
* where N is the total number of anchors from all classes.
* @param score_tensor: input score tensor
* @param coord_tensor: input coordinates tensor
* @param postprocessed_class: output class
* @param postprocessed_score: output score
* @param postprocessed_coord: output coord
*/
bool PPerceptionPostProcessingNMS::NonMaxSuppressionMultiClassCrossClassHelper(
        const PPerceptionTensorSharedPtr& score_tensor,
        const PPerceptionTensorSharedPtr& coord_tensor,
        std::vector<float>&             postprocessed_class,
        std::vector<float>&             postprocessed_score,
        std::vector<BoxCornerEncoding>& postprocessed_coord,
        std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata) {
    std::vector<PPerceptionModelAuxOutputDetectionHKD> initial_processed_auxdata;
    std::vector<float> initial_processed_class,
                       initial_processed_score;
    std::vector<BoxCornerEncoding> initial_processed_coord;

    // NMS inside each class
    std::vector<int> anchor_indices;  // dummy anchor indices
    if (!NonMaxSuppressionMultiClassRegularHelper(score_tensor, coord_tensor, 0,
               initial_processed_class, initial_processed_score, initial_processed_coord, initial_processed_auxdata, anchor_indices)) {
        return false;
    }

    // NMS across all classes
    int num_scores = initial_processed_score.size();
    std::vector<int> sorted_indices;
    sorted_indices.resize(num_scores);
    DecreasingPartialArgSort(initial_processed_score.data(), num_scores, num_scores, sorted_indices.data());

    std::vector<int> kept_indices;
    for (auto idx : sorted_indices) {
        bool keep_detection = true;
        for (auto kept_idx : kept_indices) {
            const auto iou = ComputeIntersectionOverUnion(initial_processed_coord.data(), idx, kept_idx);
            if (iou > nms_param_.cross_class_iou_threshold) {
                keep_detection = false;
                break;
            }
        }
        if (keep_detection) {
            kept_indices.emplace_back(idx);
        }
    }
    int auxdata_size = static_cast< int >(initial_processed_auxdata.size());
    for(auto idx : kept_indices) {
        postprocessed_class.emplace_back(initial_processed_class[idx]);
        postprocessed_score.emplace_back(initial_processed_score[idx]);
        postprocessed_coord.emplace_back(initial_processed_coord[idx]);
        if(idx < auxdata_size) {
            postprocessed_auxdata.emplace_back(initial_processed_auxdata[idx]);
        }
    }
    return true;
}

/* This function does NMS.
 * @param score_tensor: input score tensor. One box contains #class scores
 *                      Dimension example: hdfd 1,6048,3,1(3 classes), superpoint 1,160,1,256(1 classes)
 * @param coord_tensor: input coord tensor. The data contains coordinates plus optional aux data
 *                      Dimension example: hdfd 1,6048,1,4 (without HKD), 1,6048,1,55(with HKD)
 * @param postprocessed_class: output box class
 * @param postprocessed_score: output score
 * @param postprocessed_coord: output coordinates
 * @param postprocessed_auxdata: output optional auxdata, taken from the coord_tensor
 */
bool PPerceptionPostProcessingNMS::NonMaxSuppressionMultiClassHelper(
        const PPerceptionTensorSharedPtr& score_tensor,
        const PPerceptionTensorSharedPtr& coord_tensor,
        std::vector<float>&             postprocessed_class,
        std::vector<float>&             postprocessed_score,
        std::vector<BoxCornerEncoding>& postprocessed_coord,
        std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata) {

    /* all of these needs anchor box, so check if anchor file is read in
     */
    if (nms_param_.anchor_valid == false) {
        //EDGEFLOW_LOG_ERROR("Error: anchor was not valid");
        return false;
    }

    /* allocate enough memory for decoded_boxes_, its size is same as anchors
     * this resize should only cost time once on first frame
     * If anchors_ is empty, means no need to decode box, so decoded_boxes_ is not needed as well
     */
    if (anchor_exist_) {
        decoded_boxes_.resize(anchors_.size());

        // clear decoded_boxes_, since 0 means not calculated
        memset(decoded_boxes_.data(), 0, decoded_boxes_.size()*sizeof(BoxCornerEncoding));
    }

    std::vector<int> anchor_indices;  // dummy anchor indices
    return NonMaxSuppressionMultiClassRegularHelper(score_tensor, coord_tensor, 0,
            postprocessed_class, postprocessed_score, postprocessed_coord, postprocessed_auxdata, anchor_indices);

}

/* This function does NMS.
 * @param score_tensor: input score tensor. One box contains #class scores
 * @param coord_tensor: input coord tensor. The data contains coordinates plus optional aux data
 * @param postprocessed_class: output box class
 * @param postprocessed_score: output score
 * @param postprocessed_coord: output coordinates
 */
bool PPerceptionPostProcessingNMS::NonMaxSuppressionMultiClassHelper(
        const PPerceptionTensorSharedPtr& score_tensor,
        const PPerceptionTensorSharedPtr& coord_tensor,
        std::vector<float>&             postprocessed_class,
        std::vector<float>&             postprocessed_score,
        std::vector<BoxCornerEncoding>& postprocessed_coord) {

    /* allocate enough memory for decoded_boxes_, its size is same as anchors
     * this resize should only cost time once on first frame
     * If anchors_ is empty, means no need to decode box, so decoded_boxes_ is not needed as well
     */
    if (anchor_exist_) {
        decoded_boxes_.resize(anchors_.size());

        // clear decoded_boxes_, since 0 means not calculated
        memset(decoded_boxes_.data(), 0, decoded_boxes_.size()*sizeof(BoxCornerEncoding));
    }


    //TODO: VESTA-83882, refactor NonMaxSuppressionMultiClassHelper function
    //pass a dummy postprocessed_auxdata for now
    std::vector<PPerceptionModelAuxOutputDetectionHKD> postprocessed_auxdata;
    std::vector<int> anchor_indices;  // dummy anchor indices
    return NonMaxSuppressionMultiClassRegularHelper(score_tensor, coord_tensor, 0,
            postprocessed_class, postprocessed_score, postprocessed_coord, postprocessed_auxdata, anchor_indices);

}


/* This function does single class NMS, and also takes CenterSizeEncoding as input.
 * Complexity is O(N^2) pairwise comparison between boxes
 * @param score_vec:    input score vector. One box contains one score
 * @param coord_tensor: input coord vector, CenterSizeEncoding type.
 * @param postprocessed_score: output score vector, these are expected to be non-sigmoid scores
 * @param postprocessed_coord: output coordinates vector, BoxCornerEncoding type
 */
bool PPerceptionPostProcessingNMS::NonMaxSuppressionSingleClassHelper(
    const std::vector<float>& score_vec,
    const std::vector<CenterSizeEncoding>& coord_vec,
    std::vector<float>&             postprocessed_score,
    std::vector<BoxCornerEncoding>& postprocessed_coord) {

    /* allocate enough memory for decoded_boxes_, its size is same as input coord_vec
     * this resize should only cost time once on first frame
     */
    decoded_boxes_.resize(coord_vec.size());

    // clear decoded_boxes_, since 0 means not calculated
    memset(decoded_boxes_.data(), 0, decoded_boxes_.size()*sizeof(BoxCornerEncoding));

    std::vector<int> selected;
    // note the score threshold was inverse-sigmoid() when read in, so need to sigmoid() to revert back
    const float non_max_suppression_score_threshold = SIGMOID(nms_param_.score_threshold);
    const float intersection_over_union_threshold   = nms_param_.iou_threshold;

    // threshold scores
    std::vector<int> keep_indices;
    std::vector<float> keep_scores;
    SelectDetectionsAboveScoreThreshold(
        score_vec, non_max_suppression_score_threshold, &keep_scores, &keep_indices);

    int num_scores_kept = keep_scores.size();
    std::vector<int> sorted_indices;
    sorted_indices.resize(num_scores_kept);
    DecreasingPartialArgSort(keep_scores.data(), num_scores_kept, num_scores_kept,
                             sorted_indices.data());

    const int num_boxes_kept = num_scores_kept;
    const int output_size = std::min(num_boxes_kept, nms_param_.max_detections);

    selected.clear();

    int num_active_candidate = num_boxes_kept;
    std::vector<uint8_t> active_box_candidate(num_boxes_kept);
    for (int row = 0; row < num_boxes_kept; row++) {
        active_box_candidate[row] = 1;
    }

    for (int i = 0; i < num_boxes_kept; ++i) {
        if (num_active_candidate == 0 || selected.size() >= (size_t)output_size) break;
        if (active_box_candidate[i] == 1) {
            // Need to decode box i here, not in the DIFF2 below.
            // Otherwise if box i is the only box then it can't got decoded
            DecodeBox(decoded_boxes_.data(), coord_vec.data(), keep_indices[sorted_indices[i]], nms_param_.anchor_scale, nms_param_.anchor_valid);
            selected.push_back(keep_indices[sorted_indices[i]]);
            active_box_candidate[i] = 0;
            num_active_candidate--;
        } else {
            continue;
        }
        for (int j = i + 1; j < num_boxes_kept; ++j) {
            if (active_box_candidate[j] == 1) {

                // DIFF2: decode box
                DecodeBox(decoded_boxes_.data(), coord_vec.data(), keep_indices[sorted_indices[j]], nms_param_.anchor_scale, nms_param_.anchor_valid);

                float intersection_over_union = ComputeIntersectionOverUnion(
                    decoded_boxes_.data(), keep_indices[sorted_indices[i]], keep_indices[sorted_indices[j]]);

                if (intersection_over_union > intersection_over_union_threshold) {
                    active_box_candidate[j] = 0;
                    num_active_candidate--;
                }
            }
        }
    }

    /* now assign outputs */
    for (const auto& selected_index : selected) {
        postprocessed_coord.push_back(decoded_boxes_[selected_index]);
        postprocessed_score.push_back(score_vec[selected_index]);
        //EDGEFLOW_LOG_VERBOSE("NonMaxSuppressionSingleClassHelper Final box selected %d, score %f ymin/xmin/ymax/xmax %f %f %f %f",
                // selected_index,
                // score_vec[selected_index],
                // decoded_boxes_[selected_index].ymin, decoded_boxes_[selected_index].xmin,
                // decoded_boxes_[selected_index].ymax, decoded_boxes_[selected_index].xmax);
    }

    return true;
}
