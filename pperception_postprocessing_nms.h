#ifndef INCLUDE_PPERCEPTION_POSTPROCESSING_NMS_H_
#define INCLUDE_PPERCEPTION_POSTPROCESSING_NMS_H_

#include <cmath>
#include <algorithm>
#include "pperception_model_fdbd_mobilenetv1ssd.h"

#pragma once

namespace pperception {

#define SIGMOID(raw_score)     (1/(1 + std::exp(-1 * (raw_score))))
#define INVERSE_SIGMOID(score) (-1 * std::log(1/(score) - 1)) /* Note log() function returns ln() */

// Object Detection model produces axis-aligned boxes in two formats:
// BoxCorner represents the lower left corner (xmin, ymin) and
// the upper right corner (xmax, ymax).
// CenterSize represents the center (xcenter, ycenter), height and width.
// BoxCornerEncoding and CenterSizeEncoding are related as follows:
// ycenter = y / y_scale * anchor.h + anchor.y;
// xcenter = x / x_scale * anchor.w + anchor.x;
// half_h = 0.5*exp(h/ h_scale)) * anchor.h;
// half_w = 0.5*exp(w / w_scale)) * anchor.w;
// ymin = ycenter - half_h
// ymax = ycenter + half_h
// xmin = xcenter - half_w
// xmax = xcenter + half_w
struct BoxCornerEncoding {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
};

typedef std::shared_ptr<pperception::PPerceptionTensor> PPerceptionTensorSharedPtr;

/**
 * NMS parameter
 * values will be read from nms json config file in SetNMSConfigs()
 */
struct PPerceptionParamNMS {
    uint32_t approach          = 0;   //!< 0=Regular; 1=hybrid heads (first head for detection, and other heads for classification); 2=fast NMS; 3=corss-class NMS
    int   max_detections       = 100; //!< max number of boxes in all classes
    float score_threshold      = 0.1; //!< only consider those with a larger score. 0.1 means 10%
    float iou_threshold        = 0.3; //!< Intersection over uion threshold
    float cross_class_iou_threshold = 0.7; //!< Cross class intersection over uion threshold
    int   detections_per_class = 10;  //!< max number of boxes per class. Specific to regular NMS
    int   max_categories_per_anchor = 1; //!< how many category box per anchor box can generate. Specific to Fast NMS
    int   num_background_class = 1;   //!< how many background classes in the model
    bool  return_sigmoid_score = true;//!< if the score input to NMS are sigmoided
    int   pixel_box_size       = 4;   //!< For use case that generates bbox from pixel coordinates
    bool  anchor_valid         = false;  //!< if there's a valid anchor file
    std::vector< float > class_score_bias = {};  //!< the class score bias for hybrid_head NMS to achieve optimal classification accuracy
    CenterSizeEncoding anchor_scale = {1, 1, 1, 1};
};

struct PPerceptionModelAuxOutputDetectionHKD;

/**
 * main class for post processing NMS
 */
struct PPerceptionPostProcessingNMS {
    /* This function reads NMS config json file and set the NMS params to private variable nms_param_
     * @param nms_json_path: input json file path
     * @param model_dir: model path
     */
    bool ReadNMSConfigs(const std::string& nms_json_path, const std::string& model_path);

    /* This function does NMS.
     * @param score_tensor: input score tensor. One box contains #class scores
     * @param coord_tensor: input coord tensor. The data contains coordinates plus optional aux data
     * @param postprocessed_class: output box class
     * @param postprocessed_score: output score
     * @param postprocessed_coord: output coordinates
     * @param postprocessed_auxdata: output optional auxdata, taken from the coord_tensor and converted
     */
    bool NonMaxSuppressionMultiClassHelper(
            const PPerceptionTensorSharedPtr& score_tensor,
            const PPerceptionTensorSharedPtr& coord_tensor,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata);

    // Overload NonMaxSuppressionMultiClassHelper function
    bool NonMaxSuppressionMultiClassHelper(
            const PPerceptionTensorSharedPtr& score_tensor,
            const PPerceptionTensorSharedPtr& coord_tensor,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord);

    int  GetPixelBoxSize() const { return nms_param_.pixel_box_size; }

    float GetScoreThreshold() const { return nms_param_.score_threshold; }

    float GetIOUThreshold() const { return nms_param_.iou_threshold; }

    int   GetMaxDetections() const { return nms_param_.max_detections; }

    bool setNMSConfig();

    /* This function does single class NMS, and also takes CenterSizeEncoding as input.
     * @param score_vec:    input score vector. One box contains one score
     * @param coord_tensor: input coord vector, CenterSizeEncoding type.
     * @param postprocessed_score: output score vector
     * @param postprocessed_coord: output coordinates vector, BoxCornerEncoding type
     */
    bool NonMaxSuppressionSingleClassHelper(
            const std::vector<float>& score_vec,
            const std::vector<CenterSizeEncoding>& coord_vec,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord);

  private:

    PPerceptionParamNMS nms_param_;                    //!< parameter, set through SetNMSConfigs
    std::vector<CenterSizeEncoding> anchors_;       //!< dequantized anchor coordinates
    bool anchor_exist_ = false;                             //!< if there's anchor box, e.g. anchors_.size() != 0
    std::vector<BoxCornerEncoding>  decoded_boxes_; //!< box corners calculated from model output centers and anchor box

    void DecreasingPartialArgSort(
            const float* values, int num_values,
            int num_to_sort, int* indices);

    void SelectDetectionsAboveScoreThreshold(
            const std::vector<float>& values,
            const float threshold,
            std::vector<float>* keep_values,
            std::vector<int>* keep_indices);

    bool ReadAnchorBoxEncodings(
            const std::string& anchor_path,
            const int   anchor_quant_zero_point,
            const float anchor_quant_scale,
            std::vector<CenterSizeEncoding>& anchors);

    float ComputeIntersectionOverUnion(
        const BoxCornerEncoding *decoded_boxes,
        const int i, const int j);

    void DecodeBox(
        BoxCornerEncoding* decoded_boxes,
        const CenterSizeEncoding *raw_boxes,
        const int i,
        const CenterSizeEncoding& scale_values,
        bool anchor_valid);

    void DecodeBox(
        BoxCornerEncoding*  decoded_boxes,
        const PPerceptionTensorSharedPtr& coord_tensor,
        const int i,
        const CenterSizeEncoding& scale_values);

    bool NonMaxSuppressionSingleClassHelper(
            const PPerceptionParamNMS& param,
            const std::vector<float>& scores,
            const PPerceptionTensorSharedPtr& coord_tensor,
            std::vector<int>* selected,
            int max_detections);


    /* This function implements a regular version of Non Maximal Suppression (NMS)
     * for multiple classes where
     * 1) we do NMS separately for each class across all anchors and
     * 2) keep only the highest anchor scores across all classes
     * 3) The worst runtime of the regular NMS is O(K*N^2)
     * where N is the number of anchors and K the number of classes.
     * @param score_tensor: input score tensor
     * @param coord_tensor: input coordinates tensor
     * @param num_classes_to_consider: number of classes to consider. If <= 0, consider all the classes
     * @param postprocessed_class: output class
     * @param postprocessed_score: output score
     * @param postprocessed_coord: output coord
     * @param postprocessed_auxdata: output optional auxdata, contains the human keypoints data
     * @param anchor_indices: output anchor indices of detections
     */
    bool NonMaxSuppressionMultiClassRegularHelper(
            const PPerceptionTensorSharedPtr& score_tensor,
            const PPerceptionTensorSharedPtr& coord_tensor,
            const int                      num_classes_to_consider,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata,
            std::vector<int>&               anchor_indices);

    /* This function implements Non Maximal Suppression (NMS) for hybrid heads where
     * 1) The first head is for detection, and the other heads are for classification
     * 2) We do NMS only on the detections from the first head
     * 3) The classes of detections are determined by the other heads after NMS
     * 4) Keep only the highest anchor detection scores across all classes
     * 5) The worst runtime of the regular NMS is O(N^2 + K*N)
     * where N is the number of anchors and K the number of classes.
     * @param score_tensor: input score tensor
     * @param coord_tensor: input coordinates tensor
     * @param postprocessed_class: output class
     * @param postprocessed_score: output score
     * @param postprocessed_coord: output coord
     * @param postprocessed_auxdata: output optional auxdata, contains the human keypoints data
     */
    bool NonMaxSuppressionMultiClassWithHybridHeadsHelper(
            const PPerceptionTensorSharedPtr& score_tensor,
            const PPerceptionTensorSharedPtr& coord_tensor,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata);

    /* This function implements a fast version of Non Maximal Suppression for
     * multiple classes where
     * 1) we keep the top-k scores for each anchor and
     * 2) during NMS, each anchor only uses the highest class score for sorting.
     * 3) Compared to standard NMS, the worst runtime of this version is O(N^2)
     * instead of O(KN^2) where N is the number of anchors and K the number of
     * classes.
     * @param score_tensor: input score tensor
     * @param coord_tensor: input coordinates tensor
     * @param postprocessed_class: output class
     * @param postprocessed_score: output score
     * @param postprocessed_coord: output coord
     */
    bool NonMaxSuppressionMultiClassFastHelper(
            const PPerceptionTensorSharedPtr& score_tensor,
            const PPerceptionTensorSharedPtr& coord_tensor,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord);

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
     * @param postprocessed_auxdata: output optional auxdata, contains the human keypoints data
     */
    bool NonMaxSuppressionMultiClassCrossClassHelper(
            const PPerceptionTensorSharedPtr& score_tensor,
            const PPerceptionTensorSharedPtr& coord_tensor,
            std::vector<float>&             postprocessed_class,
            std::vector<float>&             postprocessed_score,
            std::vector<BoxCornerEncoding>& postprocessed_coord,
            std::vector<PPerceptionModelAuxOutputDetectionHKD>& postprocessed_auxdata);
};

}

#endif /* INCLUDE_PPERCEPTION_POSTPROCESSING_NMS_H_ */
