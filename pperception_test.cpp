#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <memory>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "pperception_model_fdbd_mobilenetv1ssd.h"
#include "edgeflow_test_util.h"

using namespace pperception;
using namespace edgeflow;
using namespace std;

int main(int argc, char** argv) {
    try {

        const char* input_file = "tests/raw_list.txt";
        const int loop_outer = 1;
        const char* model_config_dir = "/opt/edgeflow/etc";
        const char* model_dir = "/opt/edgeflow/model";
        const float detection_confidence_threshold = 0.6f;
        const bool output_on_screen = true;

        std::ifstream inputList(input_file);
        if (!inputList) {
            std::cout << "Input list is not valid. Please ensure that you have provided a valid input list for processing." << std::endl;
            std::exit(0);
        }

        for (int i = 0; (i < loop_outer) || (loop_outer == -1); i++) {
            std::shared_ptr<edgeflow::EdgeflowModel> base_model = std::make_shared< edgeflow::EdgeflowModelMobilenetv1Ssd >();
            std::cout << "EdgeFlow model created!" << std::endl;
            auto model = std::static_pointer_cast< edgeflow::EdgeflowModelMobilenetv1Ssd >(base_model);

            if (!model->EdgeflowModelSetup(
                        EdgeflowSupportedModel::EDGEFLOW_MODEL_FDBD_MobileNet_SSD_v1,
                        model_config_dir,
                        model_dir)) {
                cout << "edgeflow_setup_network fail" << endl;
                exit(1);
            }

            // Open the input file listing and for each input file load its contents
            // into a user buffer, execute the network
            // with the input and save each of the returned output tensors to a file.
            size_t inputListNum = 0;
            std::string fileLine;

            while (std::getline(inputList, fileLine)) {
                if (fileLine.empty()) continue;
                cout << "Processing " << fileLine << endl;

                // treat each line as a space-separated list of input files
                std::vector<std::string> filePaths;
                split(filePaths, fileLine, ' ');
                size_t i = 0;
                std::string filePath(filePaths[i]);
                cout << "load buffer for " << filePath << endl;
                std::vector<uint8_t> input_image;
                if (loadByteDataFile(filePath, input_image) == false)
                    continue; // skip non-exist files

                // preprocess
                EdgeflowModelPreprocessParamMobilenetv1Ssd pre_param;
                EdgeflowModelInputMobilenetv1Ssd           inputssd;
                if (input_image.size() == 640 * 480) {
                    inputssd.width = 640;
                    inputssd.height = 480;
                    inputssd.channel = 1;
                    inputssd.pixelbyte = 1;
                } else if (input_image.size() == 640 * 480 * 3) {
                    inputssd.width = 640;
                    inputssd.height = 480;
                    inputssd.channel = 3;
                    inputssd.pixelbyte = 1;
                }
                inputssd.image = input_image.data();

                std::vector<EdgeflowModelOutputMobilenetv1Ssd> bbox_output;
                std::vector<EdgeflowModelAuxOutputDetectionHKD> kp_output;
                bbox_output.reserve(100);
                kp_output.reserve(100);

                EdgeflowModelPostprocessParamMobilenetv1Ssd post_param;
                post_param.confidence = detection_confidence_threshold;

                auto start = std::chrono::high_resolution_clock::now();    
                
                if (false == model->EdgeflowModelSetInputs(inputssd, pre_param)) {
                    cout << "preprocess fail" << endl;
                    exit(1);
                }
                // execute network
                if (false == model->EdgeflowModelExecute(NULL, 0)) {
                    cout << "execute fail" << endl;
                    exit(1);
                }
                bbox_output.clear();
                kp_output.clear();
                
                if (false == model->EdgeflowModelGetOutputs(bbox_output, kp_output, post_param)) {
                    cout << "postprocess fail" << endl;
                    exit(1);
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                
                std::chrono::duration<double, std::milli> duration_ms = end - start;
                
                std::cout << "Execution time is " << duration_ms.count() << " milliseconds" << std::endl;
                
 

                // The results are unnormalized, i.e. the bounding box coordinates are absolute coordinates in the
                // image rather than relative values within [0, 1].
                for (auto& ptr : bbox_output) {
                    ptr.box_coord.x *= inputssd.width;
                    ptr.box_coord.y *= inputssd.height;
                    ptr.box_coord.w *= inputssd.width;
                    ptr.box_coord.h *= inputssd.height;
                }

                for (auto& ptr : kp_output) {
                    for(int ki = 0; ki < DETECTION_HKD_NUM_KP; ki++) {
                        ptr.kp[ki].x *= inputssd.width;
                            ptr.kp[ki].y *= inputssd.height;
                    }
                }


                if (output_on_screen) {
                    for (auto& ptr: bbox_output) {
                        std::cout << " class: " << ptr.box_class << " Score: " << ptr.box_score << " bbbox (x y w h): "
                                << ptr.box_coord.x << ", " << ptr.box_coord.y << ", " << ptr.box_coord.w << ", " << ptr.box_coord.h << endl;
                    }
                    std::cout << std::endl;

		    
                    for (auto& ptr: kp_output) {
                        for (int ki = 0; ki<DETECTION_HKD_NUM_KP; ki++) {
                            float v = ptr.visibility[ki];
                            // The visibility score is normalized using sigmoid function.
                            v = 1.0 / (1 + exp(-v));

                            std::cout << ki << ": " << ptr.kp[ki].x << " " << ptr.kp[ki].y <<" " << ptr.visibility[ki] << std::endl;
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }

                std::cout << std::endl;
                ++inputListNum;
            } // while getline
        }     // outer loop
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }
    return 0;
}

