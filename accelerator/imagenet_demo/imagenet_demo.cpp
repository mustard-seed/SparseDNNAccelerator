#include "graph_factory.hpp"
#include "yaml-cpp/yaml.h"
#include "timer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#define EMULATE
std::string keys =
    "{ help  h     |                        | Print help message. }"
    "{ val   v     |                        | Optional. Set the flag for running inference on the entire ImageNet validation set.}"
    "{ truth t     | demo_ground_truth.yaml | Path to the ground truth yaml file. Required if flag val is set.}"
    "{ folder f    | ILSVRC2012_img_val     | Folder of the validation images. Required if flag val is set.}"
    "{ legend l     | caffe_words.yaml      | Path to the map from label ID to words.}"
    "{ image i     |                        | Required if flag val is NOT set. Path to the single image for testing.}"
    "{ preproc p   | preprocess.yaml        | Required. Path to the preprocessing configuration YAML file. }"
    "{ model m     | <none>                 | Required. Path to the model trace YAML file.}"
    "{ param w     | <none>                 | Required. Path the the model parameter NPZ file.}";

using namespace cv;
using namespace dnn;
using namespace GraphRuntime;
using namespace YAML;

typedef struct {
    float scale;
    Scalar bgrMean;
    Scalar vars;
    Size processSize;
    Size loadSize;
} t_preprocess;

typedef struct {
    int id;
    float prob;
} t_prediction;

t_preprocess parsePreProcess(YAML::Node &_node);

std::vector<t_prediction> inference(AcceleratorWrapper &_accelerator, t_preprocess _preprocess, std::string _imgPath, int _k);
int inferenceOnValidationSet(AcceleratorWrapper &_accelerator, std::string _groundTruthFile, std::string _valFolderPath, t_preprocess _preprocess, YAML::Node &_legend);
int inferenceOnSingleImage(AcceleratorWrapper &_accelerator, std::string _imagePath, t_preprocess _preprocess, YAML::Node &_legend);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
#ifdef C5SOC
    std::string aocxBinaryFile = "device_utils.aocx";
#else
#if defined(EMULATE)
    std::string aocxBinaryFile = "c5_mac8bitx4_c_model.aocx";
#else
    std::string aocxBinaryFile = "sparse_pe_system.aocx";
#endif
#endif

#if defined(EMULATE)
    std::string platformName = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
#else
    std::string platformName = "Intel(R) FPGA SDK for OpenCL(TM)";
#endif

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks on ImageNet using the FPGA accelerator");
    if (argc == 1 || parser.has("help") || parser.has("model") == false || parser.has("param") == false || parser.has("legend") == false)
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("val") && !parser.has("image")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    //Load the model topology and the parameters. Generate the accelerator
    std::string modelFile = parser.get<String>("model");
    std::string modelParaFile = parser.get<String>("param");
    bool scatterInput = true;
    GraphRuntime::t_accelerator_info acceleratorInfo =
        {   .numPERows=PE_ROWS,
            .numPECols=PE_COLS,
            .numScalarInCluster=CLUSTER_SIZE
        };
    AcceleratorWrapper accelerator(aocxBinaryFile, platformName, acceleratorInfo, 0);
    GraphFactory graph(modelFile, modelParaFile, scatterInput);
    auto pGraph = std::move(graph.generateGraph());
    accelerator.resetGraph();
    accelerator.loadGraph(*(pGraph.get()));

    //Parse the preprocessing fields
    std::string preprocessFile = parser.get<String>("preproc");
    YAML::Node preprocessNode = LoadFile(preprocessFile);
    t_preprocess preprocess = parsePreProcess(preprocessNode);
    std::string legendFile = parser.get<String>("legend");
    YAML::Node legendNode = LoadFile(legendFile);

    //Run on validation set
    if (parser.has("val")) {
        std::string groundTruthFile = parser.get<String>("truth");
        std::string valFolder = parser.get<String>("folder");
        return inferenceOnValidationSet(accelerator, groundTruthFile, valFolder, preprocess, legendNode);
    }
    else {
        std::string imagePath = parser.get<String>("image");
        return inferenceOnSingleImage(accelerator, imagePath, preprocess, legendNode);
    }
    return 0;
}

t_preprocess parsePreProcess(YAML::Node &_node)
{
    t_preprocess proc;
    proc.scale = _node["scale"].as<float>();
    YAML::Node means = _node["means"];
    YAML::Node vars = _node["vars"];
    for (int i=0; i<3; i++) {
        proc.bgrMean[i] = means[i].as<float>();
        proc.vars[i] = vars[i].as<float>();
    }
    proc.loadSize.height = _node["loadHeight"].as<int>();
    proc.loadSize.width = _node["loadWidth"].as<int>();
    proc.processSize.height = _node["procHeight"].as<int>();
    proc.processSize.width = _node["procWidth"].as<int>();

    return proc;
}

std::vector<t_prediction> inference(AcceleratorWrapper &_accelerator, t_preprocess _preprocess, std::string _imgPath, int _k=5)
{
    cv::Mat _image = cv::imread(_imgPath);
    Mat blob;
    if (_preprocess.loadSize.height != 0 && _preprocess.loadSize.width !=0) {
        resize(_image, _image, _preprocess.loadSize);
    }
    //Preprocess the image. Set channel swap and crop to true
    blobFromImage(_image, blob, _preprocess.scale, _preprocess.processSize, _preprocess.bgrMean, true, true);
    //cv::dnn::blobFromImage(_image, blob, 1.0f);

//    imshow("abc", blob);
//    int g = waitKey(0); // Wait for a keystroke in the window
    if (_preprocess.vars[0] != 0.0 && _preprocess.vars[1] != 0.0 && _preprocess.vars[2] != 0.0) {
        cv::divide(blob, _preprocess.vars, blob);
    }
    if (!blob.isContinuous()) {
        std::runtime_error except("OpenCV Blob is not continuous.");
        throw except;
    }
    //Transfer the image to inference
    auto size = blob.size;
    //std::cout<<size[0]<<" "<<size[1]<<" "<<size[2]<<" "<<size[3]<<std::endl;
    int channels = size[1];
    int rows = size[2];
    int cols = size[3];
    int numEle = channels * rows* cols;
    std::vector<float> inputReordered(numEle, 0.0f);
    float *data = blob.ptr<float>();
    for (int k=0; k<channels; k++) {
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                int dstIdx = k + (j + i * cols) * blob.channels();
                int srcIdx = j + (i + k * cols) * rows;

                inputReordered.at(dstIdx) = data[srcIdx];
            }
        }
    }
    _accelerator.prepareInputBlob(inputReordered, 0);
    _accelerator.inference(true);

    //Post processing: compute softmax;
    std::vector<float> rawResult = _accelerator.extractOutputBlob(0);
    float tempSum = 0;
    float maxRaw = std::max_element(rawResult.begin(), rawResult.end())[0];
    std::vector<float> softmaxResult(rawResult.size(), 0.0f);
    for (int i=0; i<rawResult.size(); i++) {
        std::cout<<"rawResult["<<i<<"]: "<<rawResult.at(i)<<std::endl;
        float z = std::exp(rawResult.at(i) - maxRaw);
        softmaxResult.at(i) = z;
        tempSum += z;
    }
    tempSum = 1.0 / tempSum;
    for (int i=0; i<softmaxResult.size(); i++) {
        softmaxResult.at(i) *= tempSum;
    }
    //Post processing: sort the result from the most likely to the least likely based on softmax-output
    std::vector<int> idx(rawResult.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(
                idx.begin(),
                idx.end(),
                [&softmaxResult](size_t i1, size_t i2) {return softmaxResult[i1] > softmaxResult[i2];}
                );
    //Post processing: form the output
    std::vector<t_prediction> topK;
    int k = std::min(_k, (int) idx.size());
    topK.resize(k);
    for (int i=0; i<k; i++) {
        t_prediction pred;
        pred.id = idx.at(i);
        pred.prob = softmaxResult.at(pred.id);
        //std::cout <<"top ["<<i<<"]: "<<pred.prob<<std::endl;
        topK.at(i) = pred;
    }

    return topK;
}

int inferenceOnValidationSet(AcceleratorWrapper &_accelerator, std::string _groundTruthFile, std::string _valFolderPath, t_preprocess _preprocess, YAML::Node &_legend)
{
    YAML::Node groundTruth = YAML::LoadFile(_groundTruthFile);
    int num = groundTruth.size();
    int count = 0;

    typedef struct {
        std::string path;
        std::string correctLabel;
        std::string incorrectLabel;
    } t_negative;

    std::vector<t_negative> vecNeg;
    int posTop1Count = 0;
    int posTop5Count = 0;
    for (auto it=groundTruth.begin(); it!=groundTruth.end(); it++) {
        std::string fileName = it->first.as<std::string>();
        int trueLabel = it->second.as<int>();
        std::string filePath = _valFolderPath + "/" + fileName;
        //std::cout <<"Loading file "<<filePath<<std::endl;

        std::vector<t_prediction> vecPred;
        try {

            vecPred = inference(_accelerator, _preprocess, filePath);
        }
        catch (std::runtime_error) {
            std::cout <<"Encountered runtime error"<<std::endl;
            return -1;
        }

        //update stats
        int first = 0;
        for (const auto & pred : vecPred) {
            if (first == 0) {
                if (pred.id == trueLabel) {
                    posTop1Count++;
                }
                else
                {
                    t_negative neg;
                    neg.path = filePath;
                    neg.correctLabel = _legend[trueLabel].as<std::string>();
                    neg.incorrectLabel = _legend[pred.id].as<std::string>();
                    vecNeg.push_back(neg);
                }
            }
            if (pred.id == trueLabel) {
                posTop5Count++;
                break;
            }
            first++;
        }

        count++;
        //Print progress bar, see https://stackoverflow.com/posts/14539953/timeline
        int BARWIDTH = 50;
        if (count % 100 == 0) {
            std::cout <<"[";
            float progress = (float) count / (float) num;
            int pos = (int) ((float) BARWIDTH * progress);
            for (int i=0; i<BARWIDTH; i++) {
                if (i<pos) {
                    std::cout <<"=";
                }
                else if (i==pos) {
                    std::cout << ">";
                }
                else {
                    std::cout <<" ";
                }
            }
            std::cout <<"] "<<int (progress * 100) <<" %\r";
            std::cout.flush();
        }
    }

    float top1Accuracy = (float) posTop1Count / (float) count * 100.0;
    float top5Accuracy = (float) posTop5Count / (float) count * 100.0;
    int MAX_ERR = 5;
    int errSize = std::min((int) vecNeg.size(), MAX_ERR);
    std::cout<<std::endl;
    std::cout<<"Top1, Top5 accuracies: "<<top1Accuracy<<"%, "<<top5Accuracy<<"%\n"<<std::endl;
    std::cout<<"First "<<errSize<<" negatives: "<<std::endl;
    for (int i=0; i<errSize; i++) {
        t_negative neg = vecNeg.at(i);
        std::cout<<i<<": ";
        std::cout<<"Path: "<<neg.path<<", ";
        std::cout<<"Correct label: "<<neg.correctLabel<<", ";
        std::cout<<"Acutal label: "<<neg.incorrectLabel<<std::endl;
    }
    std::cout <<_accelerator.reportRuntime();
    return 0;
}

int inferenceOnSingleImage(AcceleratorWrapper &_accelerator, std::string _imagePath, t_preprocess _preprocess, YAML::Node &_legend)
{
    std::vector<t_prediction> vecPred;
    Timer t;
    t.start();
    try {
        vecPred = inference(_accelerator, _preprocess, _imagePath);
    }
    catch (std::runtime_error) {
        std::cout <<"Encountered runtime error"<<std::endl;
        return -1;
    }
    t.stop();
    auto frame = imread(_imagePath);
    std::string labelFull = format("%s: %.4f", _legend[vecPred.at(0).id].as<std::string>().c_str(), vecPred.at(0).prob);
    labelFull += "\n" + format("%s: %.4f", _legend[vecPred.at(1).id].as<std::string>().c_str(), vecPred.at(1).prob);
    labelFull += "\n" + format("%s: %.4f", _legend[vecPred.at(2).id].as<std::string>().c_str(), vecPred.at(2).prob);
    labelFull += "\n" + format("%s: %.4f", _legend[vecPred.at(3).id].as<std::string>().c_str(), vecPred.at(3).prob);
    labelFull += "\n" + format("%s: %.4f", _legend[vecPred.at(4).id].as<std::string>().c_str(), vecPred.at(4).prob);
    std::string labelSingle = format("%s: %.4f", _legend[vecPred.at(0).id].as<std::string>().c_str(), vecPred.at(0).prob);
    putText(frame, labelSingle, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    std::cout <<labelFull<<std::endl;
    std::cout <<_accelerator.reportRuntime();
    //std::cout <<"Total inference time (us): "<<t.get_time_s() * 1000000.0<<std::endl;
    imshow("Inference", frame);
    waitKey(0);
    return 0;
}
