#ifndef ACCELERATOR_WRAPPER
#define ACCELERATOR_WRAPPER
#include <vector>
#include <iostream>
#include <string>
#include <memory>

#include "aocl_utils_cpp.hpp"

#include "floatFixedPointConversion.hpp"
#include "tensorCompression.hpp"
#include "vectorType.hpp"
#include "layerInstructionGenerator.hpp"

#include "params.hpp"

namespace GraphRuntime {
    typedef struct {
        //Which region in the accelerator's global memory does the blob start with?
        unsigned int memoryRegionID;
        unsigned int channelPerGroup;
        unsigned int group;
        unsigned int height;
        unsigned int width;
        signed int numFracBits;
        bool flagCanBeSparse;
        std::string blobName;
    } t_blob_info;

    typedef struct {
        //Name of the layer
        std::string layerName;
        //Start index of the layer's IA mover instructions in the IA mover instruction vector
        int     offsetIAMoverInstruction;
        //Number of IA mover instructions
        int     numIAMoverInstruction;
        //Start index of the layer's OA mover instructions in the OA mover instruction vector
        int     offsetOAMoverInstruction;
        //Number of OA mover instructions
        int     numOAMoverInstructions;
//        //Type of the layer
//        int     layerType;
//        //Input width
//        int     inputSPWidth;
//        //Input Height
//        int     inputSPHeight;
//        int     inputChannels;
//        int     outputChannels;
//        int     kernelSize;
//        int     kernelStride;
//        int     stridePadUnitSize;
//        int     numGroups;
        int     outputTileHeight;
        int     outputTileWidthPerCol;
        int     numActiveColsPartialOutputTile;
        int     expectedLatency;
        int     isComputeBound;
        unsigned int ops;
    } t_layer_info;

    typedef struct {
        //PE dimenions
        int numPERows;
        int numPECols;
        int numClusterInCompressionBlock;
        int numClusterInTransferBlock;
        int numScalarInCluster;
    } t_accelerator_info;

    typedef struct {
        //Input blob information
        std::vector<t_blob_info> vecInputInfo;

        //Output blob information
        std::vector<t_blob_info> vecOutputInfo;

        //Pointer to each layer's aligned/compressed and quantized weights
        std::vector<std::shared_ptr<AlignedTensor>> pWeights;

        //Starting dram block index and the number of dram block of each weight tensor in the
        //FPGA off-chip memory
        //Tensor is in the same order as the pointers appear in the pWeights vector
        std::vector<int> vecWeightDramBlockStart;
        std::vector<int> vecBiasStart;
#if defined(SPARSE_SYSTEM)
        std::vector<int> vecWeightTBCountStart;
#endif

        //Pointers to each layer's quantized biases
        std::vector<std::shared_ptr<t_aligned_short_vector>> pBiasVector;

        //Instructions of layers
        t_aligned_ia_mover_instruction_vector vecIAMoverInstruction;
        t_aligned_ia_tile_controller_instruction_vector vecIATileControllerInstruction;
        t_aligned_oa_mover_instruction_vector vecOAMoverInstruction;
        t_aligned_oa_tile_controller_instruction_vector vecOATileControllerInstruction;
        t_aligned_weight_mover_instruction_vector vecWMoverInstruction;
        t_aligned_misc_instruction_vector vecMiscInstruction;

        //Helper information that faciliate the execution of the graph
        std::vector<t_layer_info> vecLayerInfo;
    } t_execution_graph;

    class AcceleratorWrapper {
        private:
            std::string binaryFile;
            cl::Program program;
            cl::Platform clPlatform;
            cl::Context clContext;
            cl::Device clDevice;

            //Command queues
            cl::CommandQueue clCQIAMover;
            cl::CommandQueue clCQOAMover;
            cl::CommandQueue clCQWMover;
            cl::CommandQueue clCQIATileController;
            cl::CommandQueue clCQOATileController;
            cl::CommandQueue clCQMKController;
            #if defined(NOOP)
            cl::CommandQueue clCQNoop;
            #endif

            //The kernels
            cl::Kernel kernelIAMover;
            cl::Kernel kernelOAMover;
            cl::Kernel kernelWMover;
            cl::Kernel kernelMKInstructionMover;
            cl::Kernel kernelIATileController;
            cl::Kernel KernelOATileController;
            #if defined(NOOP)
            cl::Kernel kernelNoop;
            #endif

            //Buffer members associated with the IA Mover kernel
            cl::Buffer bufferIAMoverInstructions;
            cl::Buffer bufferActivationDramBlocks;
        #if defined(SPARSE_SYSTEM)
            cl::Buffer bufferActivationTBCounts;
        #endif

            //Buffer members associated with the IA tile controller
            cl::Buffer bufferIATileControllerInstructions;

            //Buffer members associated with the OA Mover kernel
            cl::Buffer bufferOAMoverInstructions;

            //Buffer members associated with the OA tile controller
            cl::Buffer bufferOATileControllerInstructions;

            //Buffer members associated with the W Mover kernel
            cl::Buffer bufferWMoverInstructions;
            cl::Buffer bufferWMoverWDramBlocks;
            cl::Buffer bufferWMoverBias;
        #if defined(SPARSE_SYSTEM)
            cl::Buffer bufferWMoverWTBCounts;
        #endif

            //Buffer members associated with the MK instruction kernel
            cl::Buffer bufferMKInstructions;

            std::vector<std::shared_ptr<AlignedTensor>> vecInputBlobsInternal;
            std::vector<std::shared_ptr<AlignedTensor>> vecOutputBlobsInternal;

            std::vector<t_blob_info> vecInputBlobsInfo;
            std::vector<t_blob_info> vecOutputBlobsInfo;

            std::vector<t_layer_info> vecLayerInfo;
            unsigned int numIAInstructions;

            t_accelerator_info acceleratorInfo;

            std::vector<cl_double> vecInputTransferTime;
            std::vector<cl_double> vecOutputTransferTime;
            std::vector<cl_double> vecLayerExecutionTime;
            int numRunExecuted;

            double minInferenceDuration, maxInferenceDuration, averageInferenceDuration;

            bool launchIATileController, launchWMover, launchMKController;


        public:
            AcceleratorWrapper() = default;
            AcceleratorWrapper(std::string _bitstremFileName, std::string _platformName, t_accelerator_info _acceleratorInfo, int _fpgaID);
            AcceleratorWrapper(std::string _bitstreamFileName,
                               std::string _platformName,
                               t_execution_graph& _executionGraph,
                               t_accelerator_info& _acceleratorInfo,
                               int _fpgaID);
            ~AcceleratorWrapper() = default;

            /*!
             * \brief loadGraph
             * \details Load the execution graph and transfer information to the accelerator
             *          Stores copies of the input/output blob information, and the IA/OA
             *          mover instructions start/length in the accelerator memory
             * \param _executionGraph The DNN graph to be executed
             */
            void loadGraph (t_execution_graph& _executionGraph);

            void resetGraph();

            /*!
             * \brief pprepareInputBlob
             * \details Quantize and compress an input blob of the neural network
             * \param floatBlob Single batch input blob in fp32, in HWC layout
             * \param intputBlobID The input blob id. Range: [0, num input blobs)
             */
            void prepareInputBlob (std::vector<float>& floatBlob, int inputBlobID);

            /*!
             * \brief extractOutputBlob
             * \details Decompress and dequantize an outout blob of the neural network
             * \param outputBlobID The outout blob id. Range: [0, num output blobs)
             * \return The output tensor in HWC layout as fp32
             */
            std::vector<float> extractOutputBlob (int outputBlobID);


            std::vector<t_blob_info> getInputBlobsInfo();
            std::vector<t_blob_info> getOutputBlobsInfo();

            /*!
             * \brief inference
             * \details Perform inference using the content of the current inference buffers
             */
            void inference(bool flagEnableProfile=false);

            /*!
             * \brief getInvocationOverhead
             * \return The invocation overhead in us
             */
            float getInvocationOverhead();

            std::string reportRuntime();

            /*!
             * \brief dumpRuntimeToCSV
             */
            void dumpRuntimeToCSV(std::string csvFilePath);
    };
}
#endif
