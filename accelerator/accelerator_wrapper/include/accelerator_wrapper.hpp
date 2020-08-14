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
        unsigned int numFracBits;
        bool flagCanBeSparse;
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
        int     numOAInstructions;
    } t_layer_info;

    typedef struct {
        //Input blob information
        std::vector<t_blob_info> vecInputInfo;

        //Output blob information
        std::vector<t_blob_info> vecOutputInfo;

        //Aligned/compressed and quantized weights
        std::shared_ptr<AlignedTensor> pWeights;

        //Quantized biases
        t_aligned_short_vector biasVector;

        //Instructions
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

            //
        public:
            AcceleratorWrapper() = default;
            AcceleratorWrapper(std::string _fileName);
            ~AcceleratorWrapper() = default;

            /*!
             * \brief loadGraph
             * \details Load the execution graph and transfer information to the accelerator
             *          Also stores copies of the input/output blob information, and the IA/OA
             *          mover instructions start/length in the accelerator memory
             * \param _executionGraph The DNN graph to be executed
             */
            void loadGraph (t_execution_graph& _executionGraph);


    };
}
