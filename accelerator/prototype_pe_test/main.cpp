#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/cl.hpp"
#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"
#include "params.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string> //for std::to_string
#include <unistd.h> //usleep
#include <random>
#include <bitset> //Print friendly binary number

#include "device_structures.hpp"
#include "boost/align/aligned_allocator.hpp"
#include "floatFixedPointConversion.hpp"
#include "gtest/gtest.h"
#include "prototypePE_structs.hpp"
#include "tensorCompression.hpp"

#define VECTOR_LENGTH 1111
#define VECTOR_A_SEED 10
#define VECTOR_B_SEED 5
#define BERN_SEED 7
#define BERN_P 0.01
#define EPSILON 1e-5
#define VECTOR_MIN -0.2
#define VECTOR_MAX 0.2
#define FRAC_WIDTH 8
#define INT_WIDTH 3
#define MAX_INSTRUCTION_SIZE 64
#define MAX_DATA_LENGTH 32784
//#define MAX_DATA_LENGTH 1024

#define MAX_IDX 1
#define MAX_IDY 1
//#define PLAY
typedef
std::vector<cl_ushort, boost::alignment::aligned_allocator<cl_ushort, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
aligned_ushort_vector;

typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
aligned_short_vector;

typedef
std::vector<cl_int, boost::alignment::aligned_allocator<cl_int, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
aligned_int_vector;

typedef
std::vector<cl_char, boost::alignment::aligned_allocator<cl_char, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
aligned_char_vector;

typedef
std::vector<t_pe_prototype_instruction_host,
boost::alignment::aligned_allocator<t_pe_prototype_instruction_host, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_instruction_vector;

typedef
std::vector<t_output_instruction_host,
boost::alignment::aligned_allocator<t_output_instruction_host, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_output_instruction_vector;

typedef short t_bias;

class peTestFixture : public ::testing::Test
{
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    cl::CommandQueue clCQTestInterface;

    cl::CommandQueue clCQPE;

    cl::Kernel kernelTestInterface;

    cl::Kernel kernelPE;

    cl::Buffer bufferActivationInput;
    cl::Buffer bufferActivationOutput;
    cl::Buffer bufferWeightInput;
    cl::Buffer bufferWeightOutput;
    cl::Buffer bufferDrainInput;
    cl::Buffer bufferDrainOutput;
    cl::Buffer bufferOutputInstruction;

#ifdef DIRECT_COMPRESSION_SIMD
    t_aligned_compression_simd_vector inputActivationVector;
    t_aligned_compression_simd_vector outputActivationVector;
    t_aligned_compression_simd_vector inputWeightVector;
    t_aligned_compression_simd_vector outputWeightVector;
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    t_aligned_transfer_block_vector inputActivationVector;
    t_aligned_transfer_block_vector outputActivationVector;
    t_aligned_transfer_block_vector inputWeightVector;
    t_aligned_transfer_block_vector outputWeightVector;
#endif
    aligned_short_vector inputDrainVector;
    aligned_char_vector outputDrainVector;

    //Profile function
    cl_int (*get_profile_fn)(cl_device_id, cl_program, cl_bool,cl_bool,cl_bool,size_t, void *,size_t *,cl_int *);

    //Reset function
    //cl_int (*reset_fn)(cl_context, cl_uint, const cl_device_id*);

    void SetUp() override {
#ifdef ARRIA10
        binaryFile = "prototypePE_aoc_emulation.aocx";
#else
        //std::cout <<"Please type in the FPGA image (e.g. foo.aocx): "<<std::endl;
        //std::cin >> binaryFile;
        binaryFile = "prototypePE_aoc_release_hw.aocx";
#endif
        //Setup and platform and the context
        cl_int status = CL_SUCCESS;
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
        std::vector<cl::Device> devices;
        status = clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        aocl_utils_cpp::checkError(status, "Failed to query the devices");
        clDevice = devices[0];
        clContext = cl::Context({devices[0]}
                                ,NULL
                                ,&aocl_utils_cpp::oclContextCallback
                                ,NULL
                                ,&status);
        aocl_utils_cpp::checkError(status, "Failed to create context");

        //Parse the binary and invoke the kernel
        program = aocl_utils_cpp::createProgramFromBinary(
                    clContext,
                    binaryFile.c_str(),
                    {clDevice}
                    );
        status = program.build({clDevice});
        aocl_utils_cpp::checkError(status, "Failed to build program");

        kernelTestInterface = cl::Kernel(program, "kernelTestInterface", &status);
        aocl_utils_cpp::checkError(status, "Failed to create the test interface kernel!");

        //kernelPE = cl::Kernel(program, "kernelPE", &status);
        //aocl_utils_cpp::checkError(status, "Failed to create the PE kernel!");

        clCQTestInterface = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the test interface command queue!");


        bufferActivationInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(inputActivationVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input activation buffer!");

        bufferActivationOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(outputActivationVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output activation buffer!");

        bufferWeightInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(inputWeightVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input weight buffer!");

        bufferWeightOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(outputWeightVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output weight buffer!");

        bufferDrainInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(inputDrainVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input drain buffer!");

        bufferDrainOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(outputDrainVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output drain buffer!");

        bufferOutputInstruction = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        1 * sizeof(t_output_instruction_host),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output instruction buffer!");

        get_profile_fn = (cl_int (*) (cl_device_id, cl_program, cl_bool,cl_bool,cl_bool,size_t, void *,size_t *,cl_int *))clGetExtensionFunctionAddress("clGetProfileDataDeviceIntelFPGA");

        //reset_fn = (cl_int (*) (cl_context, cl_uint, const cl_device_id*))clGetExtensionFunctionAddress("clResetKernelsIntelFPGA");
        std::cout <<"AOCL setup compelete"<<std::endl;

        //Need to setup numInstructions, idx, and idy separately
    }

    void launch (int idx, int idy, int maxIdx, int maxIdy, t_bias bias, t_output_instruction_host outputInstruction, bool drainResult=true) {
        cl_int status;

        //Fill the buffers
        //Transfer the input activation
        if (inputActivationVector.size() > 0) {
            status = clCQTestInterface.enqueueWriteBuffer(bufferActivationInput,
                                                 CL_TRUE,
                                                 0,
                                                 sizeof(typeof(inputActivationVector.at(0))) * inputActivationVector.size(),
                                                 inputActivationVector.data(),
                                                 NULL);
            aocl_utils_cpp::checkError(status, "Failed to write bufferActivationInput");
        }

        //Transfer the input weight
        if (inputWeightVector.size() > 0) {
            status = clCQTestInterface.enqueueWriteBuffer(bufferWeightInput,
                                                 CL_TRUE,
                                                 0,
                                                 sizeof(typeof(inputWeightVector.at(0))) * inputWeightVector.size(),
                                                 inputWeightVector.data(),
                                                 NULL);
            aocl_utils_cpp::checkError(status, "Failed to write bufferWeightInput");
        }

        //Transfer the drain intput
        cl_ushort numInputDrain = drainResult ? PE_ROWS  - idy - 1 : 0;
        for (int i=0; i<numInputDrain; i++) {
            short val = i;
            inputDrainVector.push_back(val);
        }
        if (inputDrainVector.size() > 0){
            status = clCQTestInterface.enqueueWriteBuffer(bufferDrainInput,
                                                 CL_TRUE,
                                                 0,
                                                 sizeof(typeof(inputDrainVector.at(0))) * inputDrainVector.size(),
                                                 inputDrainVector.data(),
                                                 NULL);
            aocl_utils_cpp::checkError(status, "Failed to write bufferDrainInput");
            EXPECT_TRUE (status == CL_SUCCESS);
        }

        status = clCQTestInterface.enqueueWriteBuffer(bufferOutputInstruction,
                                             CL_TRUE,
                                             0,
                                             sizeof(t_output_instruction_host),
                                             &outputInstruction,
                                             NULL);
        aocl_utils_cpp::checkError(status, "Failed to write bufferOutputInstruction");
        EXPECT_TRUE (status == CL_SUCCESS);


        //Setup the buffer arguments and number of transfer for the test interface
        kernelTestInterface.setArg(0, bufferActivationInput);
        kernelTestInterface.setArg(1, bufferActivationOutput);
        kernelTestInterface.setArg(2, bufferWeightInput);
        kernelTestInterface.setArg(3, bufferWeightOutput);
        kernelTestInterface.setArg(4, bufferDrainInput);
        kernelTestInterface.setArg(5, bufferDrainOutput);
        kernelTestInterface.setArg(6, (cl_short) bias);
        //kernelTestInterface.setArg(7, outputInstruction);
        kernelTestInterface.setArg(7, bufferOutputInstruction);

        kernelTestInterface.setArg(8, (cl_ushort) inputActivationVector.size()); //numInputActivationBlocks
       // kernelTestInterface.setArg(12, (cl_ushort) startIndexActivationBlocks); //startIndexActivationBlocks,
        cl_ushort numOutputActivationBlocks =
                idy < (maxIdy) ? inputActivationVector.size() : 0;
        kernelTestInterface.setArg(9, numOutputActivationBlocks); //numOutputActivationBlocks
        //Allocate space for the output activation vector
        outputActivationVector.resize(numOutputActivationBlocks);

        kernelTestInterface.setArg(10, (cl_ushort) inputWeightVector.size()); //numInputWeightBlocks
        //kernelTestInterface.setArg(15, (cl_ushort) startIndexWeightBlocks); //startIndexWeightBlocks,
        cl_ushort numOutputWeightBlocks =
                idx < (maxIdx) ? inputWeightVector.size() : 0;
        kernelTestInterface.setArg(11, numOutputWeightBlocks); //numOutputWeightBlocks
        //Allocate space for the output activation vector
        outputWeightVector.resize(numOutputWeightBlocks);


        kernelTestInterface.setArg(12, (cl_ushort) numInputDrain); //numInputDrain
        cl_ushort numOutputDrain = drainResult ? PE_ROWS  - idy : 0;
        kernelTestInterface.setArg(13, numOutputDrain); //numOutputDrain
        for (int i=0; i<numOutputDrain; i++) {
            short val;
            outputDrainVector.push_back(val);
        }

        kernelTestInterface.setArg(14, (cl_uchar)maxIdx);
        kernelTestInterface.setArg(15, (cl_uchar)maxIdy);

        cl::Event event;
        //Launch kernels

        std::cout <<"Launching"<<std::endl;
        clCQTestInterface.enqueueTask(kernelTestInterface, NULL, &event);
        //Retrieve data
        clCQTestInterface.finish();

        status = (cl_int)(*get_profile_fn)(this->clDevice(), this->program(), true, true, false, 0, NULL, NULL, NULL);

        if (outputActivationVector.size() > 0) {
            clCQTestInterface.enqueueReadBuffer(
                        bufferActivationOutput,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputActivationVector.at(0))) * outputActivationVector.size(),
                        outputActivationVector.data()
                        );
         }

        if (outputWeightVector.size() > 0) {
            clCQTestInterface.enqueueReadBuffer(
                        bufferWeightOutput,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputWeightVector.at(0))) * outputWeightVector.size(),
                        outputWeightVector.data()
                        );
        }

        if (outputDrainVector.size() > 0) {
            clCQTestInterface.enqueueReadBuffer(
                        bufferDrainOutput,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputDrainVector.at(0))) * outputDrainVector.size(),
                        outputDrainVector.data()
                        );
        }

        cl_ulong kernelStartTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong kernelEndTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double kernelRunTime = (cl_double)((kernelEndTime - kernelStartTime) * (cl_double)(1e-3));

        std::cout <<"Test kernel time (us): "<<kernelRunTime<<std::endl;
        std::cout <<"Number of weight transfer blocks: "<<inputWeightVector.size()<<std::endl;
        std::cout <<"Number of activation transfer blocks: "<<inputActivationVector.size()<<std::endl;
    }
};

/*!
 * \brief initialize_vector
 * \param seed
 * \param numElements
 * \param bernProb
 * \param min
 * \param max
 * \return std::vector<float>
 */
std::vector<float> initialize_vector (
        unsigned seed,
        unsigned int numElements,
        float bernProb,
        float min,
        float max
        );



float dot_product_regular_vectors (
        std::vector <float> & inputVectorA,
        std::vector <float> & inputVectorB
        );


//TEST_F (peTestFixture, testFixture) 0, //tilingSizeWidth
//{
//    launch(IDX,IDY,0,0, false);
//    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);
//    EXPECT_TRUE(true);
//}

#ifdef PLAY
TEST_F (peTestFixture, testPlayfield) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 1.0;

    unsigned int numElements = 16;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 0.0;

    //Then convert the bias into a fixed point number;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));

    // Generate a block of activations
//    std::vector<float> activationRealInput = initialize_vector(
//                VECTOR_A_SEED,
//                numElements,
//                probOne,
//                0.90,
//                0.95
//                );
    //std::vector<float> activationRealInput = {-3.14f};
    std::vector<float> activationRealInput (numElements, 0.00);

    std::vector<float> weightRealInput (numElements, 0.00);


    for (int i=0; i<4; i++)
    {
        //if (i == 0) {
            for (int j=4*i; j < 4*i+4; j++)
            {
                activationRealInput.at(j) = 0.25*(i+1);
                weightRealInput.at(j) = 0.25*(i+1);
            }
        //}
    }


    // Generate a block of activations
//    std::vector<float> weightRealInput = initialize_vector(
//                VECTOR_A_SEED,
//                numElements,
//                probOne,
//                0.90,
//                0.95
//                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                 0, //tilingSizeWidth
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height,
                0, //tilingSizeWidth
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

     t_output_instruction_host outputInstruction =
     {
        .numPSumToProcess = maxIdy - targetIDY,
        .numBitsToRightShift = numBitsToRightShift,
        .enableRelu = false
     };


    launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;

}

#else
TEST_F (peTestFixture, testLoadBiasDotProductAndDrainageZero) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 0.0;

    unsigned int numElements = 8196*4+3;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 1.0;

    //Then convert the bias into a fixed point number;
    //fixedPointNumber biasFPInput (biasFloat, fracW, intWidthWeight);
    //int biasFPInt = ((int) biasFPInput.getBits()) << numBitsToRightShift;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));


    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> activationRealInput = {-3.14f};

    // Generate a block of activations
    std::vector<float> weightRealInput = initialize_vector(
                VECTOR_B_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    //Prepare the input buffers
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef DIRECT_COMPRESSION_SIMD
    std::cout<<"Comrpessing the weights"<<std::endl;
    directCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    directCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

    t_output_instruction_host outputInstruction =
    {
       .numPSumToProcess = maxIdy - targetIDY,
       .numBitsToRightShift = numBitsToRightShift,
       .enableRelu = false
    };


   launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainageHalfLong) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 0.5;

    unsigned int numElements = 8196*4+3;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 1.0;

    //Then convert the bias into a fixed point number;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));


    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> activationRealInput = {-3.14f};

    // Generate a block of activations
    std::vector<float> weightRealInput = initialize_vector(
                VECTOR_B_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    //Prepare the input buffers
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef DIRECT_COMPRESSION_SIMD
    std::cout<<"Comrpessing the weights"<<std::endl;
    directCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    directCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

    t_output_instruction_host outputInstruction =
    {
       .numPSumToProcess = maxIdy - targetIDY,
       .numBitsToRightShift = numBitsToRightShift,
       .enableRelu = false
    };


   launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainage025Long) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 0.25;

    unsigned int numElements = 8196*4+3;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 1.0;

    //Then convert the bias into a fixed point number;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));

    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> activationRealInput = {-3.14f};

    // Generate a block of activations
    std::vector<float> weightRealInput = initialize_vector(
                VECTOR_B_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    //Prepare the input buffers
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef DIRECT_COMPRESSION_SIMD
    std::cout<<"Comrpessing the weights"<<std::endl;
    directCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    directCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

    t_output_instruction_host outputInstruction =
    {
       .numPSumToProcess = maxIdy - targetIDY,
       .numBitsToRightShift = numBitsToRightShift,
       .enableRelu = false
    };


   launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainageOneLong) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 1.0;

    unsigned int numElements = 8196*4+3;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 1.0;

    //Then convert the bias into a fixed point number;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));


    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> activationRealInput = {-3.14f};

    // Generate a block of activations
    std::vector<float> weightRealInput = initialize_vector(
                VECTOR_B_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    //Prepare the input buffers
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef DIRECT_COMPRESSION_SIMD
    std::cout<<"Comrpessing the weights"<<std::endl;
    directCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    directCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

    t_output_instruction_host outputInstruction =
    {
       .numPSumToProcess = maxIdy - targetIDY,
       .numBitsToRightShift = numBitsToRightShift,
       .enableRelu = false
    };


   launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainageOneShort) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 1.0;

    unsigned int numElements = 32;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 0.0;

    //Then convert the bias into a fixed point number;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));


    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                0.49,
                0.51
                );
    //std::vector<float> activationRealInput = {-3.14f};

    // Generate a block of activations
    std::vector<float> weightRealInput = initialize_vector(
                VECTOR_B_SEED,
                numElements,
                probOne,
                0.49,
                0.51
                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    //Prepare the input buffers
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef DIRECT_COMPRESSION_SIMD
    std::cout<<"Comrpessing the weights"<<std::endl;
    directCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    directCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel    inputBiasVector.push_back((short) biasFPInput.getBits());
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

    t_output_instruction_host outputInstruction =
    {
       .numPSumToProcess = maxIdy - targetIDY,
       .numBitsToRightShift = numBitsToRightShift,
       .enableRelu = false
    };


   launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainage025Short) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    int maxIdx = PE_COLS, maxIdy = PE_ROWS;
    float probOne = 0.25;

    unsigned int numElements = 32;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 0.0;

    //Then convert the bias into a fixed point number;
    t_bias biasFP = ((t_bias) std::round(biasFloat * (1 << (fracIn + fracW))));

    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> activationRealInput = {-3.14f};

    // Generate a block of activations
    std::vector<float> weightRealInput = initialize_vector(
                VECTOR_B_SEED,
                numElements,
                probOne,
                -3.14,
                3.14
                );
    //std::vector<float> weightRealInput = {3.14f};

    //Compute the expected output;
    float expectedResultReal = dot_product_regular_vectors(activationRealInput, weightRealInput) + biasFloat;
    fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

    //Prepare the input buffers
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;

    fpActivationVector.resize(activationRealInput.size());
    fpWeightVector.resize(weightRealInput.size());

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpWeight(weightRealInput.at(i), fracW, intWidthWeight);
        fpWeightVector.at(i) = fpWeight;
        fixedPointNumber fpActivation(activationRealInput.at(i), fracIn, intWidthIn);
        fpActivationVector.at(i) = fpActivation;
    }

#ifdef DIRECT_COMPRESSION_SIMD
    std::cout<<"Comrpessing the weights"<<std::endl;
    directCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    directCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                7, //maxSimdBlockIndexInStreamBlock
                3, //maxScalarIndexInSimdBlock
                true //isKernel
                );
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
    std::cout<<"Comrpessing the weights"<<std::endl;
    flexibleDirectCompressedTensor compWTensor(
                fpWeightVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );

    std::cout<<"Comrpessing the activations"<<std::endl;
    flexibleDirectCompressedTensor compATensor(
                fpActivationVector,
                1, //numTensors
                numElements, //channel
                1, //width
                1, //height
                0,
                numElements - 1, //_maxScalarIndexInChannelGroup
                7, //_maxClusterIndexInCompressionBlock
                1, //_maxClusterIndexInTransferBlock
                CLUSTER_SIZE-1, //_maxScalarIndexInClusterBlock
                true //isKernel
                );
#endif

    std::cout <<"Transfer the compressed activations to the test harness"<<std::endl;
    inputActivationVector.resize(compATensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compATensor.streamBlockAddressVector.at(0); i++) {
        inputActivationVector.at(i) = compATensor.valueVector.at(i);
    }

    std::cout <<"Transfer the compressed weights to the test harness"<<std::endl;
    inputWeightVector.resize(compWTensor.streamBlockAddressVector.at(0));
    for (unsigned int i=0; i<compWTensor.streamBlockAddressVector.at(0); i++) {
        inputWeightVector.at(i) = compWTensor.valueVector.at(i);
    }

    //Prepare the instruction

    t_output_instruction_host outputInstruction =
    {
       .numPSumToProcess = maxIdy - targetIDY,
       .numBitsToRightShift = numBitsToRightShift,
       .enableRelu = false
    };


   launch(targetIDX, targetIDY, maxIdx, maxIdy, biasFP, outputInstruction);

    //Compare the result
    char actualOutputFP = outputDrainVector.at(0);

    float actualOutputReal = fixedPointNumber(actualOutputFP, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

    std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
    std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((actualOutputFP & WEIGHT_MASK))<<std::endl;

    EXPECT_TRUE(
         std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
         << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
         <<"Golden output: "<<expectedResultReal<<std::endl;
}
#endif

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

std::vector<float> initialize_vector(unsigned seed,
                       unsigned int numElements,
                       float bernProb,
                       float min,
                       float max) {
    std::mt19937 generator(seed);
    std::bernoulli_distribution bernDistribution(bernProb);
    std::uniform_real_distribution<float> uniDistribution(min, max);

    std::vector<float> vector{};
    for (unsigned i=0; i<numElements; i++) {
        float val = bernDistribution(generator) ?
                    uniDistribution(generator) : 0.0f;
        vector.push_back(val);
    }

    return vector;
}

float dot_product_regular_vectors (std::vector<float> &inputVectorA
                                   ,std::vector<float> &inputVectorB) {
    float result = 0.0f;
    for (unsigned i=0; i<inputVectorA.size(); i++) {
        result += inputVectorA.at(i) * inputVectorB.at(i);
    }
    return result;
}
