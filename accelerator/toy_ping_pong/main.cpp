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
#define MAX_DATA_LENGTH 2048

#define MAX_IDX 1
#define MAX_IDY 1

typedef
std::vector<cl_ushort, boost::alignment::aligned_allocator<cl_ushort, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
aligned_ushort_vector;

typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
aligned_short_vector;

typedef
std::vector<t_vecSpValueAndZCount, boost::alignment::aligned_allocator<t_vecSpValueAndZCount, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_compression_vector;

typedef
std::vector<t_spValueAndZCount, boost::alignment::aligned_allocator<t_spValueAndZCount, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_compression_scalar_vector;

typedef
std::vector<t_simdblock_host, boost::alignment::aligned_allocator<t_simdblock_host, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_compression_simd_vector;

typedef
std::vector<t_vecUnpackedHost, boost::alignment::aligned_allocator<t_vecUnpackedHost, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_compression_host_vector;

typedef
std::vector<t_spValueAndZCountUnpackedHost, boost::alignment::aligned_allocator<t_spValueAndZCountUnpackedHost, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_compression_scalar_host_vector;

typedef
std::vector<t_pe_prototype_instruction_host,
boost::alignment::aligned_allocator<t_pe_prototype_instruction_host, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_instruction_vector;

class peTestFixture : public ::testing::Test {
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    //cl::CommandQueue clCQDrainTransport;
    //cl::CommandQueue clCQInstructionTransport;
    //cl::CommandQueue clCQBiasTransport;
    //cl::CommandQueue clCQActivationTransport;
    //cl::CommandQueue clCQWeightTransport;
    cl::CommandQueue clCQTestInterface;
    //cl::CommandQueue clCQPE;

    //cl::Kernel kernelDrainTransport;
    //cl::Kernel kernelInstructionTransport;
    //cl::Kernel kernelBiasTransport;
    //cl::Kernel kernelActivationTransport;
    //cl::Kernel kernelWeightTransport;
    cl::Kernel kernelTestInterface;
    //cl::Kernel kernelPE;

    cl::Buffer bufferInstructionInput;
    cl::Buffer bufferInstructionOutputH;
    cl::Buffer bufferInstructionOutputV;
    cl::Buffer bufferActivationInput;
    cl::Buffer bufferActivationOutput;
    cl::Buffer bufferWeightInput;
    cl::Buffer bufferWeightOutput;
    cl::Buffer bufferBiasInput;
    cl::Buffer bufferBiasOutput;
    cl::Buffer bufferDrainInput;
    cl::Buffer bufferDrainOutput;

    t_aligned_compression_simd_vector inputActivationVector;
    t_aligned_compression_simd_vector outputActivationVector;
    t_aligned_compression_simd_vector inputWeightVector;
    t_aligned_compression_simd_vector outputWeightVector;
    aligned_short_vector inputBiasVector;
    aligned_short_vector outputBiasVector;
    aligned_short_vector inputDrainVector;
    aligned_short_vector outputDrainVector;
    t_aligned_instruction_vector inputInstructionVector;
    t_aligned_instruction_vector outputInstructionHVector;
    t_aligned_instruction_vector outputInstructionVVector;


    void SetUp() override {
        binaryFile = "prototypePE_aoc_emulation.aocx";
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

        clCQTestInterface = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the test interface command queue!");

        bufferInstructionInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(inputInstructionVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the instruction input buffer!");

        bufferInstructionOutputH = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(outputInstructionHVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the instruction output horizontal buffer!");

        bufferInstructionOutputV = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(outputInstructionVVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the instruction output vertical buffer!");


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

        bufferBiasInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(inputBiasVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input bias buffer!");

        bufferBiasOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(typeof(outputBiasVector.at(0))),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output bias buffer!");

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
        std::cout <<"AOCL setup compelete"<<std::endl;

        //Need to setup numInstructions, idx, and idy separately
    }

    void launch (int idx, int idy, unsigned short startIndexActivationBlocks, unsigned short startIndexWeightBlocks, bool drainResult=true) {
        cl_int status;
        //Set the kernels' ids
        /*
        kernelActivationTransport.setArg(0, idx);
        kernelActivationTransport.setArg(1, idy);

        kernelWeightTransport.setArg(0, idx);
        kernelWeightTransport.setArg(1, idy);

        kernelBiasTransport.setArg(0, idx);
        kernelBiasTransport.setArg(1, idy);

        kernelDrainTransport.setArg(0, idx);
        kernelDrainTransport.setArg(1, idy);

        kernelInstructionTransport.setArg(0, idx);
        kernelInstructionTransport.setArg(1, idy);

        kernelPE.setArg(0, idx);
        kernelPE.setArg(1,idy);
        */

        //Fill the buffers
        //Transfer the instruction
        if (inputInstructionVector.size() > 0) {
            status = clCQTestInterface.enqueueWriteBuffer(bufferInstructionInput,
                                                 CL_TRUE,
                                                 0,
                                                 sizeof(typeof(inputInstructionVector.at(0))) * inputInstructionVector.size(),
                                                 inputInstructionVector.data(),
                                                 NULL);
            aocl_utils_cpp::checkError(status, "Failed to write bufferInstructionInput");
        }

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

        //Transfer the input bias
        if (inputBiasVector.size() > 0) {
            status = clCQTestInterface.enqueueWriteBuffer(bufferBiasInput,
                                                 CL_TRUE,
                                                 0,
                                                 sizeof(typeof(inputBiasVector.at(0))) * inputBiasVector.size(),
                                                 inputBiasVector.data(),
                                                 NULL);
            aocl_utils_cpp::checkError(status, "Failed to write bufferDrainInput");
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


        //Setup the buffer arguments and number of transfer for the test interface
        kernelTestInterface.setArg(0, bufferActivationInput);
        kernelTestInterface.setArg(1, bufferActivationOutput);
        kernelTestInterface.setArg(2, bufferWeightInput);
        kernelTestInterface.setArg(3, bufferWeightOutput);
        kernelTestInterface.setArg(4, bufferBiasInput);
        kernelTestInterface.setArg(5, bufferBiasOutput);
        kernelTestInterface.setArg(6, bufferDrainInput);
        kernelTestInterface.setArg(7, bufferDrainOutput);
        kernelTestInterface.setArg(8, bufferInstructionInput);
        kernelTestInterface.setArg(9, bufferInstructionOutputH);
        kernelTestInterface.setArg(10, bufferInstructionOutputV);

        kernelTestInterface.setArg(11, (cl_ushort) inputActivationVector.size()); //numInputActivationBlocks
        kernelTestInterface.setArg(12, (cl_ushort) startIndexActivationBlocks); //startIndexActivationBlocks,
        cl_ushort numOutputActivationBlocks =
                idy < (PE_ROWS - 1) ? inputActivationVector.size() : 0;
        kernelTestInterface.setArg(13, numOutputActivationBlocks); //numOutputActivationBlocks
        //Allocate space for the output activation vector
        for (int i=0; i<numOutputActivationBlocks; i++) {
            t_simdblock_host brick;
            outputActivationVector.push_back(brick);
        }

        kernelTestInterface.setArg(14, (cl_ushort) inputWeightVector.size()); //numInputWeightBlocks
        kernelTestInterface.setArg(15, (cl_ushort) startIndexWeightBlocks); //startIndexWeightBlocks,
        cl_ushort numOutputWeightBlocks =
                idy < (PE_COLS - 1) ? inputWeightVector.size() : 0;
        kernelTestInterface.setArg(16, numOutputWeightBlocks); //numOutputWeightBlocks
        //Allocate space for the output activation vector
        for (int i=0; i<numOutputWeightBlocks; i++) {
            t_simdblock_host brick;
            outputWeightVector.push_back(brick);
        }
        kernelTestInterface.setArg(17, (cl_ushort) inputBiasVector.size()); //numInputBias
        cl_ushort numOutputBias =
                idx < (PE_COLS - 1) ? inputBiasVector.size() : 0;
        kernelTestInterface.setArg(18, numOutputBias); //numOutputBias
        for (int i=0; i<numOutputBias; i++) {
            short val;
            outputBiasVector.push_back(val);
        }

        kernelTestInterface.setArg(19, (cl_ushort) numInputDrain); //numInputDrain
        cl_ushort numOutputDrain = drainResult ? PE_ROWS  - idy : 0;
        kernelTestInterface.setArg(20, numOutputDrain); //numOutputDrain
        for (int i=0; i<numOutputDrain; i++) {
            short val;
            outputDrainVector.push_back(val);
        }

        kernelTestInterface.setArg(21, (cl_ushort) inputInstructionVector.size()); //numInputInstruction
        cl_ushort numOutputInstructionH =
                  ((idy == 0) && (idx < (PE_COLS - 1) )) ? inputInstructionVector.size() : 0;
        kernelTestInterface.setArg(22, (cl_ushort) numOutputInstructionH); //numOutputInsructionHorizontal,
        for (int i=0; i<numOutputInstructionH; i++) {
            t_pe_prototype_instruction_host val;
            outputInstructionHVector.push_back(val);
        }

        cl_ushort numOutputInstructionV =
                  idy < (PE_ROWS - 1) ? inputInstructionVector.size() : 0;
        kernelTestInterface.setArg(23, numOutputInstructionV); //numOutputInstructionVertical
        for (int i=0; i<numOutputInstructionV; i++) {
            t_pe_prototype_instruction_host val;
            outputInstructionVVector.push_back(val);
        }

        kernelTestInterface.setArg(24, (cl_uchar)MAX_IDX);
        kernelTestInterface.setArg(25, (cl_uchar)MAX_IDY);

        //Launch kernels
        /*
        clCQActivationTransport.enqueueTask(kernelActivationTransport);
        clCQWeightTransport.enqueueTask(kernelWeightTransport);
        clCQBiasTransport.enqueueTask(kernelBiasTransport);
        clCQDrainTransport.enqueueTask(kernelDrainTransport);
        clCQInstructionTransport.enqueueTask(kernelInstructionTransport);
        clCQPE.enqueueTask(kernelPE);
        */
        clCQTestInterface.enqueueTask(kernelTestInterface);

        //Retrieve data
        clCQTestInterface.finish();

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

        if (outputBiasVector.size() > 0) {
            clCQTestInterface.enqueueReadBuffer(
                        bufferBiasOutput,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputBiasVector.at(0))) * outputBiasVector.size(),
                        outputBiasVector.data()
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

        if (outputInstructionHVector.size() > 0) {
            clCQTestInterface.enqueueReadBuffer(
                        bufferInstructionOutputH,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputInstructionHVector.at(0))) * outputInstructionHVector.size(),
                        outputInstructionHVector.data()
                        );
        }

        if (outputInstructionVVector.size() > 0) {
            clCQTestInterface.enqueueReadBuffer(
                        bufferInstructionOutputV,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputInstructionVVector.at(0))) * outputInstructionVVector.size(),
                        outputInstructionVVector.data()
                        );
        }
    }
};

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainage) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);

    //This test won't pass if fracIn > fractOut
    char fracIn = 5, fracOut = 6, fracW = 5;
    char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
    char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;
    int targetIDX = IDX, targetIDY = IDY;
    float probOne = BERN_P;

    unsigned int numElements = 1444;
    unsigned short transmissionStartIndex = 0;
    unsigned short transmissionEndIndex = numElements - 1;
    unsigned short selectStartIndex = transmissionStartIndex; //must match the startIndex!

    EXPECT_TRUE(PE_COLS > targetIDX);
    EXPECT_TRUE(PE_ROWS > targetIDY);

    //First prepare the bias;
    //float biasFloat = 3.1415926;
    float biasFloat = 0.0;

    //Then convert the bias into a fixed point number;
    fixedPointNumber biasFPInput (biasFloat, fracW, WEIGHT_BITWIDTH);

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
    inputBiasVector.push_back((short) biasFPInput.getBits());
    // Compress the activaion block
    std::vector<fixedPointNumber> fpActivationVector;
    std::vector<fixedPointNumber> fpWeightVector;
    compress_vector (
                activationRealInput,
                intWidthIn,
                fracIn,
                fpActivationVector,
                inputActivationVector
                );
    // Compress the weight block
    compress_vector (
                weightRealInput,
                intWidthWeight,
                fracW,
                fpWeightVector,
                inputWeightVector
                );

    //Prepare the instruction

     t_pe_prototype_instruction_host dotProductInstruction =
     {
      .maxIDX = PE_COLS - 1,
      .maxIDY = PE_ROWS - 1,
      .fracW = fracW,
      .fracDin = fracIn,
      .fracDout = fracOut};
      inputInstructionVector.push_back(dotProductInstruction);

    launch(targetIDX, targetIDY, transmissionStartIndex, 0);

    //Compare the result
    short actualOutputFP = outputDrainVector.at(0);
    EXPECT_TRUE(
         (actualOutputFP & WEIGHT_MASK)
         == ((short) (expectedOutputFP.getBits() & WEIGHT_MASK) ))
         << "Actual output: "<<std::bitset<WEIGHT_BITWIDTH>(actualOutputFP & WEIGHT_MASK)
         <<std::endl<<"Expected output: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;

}

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
