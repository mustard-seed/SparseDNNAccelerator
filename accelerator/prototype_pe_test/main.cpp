#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#define MAX_DATA_LENGTH 65536
//#define MAX_DATA_LENGTH 1024

#define EMULATE
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
        const std::vector <float> & inputVectorA,
        const std::vector <float> & inputVectorB
        );

class peTestFixture : public ::testing::Test
{
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    cl::CommandQueue clCQTestInterface;

    cl::Kernel kernelTestInterface;

    cl::Kernel kernelPE;

    cl::Buffer bufferActivationInput;
    cl::Buffer bufferActivationOutput;
    cl::Buffer bufferWeightInput;
    cl::Buffer bufferWeightOutput;
    cl::Buffer bufferDrainInput;
    cl::Buffer bufferDrainOutput;
    cl::Buffer bufferOutputInstruction;

    //Profile function
    cl_int (*get_profile_fn)(cl_device_id, cl_program, cl_bool,cl_bool,cl_bool,size_t, void *,size_t *,cl_int *);

    //Reset function
    //cl_int (*reset_fn)(cl_context, cl_uint, const cl_device_id*);

    void SetUp() override;

    void launch (std::vector<float>& vecRealWeight
                                ,std::vector<float>& vecRealActivation
                                ,float bias
                                ,char fracIn
                                ,char fracOut
                                ,char fracW);
};

//TEST_F (peTestFixture, testFixture) 0, //tilingSizeWidth
//{
//    launch(IDX,IDY,0,0, false);
//    EXPECT_TRUE (COMPRESSION_VEC_SIZE == 4);
//    EXPECT_TRUE(true);
//}
#define PLAY
#ifdef PLAY
TEST_F (peTestFixture, testPlayfield) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/

    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
//    float probOne = 1.0;

//    float biasFloat = 0.0;

//    std::vector<float> activationRealInput (16, 0.00);
//    std::vector<float> weightRealInput (16, 0.00);


//    for (int i=0; i<4; i++)
//    {
//        //if (i == 0) {
//            for (int j=4*i; j < 4*i+4; j++)
//            {
//                activationRealInput.at(j) = 0.25*(i+1);
//                weightRealInput.at(j) = 0.25*(i+1);
//            }
//        //}
//    }
    float probOne = 1.0;

    unsigned int numElements = 32;
    float biasFloat = 1.0;

    // Generate a block of activations
    std::vector<float> activationRealInput = initialize_vector(
                VECTOR_A_SEED,
                numElements,
                probOne,
                -1.14,
                1.14
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
    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );

}
#else
TEST_F (peTestFixture, testLoadBiasDotProductAndDrainageZero) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    float probOne = 0.0;

    unsigned int numElements = 8196*4+3;

    float biasFloat = 1.0;

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

    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );
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
    float probOne = 0.5;

    unsigned int numElements = 8196*4+3;
    float biasFloat = 1.0;

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

    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainage025Long) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    float probOne = 0.25;

    unsigned int numElements = 8196*4+3;
    float biasFloat = 1.0;

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



    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );
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
    float probOne = 1.0;

    unsigned int numElements = 8196*4+3;
    float biasFloat = 1.0;

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



    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );
}

TEST_F (peTestFixture, testLoadBiasDotProductAndDrainageOneShort) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    //This test won't pass if fracIn > fractOut
    char fracIn = 2, fracOut = 3, fracW = 2;
    float probOne = 1.0;

    unsigned int numElements = 32;
    float biasFloat = 1.0;

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



    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );
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
    float probOne = 0.25;

    unsigned int numElements = 32;
    float biasFloat = 1.0;

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



    launch (
           activationRealInput,
           weightRealInput,
           biasFloat,
           fracIn,
           fracOut,
           fracW
           );
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

float dot_product_regular_vectors (const std::vector<float> & inputVectorA
                                   , const std::vector<float> &inputVectorB) {
    float result = 0.0f;
    for (unsigned i=0; i<inputVectorA.size(); i++) {
        result += inputVectorA.at(i) * inputVectorB.at(i);
    }
    return result;
}

void peTestFixture::SetUp() {
#ifdef ARRIA10
        binaryFile = "smallBuffer.aocx";
#else
        //std::cout <<"Please type in the FPGA image (e.g. foo.aocx): "<<std::endl;
        //std::cin >> binaryFile;
        binaryFile = "prototypePE_aoc_release_hw.aocx";
#endif
        //Setup and platform and the context
        cl_int status = CL_SUCCESS;
#if defined(EMULATE)
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
#else
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#endif
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
                        MAX_DATA_LENGTH * sizeof(char),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input activation buffer!");

        bufferActivationOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(char),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output activation buffer!");

        bufferWeightInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(char),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input weight buffer!");

        bufferWeightOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(char),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output weight buffer!");

        bufferDrainInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(char),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input drain buffer!");

        bufferDrainOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(char),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output drain buffer!");

        bufferOutputInstruction = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
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

void peTestFixture::launch (std::vector<float>& vecRealWeight
                            ,std::vector<float>& vecRealActivation
                            ,float bias
                            ,char fracIn
                            ,char fracOut
                            ,char fracW) {

                   std::cout <<"1. Quantize the input weight, activation, and bias to fixed point number"<<std::endl;
                   EXPECT_EQ(vecRealActivation.size(), vecRealActivation.size()) << "Input weight and activaiton have unequal sizes"<<std::endl;

                   int numElements = vecRealActivation.size();
                   unsigned char numBitsToRightShift = fracIn+fracW-fracOut;
                   char intWidthIn = WEIGHT_BITWIDTH - fracIn - 1;
                   char intWidthWeight = WEIGHT_BITWIDTH - fracW - 1;

                   t_bias biasFP = ((t_bias) std::round(bias * (1 << (fracIn + fracW))));
                   std::vector<fixedPointNumber> fpActivationVector;
                   std::vector<fixedPointNumber> fpWeightVector;
                   fpActivationVector.resize(vecRealActivation.size());
                   fpWeightVector.resize(vecRealWeight.size());

                   for (int i=0; i<numElements; i++) {
                       fixedPointNumber fpWeight(vecRealWeight.at(i), fracW, intWidthWeight);
                       fpWeightVector.at(i) = fpWeight;
                       fixedPointNumber fpActivation(vecRealActivation.at(i), fracIn, intWidthIn);
                       fpActivationVector.at(i) = fpActivation;
                   }

                   std::cout <<"2. Compute the expected output."<<std::endl;
                   float expectedResultReal = dot_product_regular_vectors(vecRealActivation, vecRealWeight) + bias;
                   fixedPointNumber expectedOutputFP (expectedResultReal, fracOut, WEIGHT_BITWIDTH - fracOut - 1);

                   std::cout <<"3. Prepare the input weight and activation vectors."<<std::endl;

                   std::unique_ptr<AlignedTensor> pWeight, pActivation;
           #if defined(SPARSE_SYSTEM)
                   pWeight.reset(new FlexibleDirectCompressedTensor (
                                   fpWeightVector, //fixedPointVector
                                   1, //_num3DTensors
                                   numElements, //_channel
                                   1, //_width
                                   1, //_height
                                   numElements-1, //_maxScalarIndexInChannelGroup
                                   COMPRESSION_WINDOW_SIZE-1, //_maxClusterIndexInCompressionBlock
                                   TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   CLUSTER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   true //isKernel
                                     )
                               );
                   pActivation.reset(new FlexibleDirectCompressedTensor (
                                   fpActivationVector, //fixedPointVector
                                   1, //_num3DTensors
                                   numElements, //_channel
                                   1, //_width
                                   1, //_height
                                   numElements-1, //_maxScalarIndexInChannelGroup
                                   COMPRESSION_WINDOW_SIZE-1, //_maxClusterIndexInCompressionBlock
                                   TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   CLUSTER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   true //isKernel. Indeed, set this to TRUE in this test
                                     )
                               );
           #else
                   pWeight.reset(new AlignedTensor (
                                   fpWeightVector, //fixedPointVector
                                   1, //_num3DTensors
                                   numElements, //_channel
                                   1, //_width
                                   1, //_height
                                   numElements-1, //_maxScalarIndexInChannelGroup
                                   TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   CLUSTER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   true //isKernel
                                     )
                               );
                   pActivation.reset(new AlignedTensor (
                                   fpActivationVector, //fixedPointVector
                                   1, //_num3DTensors
                                   numElements, //_channel
                                   1, //_width
                                   1, //_height
                                   numElements-1, //_maxScalarIndexInChannelGroup
                                   TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   CLUSTER_SIZE-1, //_maxClusterIndexInTransferBlock
                                   true //isKernel. Indeed, set this to TRUE in this test
                                     )
                               );
           #endif

                   std::cout <<"4. Transfer the weights, activations, and biases."<<std::endl;

                   cl_int status;

                   //Fill the buffers
                   //Transfer the input activation
                   {
                       auto numTransferBlocks = (pActivation->getTransferBlockVector()).size();
                       auto sizeTransferBlockElement = sizeof(typeof((pActivation->getTransferBlockVector()).at(0)));
                       auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;
                       status = clCQTestInterface.enqueueWriteBuffer(bufferActivationInput, //buffer
                                                            CL_TRUE, //blocking_write
                                                            0, //offset
                                                            valueVectorSizeBytes, //size in bytes
                                                            (pActivation->getTransferBlockVector()).data(), //data pointer
                                                            NULL);
                       aocl_utils_cpp::checkError(status, "Failed to write bufferActivationInput");
                    }

                   //Transfer the input weight
                   {
                       auto numTransferBlocks = (pWeight->getTransferBlockVector()).size();
                       auto sizeTransferBlockElement = sizeof(typeof((pWeight->getTransferBlockVector()).at(0)));
                       auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;
                       status = clCQTestInterface.enqueueWriteBuffer(bufferWeightInput, //buffer
                                                            CL_TRUE, //blocking_write
                                                            0, //offset
                                                            valueVectorSizeBytes, //size in bytes
                                                            (pWeight->getTransferBlockVector()).data(), //data pointer
                                                            NULL);
                       aocl_utils_cpp::checkError(status, "Failed to write bufferWeightInput");
                    }

                   t_output_instruction_host outputInstruction =
                   {
                      .numPSumToProcess = 1,
                      .numBitsToRightShift = numBitsToRightShift,
                      .enableRelu = false
                   };

                   status = clCQTestInterface.enqueueWriteBuffer(bufferOutputInstruction,
                                                        CL_TRUE,
                                                        0,
                                                        sizeof(t_output_instruction_host),
                                                        &outputInstruction,
                                                        NULL);
                   aocl_utils_cpp::checkError(status, "Failed to write bufferOutputInstruction");
                   EXPECT_TRUE (status == CL_SUCCESS);

           #if defined(SPARSE_SYSTEM)
                   unsigned short numInputActivationBlocks = (pActivation->getTransferBlockCountVector()).at(0);
                   unsigned short numOutputActivationBlocks = numInputActivationBlocks;

                   unsigned short numInputWeightBlocks = (pWeight->getTransferBlockCountVector()).at(0);
                   unsigned short numOutputWeightBlocks = numInputWeightBlocks;
           #else
                   unsigned short numInputActivationBlocks = 1 + (numElements-1) / (TRANSFER_SIZE * CLUSTER_SIZE);
                   unsigned short numOutputActivationBlocks = numInputActivationBlocks;

                   unsigned short numInputWeightBlocks = 1 + (numElements-1) / (TRANSFER_SIZE * CLUSTER_SIZE);
                   unsigned short numOutputWeightBlocks = numInputWeightBlocks;
           #endif
                   unsigned short numOutputDrain = 1;

                   t_aligned_transfer_block_vector outWeightVector;
                   t_aligned_transfer_block_vector outActivationVector;
                   signed char drain;

                   outWeightVector.resize(numInputWeightBlocks);
                   outActivationVector.resize(numInputActivationBlocks);

                   std::cout <<"5. Set the kernel arguments and launch."<<std::endl;
                   //Setup the buffer arguments and number of transfer for the test interface
                   kernelTestInterface.setArg(0, bufferActivationInput);
                   kernelTestInterface.setArg(1, bufferActivationOutput);
                   kernelTestInterface.setArg(2, bufferWeightInput);
                   kernelTestInterface.setArg(3, bufferWeightOutput);
                   kernelTestInterface.setArg(4, bufferDrainOutput);
                   kernelTestInterface.setArg(5, (cl_short) biasFP);
                   kernelTestInterface.setArg(6, bufferOutputInstruction);

                   kernelTestInterface.setArg(7, (cl_ushort) numInputActivationBlocks); //numInputActivationBlocks
                   kernelTestInterface.setArg(8, (cl_ushort) numOutputActivationBlocks); //numOutputActivationBlocks

                   kernelTestInterface.setArg(9, (cl_ushort) numInputWeightBlocks); //numInputWeightBlocks
                   kernelTestInterface.setArg(10, (cl_ushort) numOutputWeightBlocks); //numOutputWeightBlocks

                   kernelTestInterface.setArg(11, (cl_ushort) numOutputDrain); //numOutputDrain
                   cl::Event event;
                   //Launch kernels

                   status = clCQTestInterface.enqueueTask(kernelTestInterface, NULL, &event);
                   aocl_utils_cpp::checkError(status, "Failed to launch the kernel");
                   //Retrieve data
                   clCQTestInterface.finish();

                   cl_ulong kernelStartTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                   cl_ulong kernelEndTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                   cl_double kernelRunTime = (cl_double)((kernelEndTime - kernelStartTime) * (cl_double)(1e-3));


                   //status = (cl_int)(*get_profile_fn)(this->clDevice(), this->program(), true, true, false, 0, NULL, NULL, NULL);

                   std::cout <<"6. Drain the outputs."<<std::endl;
                   clCQTestInterface.enqueueReadBuffer(
                               bufferActivationOutput,
                               CL_TRUE,
                               0,
                               sizeof(typeof(outActivationVector.at(0))) * outActivationVector.size(),
                               outActivationVector.data()
                               );

                   clCQTestInterface.enqueueReadBuffer(
                               bufferWeightOutput,
                               CL_TRUE,
                               0,
                               sizeof(typeof(outWeightVector.at(0))) * outWeightVector.size(),
                               outWeightVector.data()
                               );

                   clCQTestInterface.enqueueReadBuffer(
                               bufferDrainOutput,
                               CL_TRUE,
                               0,
                               sizeof(char),
                               &drain
                               );


                   std::cout <<"7. Compare the results"<<std::endl;
                   float actualOutputReal = fixedPointNumber((signed char)drain, fracOut, WEIGHT_BITWIDTH-fracOut-1).convert2Float();

                   std::cout <<"Expected output bits: "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl;
                   std::cout <<"Actual output bits: "<<std::bitset<WEIGHT_BITWIDTH>((drain & WEIGHT_MASK))<<std::endl;

                   EXPECT_TRUE(
                        std::abs(actualOutputReal - expectedOutputFP.convert2Float()) <= 1.0 / (1 << fracOut))
                        << "Actual output: "<<actualOutputReal<<" "<<std::bitset<WEIGHT_BITWIDTH>(drain & WEIGHT_MASK)
                        <<std::endl<<"Expected output: "<<expectedOutputFP.convert2Float()<<" "<<std::bitset<WEIGHT_BITWIDTH>((expectedOutputFP.getBits()) & WEIGHT_MASK)<<std::endl
                        <<"Golden output: "<<expectedResultReal<<std::endl;

                   std::cout <<"Test kernel time (us): "<<kernelRunTime<<std::endl;
                   std::cout <<"Number of weight transfer blocks: "<<numInputActivationBlocks<<std::endl;
                   std::cout <<"Number of activation transfer blocks: "<<numInputWeightBlocks<<std::endl;
               }
