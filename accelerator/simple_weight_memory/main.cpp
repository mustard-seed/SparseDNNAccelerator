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

#include "gtest/gtest.h"
#include "boost/align/aligned_allocator.hpp"

#include "floatFixedPointConversion.hpp"
#include "tensorCompression.hpp"

#define W0 0.33333333
#define W1 0.33333333
#define W2 0.33333333
#define K_SIZE 3
#define MAX_DATA_LENGTH 1048576

typedef
std::vector<cl_float, boost::alignment::aligned_allocator<cl_float, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
t_aligned_float_vector;


class peTestFixture : public ::testing::Test {
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    cl::CommandQueue commandQueue;

    cl::Kernel toyPingPongConvKernel;
    cl::Kernel nopKernel;

    cl::Buffer bufferInput;
    cl::Buffer bufferOutput;

    t_aligned_float_vector inputVector;
    t_aligned_float_vector outputVector;


    void SetUp() override {
        std::cout<<"Type in the aocx file name: ";
        std::cin >> binaryFile;
        //binaryFile = "toyPingPong_aoc_emulation.aocx";
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

        toyPingPongConvKernel = cl::Kernel(program, "toyPingPongConv", &status);
        aocl_utils_cpp::checkError(status, "Failed to create the ping pong kernel!");

        nopKernel = cl::Kernel(program, "nop", &status);
        aocl_utils_cpp::checkError(status, "Failed to created the nop kernel!");

        commandQueue = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue queue!");

        bufferInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_DATA_LENGTH * sizeof(cl_float),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input buffer!");


        bufferOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        MAX_DATA_LENGTH * sizeof(cl_float),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input buffer!");


        std::cout <<"AOCL setup compelete"<<std::endl;

        //Need to setup numInstructions, idx, and idy separately
    }

    void launch (float w0, float w1, float w2, int cacheSize) {
        cl_int status;
        //Fill the buffers
        //Transfer the instruction
        if (inputVector.size() > K_SIZE) {
            //Prepare the output vector
            outputVector.resize(inputVector.size() - K_SIZE + 1);
            status = commandQueue.enqueueWriteBuffer(bufferInput,
                                                 CL_TRUE,
                                                 0,
                                                 sizeof(typeof(inputVector.at(0))) * inputVector.size(),
                                                 inputVector.data(),
                                                 NULL);
            aocl_utils_cpp::checkError(status, "Failed to write input buffer");

            //Setup the buffer arguments and number of transfer for the test interface
            toyPingPongConvKernel.setArg(0, bufferInput);
            toyPingPongConvKernel.setArg(1, bufferOutput);
            toyPingPongConvKernel.setArg(2, (cl_int) (inputVector.size()));
            toyPingPongConvKernel.setArg(3, w0);
            toyPingPongConvKernel.setArg(4, w1);
            toyPingPongConvKernel.setArg(5, w2);
            toyPingPongConvKernel.setArg(6, cacheSize);

            cl::Event kernelEvent;
            status = commandQueue.enqueueTask(toyPingPongConvKernel, NULL, &kernelEvent);
            aocl_utils_cpp::checkError(status, "Failed to launch the kernel!");

            //Retrieve data
            commandQueue.finish();

            status = commandQueue.enqueueReadBuffer(
                        bufferOutput,
                        CL_TRUE,
                        0,
                        sizeof(typeof(outputVector.at(0))) * outputVector.size(),
                        outputVector.data()
                        );

            aocl_utils_cpp::checkError(status, "Failed to read the data back!");

            cl_ulong kernelStartTime = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong kernelEndTime = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double kernelRunTime = (cl_double)((kernelEndTime - kernelStartTime) * (cl_double)(1e-3));

            std::cout <<"Convolution kernel time (us): "<<kernelRunTime<<std::endl;
            std::cout <<"Number of inputs: "<<inputVector.size()<<std::endl;
            std::cout <<"Cache size: "<<cacheSize<<std::endl;

            int numNOPLaunch = 100;
            cl_double nopKernelTotalTime = 0;
            std::cout <<"Launching the NOP kernel "<<numNOPLaunch<<" times."<<std::endl;
            for (int i=0; i<numNOPLaunch; i++) {
            cl::Event nopKernelEvent;
                status = commandQueue.enqueueTask(nopKernel, NULL, &nopKernelEvent);
                aocl_utils_cpp::checkError(status, "Failed to launch the nop kernel!");
                commandQueue.finish();

                kernelStartTime = nopKernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                kernelEndTime = nopKernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                nopKernelTotalTime += (cl_double)((kernelEndTime - kernelStartTime) * (cl_double)(1e-3));
            }
            std::cout <<"Average NOP kernel time (us): "<<nopKernelTotalTime / (cl_double (numNOPLaunch))<<std::endl;

        }
        else {
            std::cout <<"Number of input is too low!"<<std::endl;
        }
    } //launch

};

t_aligned_float_vector initialize_vector(unsigned seed,
                       float bernProb,
                       int numTensors,
                       int height,
                       int width,
                       int channel,
                       float min,
                       float max
                                     ) {
    std::mt19937 generator(seed);
    std::bernoulli_distribution bernDistribution(bernProb);
    std::uniform_real_distribution<float> uniDistribution(min, max);
    int numElements = numTensors * height * width * channel;
    t_aligned_float_vector vector(numElements, 0.0f);

    for (unsigned i=0; i<numElements; i++) {
        float val = bernDistribution(generator) ?
                    uniDistribution(generator) : 0.0f;
        vector.at(i) = val;
    }
    return vector;
}

TEST (hostInfrastructureTest, compressionTest) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.05;
    int numTensors = 1;
    int height = 64;
    int width = 64;
    int channel = 512;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short streamingBlockSize = 8;
    unsigned short simdBlockSize = 4;
    unsigned short extMemoryRowAddressStride = 1024*16;
    unsigned short numBanks = 1;
    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numTensors,
                height,
                width,
                channel,
                min,
                max
                );


    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the tensor"<<std::endl;
    compressedTensor compTensor(
                fpVector,
                numTensors,
                channel,
                width,
                height,
                streamingBlockSize,
                simdBlockSize,
                extMemoryRowAddressStride,
                numBanks
                );

    std::cout <<"Start to decode the tensor"<<std::endl;
    std::vector<float> decodedVector;
    decodeTensor(compTensor, decodedVector, fracWidth, intWidth);

    std::cout <<"Comparing the decoded tensor and the original tensor"<<std::endl;
    for (int i=0; i<numElements; i++) {
        float orig = floatVector.at(i);
        float newValue = decodedVector.at(i);
        EXPECT_TRUE(std::abs(orig-newValue) < 1.0f / (1 << fracWidth))
                << "i, Original Value, New Value: "<<i<<" "<<orig<<" "<<newValue<<std::endl;
    }
}

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
