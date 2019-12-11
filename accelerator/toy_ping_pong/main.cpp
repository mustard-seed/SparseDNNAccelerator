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

#include "gtest/gtest.h"
#include "boost/align/aligned_allocator.hpp"

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

void convolution1D (t_aligned_float_vector & inputVector, std::vector<float>& outputVector, float w0, float w1, float w2) {
    assert (inputVector.size() >= 3);

    for (int i=0; i<inputVector.size() - 2; i++) {
        float result = inputVector.at(i) * w0 + inputVector.at(i+1) * w1 + inputVector.at(i+2) * w2;
        outputVector.push_back(result);
    }
}

t_aligned_float_vector initialize_vector(unsigned seed,
                       unsigned int numElements
                       //float min,
                       //float max
                                     ) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniDistribution(-10.0, 10.0);
    t_aligned_float_vector vector;

    for (unsigned i=0; i<numElements; i++) {
        float val = uniDistribution(generator);
        vector.push_back(val);
    }
    return vector;
}
TEST_F (peTestFixture, testLoadBiasDotProductAndDrainage) {
/* Test goal: Verify the correctness of the bias loading, dot product and drainage capability
 * Procedure: Load a bias into the PE, the stream compressed activation and weights, then read the result back. Verify that the bias read back approximately mataches the bias loaded in. Consider the effect of
 * different fixed-point width
 *
*/
    int seed = 12;
    int numElements = MAX_DATA_LENGTH;
    inputVector = initialize_vector(seed, numElements);
    std::vector<float> referenceOutput;
    convolution1D(inputVector, referenceOutput, W0, W1, W2);

    for (int cacheSize=64; cacheSize <= 1024; cacheSize +=64) {
        std::cout <<"================="<<std::endl;
        launch(W0, W1, W2, cacheSize);

        //Compare the result
        for (int i=0; i<referenceOutput.size(); i++) {
            EXPECT_TRUE(
                 std::abs(referenceOutput.at(i) - outputVector.at(i)) < 1e-4)
                 << "referenceOutput["<<i<<"]: "<<referenceOutput.at(i)<<std::endl
                 << "outputVector["<<i<<"]: "<<outputVector.at(i)<<std::endl;
       }
        std::cout <<"Finished checking the result"<<std::endl;
    }

}

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
