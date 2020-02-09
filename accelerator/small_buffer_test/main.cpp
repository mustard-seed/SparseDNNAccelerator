#define CL_TARGET_OPENCL_VERSION 200

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
#include <memory> //smart pointers

#include <chrono>

#include "gtest/gtest.h"
#include "boost/align/aligned_allocator.hpp"

#include "small_buffer.hpp"

/*Limits on the buffer sizes
 * Assume the biggest test convolves a 32x32x64 tensor with a 128*32*32*64 tensor
 * Add a safety factor of 2
 * */

class testFixture : public ::testing::Test {
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    //Command queues
    cl::CommandQueue clCQTestHarness;

    //The kernels
    cl::Kernel kernel;

    cl::Buffer bufferTransferBlocks;
    cl::Buffer bufferFilteredBlocks;
    cl::Buffer bufferNextBuffer;
    cl::Buffer bufferMacValidFlags;

    void SetUp() override
    {
       cl_int status = CL_SUCCESS;
//TODO: CHANGE THE NAME
#ifdef C5SOC
        binaryFile = "device_utils.aocx";
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#else
        binaryFile = "operandMatcher_c_model.aocx";
#if defined(EMULATE)
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
#else
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#endif
#endif

        //Setup and platform and the context
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

        //Instantiate the host-side kernel objects
        kernel = cl::Kernel(program, "smallBufferTest", &status);
        aocl_utils_cpp::checkError(status, "Failed to create the small buffer test kernel!");

        //Instantiate the command queues
        clCQTestHarness = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQTestHarness!");

        //Instantiate the buffers
        cl_ulong maxBufferSizeByte = clDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE> (&status);
        aocl_utils_cpp::checkError(status, "Failed to query the maximum buffer size in bytes!");

        std::cout <<"Setting the buffer bufferTransferBlocks. Size: "<<maxBufferSizeByte<<" bytes."<<std::endl;
        bufferTransferBlocks = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        maxBufferSizeByte,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferTransferBlocks!");

        std::cout <<"Setting the buffer bufferFilteredBlocks. Size: "<<maxBufferSizeByte<<" bytes."<<std::endl;
        bufferFilteredBlocks = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        maxBufferSizeByte,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferFilteredBlocks!");

        std::cout <<"Setting the buffer bufferTransferBlocks. Size: "<<maxBufferSizeByte<<" bytes."<<std::endl;
        bufferNextBuffer = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        maxBufferSizeByte,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferNextBuffer!");

        std::cout <<"Setting the buffer bufferMacValidFlags. Size: "<<maxBufferSizeByte<<" bytes."<<std::endl;
        bufferMacValidFlags = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        maxBufferSizeByte,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMacValidFlags!");

        std::cout <<"AOCL setup compelete"<<std::endl;
    }

   void launch (
            unsigned short bitmask,
            unsigned short mutualBitmask,
            std::vector<t_smb_tb> inputTransferBlocks
            )
    {

        auto numTransferBlocks = inputTransferBlocks.size();
        std::vector<t_smb_tb> filteredTransferBlocks;
        filteredTransferBlocks.resize(numTransferBlocks);
        std::vector<t_smb_tb> nextBuffers;
        nextBuffers.resize(numTransferBlocks);
        std::vector<unsigned char> macValidFlags;
        macValidFlags.resize(numTransferBlocks);

        std::cout <<"1. Setting kernel arguments for the test harness."<<std::endl;
        {
            kernel.setArg(0, bufferTransferBlocks);
            kernel.setArg(1, bufferFilteredBlocks);
            kernel.setArg(2, bufferNextBuffer);
            kernel.setArg(3, bufferMacValidFlags);
            kernel.setArg(4, (cl_uchar) numTransferBlocks);
            kernel.setArg(5, (cl_ushort) bitmask);
            kernel.setArg(6, (cl_ushort) mutualBitmask);

        }

        /* Transfer buffer content
        */
        cl_int status;

        //Transfer the input
        std::cout <<"2. Transfer the input transfer blocks "<<std::endl;
        {
            cl::Event event;
            auto sizeTransferBlockElement = sizeof(typeof(inputTransferBlocks.at(0)));
            auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

            std::cout <<"3. Transfering "<<valueVectorSizeBytes<<" bytes in to bufferTransferBlocks"<<std::endl;

            status = clCQTestHarness.enqueueWriteBuffer(bufferTransferBlocks, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 valueVectorSizeBytes, //size
                                                 inputTransferBlocks.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the bufferTransferBlocks vector");
            clCQTestHarness.finish();
        } // Transfer the input

        //Launch the kernels
        std::cout<<"4. Launch the kernel."<<std::endl;

        status = clCQTestHarness.enqueueTask(kernel, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch test kernel!");

        clCQTestHarness.finish();


#if defined(PROFILE) && defined(C5SOC)
        std::cout <<"4.b Attempting to retrieve autorun profiling data."<<std::endl;
        status = clGetProfileDataDeviceIntelFPGA(
                  clDevice(),
                  program(),
                  CL_TRUE,
                  CL_TRUE,
                  CL_FALSE,
                  0,
                  NULL,
                  NULL,
                  NULL
                    );
        aocl_utils_cpp::checkError(status, "Failed to retrieve profiling information!");
#endif

        std::cout <<"5. Retrieve the outputs."<<std::endl;
        status = clCQTestHarness.enqueueReadBuffer(
            bufferFilteredBlocks,
            CL_TRUE,
            0,
            sizeof(typeof((filteredTransferBlocks.at(0)))) * (filteredTransferBlocks.size()),
            filteredTransferBlocks.data()
        );
        aocl_utils_cpp::checkError(status, "Failed to read filtered transfer blocks!");

        status = clCQTestHarness.enqueueReadBuffer(
            bufferNextBuffer,
            CL_TRUE,
            0,
            sizeof(typeof((nextBuffers.at(0)))) * (nextBuffers.size()),
            nextBuffers.data()
        );
        aocl_utils_cpp::checkError(status, "Failed to read the updated buffer blocks!");

        status = clCQTestHarness.enqueueReadBuffer(
            bufferMacValidFlags,
            CL_TRUE,
            0,
            sizeof(typeof((macValidFlags.at(0)))) * (macValidFlags.size()),
            macValidFlags.data()
        );
        aocl_utils_cpp::checkError(status, "Failed to read the mac valid flags!");

        //Print the input and the output
        {
            std::cout <<"Input mutual bitmask "<<std::bitset<16>(mutualBitmask)<<std::endl;
            std::cout <<"Input bitmask "<<std::bitset<16>(bitmask)<<std::endl;
            for (unsigned i=0; i<numTransferBlocks; i++)
            {
                std::cout <<"Input transfer block ["<<i<<"]: "
                         <<inputTransferBlocks.at(i).values[0]<<" "
                         <<inputTransferBlocks.at(i).values[1]<<" "
                         <<inputTransferBlocks.at(i).values[2]<<" "
                         <<inputTransferBlocks.at(i).values[3]<<std::endl;

                std::cout <<"Filtered blocks ["<<i<<"]: "
                         <<filteredTransferBlocks.at(i).values[0]<<" "
                         <<filteredTransferBlocks.at(i).values[1]<<" "
                         <<filteredTransferBlocks.at(i).values[2]<<" "
                         <<filteredTransferBlocks.at(i).values[3]<<std::endl;

                std::cout <<"New buffer blocks ["<<i<<"]: "
                         <<nextBuffers.at(i).values[0]<<" "
                         <<nextBuffers.at(i).values[1]<<" "
                         <<nextBuffers.at(i).values[2]<<" "
                         <<nextBuffers.at(i).values[3]<<std::endl;

                std::cout <<"Mac valid flags ["<<i<<"]: "
                         <<macValidFlags.at(i)<<" "
                         <<macValidFlags.at(i)<<" "
                         <<macValidFlags.at(i)<<" "
                         <<macValidFlags.at(i)<<std::endl;
            }
        } // input checking block
    } //launch

};
#define PLAY
#ifdef PLAY
TEST_F (testFixture, play) {

}
#else

#endif

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
