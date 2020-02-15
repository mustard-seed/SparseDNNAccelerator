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
#define EMULATE

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
        binaryFile = "smallBufferTest.aocx";
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#else
        binaryFile = "smallBuffer.aocx";
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
        cl_ulong maxBufferSizeByte = 2048;
        //aocl_utils_cpp::checkError(status, "Failed to query the maximum buffer size in bytes!");

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
            std::cout <<"Input mutual bitmask "<<std::bitset<8>(mutualBitmask)<<std::endl;
            std::cout <<"Input bitmask "<<std::bitset<8>(bitmask)<<std::endl;
            for (unsigned i=0; i<numTransferBlocks; i++)
            {
                std::cout <<"Input transfer block ["<<i<<"]: "
                         <<(int) inputTransferBlocks.at(i).values[0]<<" "
                         <<(int) inputTransferBlocks.at(i).values[1]<<" "
                         <<(int) inputTransferBlocks.at(i).values[2]<<" "
                         <<(int) inputTransferBlocks.at(i).values[3]<<std::endl;

                std::cout <<"Filtered blocks ["<<i<<"]: "
                         <<(int) filteredTransferBlocks.at(i).values[0]<<" "
                         <<(int) filteredTransferBlocks.at(i).values[1]<<" "
                         <<(int) filteredTransferBlocks.at(i).values[2]<<" "
                         <<(int) filteredTransferBlocks.at(i).values[3]<<std::endl;

                std::cout <<"New buffer blocks ["<<i<<"]: "
                         <<(int) nextBuffers.at(i).values[0]<<" "
                         <<(int) nextBuffers.at(i).values[1]<<" "
                         <<(int) nextBuffers.at(i).values[2]<<" "
                         <<(int) nextBuffers.at(i).values[3]<<std::endl;

                std::cout <<"Mac valid flags ["<<i<<"]: "
                         <<(int) macValidFlags.at(i)<<std::endl;
            }
        } // input checking block
    } //launch

};
//#define PLAY
#ifdef PLAY
TEST_F (testFixture, play) {
    //bitmask: 16'b01101111_11110110
    unsigned short _bitmask = 0x6FF6;
    //mutual bitmask: 16'b01100110_01100110
    unsigned short _mutualBitmask = 0x6666;
    std::vector<t_smb_tb> _testTB(6, {0,0,0,0});
    _testTB.at(0) = {0, 1, 2, 3};
    _testTB.at(1) = {4, 5, 6, 7};
    _testTB.at(2) = {8, 9, 10, 11};
    _testTB.at(3) = {12, 13, 14, 15};
    _testTB.at(4) = {16, 17, 18, 19};
    _testTB.at(5) = {20, 21, 22, 23};
    launch(
           _bitmask,
           _mutualBitmask,
           _testTB
           );
}
#else
TEST_F (testFixture, test0) {
    //bitmask: 16'b01101111_11110110
    unsigned short _bitmask = 0x6FF6;
    //mutual bitmask: 16'b01100110_01100110
    unsigned short _mutualBitmask = 0x6666;
    std::vector<t_smb_tb> _testTB(6, {0,0,0,0});
    _testTB.at(0) = {0, 1, 2, 3};
    _testTB.at(1) = {4, 5, 6, 7};
    _testTB.at(2) = {8, 9, 10, 11};
    _testTB.at(3) = {12, 13, 14, 15};
    _testTB.at(4) = {16, 17, 18, 19};
    _testTB.at(5) = {20, 21, 22, 23};
    launch(
           _bitmask,
           _mutualBitmask,
           _testTB
           );
}

TEST_F (testFixture, test1) {
    //bitmask: 8'b11111111_11111111
    unsigned char _bitmask = 0xFFFF;
    //mutual bitmask: 16'b01100110_01100110
    unsigned char _mutualBitmask = 0x6666;
    std::vector<t_smb_tb> _testTB(4, {0,0,0,0});
    _testTB.at(0) = {0, 1, 2, 3};
    _testTB.at(1) = {4, 5, 6, 7};
    _testTB.at(2) = {8, 9, 10, 11};
    _testTB.at(3) = {12, 13, 14, 15};
    _testTB.at(4) = {16, 17, 18, 19};
    _testTB.at(5) = {20, 21, 22, 23};
    _testTB.at(6) = {24, 25, 26, 27};
    _testTB.at(7) = {28, 29, 30, 31};
    launch(
           _bitmask,
           _mutualBitmask,
           _testTB
           );
}

#endif

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
