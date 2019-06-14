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

#include "device_structures.hpp"
#include "boost/align/aligned_allocator.hpp"
#include "floatFixedPointConversion.hpp"
#include "gtest/gtest.h"
#include "prototypePE_structs.hpp"

#define VECTOR_LENGTH 1024
#define VECTOR_A_SEED 10
#define VECTOR_B_SEED 5
#define BERN_SEED 7
#define BERN_P 1.0
#define EPSILON 1e-5
#define VECTOR_MIN -2
#define VECTOR_MAX 2
#define FRAC_WIDTH 6
#define INT_WIDTH 5
#define MAX_INSTRUCTION_SIZE 64

typedef
std::vector<cl_ushort, boost::alignment::aligned_allocator<cl_ushort, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
aligned_ushort_vector;

typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
aligned_short_vector;

typedef
std::vector<t_pe_prototype_instruction,
boost::alignment::aligned_allocator<t_pe_prototype_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_instruction_vector;

class peTestFixture : public ::testing::Test {
protected:
    std::string binaryFile;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;
    cl::CommandQueue clCommandQueue;
    cl::Kernel kernelPE;
    cl::Buffer bufferInstruction;
    cl::Buffer bufferActivationInput;
    cl::Buffer bufferActivationOutput;
    cl::Buffer bufferWeightInput;
    cl::Buffer bufferWeightOutput;
    cl::Buffer bufferBiasInput;
    cl::Buffer bufferBiasOutput;


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
        cl::Program program = aocl_utils_cpp::createProgramFromBinary(
                    clContext,
                    binaryFile.c_str(),
                    {clDevice}
                    );
        status = program.build({clDevice});
        aocl_utils_cpp::checkError(status, "Failed to build program");

        kernelPE = cl::Kernel(program, "kernelPrototypePE", &status);
        aocl_utils_cpp::checkError(status, "Failed to create the prototype PE kernel!");

        //Setup the command queue and buffers
        clCommandQueue = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue!");

        bufferInstruction = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        MAX_INSTRUCTION_SIZE * sizeof(t_pe_prototype_instruction),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the instruction buffer!");

        bufferActivationInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        VECTOR_LENGTH * sizeof(short),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input activation buffer!");

        bufferActivationOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        VECTOR_LENGTH * sizeof(short),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output activation buffer!");

        bufferWeightInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        VECTOR_LENGTH * sizeof(short),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input weight buffer!");

        bufferWeightOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        VECTOR_LENGTH * sizeof(short),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output weight buffer!");

        bufferBiasInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        VECTOR_LENGTH * sizeof(short),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the input bias buffer!");

        bufferBiasOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        VECTOR_LENGTH * sizeof(short),
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the output bias buffer!");

        kernelPE.setArg(0, bufferActivationInput);
        kernelPE.setArg(1, bufferActivationOutput);
        kernelPE.setArg(2, bufferWeightInput);
        kernelPE.setArg(3, bufferWeightOutput);
        kernelPE.setArg(4, bufferBiasInput);
        kernelPE.setArg(5, bufferBiasOutput);
        kernelPE.setArg(6, bufferInstruction);

        std::cout <<"AOCL setup compelete"<<std::endl;

        //Need to setup numInstructions, idx, and idy separately
    }
};


/*!
 * \brief clInit
 * \details Set up the OpenCL context, creates the command queue, and create the kernels
 * \param binaryFile
 * \return The status code
 */
cl_int clInit (const std::string binaryFile,
               cl::Platform &clPlatform,
               cl::Context & clContext,
               cl::Device & clDevice,
               cl::CommandQueue & clDMAQueue,
               cl::CommandQueue & clSequencerQueue,
               std::vector<cl::CommandQueue> & clCollectorQueue,
               cl::Kernel & krnSpWDMA,
               cl::Kernel & krnSequencer,
               std::vector<cl::Kernel> & krnWeightCollectorVec);

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

/*!
 * \brief compress_vector
 * \details Convert a floating point vector to a fixed vector, and generates the compressed version
 * \param encodingBlockSize
 * \param inputVector
 * \param fixedPointVector
 * \param compressedVector
 */
void compress_vector (
        std::vector<float> & inputVector,
        unsigned int encodingBlockSize,
        char intWidth,
        char fracWidth,
        std::vector<fixedPointNumber> & fixedPointVector,
        aligned_short_vector & compressedVector
        );

float dot_product_regular_vectors (
        std::vector <float> & inputVectorA,
        std::vector <float> & inputVectorB
        );

float dot_product_compressed_vectors (
        aligned_short_vector & compressedVectorA,
        aligned_short_vector & compressedVectorB,
        unsigned int numEncodingBlocks,
        unsigned int encodingBlockSize,
        char intWidth,
        char fracWidth
        );

TEST(commpressionTest, compressionDotProduct) {
    std::vector<float> vectorA = initialize_vector(
                VECTOR_A_SEED,
                VECTOR_LENGTH,
                BERN_P,
                VECTOR_MIN,
                VECTOR_MAX
                );
    std::vector<float> vectorB = initialize_vector(
                VECTOR_B_SEED,
                VECTOR_LENGTH,
                BERN_P,
                VECTOR_MIN,
                VECTOR_MAX
                );

    float goldenResult = dot_product_regular_vectors(
                vectorA,
                vectorB
                );

    int effectual_length = (int) (std::ceil( (float) VECTOR_LENGTH / (float) ENCODING_LENGTH) * (float) ENCODING_LENGTH);

    std::vector<fixedPointNumber> fpVectorA (effectual_length, {0, FRAC_WIDTH, INT_WIDTH});
    aligned_short_vector compressedVectorA (effectual_length, 0);
    std::vector<fixedPointNumber> fpVectorB (effectual_length, {0, FRAC_WIDTH, INT_WIDTH});
    aligned_short_vector compressedVectorB (effectual_length, 0);

    compress_vector(vectorA, ENCODING_LENGTH, INT_WIDTH, FRAC_WIDTH, fpVectorA, compressedVectorA);
    compress_vector(vectorB, ENCODING_LENGTH, INT_WIDTH, FRAC_WIDTH, fpVectorB, compressedVectorB);

    std::cout <<"Check Vector A"<<std::endl;
    for (unsigned i = 0; i < fpVectorA.size(); i++) {
        float orig = vectorA.at(i);
        float fpA = (fpVectorA.at(i)).convert2Float();
        EXPECT_TRUE(std::abs(orig-fpA) < 1.0f / (1 << FRAC_WIDTH)) << "orig, fpA: "<<orig<<" "<<fpA<<std::endl;
    }

    std::cout <<"Check Vector B"<<std::endl;
    for (unsigned i = 0; i < fpVectorB.size(); i++) {
        float orig = vectorB.at(i);
        float fpB = (fpVectorB.at(i)).convert2Float();
        EXPECT_TRUE(std::abs(orig-fpB) < 1.0f / (1 << FRAC_WIDTH)) << "orig, fpB: "<<orig<<" "<<fpB<<std::endl;
    }

    std::cout <<"Check the dot product"<<std::endl;
    float compressedResult = dot_product_compressed_vectors(
                compressedVectorA,
                compressedVectorB,
                VECTOR_LENGTH / ENCODING_LENGTH,
                ENCODING_LENGTH,
                INT_WIDTH,
                FRAC_WIDTH
                );

    EXPECT_TRUE(std::abs(compressedResult-goldenResult) < 1.0f / (1 << FRAC_WIDTH))
            << "goldenResult, compressedResult: "<<goldenResult<<" "<<compressedResult<<std::endl;

}

TEST_F (peTestFixture, testFixture) {
    EXPECT_TRUE(true);
}

void cleanup();

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();


//    aocl_utils_cpp::Options options(argc, argv);

//    std::string binaryFile;

//    if(options.has("help")) {
//        std::cout<<"Usage: "<<argv[0]
//        <<" -aocx=<abs path to .aocx>"<<std::endl;
//        return 1;
//    }

//    if (options.has("aocx")) {
//        binaryFile = options.get<std::string>("aocx");
//      }
//      else {
//        std::cout<<"Error: aocx file path is not supplied"<<std::endl;
//        return 1;
//      }

//    try{
//        cl_int status = CL_SUCCESS;
//        //Platform
//        cl::Platform clPlatform;

//        //Device ID. Assumes that there is only one device.
//        cl::Device clDevice;

//        //Context
//        cl::Context clContext;

//        //Command queue used for data transfer
//        cl::CommandQueue clDMAQueue, clSequencerQueue;
//        std::vector<cl::CommandQueue> vecClCollectorQueues;
//        for (unsigned int i=0; i < KERNEL_CACHE_LANES; i++) {
//            vecClCollectorQueues.emplace_back();
//        }

//        //The sparse weight feeder DMA kernel
//        cl::Kernel krnSpWDMA;

//        //The sequencer kernel
//        cl::Kernel krnSequencer;

//        //The weight collector kernel
//        std::vector<cl::Kernel> krnWeightCollectorVec;

//        //Set up the OpenCL environment and create the kernels
//        clInit(binaryFile,
//               clPlatform,
//               clContext,
//               clDevice,
//               clDMAQueue,
//               clSequencerQueue,
//               vecClCollectorQueues,
//               krnSpWDMA,
//               krnSequencer,
//               krnWeightCollectorVec);

//        //Create the buffers
//        cl::Buffer inputSpWBuffer(
//                    clContext,
//                    CL_MEM_READ_WRITE,
//                    MATRIX_ROWS * MATRIX_COLS * sizeof(typeof(effectualValues.at(0))),
//                    NULL,
//                    &status
//                    );
//        aocl_utils_cpp::checkError(status, "Failed to create the input sparse weight buffer");

//        cl::Buffer inputPointerBuffer(
//                    clContext,
//                    CL_MEM_READ_WRITE,
//                    (CBPerRow+1) * MATRIX_ROWS * sizeof(typeof(cbOffsets.at(0))),
//                    NULL,
//                    &status
//                    );
//        aocl_utils_cpp::checkError(status, "Failed to create the input block pointer buffer");

//        cl::Buffer outputSpWBuffer(
//                    clContext,
//                    CL_MEM_READ_WRITE,
//                    MATRIX_ROWS * MATRIX_COLS * sizeof(typeof(outputEffectualValues.at(0))),
//                    NULL,
//                    &status
//                    );
//        aocl_utils_cpp::checkError(status, "Failed to create the output sparse weight buffer");

//        cl::Buffer inputInstructionBuffer (
//                    clContext,
//                    CL_MEM_READ_WRITE,
//                    instructionVector.size() * sizeof(t_instruction),
//                    NULL,
//                    &status
//                    );
//        aocl_utils_cpp::checkError(status, "Failed to create the input sparse weight buffer");

//        std::cout <<"Set up the kernel arguments"<<std::endl;
//        status = krnSequencer.setArg(0, inputInstructionBuffer);
//        status = krnSequencer.setArg(1, (cl_ushort) numInstructions);
//        aocl_utils_cpp::checkError(status, "Failed to set up arguments for the instruction sequencer");

//        for (auto iter=krnWeightCollectorVec.begin();
//             iter < krnWeightCollectorVec.end();
//             iter++) {
//            status = iter->setArg(0, outputSpWBuffer);
//        }
//        aocl_utils_cpp::checkError(status, "Failed to set up arguments for the weight collector");

//        status = krnSpWDMA.setArg(0, inputSpWBuffer);
//        status = krnSpWDMA.setArg(1, inputPointerBuffer);
//        aocl_utils_cpp::checkError(status, "Failed to set up arguments for the sparse weight DMA");

//        std::cout <<"Transer data to the accelerator"<<std::endl;
//        cl::Event inputTransferEvent, inputPointerTransferEvent, instructionTransferEvent;
//        status = clSequencerQueue.enqueueWriteBuffer(
//                    inputSpWBuffer,
//                    CL_TRUE,
//                    0,
//                    sizeof(typeof(effectualValues.at(0))) * effectualValues.size(),
//                    effectualValues.data(),
//                    NULL,
//                    &inputTransferEvent
//                    );

//        //CAUTION: Specail trick. Pollut the output region with all 4s, and see whether the FPGA pollute it.
//        status = clSequencerQueue.enqueueWriteBuffer(outputSpWBuffer,
//                                  CL_TRUE
//                                  ,0
//                                  ,sizeof(typeof(outputEffectualValues.at(0))) * outputEffectualValues.size()
//                                  , outputEffectualValues.data()
//                                  );

//        aocl_utils_cpp::checkError(status, "Failed to transfer sparse weights to the accelerator");
//        status = clSequencerQueue.enqueueWriteBuffer(
//                    inputPointerBuffer,
//                    CL_TRUE,
//                    0,
//                    sizeof(typeof(cbOffsets.at(0))) * cbOffsets.size(),
//                    cbOffsets.data(),
//                    NULL,
//                    &inputPointerTransferEvent
//                    );
//        aocl_utils_cpp::checkError(status, "Failed to transfer weight pointers to the accelerator");
//        status = clSequencerQueue.enqueueWriteBuffer(
//                    inputInstructionBuffer,
//                    CL_TRUE,
//                    0,
//                    sizeof(typeof(instructionVector.at(0))) * instructionVector.size(),
//                    instructionVector.data(),
//                    NULL,
//                    &instructionTransferEvent
//                    );
//        clSequencerQueue.finish();
//        std::cout <<"Number of instruction is "<<instructionVector.size()<<std::endl;
//        aocl_utils_cpp::checkError(status, "Failed to transfer instructions to the accelerator");

//        std::vector<cl::Event> transferEvents{
//            inputTransferEvent, inputPointerTransferEvent, inputPointerTransferEvent};

//        std::cout <<"Launching the kernels"<<std::endl;
//        for (unsigned int i = 0;
//             i < krnWeightCollectorVec.size();
//             i ++) {
//            status = vecClCollectorQueues.at(i).enqueueTask(
//                        krnWeightCollectorVec.at(i)
//                        );
//        }
//        status = clDMAQueue.enqueueTask(krnSpWDMA);

//        cl::Event sequencerEvent;
//        status = clSequencerQueue.enqueueTask(krnSequencer, NULL, &sequencerEvent);
//        aocl_utils_cpp::checkError(status, "Failed to launch at least one kernel");
//        //usleep (10000000);
//        std::cout <<"Wait for result to be transferred back"<<std::endl;
//        clSequencerQueue.finish();

//        cl::Event outputReadEvent;
//        status = clSequencerQueue.enqueueReadBuffer(outputSpWBuffer,
//                                  CL_TRUE
//                                  ,0
//                                  ,sizeof(typeof(outputEffectualValues.at(0))) * outputEffectualValues.size()
//                                  , outputEffectualValues.data()
//                                  , NULL
//                                  , &outputReadEvent
//                                  );
//       clSequencerQueue.finish();
//       aocl_utils_cpp::checkError(status, "Failed to read the results back");

//       //Report timing
//       cl_ulong inputSpWBufferWriteStart = inputTransferEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//       cl_ulong inputSpWBufferWriteEnd = inputTransferEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//       cl_double inputSpWBufferWriteTimeMs = (cl_double)(inputSpWBufferWriteEnd-inputSpWBufferWriteStart)*(cl_double)(1e-06);

//       cl_ulong inputSpWPointerBufferWriteStart = inputPointerTransferEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//       cl_ulong inputSpWPointerBufferWriteEnd = inputPointerTransferEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//       cl_double inputSpWPointerBufferWriteTimeMs = (cl_double)(inputSpWPointerBufferWriteEnd-inputSpWPointerBufferWriteStart)*(cl_double)(1e-06);

//       cl_ulong sequencerStart = sequencerEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//       cl_ulong sequencerEnd = sequencerEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//       cl_double sequencerTimeMs = (cl_double)(sequencerEnd-sequencerStart)*(cl_double)(1e-06);

//       cl_ulong outputReadStart = outputReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//       cl_ulong outputReadEnd = outputReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//       cl_double outputReadTimeMs = (cl_double)(outputReadEnd-outputReadStart)*(cl_double)(1e-06);

//       std::cout <<"============Timing information=================="<<std::endl;
//       std::cout <<"Input SpW Buffer Write (ms): "<<inputSpWBufferWriteTimeMs<<std::endl;
//       std::cout <<"Input SpW Pointer Buffer Write (ms): "<<inputSpWPointerBufferWriteTimeMs<<std::endl;
//       std::cout <<"Ouput SpW Read Time (ms): "<<outputReadTimeMs<<std::endl;
//       std::cout <<"Sequencer Execution Time (ms): "<<sequencerTimeMs<<std::endl;
//      }
//    catch (const std::runtime_error & e) {
//        std::cout <<e.what()<<std::endl;
//        return 1;
//    }
//    catch (...) {
//        std::cout <<"Unspecified error occured!"<<std::endl;
//        return 1;
//    }


    return 0;
}

cl_int clInit (const std::string binaryFile,
               cl::Platform &clPlatform,
               cl::Context & clContext,
               cl::Device & clDevice,
               cl::CommandQueue & clDMAQueue,
               cl::CommandQueue & clSequencerQueue,
               std::vector<cl::CommandQueue> & clCollectorQueue,
               cl::Kernel & krnSpWDMA,
               cl::Kernel & krnSequencer,
               std::vector<cl::Kernel> & krnWeightCollectorVec)
{
    cl_int status = CL_SUCCESS;
    clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    std::vector<cl::Device> devices;
    status = clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    aocl_utils_cpp::checkError(status, "Failed to query the devices");

    std::cout <<"Selecting the device[0]"<<std::endl;
    clDevice = devices[0];
    clContext = cl::Context({devices[0]}
                            ,NULL
                            ,&aocl_utils_cpp::oclContextCallback
                            ,NULL
                            ,&status);
    aocl_utils_cpp::checkError(status, "Failed to create context");

    clDMAQueue = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );

    clSequencerQueue = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );

    for (auto iter = clCollectorQueue.begin();
         iter != clCollectorQueue.end();
         iter++) {
        *iter = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
    }
    aocl_utils_cpp::checkError(status, "Failed to create at least one command queue");

    std::cout <<"Using AOCX: "<<binaryFile<<std::endl;
    cl::Program program = aocl_utils_cpp::createProgramFromBinary(
                clContext,
                binaryFile.c_str(),
                {clDevice}
                );
    status = program.build({clDevice});
    aocl_utils_cpp::checkError(status, "Failed to build program");

    //Instantiate the kernels
    krnSequencer = cl::Kernel(program, "kernelSequencer", &status);
    aocl_utils_cpp::checkError(status, "Failed to create the squencer kernel");

    krnSpWDMA = cl::Kernel(program, "kernelSparseWeightDMA", &status);
    aocl_utils_cpp::checkError(status, "Failed to create the sparse weight DMA kernel");

    for (unsigned int i=0; i<KERNEL_CACHE_LANES; i++){
        std::string kernelName = "kernelWeightCollector"+std::to_string(i);
        krnWeightCollectorVec.push_back(
                    cl::Kernel(program, kernelName.c_str(), &status)
                    );
    }
    aocl_utils_cpp::checkError(status, "Failed to create at least one of the weight collector kernel");

    return status;
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

void compress_vector (std::vector<float> &inputVector
                      , unsigned int encodingBlockSize
                      ,char intWidth
                      ,char fracWidth
                      ,std::vector<fixedPointNumber> &fixedPointVector
                      ,aligned_short_vector &compressedVector){
    //Pad zeros
    while (inputVector.size() % encodingBlockSize != 0) {
        inputVector.push_back(0.0f);
    }

    for (unsigned int iCompressedVector=0, iFullLengthVector=0, zOffset=0;
         iFullLengthVector < inputVector.size();
         iFullLengthVector++) {
        float origValue = inputVector.at(iFullLengthVector);
        fixedPointNumber fpValue(origValue, fracWidth, intWidth);
        fixedPointVector.at(iFullLengthVector) = fpValue;

        if (std::abs(origValue) > EPSILON
                || iFullLengthVector % encodingBlockSize == encodingBlockSize - 1
                || zOffset == ( (1 << WEIGHT_ZCOUNT_BITWIDTH) - 1)) {
            int value = fpValue.getBits();
            short shortValue =
               ( (zOffset << WEIGHT_ZCOUNT_BITOFFSET) & WEIGHT_ZCOUNT_MASK )
               | (value & fpValue.getMask());
            compressedVector.at(iCompressedVector) = shortValue;
            iCompressedVector++;
            zOffset=0;
        }
        else {
            zOffset++;
        }
    }
}

float dot_product_regular_vectors (std::vector<float> &inputVectorA
                                   ,std::vector<float> &inputVectorB) {
    float result = 0.0f;
    for (unsigned i=0; i<inputVectorA.size(); i++) {
        result += inputVectorA.at(i) * inputVectorB.at(i);
    }
    return result;
}

float dot_product_compressed_vectors (
        aligned_short_vector & compressedVectorA,
        aligned_short_vector & compressedVectorB,
        unsigned int numEncodingBlocks,
        unsigned int encodingBlockSize,
        char intWidth,
        char fracWidth
        )
{
    unsigned int indexVectorA=0, indexVectorB=0;
    auto iterVectorA = compressedVectorA.begin();
    auto iterVectorB = compressedVectorB.begin();
    unsigned int maxIndex = numEncodingBlocks * encodingBlockSize;
    float result = 0.0f;
    bool readA = true, readB = true;

    while (indexVectorA < maxIndex && indexVectorB < maxIndex) {
        short codeA = *iterVectorA;
        unsigned int offsetA = (codeA & WEIGHT_ZCOUNT_MASK) >> (WEIGHT_ZCOUNT_BITOFFSET);
        unsigned int bitsA = codeA & WEIGHT_MASK;
        fixedPointNumber fpA((int) bitsA, fracWidth, intWidth);
        float floatA = fpA.convert2Float();
        if (readA) {
            indexVectorA += (offsetA + 1);
        }

        short codeB = *iterVectorB;
        unsigned int offsetB = (codeB & WEIGHT_ZCOUNT_MASK) >> (WEIGHT_ZCOUNT_BITOFFSET);
        unsigned int bitsB = codeB & WEIGHT_MASK;
        fixedPointNumber fpB((int) bitsB, fracWidth, intWidth);
        float floatB = fpB.convert2Float();
        if (readB) {
            indexVectorB += (offsetB + 1);
        }

        if (indexVectorA == indexVectorB) {
            result += floatA * floatB;
            //indexVectorA += 1;
            iterVectorA++;
            //indexVectorB += 1;
            iterVectorB++;
            readA = true;
            readB = true;
        }
        else if (indexVectorA > indexVectorB) {
            //indexVectorB += 1;
            iterVectorB++;
            readB = true;
            readA = false;
        }
        else {
            //indexVectorA += 1;
            iterVectorA++;
            readA = true;
            readB = false;
        }
    }

    return result;
}

