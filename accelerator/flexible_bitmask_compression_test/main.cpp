#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/cl.hpp"
#include "aocl_utils_cpp.hpp"
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

/*!
 * \brief The testFixture class
 */

typedef
std::vector<cl_float, boost::alignment::aligned_allocator<cl_float, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
t_aligned_float_vector;

class testFixture : public ::testing::Test {
protected:
    void SetUp() override
    {} //Does nothing special

    void launch (
        char _fracWidth,
        char _intWidth,
        float _bernProb,
        int numTensors,
        int height,
        int width,
        int channel,
        int seed,
        float min,
        float max
        ,unsigned short maxScalarIndexInChannelGroup
        ,unsigned  char maxClusterIndexInCompressionBlock
        ,unsigned char maxClusterIndexInTransferBlock
        ,unsigned char maxScalarIndexInCluster
        ,bool flagIsKernel
        ,bool flagCompress
        );

    t_aligned_float_vector initialize_vector(unsigned seed,
                           float bernProb,
                           int numTensors,
                           int height,
                           int width,
                           int channel,
                           float min,
                           float max);
};

//#define PLAY


#ifdef PLAY
TEST_F (testFixture, playField) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.1;
    int numTensors = 1;
    int height = 1;
    int width = 2;
    int channel = 256;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = 15;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = false;
    bool flagCompress = false;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}
#else
TEST_F (testFixture, compressionTestKernel) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.05;
    int numTensors = 512;
    int height = 3;
    int width = 3;
    unsigned short tileSizeWidth = 32;
    int channel = 512;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = channel - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = true;
    bool flagCompress = true;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}

TEST_F (testFixture, normalTestKernel) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.05;
    int numTensors = 512;
    int height = 3;
    int width = 3;
    unsigned short tileSizeWidth = 32;
    int channel = 512;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = channel - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = true;
    bool flagCompress = false;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}


TEST_F (testFixture, compressionTestGroupedKernel) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.05;
    int numTensors = 256;
    int height = 3;
    int width = 3;
    unsigned short tileSizeWidth = 32;
    int channel = 256;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = channel - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = true;
    bool flagCompress = true;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}

TEST_F (testFixture, normalTestGroupedKernel) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.05;
    int numTensors = 256;
    int height = 3;
    int width = 3;
    unsigned short tileSizeWidth = 32;
    int channel = 256;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = channel - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = true;
    bool flagCompress = false;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}

TEST_F (testFixture, compressionTestGroupedActivation) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.1;
    int numTensors = 1;
    int height = 64;
    int width = 64;
    unsigned short tileSizeWidth = 32;
    int channel = 256;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = 15;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = false;
    bool flagCompress = true;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}

TEST_F (testFixture, normalTestGroupedActivation) {
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.1;
    int numTensors = 1;
    int height = 64;
    int width = 64;
    unsigned short tileSizeWidth = 32;
    int channel = 256;
    int seed = 1256;
    float min = 1.0;
    float max = 2.0;
    int numElements = numTensors * height * width * channel;

    unsigned short maxScalarIndexInChannelGroup = 15;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;
    bool flagIsKernel = false;
    bool flagCompress = false;

    this->launch(
           fracWidth
           ,intWidth
           ,bernProb
           ,numTensors
           ,height
           ,width
           ,channel
           ,seed
           ,min
           ,max
           ,maxScalarIndexInChannelGroup
           ,maxClusterIndexInCompressionBlock
           ,maxClusterIndexInTransferBlock
           ,maxScalarIndexInCluster
           ,flagIsKernel
           ,flagCompress
            );
}
#endif
int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

void testFixture::launch(
        char fracWidth
       ,char intWidth
       ,float bernProb
       ,int numTensors
       ,int height
       ,int width
       ,int channel
       ,int seed
       ,float min
       ,float max
       ,unsigned short maxScalarIndexInChannelGroup
       ,unsigned  char maxClusterIndexInCompressionBlock
       ,unsigned char maxClusterIndexInTransferBlock
       ,unsigned char maxScalarIndexInCluster
       ,bool flagIsKernel
       ,bool flagCompress
        )
{
    unsigned int numElements = numTensors*height*width*channel;

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

    std::cout <<"Start to prepare the input tensor."<<std::endl;
    std::cout <<"Compressor tensor: "<<flagCompress<<std::endl;



    AlignedTensor* pCompTensor;
    if (flagCompress) {
          pCompTensor = new FlexibleDirectCompressedTensor (
                    fpVector,
                    numTensors,
                    channel,
                    width,
                    height,
                    maxScalarIndexInChannelGroup,
                    maxClusterIndexInCompressionBlock,
                    maxClusterIndexInTransferBlock,
                    maxScalarIndexInCluster,
                    flagIsKernel
                );
           }
    else {
            pCompTensor = new AlignedTensor (
                    fpVector,
                    numTensors,
                    channel,
                    width,
                    height,
                    maxScalarIndexInChannelGroup,
                    maxClusterIndexInTransferBlock,
                    maxScalarIndexInCluster,
                    flagIsKernel
                    );
    }
    std::cout <<"Start to decode the tensor"<<std::endl;
    std::vector<fixedPointNumber> decodedFPVector;
    pCompTensor->decodeTensor(decodedFPVector, fracWidth, intWidth);

    std::cout <<"Comparing the decoded tensor and the original tensor"<<std::endl;
    for (int i=0; i<numElements; i++) {
        fixedPointNumber orig = fpVector.at(i);
        fixedPointNumber newValue = decodedFPVector.at(i);
        EXPECT_TRUE(
             (orig.getBits() == newValue.getBits()))
             << "Actual output: "<<std::bitset<WEIGHT_BITWIDTH>(newValue.getBits() & WEIGHT_MASK)
             <<std::endl<<"Expected output: "<<std::bitset<WEIGHT_BITWIDTH>((orig.getBits()) & WEIGHT_MASK)<<std::endl;
    }

    delete pCompTensor;
}

t_aligned_float_vector testFixture::initialize_vector(unsigned seed,
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
