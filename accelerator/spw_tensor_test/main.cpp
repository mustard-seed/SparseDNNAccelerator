#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#include "params.hpp"
#include "device_structures.hpp"
#include "floatFixedPointConversion.hpp"
#include "spwTensorCompression.hpp"

#define DIVIDE_CEIL(x, y) (1 + (x-1) / y)

#define SEED 27
#define FRAC_WIDTH 4
#define INT_WIDTH 3

class spwTensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {}

    void launch_activation_test (
            int _height,
            int _width,
            int _inputChannels,
            int _stripStrideInExternalMemory
            );

    void launch_dense_weight_test (
            int _outputChannels,
            int _height,
            int _width,
            int _inputChannels,
            int _peSimdSize,
            int _clusterSize
            );

    std::vector <signed char> generateRandomVector (
                int _numElements
            );

    bool compareTensor (
            std::vector<fixedPointNumber> _expectedVector,
            std::vector<fixedPointNumber> _decodedVector
            );

#if defined(SPW_SYSTEM)
    std::vector <signed char> generate_spw_compression_window (
                std::mt19937& randGenerator,
                int _clusterSize,
                int _numClustersInPruningRange,
                int _numNZClustersPerPruningRange
            );

    void launch_sparse_weight_test (
            int _outputChannel,
            int _inputChannel,
            int _width,
            int _height,
            int _peSimdSize,
            int _clusterSize,
            int _numClustersInPruningRange,
            int _numNZClustersPerPruningRange
            );
#endif //SPW_SYSTEM
};

TEST_F(spwTensorTest, activation_test)
{
    //Activation test
    int height = 4;
    int width = 4;
    int inputChannels = 13;
    int stripStride = 16;

    launch_activation_test(
                height,
                width,
                inputChannels,
                stripStride
                );
}

TEST_F(spwTensorTest, dense_filter_test)
{
    //Dense filter test
    int outputChannels = 13;
    int height = 4;
    int width = 4;
    int inputChannels = 13;
    int peSimdSize = PE_SIMD_SIZE;
    int clusterSize = CLUSTER_SIZE;

    launch_dense_weight_test (
        outputChannels,
        height,
        width,
        inputChannels,
        peSimdSize,
        clusterSize
        );
}

#if defined(SPW_SYSTEM)
TEST_F(spwTensorTest, spw_filter_test)
{
    //SpW filter test
    int outputChannels = 13;
    int height = 4;
    int width = 4;
    int inputChannels = 13;
    int peSimdSize = PE_SIMD_SIZE;
    int clusterSize = CLUSTER_SIZE;
    int numClustersInPruningRange = 4;
    int numNZClustersInPruningRange = 2;
//    int outputChannels = 1;
//    int height = 1;
//    int width = 1;
//    int inputChannels = 13;
//    int peSimdSize = PE_SIMD_SIZE;
//    int clusterSize = CLUSTER_SIZE;
//    int numClustersInPruningRange = 4;
//    int numNZClustersInPruningRange = 2;


    launch_sparse_weight_test (
        outputChannels,
        inputChannels,
        height,
        width,
        peSimdSize,
        clusterSize,
        numClustersInPruningRange,
        numNZClustersInPruningRange
        );
}
#endif


int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

void
spwTensorTest::launch_activation_test (
            int _height,
            int _width,
            int _inputChannels,
            int _stripStrideInExternalMemory
            )
{
    std::cout <<"Running activation tensor test"<<std::endl;
    std::cout <<"[Height, Width, IC, Strip stride]: "
             <<_height<<" "<<_width<<" "<<_inputChannels<<" "<<_stripStrideInExternalMemory<<std::endl;
    if (_stripStrideInExternalMemory < _inputChannels)
    {
        std::cout <<"Strip stride in external memory cannot be less than the input channel size."<<std::endl;
        throw;
    }
    int tensorSize = _height * _width * _inputChannels;
    std::vector <signed char> rawVector = generateRandomVector(tensorSize);
    fixedPointNumber emptyFpValue((signed char) 0x0, FRAC_WIDTH, INT_WIDTH);
    std::vector <fixedPointNumber> expectedFpVector(tensorSize, emptyFpValue);
    for (int i=0; i<tensorSize; i++)
    {
        expectedFpVector.at(i) = fixedPointNumber(rawVector.at(i), FRAC_WIDTH, INT_WIDTH);
    }

    DeviceActivationTensor deviceTensor(
                    expectedFpVector,
                    _inputChannels,
                    _width,
                    _height,
                    _stripStrideInExternalMemory
                );

    std::vector <fixedPointNumber> decodedTensor = deviceTensor.decodeTensor(FRAC_WIDTH, INT_WIDTH);

    compareTensor(expectedFpVector, decodedTensor);
}

void
spwTensorTest::launch_dense_weight_test (
        int _outputChannels,
        int _height,
        int _width,
        int _inputChannels,
        int _peSimdSize,
        int _clusterSize
        )
{
    std::cout <<"Running dense filter test"<<std::endl;
    std::cout <<"outputChannels: "<<_outputChannels<<std::endl;
    std::cout <<"height: "<<_height<<std::endl;
    std::cout <<"width: "<<_width<<std::endl;
    std::cout <<"inputChannels: "<<_inputChannels<<std::endl;
    std::cout <<"peSimdSize: "<<_peSimdSize<<std::endl;
    std::cout <<"WEIGHT_WIDE_SIZE: "<<WEIGHT_WIDE_SIZE<<std::endl;
    std::cout <<"WEIGHT_DRAM_SIZE_GEQ_PE_SIZE: "<<WEIGHT_DRAM_SIZE_GEQ_PE_SIZE<<std::endl;
    std::cout <<"clusterSize: "<<_clusterSize<<std::endl;

    int tensorSize = _outputChannels * _height * _width * _inputChannels;
    std::vector <signed char> rawVector = generateRandomVector(tensorSize);
    fixedPointNumber emptyFpValue((signed char) 0x0, FRAC_WIDTH, INT_WIDTH);
    std::vector <fixedPointNumber> expectedFpVector(tensorSize, emptyFpValue);
    for (int i=0; i<tensorSize; i++)
    {
        expectedFpVector.at(i) = fixedPointNumber(rawVector.at(i), FRAC_WIDTH, INT_WIDTH);
    }

    DeviceWeightTensor deviceTensor(
                    expectedFpVector,
                    _outputChannels,
                    _inputChannels,
                    _width,
                    _height,
                    _peSimdSize,
                    _clusterSize
                );

    std::vector <fixedPointNumber> decodedTensor = deviceTensor.decodeTensor(FRAC_WIDTH, INT_WIDTH);

    compareTensor(expectedFpVector, decodedTensor);
}

bool
spwTensorTest::compareTensor (
        std::vector<fixedPointNumber> _expectedVector,
        std::vector<fixedPointNumber> _decodedVector
        )
{
    int expectedSize = _expectedVector.size();
    int decodedSize = _decodedVector.size();
    EXPECT_TRUE(expectedSize == decodedSize)
            <<"Size of the decoded vector does not match the expectation. "
            <<"Actual: "<<decodedSize<<". Expected: "<<expectedSize<<std::endl;
    for (int i=0; i<decodedSize; i++)
    {
        signed char expectedValue = _expectedVector.at(i).getBits();
        signed char actualValue = _decodedVector.at(i).getBits();
        EXPECT_TRUE(expectedValue == actualValue)
                <<"Actual value does not match expected value. \n"
                <<"i: "<<i
                <<" Actual: "<<(int) actualValue<<". Expected: "<<(int)expectedValue<<std::endl;
    }

    return true;
}

std::vector <signed char>
spwTensorTest:: generateRandomVector (
            int _numElements
        )
{
    std::mt19937 randGenerator;
    std::uniform_int_distribution<signed char> distribution(-3, 3);
    std::vector<signed char> values(_numElements, 0x0);
    for (auto & val : values)
    {
        val = distribution(randGenerator);
    }
    return values;
}

#if defined(SPW_SYSTEM)
std::vector <signed char>
spwTensorTest::generate_spw_compression_window(
        std::mt19937 &randGenerator,
        int _clusterSize,
        int _numClustersInPruningRange,
        int _numNZClustersPerPruningRange)
{
    std::uniform_int_distribution<int> distribution(-3,3);
    int windowSize = _clusterSize * _numClustersInPruningRange;
    if (_numNZClustersPerPruningRange > _numClustersInPruningRange)
    {
        std::cout <<"Number of none-zero clusters in a pruning range cannot exceed "
                 <<"the number of clusters in a pruning range."<<std::endl;
        throw;
    }
    std::vector <signed char> compressionWindow(windowSize, 0x0);

    std::vector<int> idxClusters (_numClustersInPruningRange, 0x0);
    std::iota(idxClusters.begin(), idxClusters.end(), 0);
    std::shuffle(idxClusters.begin(), idxClusters.end(), randGenerator);
    for (int i=0; i<_numNZClustersPerPruningRange; i++)
    {
        int idx = idxClusters.at(i);
        for (int c=0; c<_clusterSize; c++)
        {
            int vecIdx = idx*_clusterSize + c;
            compressionWindow.at(vecIdx) = (signed char) distribution(randGenerator);
        }
    }

    return compressionWindow;
}

void
spwTensorTest::launch_sparse_weight_test(
        int _outputChannel,
        int _inputChannel,
        int _width,
        int _height,
        int _peSimdSize,
        int _clusterSize,
        int _numClustersInPruningRange,
        int _numNZClustersPerPruningRange)
{
    std::cout <<"Running spw filter test"<<std::endl;
    std::cout <<"outputChannels: "<<_outputChannel<<std::endl;
    std::cout <<"height: "<<_height<<std::endl;
    std::cout <<"width: "<<_width<<std::endl;
    std::cout <<"inputChannels: "<<_inputChannel<<std::endl;
    std::cout <<"peSimdSize: "<<_peSimdSize<<std::endl;
    std::cout <<"clusterSize: "<<_clusterSize<<std::endl;
    std::cout <<"WEIGHT_WIDE_SIZE: "<<WEIGHT_WIDE_SIZE<<std::endl;
    std::cout <<"WEIGHT_DRAM_SIZE_GEQ_PE_SIZE: "<<WEIGHT_DRAM_SIZE_GEQ_PE_SIZE<<std::endl;
    std::cout <<"numClustersInPruningRange: "<<_numClustersInPruningRange<<std::endl;
    std::cout <<"numNZClustersPerPruningRange: "<<_numNZClustersPerPruningRange<<std::endl;

    int tensorSize = _outputChannel * _height * _width * _inputChannel;
    int sizePruningRange = _clusterSize * _numClustersInPruningRange;
    int numPruningRangePerStrip = DIVIDE_CEIL(_inputChannel, sizePruningRange);
    int numStrips = _outputChannel * _height * _width;

    //Prepare the fixed point tensor
    fixedPointNumber emptyFpValue((signed char) 0x0, FRAC_WIDTH, INT_WIDTH);
    std::vector<fixedPointNumber> expectedFpVector(tensorSize, emptyFpValue);
    std::mt19937 randGenerator(SEED);
    for (int s=0; s<numStrips; s++)
    {
        for (int p=0; p<numPruningRangePerStrip; p++)
        {
            std::vector<signed char> rawCompressionWindow =
                    generate_spw_compression_window(
                            randGenerator,
                            _clusterSize,
                            _numClustersInPruningRange,
                            _numNZClustersPerPruningRange
                        );

            for (int i=0; i<rawCompressionWindow.size(); i++)
            {
                int idxIC = p*sizePruningRange + i;
                if (idxIC < _inputChannel)
                {
                    int idxFpVector = s * _inputChannel + idxIC;
                    expectedFpVector.at(idxFpVector) = fixedPointNumber(
                                    rawCompressionWindow.at(i),
                                    FRAC_WIDTH,
                                    INT_WIDTH
                                );
                }
            } //for (int i=0; i<rawCompressionWindow.size(); i++)
        } //for (int p=0; p<numPruningRangePerStrip; p++)
    } //for (int s=0; s<numStrips; s++)

    DeviceSpWTensor deviceTensor(
                    expectedFpVector,
                    _outputChannel,
                    _inputChannel,
                    _width,
                    _height,
                    _peSimdSize,
                    _clusterSize,
                    _numClustersInPruningRange,
                    _numNZClustersPerPruningRange
                );

    std::vector<fixedPointNumber> decodedTensor = deviceTensor.decodeTensor(FRAC_WIDTH, INT_WIDTH);

    compareTensor(expectedFpVector, decodedTensor);
}
#endif
