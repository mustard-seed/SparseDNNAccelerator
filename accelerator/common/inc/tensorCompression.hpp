#ifndef _TENSOR_COMPRESSION_HPP_
#define _TENSOR_COMPRESSION_HPP_
#include "vectorType.hpp"
#include "floatFixedPointConversion.hpp"

class compressedTensor {
private:

public:
    //Vectos holding the nz simdblocks, the relative channel offsets,
    //and the BRAM addresses of each streamBlock
    t_aligned_simd_value_vector valueVector;
    t_aligned_channel_offset_vector channelOffsetVector;
    //The BRAM addresses of each streamBlock can be computed on the fly in the kernel!
    //t_aligned_streamblock_address_vector streamBlockAddressVector;

    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors;
    unsigned short channel;
    unsigned short width;
    unsigned short height;

    //The following parameters should be compatiable with the hardware
    //Number of uncompressed simdblocks in a streaming block
    //TODO: Should this value should match the number of PE rows?
    unsigned short streamingBlockSize;

    //Number of uncompressed scalar value in each simdblock;
    unsigned short simdBlockSize;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryRowAddressStride;

    //Number of banks per 3D tensor. The strips will be distributed across the banks
    //in a interleaved fashion
    unsigned char numBanks;

    //Allow the default constructor to stay
    compressedTensor() {}

    //In practice, use the following constructor
    compressedTensor (
            std::vector<fixedPointNumber> & fixedPointVector,
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned short _streamingBlockSize,
            unsigned short _simdBlockSize,
            unsigned short _extMemoryRowAddressStride,
            unsigned short _numBanks
            );



};

//Helper function used to decode a compressed tensor
int decodeTensor(compressedTensor compTensor, std::vector<float> & denseTensor, char fracWidth, char intWidth);
#endif
