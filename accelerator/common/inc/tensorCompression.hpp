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
    t_aligned_streamblock_address_vector streamBlockAddressVector;
    //The BRAM addresses of each streamBlock can be computed on the fly in the kernel!
    //t_aligned_streamblock_address_vector streamBlockAddressVector;

    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors;
    unsigned short channel;
    unsigned short width;
    unsigned short height;

    //This flag controls the external memory address stride, and
    //the streaming block size.
    //If is Kernel, each filter is one streaming block
    //Otherwise, the size of each streaming block is that of one synchorinization block
    bool isKernel;

    //The max index (starts from 0) of a simd block found inside a stream block
    unsigned short maxSimdBlockIndexInStreamBlock;

    //The max index (starts from 0) of a simd block found inside a synchornization block
    unsigned char maxSimdBlockIndexInSyncBlock;

    //The maximum index (starts from 0) of a scalar found inside a simd blcok
    //Equal to the size of simd block minus 1
    unsigned char maxScalarIndexInSimdBlock;

    //Word stride between the start of contiguous compression region in the external memory
    //In the case of kernel tensor, the contiguous compression region is sized at the number of words in the
    //uncompressed kernel
    //In the case of input/output activation tensor, the contiguous compression region is sized at
    //page_size * ceil (page_size / (channel * row))
    unsigned int externalMemoryAddressStride;

    //Number of banks per 3D tensor. The strips will be distributed across the banks
    //in a interleaved fashion
    unsigned char numBanks;

    //Allow the default constructor to stay
    compressedTensor() {}

    //In practice, use the following constructor
    compressedTensor (std::vector<fixedPointNumber> & fixedPointVector,
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned char _maxSimdBlockIndexInStreamBlock,
            unsigned char _maxScalarIndexInSimdBlock,
            unsigned short _numBanks,
            bool _isKernel
            );



};

class directCompressedTensor {
private:

public:
    //Vectos holding the nz simdblocks, the relative channel offsets,
    //and the BRAM addresses of each streamBlock
    t_aligned_simd_value_vector valueVector;
    t_aligned_streamblock_address_vector streamBlockAddressVector;
    //The BRAM addresses of each streamBlock can be computed on the fly in the kernel!
    //t_aligned_streamblock_address_vector streamBlockAddressVector;

    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors;
    unsigned short channel;
    unsigned short width;
    unsigned short height;

    //This flag controls the external memory address stride, and
    //the streaming block size.
    //If is Kernel, each filter is one streaming block
    //Otherwise, the size of each streaming block is that of one synchorinization block
    bool isKernel;

    //The max index (starts from 0) of a simd block found inside a stream block
    unsigned short maxSimdBlockIndexInStreamBlock;

    //The max index (starts from 0) of a simd block found inside a synchornization block
    unsigned char maxSimdBlockIndexInSyncBlock;

    //The maximum index (starts from 0) of a scalar found inside a simd blcok
    //Equal to the size of simd block minus 1
    unsigned char maxScalarIndexInSimdBlock;

    //Word stride between the start of contiguous compression region in the external memory
    //In the case of kernel tensor, the contiguous compression region is sized at the number of words in the
    //uncompressed kernel
    //In the case of input/output activation tensor, the contiguous compression region is sized at
    //page_size * ceil (page_size / (channel * row))
    unsigned int externalMemoryAddressStride;

    //Allow the default constructor to stay
    directCompressedTensor() {}

    //In practice, use the following constructor
    directCompressedTensor (std::vector<fixedPointNumber> & fixedPointVector,
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned char _maxSimdBlockIndexInStreamBlock,
            unsigned char _maxScalarIndexInSimdBlock,
            bool _isKernel
            );



};

//Helper function used to decode a compressed tensor
int decodeTensor(compressedTensor compTensor, std::vector<float> & denseTensor, char fracWidth, char intWidth);

//Helper function used to decode a compressed tensor
int decodeDirectCompressedTensor(directCompressedTensor compTensor, std::vector<float> & denseTensor, char fracWidth, char intWidth);

#ifdef FLEXIBLE_BITMASK_COMPRESSION
class flexibleDirectCompressedTensor {
private:

public:
    //Vectos holding the nz simdblocks, the relative channel offsets,
    //and the BRAM addresses of each streamBlock
    t_aligned_transfer_block_vector valueVector;
    t_aligned_streamblock_address_vector streamBlockAddressVector;
    //The BRAM addresses of each streamBlock can be computed on the fly in the kernel!
    //t_aligned_streamblock_address_vector streamBlockAddressVector;

    //Dimension of the uncompressed, un-flattened tensor
    unsigned short num3DTensors;
    unsigned short channel;
    unsigned short width;
    unsigned short height;

    // Tiling information, only relevant for activation tensor
    unsigned short tilingSizeWidth;

    //If is Kernel, each filter is one streaming block
    //Otherwise, the size of each streaming block is that of one synchorinization block
    bool isKernel;

    //The max index (starts from 0) of a scalar inside a channel group
    //Not important for kernel filters
    unsigned short maxScalarIndexInChannelGroup;


    //The maximum index (starts from 0) of a cluster found inside a compression blcok
    unsigned char maxClusterIndexInCompressionBlock;

    //The maximum scalar index (0) inside a compression block. Derived from the other values
    unsigned char maxScalarIndexInCompressionBlock;

    //The maximum index (starts from 0) of a suriving cluster found inside a transfer block
    unsigned char maxClusterIndexInTransferBlock;

    //The maximum index of a scalar in a cluster (size of cluster - 1)
    unsigned char maxScalarIndexInCluster;

    //Word stride between the start of contiguous compression region in the external memory
    //In the case of kernel tensor, the contiguous compression region is sized at the number of words in the
    //uncompressed kernel
    //In the case of input/output activation tensor, the contiguous compression region is sized at
    //page_size * ceil (page_size / (channel * row))
    unsigned int externalMemoryAddressStride;

    //Allow the default constructor to stay
    flexibleDirectCompressedTensor() {}

    //In practice, use the following constructor
    flexibleDirectCompressedTensor (std::vector<fixedPointNumber> & fixedPointVector,
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned short _tilingSizeWidth,
            unsigned short _maxScalarIndexInChannelGroup,
            unsigned char _maxClusterIndexInCompressionBlock,
            unsigned char _maxClusterIndexInTransferBlock,
            unsigned char _maxScalarIndexInCluster,
            bool _isKernel
            );



};

//Helper function used to decode a compressed tensor
int decodeFlexibleDirectCompressedTensor(
        flexibleDirectCompressedTensor compTensor
        ,std::vector<float> & denseTensor
        ,char fracWidth
        ,char intWidth);
#endif

#endif
