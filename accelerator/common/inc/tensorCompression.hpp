#ifndef _TENSOR_COMPRESSION_HPP_
#define _TENSOR_COMPRESSION_HPP_
#include "vectorType.hpp"
#include "floatFixedPointConversion.hpp"

class AlignedTensor {
  protected:
    //Vectos holding the vector values;
    t_aligned_transfer_block_vector valueVector;

    //Dimension of the uncompressed, un-flattened tensor
    unsigned short num3DTensors;
    unsigned short channel;
    unsigned short width;
    unsigned short height;

    //If is Kernel, each filter is one streaming block
    //Otherwise, the size of each streaming block is that of one synchorinization block
    bool isKernel;

    //The max index (starts from 0) of a scalar inside a channel group
    //For filter, this will be set to the number of channels minus 1
    unsigned short maxScalarIndexInChannelGroup;

    //Number of groups in a channel
    unsigned char numChannelGroups;

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

  public:
    AlignedTensor () = delete;

    //Constructor used to initialize, but not populate, a dense aligned tensor
    AlignedTensor (
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned short _maxScalarIndexInChannelGroup,
            unsigned char _maxClusterIndexInTransferBlock,
            unsigned char _maxScalarIndexInCluster,
            bool _isKernel
            );

    //Constructor for converting a dense vector
    AlignedTensor (std::vector<fixedPointNumber> & fixedPointVector,
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned short _maxScalarIndexInChannelGroup,
            unsigned char _maxClusterIndexInTransferBlock,
            unsigned char _maxScalarIndexInCluster,
            bool _isKernel
            );

    //Decode the aligned tenor values into a compact tensor of fixed-point numbers
    virtual void decodeTensor (
            std::vector<fixedPointNumber> & fixedPointVector,
            char _fracWidth,
            char _intWidth
            );

    t_aligned_transfer_block_vector& getTransferBlockVector();

    virtual t_aligned_streamblock_address_vector& getTransferBlockCountVector();

    unsigned int getExternalMemoryAddressStride();




};

class FlexibleDirectCompressedTensor : public AlignedTensor {
protected:
    //Vectos holding the BRAM addresses of each streamBlock
    t_aligned_streamblock_address_vector streamBlockAddressVector;

    //The maximum index (starts from 0) of a cluster found inside a compression blcok
    unsigned char maxClusterIndexInCompressionBlock;

    //The maximum scalar index (0) inside a compression block. Derived from the other values
    unsigned char maxScalarIndexInCompressionBlock;

public:
    //Allow the default constructor to stay
    FlexibleDirectCompressedTensor() = delete;

    //Constructor for converting a dense vector
    FlexibleDirectCompressedTensor (std::vector<fixedPointNumber> & fixedPointVector,
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned short _maxScalarIndexInChannelGroup,
            unsigned char _maxClusterIndexInCompressionBlock,
            unsigned char _maxClusterIndexInTransferBlock,
            unsigned char _maxScalarIndexInCluster,
            bool _isKernel
            );

    //Constructor for initialize a sparse vector
    FlexibleDirectCompressedTensor (
            unsigned short _num3DTensors,
            unsigned short _channel,
            unsigned short _width,
            unsigned short _height,
            unsigned short _maxScalarIndexInChannelGroup,
            unsigned char _maxClusterIndexInCompressionBlock,
            unsigned char _maxClusterIndexInTransferBlock,
            unsigned char _maxScalarIndexInCluster,
            bool _isKernel
            );

    void decodeTensor (
                std::vector<fixedPointNumber> & _fixedPointVector,
                char _fracWidth,
                char _intWidth
                ) override;

    t_aligned_streamblock_address_vector& getTransferBlockCountVector() override;

};
#endif
