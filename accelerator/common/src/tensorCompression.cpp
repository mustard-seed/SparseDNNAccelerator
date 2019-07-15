#include "tensorCompression.hpp"
#include "params.hpp"
#include <cmath> //std::ceil
#include <iostream> //cout

compressedTensor::compressedTensor(std::vector<fixedPointNumber> &fixedPointVector
                                   ,unsigned short _num3DTensors
                                   ,unsigned short _channel
                                   ,unsigned short _width
                                   ,unsigned short _height
                                   ,unsigned short _streamingBlockSize
                                   ,unsigned short _simdBlockSize
                                   ,unsigned short _extMemoryRowAddressStride
                                   ,unsigned short _numBanks) {
    num3DTensors = _num3DTensors;
    channel = _channel;
    width = _width;
    height = _height;
    streamingBlockSize = _streamingBlockSize;
    simdBlockSize = _simdBlockSize;
    externalMemoryRowAddressStride = _extMemoryRowAddressStride;
    numBanks = _numBanks;

    valueVector.resize(height * externalMemoryRowAddressStride);
    channelOffsetVector.resize(height * externalMemoryRowAddressStride);
    //streamBlockAddressVector.resize(
    //            height*width* ((int) std::ceil(
    //                ((float) channel) / ((float) simdBlockSize * (float) streamingBlockSize)
    //                ))
    //            );

    int iCompressVectorBase = 0;
    for (int iTensor=0, iFullVector=0; iTensor < num3DTensors; iTensor++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            int iCompressVector = iCompressVectorBase;
            unsigned short countNumSimdBlockStored = 0;
            //Index of the external memory location
            for (int iWidth=0; iWidth < width; iWidth++) {
                t_simdblock_value compressionBlock;
                bool retainFlag = false;
                //Number of uncompressed simdblocks in the current streaming block
                int iChannelOffset = 0;
                //Number of scar within the current simd block
                int simdCount = 0;
                //Depth of the BRAM port we are updating;
                //unsigned short addressDepth = streamBlockAddressTracker.at(iWidth % numBanks);
                for (int iChannel=0;
                     iChannel < channel;
                     iChannel++, iFullVector++) {
                    fixedPointNumber fpNumber = fixedPointVector.at(iFullVector);
                    char fpValue = ( (fpNumber.getBits()) & (fpNumber.getMask()) );
                    //If at least one value needs to be retained, then the entire block is retained.
                    retainFlag = (fpValue != 0x0) || retainFlag;
                    compressionBlock.values[simdCount++] = fpValue;

                    //If we are at the end of the channel, at a simd block is yet formed
                    //then we need to pad 0s
                    if ( iChannel == (channel - 1) ) {
                           while (simdCount < simdBlockSize) {
                               compressionBlock.values[simdCount++] = 0x0;
                            }
                    }

                    //Update the simdCound
                    simdCount =  (simdCount == simdBlockSize) ? 0 : simdCount;

                    //If a simd block has been formed, and it
                    // a. contains at least one non-zero element OR
                    // b. is the last simd block in a streaming block OR
                    // c. is the last simd block in the channel
                    // Then we need to store it.
                    // If we come across case b or c, we also need to mark the block as "last"

                    bool storeSimdBlock =
                            (simdCount == 0) && (retainFlag || (iChannelOffset == (streamingBlockSize - 1) ) || (iChannel == (channel - 1)) ) ?
                                true : false;
                    unsigned char simdBlocIsLast =
                           (simdCount == 0) && ((iChannelOffset == (streamingBlockSize - 1) ) || (iChannel == (channel - 1)) ) ?
                                0x1 : 0x0;

                    if (storeSimdBlock) {
                        valueVector.at(iCompressVector) = compressionBlock;
                        char value = (
                                    ((iChannelOffset & CHANNEL_OFFSET_MASK) << CHANNEL_OFFSET_BITOFFSET)
                                    | ((simdBlocIsLast & IS_LAST_BLOCK_MASK) << IS_LAST_BLOCK_BITOFFSET)
                                    );
                        channelOffsetVector.at(iCompressVector+2) = value;
                        iCompressVector++;
                        countNumSimdBlockStored++;
                        retainFlag = false;
                    }

                    //Update the iChannelOffset
                    if (simdCount == 0) {
                        iChannelOffset = (iChannelOffset == streamingBlockSize - 1)
                                ? 0 : (iChannelOffset+1);
                    }

                } // for channel
            } // for width
            channelOffsetVector.at(iHeight * externalMemoryRowAddressStride) = (countNumSimdBlockStored & 0xFF);
            channelOffsetVector.at(iHeight * externalMemoryRowAddressStride + 1) = ((countNumSimdBlockStored & 0xFF00) >> 8);
            iCompressVectorBase += externalMemoryRowAddressStride;
        } // for height
     } // for Tensor
}

int decodeTensor(compressedTensor compTensor, std::vector<float> & denseTensor, char fracWidth, char intWidth) {
    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors = compTensor.num3DTensors;
    unsigned short channel = compTensor.channel;
    unsigned short width = compTensor.width;
    unsigned short height = compTensor.height;

    //The following parameters should be compatiable with the hardware
    //Number of uncompressed simdblocks in a streaming block
    //TODO: Should this value should match the number of PE rows?
    unsigned short streamingBlockSize = compTensor.streamingBlockSize;

    //Number of uncompressed scalar value in each simdblock;
    unsigned short simdBlockSize = compTensor.simdBlockSize;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryRowAddressStride = compTensor.externalMemoryRowAddressStride;

    //Allocate space for the dense tensor
    denseTensor.resize(num3DTensors*channel*width*height);
    int iDenseTensor = 0;
    int iCompressVectorBase = 0;

    int numSimdBlock = 0;
    for (int iTensor=0; iTensor < num3DTensors; iTensor++) {
        for (int iH = 0; iH < height; iH++) {
            int iCompressVector = iCompressVectorBase;
            unsigned short numCompressedBlocks =
                    (unsigned short) compTensor.channelOffsetVector.at(iCompressVector)
                    | (((unsigned short) (compTensor.channelOffsetVector.at(iCompressVector + 1))) << 8);
            int iDenseTensorChannel = 0;
            std::vector<float> streamingBlock (streamingBlockSize*simdBlockSize, 0.0f);
            for (int iSimdBlocks = 0; iSimdBlocks < numCompressedBlocks; iSimdBlocks++) {
                char channelOffsetBlob = compTensor.channelOffsetVector.at(iCompressVector+2);
                bool isLastBlock = (((channelOffsetBlob >> IS_LAST_BLOCK_BITOFFSET) & IS_LAST_BLOCK_MASK) == 0x1) ?
                            true : false;
                char channelOffset = ((channelOffsetBlob >> CHANNEL_OFFSET_BITOFFSET) & CHANNEL_OFFSET_MASK);
                t_simdblock_value compressionBlock = (compTensor.valueVector).at(iCompressVector);
                for (int i=0, idx=channelOffset*simdBlockSize; i<simdBlockSize; i++, idx++) {
                    fixedPointNumber fpValue ((short) compressionBlock.values[i], fracWidth, intWidth);
                    streamingBlock.at(idx) = fpValue.convert2Float();
                }
                iCompressVector++;

                //Update the output vector if necessary
                if (isLastBlock) {
                    for (int ii=0; ii<streamingBlockSize*simdBlockSize; ii++) {
                        if (iDenseTensorChannel < channel) {
                           denseTensor.at(iDenseTensor++) = streamingBlock.at (ii);
                           iDenseTensorChannel++;
                        }
                    }
                    streamingBlock.clear();
                    streamingBlock.resize(streamingBlockSize*simdBlockSize, 0.0f);
                }

                if (iDenseTensorChannel == channel) {
                    iDenseTensorChannel = 0;
                }
                //std::cout <<"Decoded simd block "<<iSimdBlocks<<" in row "<<iH<<std::endl;
            } // for surviving simd blocks
            numSimdBlock += numCompressedBlocks;
            iCompressVectorBase += externalMemoryRowAddressStride;
        } // for height
    } // for tensors
    std::cout <<"Number of simd blocks survived "<<numSimdBlock<<std::endl;
    std::cout <<"Number of dense value written: "<<iDenseTensor<<std::endl;
}
