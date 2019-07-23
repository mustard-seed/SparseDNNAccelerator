#include "tensorCompression.hpp"
#include "params.hpp"
#include <cmath> //std::ceil
#include <iostream> //cout
#include <cassert>

unsigned char countNumLeadingZeros (unsigned char bitmask);

compressedTensor::compressedTensor (std::vector<fixedPointNumber> & fixedPointVector,
                                    unsigned short _num3DTensors,
                                    unsigned short _channel,
                                    unsigned short _width,
                                    unsigned short _height,
                                    unsigned char _maxSimdBlockIndexInStreamBlock,
                                    unsigned char _maxScalarIndexInSimdBlock,
                                    unsigned short _numBanks,
                                    bool _isKernel
                                    ) {
    num3DTensors = _num3DTensors;
    channel = _channel;
    width = _width;
    height = _height;
    maxScalarIndexInSimdBlock = _maxScalarIndexInSimdBlock;
    numBanks = _numBanks;
    isKernel = _isKernel;
    unsigned short numSimdBlocksInChannel =
            (unsigned short) std::ceil ( (float) channel / (float) (maxScalarIndexInSimdBlock + 1) );
    externalMemoryAddressStride = isKernel ? width * height * numSimdBlocksInChannel * num3DTensors : width * numSimdBlocksInChannel;

    maxSimdBlockIndexInStreamBlock = isKernel ?
                height * width * (unsigned short) std::ceil ( (float) channel / (float) (maxScalarIndexInSimdBlock + 1) ) - 1
              : _maxSimdBlockIndexInStreamBlock;
    maxSimdBlockIndexInSyncBlock = std::min(maxSimdBlockIndexInStreamBlock, (unsigned short) MAX_SIMD_BLOCK_INDEX);


    if (isKernel) {
        valueVector.resize(externalMemoryAddressStride);
        channelOffsetVector.resize(externalMemoryAddressStride);
        streamBlockAddressVector.resize(num3DTensors);
     }
    else {
        valueVector.resize(height * externalMemoryAddressStride);
        channelOffsetVector.resize(height * externalMemoryAddressStride);
        assert( numSimdBlocksInChannel % (_maxSimdBlockIndexInStreamBlock + 1) == 0 );
        unsigned int numStreamBlocksPerChannel = numSimdBlocksInChannel / (_maxSimdBlockIndexInStreamBlock + 1);
        streamBlockAddressVector.resize(num3DTensors * width * height * numStreamBlocksPerChannel );
    }
    //streamBlockAddressVector.resize(
    //            height*width* ((int) std::ceil(
    //                ((float) channel) / ((float) simdBlockSize * (float) streamingBlockSize)
    //                ))
    //            );

    int iCompressVectorBase = 0;
    int iStreamBlockAddressVector = 0;
    cl_ushort streamBlockAddress = 0;
    //Index of the current uncompressed simd block in the current stream block
    cl_ushort iSimdBlockInStreamBlock = 0;
    for (int iTensor=0, iFullVector=0; iTensor < num3DTensors; iTensor++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            int iCompressVector = iCompressVectorBase;
            //Index of the external memory location
            for (int iWidth=0; iWidth < width; iWidth++) {
                t_simdblock_value compressionBlock;
                bool retainFlag = false;
                //Index of the current uncompressed simd block in the current sync Block;
                int iSimdBlockInSyncBlock = 0;
                //Number of scar within the current simd block
                int iScalarInSimdBlock = 0;
                //Depth of the BRAM port we are updating;

                //Stream block address is reset to zero at the beginning of every strip
                //If the tensor is an I/O
                if (!isKernel) {
                    streamBlockAddress = 0;
                }
                for (int iChannel=0;
                     iChannel < channel;
                     iChannel++, iFullVector++) {
                    //Float to fixed point conversion
                    fixedPointNumber fpNumber = fixedPointVector.at(iFullVector);
                    char fpValue = ( (fpNumber.getBits()) & (fpNumber.getMask()) );

                    //If at least one value needs to be retained, then the entire block is retained.
                    retainFlag = (fpValue != 0x0) || retainFlag;
                    compressionBlock.values[iScalarInSimdBlock] = fpValue;

                    //If we are at the end of the channel, at a simd block is yet formed
                    //then we need to pad 0s
                    if ( iChannel == (channel - 1) ) {
                           while (iScalarInSimdBlock < maxScalarIndexInSimdBlock) {
                               compressionBlock.values[++iScalarInSimdBlock] = 0x0;
                            }
                    }

                    // Condition for storing a simd block and its position in a sync block
                    // If a simd block has been formed, and it
                    // a. contains at least one non-zero element OR
                    // b. is the last simd block in a sync block OR
                    // c. is the last simd block in the channel
                    // Then we need to store it.
                    // If we come across case b or c, we also need reset iSimdBlockInSyncBlock

                    bool simdBlockFormed = (iScalarInSimdBlock == maxScalarIndexInSimdBlock);

                    bool simdBlockIndexInSyncBlockReset = ( iSimdBlockInSyncBlock == maxSimdBlockIndexInSyncBlock )
                            || (iChannel == (channel - 1));

                    bool simdBlockIndexInStreamBlockReset = (iSimdBlockInStreamBlock == maxSimdBlockIndexInStreamBlock);

                    bool storeSimdBlock =
                            simdBlockFormed && (retainFlag || simdBlockIndexInSyncBlockReset ) ?
                                true : false;

                    //Logic for storing the simd block
                    if (storeSimdBlock) {
                        valueVector.at(iCompressVector) = compressionBlock;
                        channelOffsetVector.at(iCompressVector) = iSimdBlockInSyncBlock;
                        streamBlockAddress++;

                        iCompressVector++;
                        retainFlag = false;
                    }

                    //Logic for updating
                    //1. iSimdBlockInSyncBlock
                    //2. iSimdBlockInStreamBlock
                    //3. streamAddress
                    if (simdBlockFormed) {
                        iSimdBlockInSyncBlock = simdBlockIndexInSyncBlockReset ?
                                    0 : iSimdBlockInSyncBlock + 1;
                        iSimdBlockInStreamBlock = simdBlockIndexInStreamBlockReset ?
                                    0 : iSimdBlockInStreamBlock + 1;
                        //We've crossed a stream block boundary, need to record the end pointer to the stream block
                        //In the compressed tensor
                        if (simdBlockIndexInStreamBlockReset) {
                            streamBlockAddressVector[iStreamBlockAddressVector++] = streamBlockAddress;
                            if ((!isKernel) && (iChannel == (channel - 1))) {
                                streamBlockAddress = 0;
                            }
                        }
                    }

                    //Update the scalarIndexInSimdBlock
                    iScalarInSimdBlock =  (iScalarInSimdBlock == maxScalarIndexInSimdBlock) ? 0 : iScalarInSimdBlock + 1;

                } // for channel
            } // for width
            if (!isKernel) {
                iCompressVectorBase += externalMemoryAddressStride;
             }
            else {
                iCompressVectorBase = iCompressVector;
            }
        } // for height
        streamBlockAddress = 0;
     } // for Tensor
}

int decodeTensor(compressedTensor compTensor, std::vector<float> & denseTensor, char fracWidth, char intWidth) {
    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors = compTensor.num3DTensors;
    unsigned short channel = compTensor.channel;
    unsigned short width = compTensor.width;
    unsigned short height = compTensor.height;
    bool isKernel = compTensor.isKernel;

    //The following parameters should be compatiable with the hardware
    //Number of uncompressed simdblocks in a streaming block
    //TODO: Should this value should match the number of PE rows?
    unsigned short maxSimdBlockIndexInStreamBlock = compTensor.maxSimdBlockIndexInStreamBlock;

    //Number of uncompressed scalar value in each simdblock;
    unsigned short simdBlockSize = compTensor.maxScalarIndexInSimdBlock + 1;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryAddressStride = compTensor.externalMemoryAddressStride;

    unsigned short numSimdBlockPerChannel = (unsigned short) std::ceil( ((float) channel) / ((float) simdBlockSize) );

    unsigned short numStreamBlocksInTensor =
            isKernel ? num3DTensors : num3DTensors * height * width * numSimdBlockPerChannel / (maxSimdBlockIndexInStreamBlock + 1);

    //Allocate space for the dense tensor
    denseTensor.resize(num3DTensors*channel*width*height, 0.0f);
    int numSimdBlock = 0;
    int iCompressVectorBase = 0;
    int iCompressVector = 0;
    int iWidth = 0;
    int iHeight = 0;
    int iTensor = 0;
    int iChannelBase = 0;

    for (unsigned short iStreamBlock = 0; iStreamBlock<numStreamBlocksInTensor; iStreamBlock++) {
        int numCompressedSimdBlocksInStreamBlock = 0;

        //Logic for computing the number of compressed simdblocks in the stream block
        if (isKernel) {
            numCompressedSimdBlocksInStreamBlock = compTensor.streamBlockAddressVector.at(iStreamBlock);
        }
        else {
            if (iChannelBase == 0) {
                numCompressedSimdBlocksInStreamBlock = compTensor.streamBlockAddressVector.at(iStreamBlock);
            }
            else {
                numCompressedSimdBlocksInStreamBlock =
                        compTensor.streamBlockAddressVector.at(iStreamBlock)
                        - compTensor.streamBlockAddressVector.at(iStreamBlock - 1);
            }
        }

        //Perform some operations to initialize the iterators
        for (int iSimdBlock = 0; iSimdBlock<numCompressedSimdBlocksInStreamBlock; iSimdBlock++) {
            t_simdblock_value simdBlock =compTensor.valueVector.at(iCompressVector);
            t_simdblock_channel_offset iSimdBlockInSyncBlock = compTensor.channelOffsetVector.at(iCompressVector);
            iCompressVector++;
            numSimdBlock++;

            int iChannel = iChannelBase + iSimdBlockInSyncBlock * simdBlockSize;
            int iDenseVector = iTensor * height * width * channel + iHeight * width * channel + iWidth * channel + iChannel;

            //Write the scalars in the simd block to the dense vector
            for (int i=0; i< simdBlockSize; i++) {
                if (iChannel < channel) {
                    fixedPointNumber fpValue ( (short) (simdBlock.values[i]), fracWidth, intWidth);
                    denseTensor.at(iDenseVector++) = fpValue.convert2Float();
                    iChannel++;
                }
            }

            if (iChannel < channel) {
                if (iSimdBlockInSyncBlock == compTensor.maxSimdBlockIndexInSyncBlock) {
                    iChannelBase += (compTensor.maxSimdBlockIndexInSyncBlock + 1) * simdBlockSize;
                }
            }
            else {
                iChannelBase = 0;
                iWidth = (iWidth == (width - 1)) ? 0 : iWidth + 1;
                if (iWidth == 0) {
                    iHeight = (iHeight == (height - 1)) ? 0: iHeight + 1;
                    if (iHeight == 0) {
                        iTensor++;
                    }
                    if (!isKernel) {
                        iCompressVectorBase += externalMemoryAddressStride;
                        iCompressVector = iCompressVectorBase;
                    }
                } // height update
            } // width update
        } // for simdblocks in one Stream block

    }
    std::cout <<"Number of simd blocks survived "<<numSimdBlock<<std::endl;
    std::cout <<"Size of the dense vector "<<denseTensor.size()<<std::endl;
}

directCompressedTensor::directCompressedTensor (std::vector<fixedPointNumber> & fixedPointVector,
                                    unsigned short _num3DTensors,
                                    unsigned short _channel,
                                    unsigned short _width,
                                    unsigned short _height,
                                    unsigned char _maxSimdBlockIndexInStreamBlock,
                                    unsigned char _maxScalarIndexInSimdBlock,
                                    bool _isKernel
                                    ) {
    num3DTensors = _num3DTensors;
    channel = _channel;
    width = _width;
    height = _height;
    maxScalarIndexInSimdBlock = _maxScalarIndexInSimdBlock;
    isKernel = _isKernel;
    unsigned short numSimdBlocksInChannel =
            (unsigned short) std::ceil ( (float) channel / (float) (maxScalarIndexInSimdBlock + 1) );


    //Sould probably change the following assignment
    externalMemoryAddressStride = isKernel ? width * height * numSimdBlocksInChannel * num3DTensors : width * numSimdBlocksInChannel;

    maxSimdBlockIndexInStreamBlock = isKernel ?
                height * width * (unsigned short) std::ceil ( (float) channel / (float) (maxScalarIndexInSimdBlock + 1) ) - 1
              : _maxSimdBlockIndexInStreamBlock;
    maxSimdBlockIndexInSyncBlock = std::min((unsigned int) maxSimdBlockIndexInStreamBlock, (unsigned int) 7);



    if (isKernel) {
        unsigned short numSyncBlocksInChannel = (unsigned short) std::ceil ( (float) numSimdBlocksInChannel / (float) (maxSimdBlockIndexInSyncBlock + 1) );
        externalMemoryAddressStride = width*height*(numSimdBlocksInChannel + numSyncBlocksInChannel);
        valueVector.resize(externalMemoryAddressStride*num3DTensors);
        streamBlockAddressVector.resize(num3DTensors);
     }
    else {
        //An integer number of stream blocks must fit inside the channel
        assert( numSimdBlocksInChannel % (_maxSimdBlockIndexInStreamBlock + 1) == 0 );
        unsigned int numStreamBlocksPerChannel = numSimdBlocksInChannel / (_maxSimdBlockIndexInStreamBlock + 1);
        unsigned int numSyncBlockPerStreamBlcok =
                (unsigned int) std::ceil ((float) (maxSimdBlockIndexInStreamBlock + 1) / (float) (maxSimdBlockIndexInSyncBlock + 1) );
        unsigned short numSyncBlocksInChannel = numSyncBlockPerStreamBlcok * numStreamBlocksPerChannel;
        externalMemoryAddressStride = width *(numSimdBlocksInChannel + numSyncBlocksInChannel);
        valueVector.resize(height * externalMemoryAddressStride);
        streamBlockAddressVector.resize(num3DTensors * width * height * numStreamBlocksPerChannel );
    }
    //streamBlockAddressVector.resize(
    //            height*width* ((int) std::ceil(
    //                ((float) channel) / ((float) simdBlockSize * (float) streamingBlockSize)
    //                ))
    //            );

    int iCompressVectorBase = 0;
    int iStreamBlockAddressVector = 0;
    //Index of the current uncompressed simd block in the current stream block
    cl_ushort iSimdBlockInStreamBlock = 0;
    cl_ushort streamBlockAddress = 0;
    for (int iTensor=0, iFullVector=0; iTensor < num3DTensors; iTensor++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            int iCompressVector = iCompressVectorBase;
            //Index of the external memory location
            for (int iWidth=0; iWidth < width; iWidth++) {
                t_simdblock_value compressionBlock;
                bool retainFlag = false;
                //Index of the current uncompressed simd block in the current sync Block;
                unsigned int iSimdBlockInSyncBlock = 0;
                //Number of scar within the current simd block
                int iScalarInSimdBlock = 0;

                //Effectual simd block bitmask within a synb block
                unsigned char bitmask = 0;

               //Number of surviving simd blocks in the current sync block
               unsigned char numSurivingBlocksInSyncBlock;

               //buffer for the current sync block
               std::vector<t_simdblock_value> syncBlockBuffer;

                //Stream block address is reset to zero at the beginning of every strip
                //If the tensor is an I/O
                if (!isKernel) {
                    streamBlockAddress = 0;
                }
                for (int iChannel=0;
                     iChannel < channel;
                     iChannel++, iFullVector++) {
                    //Float to fixed point conversion
                    fixedPointNumber fpNumber = fixedPointVector.at(iFullVector);
                    char fpValue = ( (fpNumber.getBits()) & (fpNumber.getMask()) );

                    //If at least one value needs to be retained, then the entire block is retained.
                    retainFlag = (fpValue != 0x0) || retainFlag;
                    compressionBlock.values[iScalarInSimdBlock] = fpValue;

                    //If we are at the end of the channel, at a simd block is yet formed
                    //then we need to pad 0s
                    if ( iChannel == (channel - 1) ) {
                           while (iScalarInSimdBlock < maxScalarIndexInSimdBlock) {
                               compressionBlock.values[++iScalarInSimdBlock] = 0x0;
                            }
                    }

                    // Condition for storing a simd block and its position in a sync block
                    // If a simd block has been formed, and it
                    // a. contains at least one non-zero element
                    // Then we need to store it.
                    // If we come across case b or c, we also need reset iSimdBlockInSyncBlock

                    bool simdBlockFormed = (iScalarInSimdBlock == maxScalarIndexInSimdBlock);

                    bool simdBlockIndexInSyncBlockReset = ( iSimdBlockInSyncBlock == maxSimdBlockIndexInSyncBlock )
                            || (iChannel == (channel - 1));

                    bool simdBlockIndexInStreamBlockReset = (iSimdBlockInStreamBlock == maxSimdBlockIndexInStreamBlock);

                    bool storeSimdBlock =
                            simdBlockFormed && retainFlag  ?
                                true : false;

                    //Logic for storing the simd block
                    if (storeSimdBlock) {
                        //valueVector.at(iCompressVector) = compressionBlock;

                        //streamBlockAddress++;

                        //iCompressVector++;
                        retainFlag = false;
                        syncBlockBuffer.push_back(compressionBlock);
                        bitmask = bitmask | (0x01 << iSimdBlockInSyncBlock);
                    }

                    //Logic for updating
                    //1. iSimdBlockInSyncBlock
                    //2. iSimdBlockInStreamBlock
                    //3. streamAddress
                    if (simdBlockFormed) {
                        //Write the bit mask and the buffered simd block into the compressed tensor
                        if (simdBlockIndexInSyncBlockReset) {
                            t_simdblock_value bitmaskBlock;
                            bitmaskBlock.values[0] = bitmask;
                            valueVector.at(iCompressVector++) = bitmaskBlock;
                            streamBlockAddress++;
                            for (unsigned char i=0; i<syncBlockBuffer.size(); i++) {
                                valueVector.at(iCompressVector++) = syncBlockBuffer.at(i);
                                streamBlockAddress++;
                            }

                            //reset the bitmask and the sync block buffer
                            syncBlockBuffer.resize(0);
                            bitmask = 0;

                        }
                        iSimdBlockInSyncBlock = simdBlockIndexInSyncBlockReset ?
                                    0 : iSimdBlockInSyncBlock + 1;
                        iSimdBlockInStreamBlock = simdBlockIndexInStreamBlockReset ?
                                    0 : iSimdBlockInStreamBlock + 1;
                        //We've crossed a stream block boundary, need to record the end pointer to the stream block
                        //In the compressed tensor
                        if (simdBlockIndexInStreamBlockReset) {
                            streamBlockAddressVector[iStreamBlockAddressVector++] = streamBlockAddress;
                            if ((!isKernel) && (iChannel == (channel - 1))) {
                                streamBlockAddress = 0;
                            }
                        }
                    }

                    //Update the scalarIndexInSimdBlock
                    iScalarInSimdBlock =  (iScalarInSimdBlock == maxScalarIndexInSimdBlock) ? 0 : iScalarInSimdBlock + 1;

                } // for channel
            } // for width
            if (!isKernel) {
                iCompressVectorBase += externalMemoryAddressStride;
             }
            else {
                iCompressVectorBase = iCompressVector;
            }
        } // for height
        streamBlockAddress = 0;
     } // for Tensor
}

int decodeDirectCompressedTensor(directCompressedTensor compTensor, std::vector<float> & denseTensor, char fracWidth, char intWidth) {
    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors = compTensor.num3DTensors;
    unsigned short channel = compTensor.channel;
    unsigned short width = compTensor.width;
    unsigned short height = compTensor.height;
    bool isKernel = compTensor.isKernel;

    //The following parameters should be compatiable with the hardware
    //Number of uncompressed simdblocks in a streaming block
    //TODO: Should this value should match the number of PE rows?
    unsigned short maxSimdBlockIndexInStreamBlock = compTensor.maxSimdBlockIndexInStreamBlock;

    //Number of uncompressed scalar value in each simdblock;
    unsigned short simdBlockSize = compTensor.maxScalarIndexInSimdBlock + 1;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryAddressStride = compTensor.externalMemoryAddressStride;

    unsigned short numSimdBlockPerChannel = (unsigned short) std::ceil( ((float) channel) / ((float) simdBlockSize) );

    unsigned short numStreamBlocksInTensor =
            isKernel ? num3DTensors : num3DTensors * height * width * numSimdBlockPerChannel / (maxSimdBlockIndexInStreamBlock + 1);

    //Allocate space for the dense tensor
    denseTensor.resize(num3DTensors*channel*width*height, 0.0f);
    int numSimdBlock = 0;
    int iCompressVectorBase = 0;
    int iCompressVector = 0;
    int iWidth = 0;
    int iHeight = 0;
    int iTensor = 0;
    int iChannelBase = 0;

    for (unsigned short iStreamBlock = 0; iStreamBlock<numStreamBlocksInTensor; iStreamBlock++) {
        int numCompressedSimdBlocksInStreamBlock = 0;

        //Logic for computing the number of compressed simdblocks in the stream block
        if (isKernel) {
            numCompressedSimdBlocksInStreamBlock = compTensor.streamBlockAddressVector.at(iStreamBlock);
        }
        else {
            if (iChannelBase == 0) {
                numCompressedSimdBlocksInStreamBlock = compTensor.streamBlockAddressVector.at(iStreamBlock);
            }
            else {
                numCompressedSimdBlocksInStreamBlock =
                        compTensor.streamBlockAddressVector.at(iStreamBlock)
                        - compTensor.streamBlockAddressVector.at(iStreamBlock - 1);
            }
        }

        typedef enum ReadState {ReadBitMask, ReadSimdBlock} e_readState;

        e_readState state = ReadBitMask;

        unsigned char positionInSyncBlock;
        unsigned char bitmask;

        //Perform some operations to initialize the iterators
        for (int iSimdBlock = 0; iSimdBlock<numCompressedSimdBlocksInStreamBlock; iSimdBlock++) {
            t_simdblock_value simdBlock = compTensor.valueVector.at(iCompressVector);
            iCompressVector++;
            numSimdBlock++;
            bool updateChannel = false;

            switch (state) {
            case ReadBitMask: {
                    bitmask = simdBlock.values[0];
                    positionInSyncBlock = 0;
                    state = (bitmask != 0) ? ReadSimdBlock : ReadBitMask;
                    updateChannel = (bitmask != 0) ? false: true;
                }

                break;
                case ReadSimdBlock: {
                    unsigned char numLeadingZeros = countNumLeadingZeros(bitmask);
                    unsigned char indexInSyncBlock = positionInSyncBlock + numLeadingZeros;
                    bitmask = bitmask >> (numLeadingZeros + 1);
                    state = (bitmask == 0) ? ReadBitMask : ReadSimdBlock;
                    positionInSyncBlock = indexInSyncBlock + 1;

                    int iChannel = iChannelBase + (int) indexInSyncBlock * (int) simdBlockSize;
                    int iDenseVector = iTensor * height * width * channel + iHeight * width * channel + iWidth * channel + iChannel;

                    //Write the scalars in the simd block to the dense vector
                    for (int i=0; i< simdBlockSize; i++) {
                        if (iChannel < channel) {
                            fixedPointNumber fpValue ( (short) (simdBlock.values[i]), fracWidth, intWidth);
                            denseTensor.at(iDenseVector++) = fpValue.convert2Float();
                            iChannel++;
                        }
                    }

                    if (state == ReadBitMask) {
                        updateChannel = true;
                    }

                } // case ReadSimdBlock
                break;
                default:
                break;
            }

            if (updateChannel) {
                iChannelBase += (compTensor.maxSimdBlockIndexInSyncBlock + 1) * simdBlockSize;
                if (iChannelBase >= channel) {
                    iChannelBase = 0;
                    iWidth = (iWidth == (width - 1)) ? 0 : iWidth + 1;
                        if (iWidth == 0) {
                            iHeight = (iHeight == (height - 1)) ? 0: iHeight + 1;
                            if (iHeight == 0) {
                                iTensor++;
                            }
                            if (!isKernel) {
                                iCompressVectorBase += externalMemoryAddressStride;
                                iCompressVector = iCompressVectorBase;
                            }
                        } // height update
                    } // width update
             }

        } // for simdblocks in one Stream block

    }
    std::cout <<"Number of simd blocks survived "<<numSimdBlock<<std::endl;
    std::cout <<"Size of the dense vector "<<denseTensor.size()<<std::endl;
}

unsigned char countNumLeadingZeros (unsigned char bitmask) {
    unsigned char count = 0;
    while ( ( (bitmask & 0x01 )== 0) && count < 8) {
        count++;
        bitmask = bitmask >> 0x1;
    }
    return count;
}
