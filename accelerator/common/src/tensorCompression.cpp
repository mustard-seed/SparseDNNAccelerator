#include "tensorCompression.hpp"
#include "params.hpp"
#include <cmath> //std::ceil
#include <iostream> //cout
#include <cassert>

unsigned char countNumLeadingZeros (unsigned char bitmask);

unsigned char countNumOnes (unsigned char bitmask);

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
                    fixedPointNumber fpValue ( (char) (simdBlock.values[i]), fracWidth, intWidth);
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
                            fixedPointNumber fpValue ( (char) (simdBlock.values[i]), fracWidth, intWidth);
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

unsigned char countNumOnes (unsigned char bitmask) 
{
    unsigned char count = 0;
    for (int i=0; i<8; i++) {
        if ((bitmask & 0x01) == 0x1) {
            count++;
        }
        bitmask >>= 0x1;
    }
    return count;
}

unsigned int lcm (unsigned int a, unsigned int b)
{
    unsigned int gcd = a, temp = b;
    while (gcd != temp)
    {
        if (gcd > temp)
        {
            gcd -= temp;
        }
        else
        {
            temp -= gcd;
        }
    }

    return ((a*b) / gcd);
}

#ifdef FLEXIBLE_BITMASK_COMPRESSION
flexibleDirectCompressedTensor::flexibleDirectCompressedTensor (
                        std::vector<fixedPointNumber> & fixedPointVector,
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
                        )
{
    //Assign values to the member variables
    num3DTensors = _num3DTensors;
    channel = _channel;
    width = _width;
    height = _height;
    tilingSizeWidth = _tilingSizeWidth;
    maxScalarIndexInChannelGroup = _maxScalarIndexInChannelGroup;
    maxClusterIndexInCompressionBlock = _maxClusterIndexInCompressionBlock;
    maxClusterIndexInTransferBlock = _maxClusterIndexInTransferBlock;
    maxScalarIndexInCluster = _maxScalarIndexInCluster;
    maxScalarIndexInCompressionBlock = (maxClusterIndexInCompressionBlock+1)*(maxScalarIndexInCluster+1) - 1;
    isKernel = _isKernel;

    //Compute the number of transfer blocks in a compression block
    //Need an extra one at the end to account for the bitmask
    // which occupies its own transfer block
    unsigned int numTransferBlocksPerCompressionBlock
            =  (unsigned int) std::ceil ( (float) (1 + maxClusterIndexInCompressionBlock + 1) / (float) (maxClusterIndexInTransferBlock + 1));

    //==========================================================
    //Compute the memory stride and allocate space in the vectors
    if (isKernel) {
        unsigned int numCompressionBlocksInChannel =
                (unsigned int) std::ceil ( (float) channel /
                                             (float) ((_maxClusterIndexInCompressionBlock + 1) * (_maxScalarIndexInCluster + 1)) );

        unsigned int tempStride = width*height*((unsigned int) numTransferBlocksPerCompressionBlock* (unsigned int) numCompressionBlocksInChannel);
        //externalMemoryAddressStride = lcm(tempStride, (unsigned int) WIDE_SIZE); //DRAM stride needs to be a multiple of DRAM width and the storage requirement per filter
        externalMemoryAddressStride = (unsigned int) std::ceil( ((float) (tempStride) ) / ((float) (WIDE_SIZE)) ) * WIDE_SIZE;
        valueVector.resize(externalMemoryAddressStride*num3DTensors);
        streamBlockAddressVector.resize(num3DTensors);
     }
    else {
        //Channel group size should divide channel size
        assert( channel % (maxScalarIndexInChannelGroup + 1) == 0 );
        assert (num3DTensors == 1);
        unsigned int numChannelGroups = channel / (maxScalarIndexInChannelGroup + 1);
        unsigned int numCompressionBlocksInChannelGroup =
                (unsigned int) std::ceil ( (float) (maxScalarIndexInChannelGroup + 1)
                               / (float) ((maxScalarIndexInCluster + 1) * (maxClusterIndexInCompressionBlock + 1)) );
        unsigned int tempStride = (numCompressionBlocksInChannelGroup * numChannelGroups * numTransferBlocksPerCompressionBlock);
        externalMemoryAddressStride = (unsigned int) std::ceil( ((float) (tempStride) ) / ((float) (WIDE_SIZE)) ) * WIDE_SIZE;
        valueVector.resize( width * height * externalMemoryAddressStride);
        streamBlockAddressVector.resize(num3DTensors * width * height * numChannelGroups );
    }
    //============================================================

    //Trackers of the position in the compressed vector and the stream block address vector
    int iCompressVectorBase = 0;
    int iStreamBlockAddressVector = 0;

    //Keep track of the address of each stream block along the channels
    cl_ushort streamBlockAddress = 0;
    int iFullVector = 0;
    for (int iTensor=0; iTensor < num3DTensors; iTensor++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            for (int iWidth=0; iWidth<width; iWidth++)
            {
            //for (int iTileWidth=0; iTileWidth < (int) (std::ceil( (float) width / (float) tilingSizeWidth)); iTileWidth++) {
            //    int maxLocalWidth = (tilingSizeWidth <= (width - iTileWidth*tilingSizeWidth)) ?
            //                tilingSizeWidth : (width - iTileWidth*tilingSizeWidth);
                //Set the pointer of the compression bector
                int iCompressVector = iCompressVectorBase;
                //for (int iLocalWidth=0; iLocalWidth<maxLocalWidth; iLocalWidth++) {

                   //Buffer for the current compression window. Default values are zero
                   std::vector<cl_char> compressionBlock(maxScalarIndexInCompressionBlock+1, 0);

                   //Tracker of the scalar in the compression window
                   int iScalarInCompressionBlock = 0;

                    //Stream block address is reset to zero at the beginning of every strip
                    //If the tensor is an I/O
                    if (!isKernel) {
                        streamBlockAddress = 0;
                    }

                    for (int iChannel=0, iChannelInGroup=0;
                         iChannel < channel;
                         iChannel++, iFullVector++) {
                        //Float to fixed point conversion
                        fixedPointNumber fpNumber = fixedPointVector.at(iFullVector);
                        char fpValue = ( (fpNumber.getBits()) & (fpNumber.getMask()) );

                        //std::cout <<"iFullVector is "<<iFullVector<<std::endl;
                        //std::cout <<"iScalarInCompressionBlock is "<<iScalarInCompressionBlock<<std::endl;

                        compressionBlock.at(iScalarInCompressionBlock++) = fpValue;

                        //Indicator of whether a compression block has been formed.
                        bool compressionBlockFormed =
                                (iScalarInCompressionBlock > maxScalarIndexInCompressionBlock)
                                || (iChannel == (channel - 1))
                                || ((iChannelInGroup == maxScalarIndexInChannelGroup) && (!isKernel));


                        if ( compressionBlockFormed ) {
                            //If we are at the end of the channel or a channel group (only matter for kernel),
                            //and a compression block is yet formed
                            //then we need to pad 0s
                           while (iScalarInCompressionBlock <= maxScalarIndexInCompressionBlock) {
                               compressionBlock.at(iScalarInCompressionBlock++) = 0x0;
                            }

                           iScalarInCompressionBlock = 0;

                           //Compute the bitmask, and form the blocks
                           unsigned char bitmask = 0x0;
                           for (unsigned int i=0; i<=maxClusterIndexInCompressionBlock; i++) {
                               bool preserve = false;
                               for (unsigned int j=0; j<=maxScalarIndexInCluster; j++)
                               {
                                    char fpValue = compressionBlock.at(i*(maxScalarIndexInCluster+1) + j);
                                    preserve  = preserve || (fpValue != 0x0);
                               }
                               unsigned char bit = (preserve) ?
                                           0x01 : 0x00;
                               bitmask |= (bit << i);
                           } // for bitmask

                           //Populate the value vector
                           t_transfer_block transferBlock;
                           //First load the bitmask
                           //Assume it is 8 bits
                           transferBlock.values[0].cluster_values[0] = bitmask;

                           //std::cout <<"bitmask L: "<<(int)(bitmask & 0xFF)<<std::endl;


                           //valueVector.at(iCompressVector++) = transferBlock;
                            unsigned char iTransferBlock = 1; //account for the bitmask;
                           //streamBlockAddress++;

                           //iterate through the compression block and transfer the non-zero values
                           for (int i=0; i<=maxClusterIndexInCompressionBlock; i++) {
                               bool preserve = false;
                               t_cluster cluster;
                               for (unsigned int j=0; j<=maxScalarIndexInCluster; j++)
                               {
                                    auto fpValue = compressionBlock.at(i*(maxScalarIndexInCluster+1) + j);
                                    preserve  = preserve || (fpValue != 0x0);
                                    cluster.cluster_values[j] = fpValue;
                               }
                               if (preserve) {
                                   transferBlock.values[iTransferBlock++] = cluster;
                                   //std::cout <<"Preserved i = "<<i<<std::endl;
                               }

                               //GOTTCHA!!!!
                               //Push the transfer block if we have filled it or we have reached the end of the
                               //compression block and in the processing of filling one
                               if ((iTransferBlock > maxClusterIndexInTransferBlock)
                                    || ((i == maxClusterIndexInCompressionBlock) && (iTransferBlock > 0)))
                               {
                                    valueVector.at(iCompressVector++) = transferBlock;
                                    streamBlockAddress++;
                                    iTransferBlock = 0;
                               }
                           } // for element in compression block



                           //If we are compressing an activtion tensor and we are at its end
                           //or at the end of one of its channel groups
                           //Then we need to store the number of transfer blocks that the
                           //Compressed channel group takes
                           if ((!isKernel)
                                   && (iChannelInGroup == maxScalarIndexInChannelGroup)
                                   ) {
                               //std::cout <<"Updating streamBlockAddressVector. Activation. iChannelInGroup = "
                               //         <<iChannelInGroup<<std::endl;
                               streamBlockAddressVector.at(iStreamBlockAddressVector++)
                                       = streamBlockAddress;
                           }
                        } // if compression block is formed

                        //Some indices updates
                        iChannelInGroup++;
                        if (iChannelInGroup > maxScalarIndexInChannelGroup) {
                            iChannelInGroup = 0;
                        }
                    } // for channel
                //} //for iLocalWidth
                if (!isKernel) {
                    iCompressVectorBase += externalMemoryAddressStride;
                 }
                else {
                    iCompressVectorBase = iCompressVector;
                }
            //} // for iTileWidth
            }//iWidth
            //Need to stride the value vector here if we are compressing an activation

        } // for height
        //Need to stride the value vector here if we are compressing a filters
        if (isKernel) {
            //If we are compressing a filter and we are at its end
            //Then we need to store the number of transfer blocks that the
            //Compressed filter take
            //std::cout <<"Updating streamBlockAddressVector. Filter"<<std::endl;
            streamBlockAddressVector.at(iStreamBlockAddressVector++)
                    = streamBlockAddress;


            iCompressVectorBase = (iTensor+1)*externalMemoryAddressStride;
            streamBlockAddress = 0x0;
        }
     } // for Tensor
}

//Helper function used to decode a compressed tensor
int decodeFlexibleDirectCompressedTensor(
        flexibleDirectCompressedTensor compTensor
        ,std::vector<float> & denseTensor
        ,char fracWidth
        ,char intWidth) 
{


    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors = compTensor.num3DTensors;
    unsigned short channel = compTensor.channel;
    unsigned short width = compTensor.width;
    unsigned short height = compTensor.height;
    unsigned short tileSizeWidth = compTensor.tilingSizeWidth;
    bool isKernel = compTensor.isKernel;

    //Compression parameters
    unsigned short maxScalarIndexInChannelGroup = compTensor.maxScalarIndexInChannelGroup;
    unsigned char maxScalarIndexInCompressionBlock
        = compTensor.maxScalarIndexInCompressionBlock;
    unsigned char maxClusterIndexInCompressionBlock
        = compTensor.maxClusterIndexInCompressionBlock;
    unsigned char maxClusterIndexInTransferBlock
        = compTensor.maxClusterIndexInTransferBlock;
    unsigned char maxScalarIndexInCluster =
            compTensor.maxScalarIndexInCluster;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryAddressStride 
        = compTensor.externalMemoryAddressStride;

    //Allocate space for the dense tensor
    denseTensor.resize(num3DTensors*channel*width*height, 0.0f);

    //Computer same bounds
    unsigned int numStreamBlocksInTensor =
            isKernel ? 
            num3DTensors 
            : height * width * channel / (maxScalarIndexInChannelGroup + 1);

    //Initilize some counters
    int iCompressVectorBase = 0;
    int iCompressVector = 0;
    //int iLocalWidth = 0;
    //int iTileWidth = 0;
    int iWidth = 0;
    int numTileWidth = (int) std::ceil( (float) width / (float) tileSizeWidth);
    int iHeight = 0;
    int iTensor = 0;
    int iChannelBase = 0;

    int countTransferBlocks = 0;

    //Read the stream blocks, one at a time
    for (unsigned int iStreamBlock=0; 
        iStreamBlock < numStreamBlocksInTensor;
        iStreamBlock++)
    {
        //std::cout <<"flexibleDirectDecompression. iStreamBlock="<<iStreamBlock<<std::endl;
        //Need to know how many transfer blocks are there in the stream block that we 
        //are about to read
        int numTransferBlockInStreamBlock;
        if (isKernel) 
        {
           numTransferBlockInStreamBlock = 
            compTensor.streamBlockAddressVector.at(iStreamBlock);
        }
        else
        {
            if (iChannelBase==0) 
            {
                numTransferBlockInStreamBlock = 
                    compTensor.streamBlockAddressVector.at(iStreamBlock);
            }
            else 
            {
                 numTransferBlockInStreamBlock =
                    compTensor.streamBlockAddressVector.at(iStreamBlock)
                    - compTensor.streamBlockAddressVector.at(iStreamBlock - 1);
            }

        } // if/else blocks that assign to numTransferBlockInStreamBlock

        //============Expands a compression block============
        // A assemble vector for the decompression
        std::vector<char> vectorCompressionBlock;
        unsigned char bitmask;

        bool updateBitmask = true;
        unsigned char numNZClustersInCompressionBlock;
        unsigned char countNZClustersInCompressionBlock;

        //Decode each stream block
        for (int i=0; i<numTransferBlockInStreamBlock; i++) {
            t_transfer_block transferBlock =
                compTensor.valueVector.at(iCompressVector++);

            countTransferBlocks++;

            if (updateBitmask) {
                bitmask = transferBlock.values[0].cluster_values[0];
                numNZClustersInCompressionBlock = countNumOnes(bitmask);
                countNZClustersInCompressionBlock = 0;
                vectorCompressionBlock.resize(0);
                //std::cout <<"bitmask R: "<<(int)(bitmask & 0xFF)<<std::endl;
                //std::cout <<"numNZClustersInCompressionBlock: "<<(int)(numNZClustersInCompressionBlock)<<std::endl;
                updateBitmask = false;
                for (int i=1; i<=maxClusterIndexInTransferBlock; i++)
                {
                    for (int j=0; j<=maxScalarIndexInCluster; j++)
                    {
                        vectorCompressionBlock.push_back(transferBlock.values[i].cluster_values[j]);
                    }
                }
                 countNZClustersInCompressionBlock += (maxClusterIndexInTransferBlock);
            }
            else {
                for (int i=0; i<=maxClusterIndexInTransferBlock; i++)
                {
                    for (int j=0; j<=maxScalarIndexInCluster; j++)
                    {
                        vectorCompressionBlock.push_back(transferBlock.values[i].cluster_values[j]);
                    }
                }
                countNZClustersInCompressionBlock += (maxClusterIndexInTransferBlock + 1);
            }

            //If a compression block has been formed then we need to decompress it
            if (countNZClustersInCompressionBlock >= numNZClustersInCompressionBlock) {
                //Setup for the next round
                updateBitmask = true;

                //=================================================
                //Transfer scalars from the compression block to the dense vector
                unsigned char positionInCompressionBlock = 0;
                int iVectorCompressionBlock = 0;
                while (bitmask != 0)
                {
                    unsigned char numLeadingZeros = countNumLeadingZeros(bitmask);
                    unsigned char indexInCompressionBlock = positionInCompressionBlock + numLeadingZeros;
                    bitmask = bitmask >> (numLeadingZeros + 1);
                    positionInCompressionBlock = indexInCompressionBlock + 1;

                    for (int i=0; i<=maxScalarIndexInCluster; i++) {
                        int iChannel = iChannelBase + (int) indexInCompressionBlock*(maxScalarIndexInCluster+1) + i;

                        if (iChannel < channel) {
                            fixedPointNumber fpValue (
                                (char) (vectorCompressionBlock.at(iVectorCompressionBlock++)), fracWidth, intWidth );
//                            int iDenseVector =
//                                iTensor * height * width * channel
//                                + iHeight * width * channel
//                                + iTileWidth * tileSizeWidth * channel
//                                + iLocalWidth * channel
//                                + iChannel;
                            int iDenseVector =
                                iTensor * height * width * channel
                                + iHeight * width * channel
                                + iWidth * channel
                                + iChannel;
                            //std::cout <<"Bitmask, iTensor, iHeight, iTileWidth, iLocalWidth, iChannel: "
                            //   <<(unsigned int) bitmask<<" "<<iTensor<<" "<<iHeight<<" "<<iTileWidth<<" "<<iLocalWidth<<" "<<iChannel<<std::endl;

                            denseTensor.at(iDenseVector) = fpValue.convert2Float();
                        }
                    }
                }

                //Update the trackers
                int numScalarsRemainInChannelGroup = 
                    (maxScalarIndexInChannelGroup + 1)
                    - (iChannelBase % (maxScalarIndexInChannelGroup + 1));
                int numScalarsAdded;
                if (!isKernel) 
                { 
                    numScalarsAdded = 
                        numScalarsRemainInChannelGroup < (maxScalarIndexInCompressionBlock + 1) ?
                        numScalarsRemainInChannelGroup : (maxScalarIndexInCompressionBlock + 1);
                }
                else 
                {
                    numScalarsAdded = maxScalarIndexInCompressionBlock + 1;
                }

                iChannelBase += numScalarsAdded;
                //std::cout <<"iChannelBase "<<iChannelBase<<std::endl;
                if (iChannelBase >= channel)
                {
                    //std::cout <<"Resetting iChannelBase"<<std::endl;
                    iChannelBase = 0;
//                    int maxLocalWidth = (tileSizeWidth < width - iTileWidth*tileSizeWidth) ?
//                                tileSizeWidth : width - iTileWidth*tileSizeWidth;
//                    iLocalWidth =  (iLocalWidth == (maxLocalWidth-1)) ? 0 : iLocalWidth + 1;
                      iWidth = (iWidth == (width - 1)) ? 0 : iWidth + 1;
                        //if (iLocalWidth == 0)
                        //{
                        //    iTileWidth = (iTileWidth == (numTileWidth - 1)) ? 0 : iTileWidth + 1;
                        //    if (iTileWidth == 0)
                        //    {
                            if (iWidth == 0)
                            {
                                iHeight = (iHeight == (height - 1)) ? 0: iHeight + 1;
                                if (iHeight == 0)
                                {
                                    iTensor++;
                                    iCompressVectorBase += externalMemoryAddressStride;
                                    iCompressVector = iCompressVectorBase;

                                }
                            } // height update
                            if (!isKernel)
                            {
                                iCompressVectorBase += externalMemoryAddressStride;
                                iCompressVector = iCompressVectorBase;
                            }
                   } // iChannelBase >= channel
                        //} // iTileWidth update
                //} // iLocalWidth update
            } // if a compression block has been formed
        }  //for-loop. Expands atream block
        //=====================================================
    }

    return countTransferBlocks;

}
#endif
