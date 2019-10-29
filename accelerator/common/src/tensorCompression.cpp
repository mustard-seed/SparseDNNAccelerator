#include "tensorCompression.hpp"
#include "params.hpp"
#include <cmath> //std::ceil
#include <iostream> //cout
#include <cassert>

unsigned char countNumLeadingZeros (unsigned char bitmask);

unsigned char countNumOnes (unsigned char bitmask);

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
                1 + (channel - 1) / ((_maxClusterIndexInCompressionBlock + 1) * (_maxScalarIndexInCluster + 1));
        unsigned int tempStride = width*height*((unsigned int) numTransferBlocksPerCompressionBlock* (unsigned int) numCompressionBlocksInChannel);
        //externalMemoryAddressStride = lcm(tempStride, (unsigned int) WIDE_SIZE); //DRAM stride needs to be a multiple of DRAM width and the storage requirement per filter
        externalMemoryAddressStride = (unsigned int) std::ceil( ((float) (tempStride) ) / ((float) (WIDE_SIZE)) ) * WIDE_SIZE;
        valueVector.resize(externalMemoryAddressStride*num3DTensors);
        streamBlockAddressVector.resize(num3DTensors);
        numChannelGroups = 1;
        maxScalarIndexInChannelGroup = _channel - 1;
     }
    else {
        //Channel group size should divide channel size
        maxScalarIndexInChannelGroup = _maxScalarIndexInChannelGroup;
        assert( channel % (maxScalarIndexInChannelGroup + 1) == 0 );
        assert (num3DTensors == 1);
        numChannelGroups = channel / (maxScalarIndexInChannelGroup + 1);
        unsigned int numCompressionBlocksInChannelGroup =
                (unsigned int) std::ceil ( (float) (maxScalarIndexInChannelGroup + 1)
                               / (float) ((maxScalarIndexInCluster + 1) * (maxClusterIndexInCompressionBlock + 1)) );
        unsigned int tempStride = (numCompressionBlocksInChannelGroup * numTransferBlocksPerCompressionBlock);
        externalMemoryAddressStride = (unsigned int) std::ceil( ((float) (tempStride) ) / ((float) (WIDE_SIZE)) ) * WIDE_SIZE;
        valueVector.resize( width * height * numChannelGroups * externalMemoryAddressStride);
        streamBlockAddressVector.resize(width * height * numChannelGroups );
    }
    //============================================================

    //Trackers of the position in the compressed vector and the stream block address vector
    int iCompressVectorBase = 0;
    int iStreamBlockAddressVector = 0;

    //Keep track of the address of each stream block along the channels
    cl_ushort streamBlockAddress = 0;
    for (int iTensor=0; iTensor < num3DTensors; iTensor++){
        for (unsigned char iGroup=0; iGroup<numChannelGroups; iGroup++)
        {
            for (int iHeight=0; iHeight < height; iHeight++) {
                for (int iWidth=0; iWidth<width; iWidth++)
                {
                    //Set the pointer of the compression bector
                    int iCompressVector = iCompressVectorBase;

                   //Buffer for the current compression window. Default values are zero
                   std::vector<cl_char> compressionBlock(maxScalarIndexInCompressionBlock+1, 0);

                   //Tracker of the scalar in the compression window
                   int iScalarInCompressionBlock = 0;

                    int iFullVector = iTensor * channel * width * height
                            + (iHeight * width + iWidth) * channel
                            + iGroup*(maxScalarIndexInChannelGroup + 1);

                    for (int iChannelInGroup=0;
                         iChannelInGroup <= maxScalarIndexInChannelGroup;
                         iChannelInGroup++) {
                        //Float to fixed point conversion
                        fixedPointNumber fpNumber = fixedPointVector.at(iFullVector);
                        char fpValue = ( (fpNumber.getBits()) & (fpNumber.getMask()) );

                        //std::cout <<"iFullVector is "<<iFullVector<<std::endl;
                        //std::cout <<"iScalarInCompressionBlock is "<<iScalarInCompressionBlock<<std::endl;
                        iFullVector++;

                        compressionBlock.at(iScalarInCompressionBlock++) = fpValue;

                        //Indicator of whether a compression block has been formed.
                        bool compressionBlockFormed =
                                (iScalarInCompressionBlock > maxScalarIndexInCompressionBlock)
                                || (iChannelInGroup == maxScalarIndexInChannelGroup);


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
                        } // if compression block is formed

                    } // for channel in a channel group

                    if (!isKernel) {
                        iCompressVectorBase += externalMemoryAddressStride;
                        streamBlockAddressVector.at(iStreamBlockAddressVector++)
                                                        = streamBlockAddress;
                        streamBlockAddress = 0;
                     }
                    else {
                        iCompressVectorBase = iCompressVector;
                    }
                }//iWidth
            //Need to stride the value vector here if we are compressing an activation
            } // for height
        } //for Groups

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
    unsigned short numChannelGroups = compTensor.numChannelGroups;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryAddressStride
        = compTensor.externalMemoryAddressStride;

    //Allocate space for the dense tensor
    denseTensor.resize(num3DTensors*channel*width*height, 0.0f);

    //Computer same bounds
    unsigned int numStreamBlocksInTensor =
            isKernel ?
            num3DTensors
            : height * width * numChannelGroups;

    //Initilize some counters
    int iCompressVectorBase = 0;
    int iCompressVector = 0;
    int iWidth = 0;
    int iHeight = 0;
    int iTensor = 0;
    int iChannelInGroupBase = 0;
    int iGroup = 0;

    int countTransferBlocks = 0;

    //Read the stream blocks, one at a time
    for (unsigned int iStreamBlock=0;
        iStreamBlock < numStreamBlocksInTensor;
        iStreamBlock++)
    {
        //std::cout <<"flexibleDirectDecompression. iStreamBlock="<<iStreamBlock<<std::endl;
        //Need to know how many transfer blocks are there in the stream block that we
        //are about to read
        int numTransferBlockInStreamBlock = compTensor.streamBlockAddressVector.at(iStreamBlock);

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
                        int iChannelInGroup = iChannelInGroupBase + (int) indexInCompressionBlock*(maxScalarIndexInCluster+1) + i;

                        if (iChannelInGroup <= maxScalarIndexInChannelGroup) {
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
                                + iGroup * (maxScalarIndexInChannelGroup + 1)
                                + iChannelInGroup;
//                            std::cout <<"Bitmask, iTensor, iGroup, iHeight, iWidth, iChannelInGroup: "
//                               <<(unsigned int) bitmask<<" "<<iTensor<<" "<<iGroup<<" "<<iHeight<<" "<<iWidth<<" "<<iChannelInGroup<<std::endl;

                            denseTensor.at(iDenseVector) = fpValue.convert2Float();
                        } //if iChannelInGroup <= maxScalarIndexInChannelGroup
                    } // for from 0 to maxScalarIndexInCluster
                } // while. decode a compression block;

                //Update the trackers
                int numScalarsRemainInChannelGroup =
                    (maxScalarIndexInChannelGroup + 1) - iChannelInGroupBase;
                int numScalarsAdded =
                    numScalarsRemainInChannelGroup < (maxScalarIndexInCompressionBlock + 1) ?
                    numScalarsRemainInChannelGroup : (maxScalarIndexInCompressionBlock + 1);
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

                iChannelInGroupBase += numScalarsAdded;
                //std::cout <<"iChannelBase "<<iChannelBase<<std::endl;
                if (iChannelInGroupBase > maxScalarIndexInChannelGroup)
                {
                    //std::cout <<"Resetting iChannelGroupBase"<<std::endl;
                    iChannelInGroupBase = 0;
                    iWidth = (iWidth == (width - 1)) ? 0 : iWidth + 1;
                    if (iWidth == 0)
                    {
                        iHeight = (iHeight == (height - 1)) ? 0: iHeight + 1;
                        if (iHeight == 0)
                        {
                            iGroup = (iGroup == (numChannelGroups-1)) ? 0 : iGroup + 1;
                            if (iGroup == 0)
                            {
                                iTensor++;
                                if (isKernel)
                                {
                                    iCompressVectorBase += externalMemoryAddressStride;
                                    iCompressVector = iCompressVectorBase;
                                }
                            }
                        }
                    } // height update
                    if (!isKernel)
                    {
                        iCompressVectorBase += externalMemoryAddressStride;
                        iCompressVector = iCompressVectorBase;
                    }
                } // iChannelBase >= channel
            } // if a compression block has been formed
        }  //for-loop. Expands atream block
        //=====================================================
    }
    return countTransferBlocks;

}

