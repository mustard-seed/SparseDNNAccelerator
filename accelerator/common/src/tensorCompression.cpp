#include "tensorCompression.hpp"
#include "params.hpp"
#include <cmath> //std::ceil
#include <iostream> //cout
#include <cassert>
#include <stdexcept> //std::runtime_error

#include <bitset> //Print friendly binary number

unsigned char findFirstNonZero (unsigned char bitmaskBytes[], unsigned char startingIndex);

unsigned char countNumOnes (unsigned char bitmaskBytes[]);

unsigned char findFirstNonZero (unsigned char bitmaskBytes[], unsigned char startingIndex) {
    unsigned char returnIndex = startingIndex;
    while (returnIndex < COMPRESSION_WINDOW_SIZE) {
        unsigned int byteIndex = returnIndex / 8;
        unsigned int bitIndex = returnIndex % 8;
        unsigned char mask = 0x01 << bitIndex;
        unsigned char bit = (bitmaskBytes[byteIndex] & mask) >> bitIndex;
        if (bit == 0x01)
        {
            break;
        }
        else
        {
           returnIndex++;
        }
    }
    return returnIndex;
}

unsigned char countNumOnes (unsigned char bitmaskBytes[])
{
    unsigned char count = 0;
    for (int i=0; i<NUM_BITMASK_BYTES; i++)
    {
        unsigned char byte = bitmaskBytes[i];
        for (int j=0; j<8; j++) {
            if ((byte & 0x01) == 0x1) {
                count++;
            }
            byte >>= 0x1;
        }
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

AlignedTensor::AlignedTensor(
        unsigned short _num3DTensors,
        unsigned short _channel,
        unsigned short _width,
        unsigned short _height,
        unsigned short _maxScalarIndexInChannelGroup,
        unsigned char _maxClusterIndexInTransferBlock,
        unsigned char _maxScalarIndexInCluster,
        bool _isKernel)
{
    //Assign values to the member variables
    num3DTensors = _num3DTensors;
    channel = _channel;
    width = _width;
    height = _height;
    maxClusterIndexInTransferBlock = _maxClusterIndexInTransferBlock;
    maxScalarIndexInCluster = _maxScalarIndexInCluster;
    isKernel = _isKernel;

    unsigned int numScalarPerTransferBlock = (1+maxScalarIndexInCluster)*(1+maxClusterIndexInTransferBlock);
    unsigned int numTransferBlockPerChannelGroup = 1 + ((1+_maxScalarIndexInChannelGroup)-1) / numScalarPerTransferBlock;

    //==========================================================
    //Compute the memory stride and allocate space in the vectors
    if (isKernel) {
        externalMemoryAddressStride = calculateExternalMemoryAddressStride(
                    _channel, //channelsPerGroup
                    1, //group
                    height, //height
                    width, //width
                    maxScalarIndexInCluster+1, //clusterSize
                    maxClusterIndexInTransferBlock+1, //transferBlockSize
                    1, //compressionWindowSize, unused
                    WIDE_SIZE, //numTransferBlockPerDramBlock
                    true, //isKernel
                    true //isDense
                    );
        valueVector.resize(externalMemoryAddressStride*num3DTensors);
        numChannelGroups = 1;
        maxScalarIndexInChannelGroup = _channel - 1;
     }
    else {
        //Channel group size should divide channel size
        maxScalarIndexInChannelGroup = _maxScalarIndexInChannelGroup;
        assert( channel % (maxScalarIndexInChannelGroup + 1) == 0 );
        assert (num3DTensors == 1);
        numChannelGroups = channel / (maxScalarIndexInChannelGroup + 1);
        externalMemoryAddressStride = calculateExternalMemoryAddressStride(
                    _maxScalarIndexInChannelGroup+1, //channelsPerGroup
                    numChannelGroups, //group
                    height, //height
                    width, //width
                    maxScalarIndexInCluster+1, //clusterSize
                    maxClusterIndexInTransferBlock+1, //transferBlockSize
                    1, //compressionWindowSize, unused
                    WIDE_SIZE, //numTransferBlockPerDramBlock
                    false, //isKernel
                    true //isDense
                    );
        valueVector.resize( width * height * numChannelGroups * externalMemoryAddressStride);
    }
}

//Initialize and poplulate the tensor
AlignedTensor::AlignedTensor(
        std::vector<fixedPointNumber> &fixedPointVector
        ,unsigned short _num3DTensors
        ,unsigned short _channel
        ,unsigned short _width
        ,unsigned short _height
        ,unsigned short _maxScalarIndexInChannelGroup
        ,unsigned char _maxClusterIndexInTransferBlock
        ,unsigned char _maxScalarIndexInCluster
        ,bool _isKernel)
    : AlignedTensor(
          _num3DTensors
          ,_channel
          ,_width
          ,_height
          ,_maxScalarIndexInChannelGroup
          ,_maxClusterIndexInTransferBlock
          ,_maxScalarIndexInCluster
          ,_isKernel
          )
{
    unsigned int numTBPerChannelGroup =
           1 + (1+maxScalarIndexInChannelGroup - 1) / ((1+maxScalarIndexInCluster) * (1+maxClusterIndexInTransferBlock));

    unsigned int iFullVector = 0;
    for (int iTensor=0; iTensor < num3DTensors; iTensor++){

        unsigned int iAlignedVector = externalMemoryAddressStride * iTensor;

        for (unsigned char iGroup=0; iGroup<numChannelGroups; iGroup++)
        {
            for (int iHeight=0; iHeight < height; iHeight++) {
                for (int iWidth=0; iWidth<width; iWidth++)
                {
                    //Overide the startign position in the aligned tensor if the tensor is
                    //an activation tensor
                    iAlignedVector = (!isKernel) ?
                                (iGroup*height*width + width*iHeight + iWidth) * externalMemoryAddressStride : iAlignedVector;

                    unsigned int iChannel = 0;
                    for (unsigned int iChannelTBInGroup=0;
                         iChannelTBInGroup < numTBPerChannelGroup;
                         iChannelTBInGroup++) {

                        t_transfer_block transferBlock;

                        for (int iClusterInTB=0; iClusterInTB<=(this->maxClusterIndexInTransferBlock); iClusterInTB++)
                        {
                            for (int iScalarInCluster =0; iScalarInCluster<=(this->maxScalarIndexInCluster); iScalarInCluster++)
                            {
                                signed char value;;
                                if (iChannel <= (this->maxScalarIndexInChannelGroup))
                                {
                                    fixedPointNumber fpNumber = fixedPointVector.at(iFullVector);
                                    value =  ( (fpNumber.getBits()) & (fpNumber.getMask()) );

                                    iFullVector++;
                                    iChannel++;
                                }
                                else
                                {
                                    value = 0x0;
                                }

                                transferBlock.values[iClusterInTB*CLUSTER_SIZE+iScalarInCluster] = value;
                            }
                        } //over one TB

                        valueVector.at(iAlignedVector++) = transferBlock;

                    } // for channel in a channel group

                }//iWidth
            //Need to stride the value vector here if we are compressing an activation
            } // for height
        } //for Groups
     } //for Tensors
}

t_aligned_transfer_block_vector& AlignedTensor::getTransferBlockVector()
{
    return valueVector;
}

unsigned int AlignedTensor::getExternalMemoryAddressStride()
{
    return externalMemoryAddressStride;
}

void AlignedTensor::decodeTensor(
        std::vector<fixedPointNumber> &_fixedPointVector,
        signed char _fracWidth,
        signed char _intWidth)
{
    fixedPointNumber fpZero(0.0f, _fracWidth, _intWidth);

    //Calculatethe number of elements in the fixedPointVector
    unsigned int numElements =
            (this->num3DTensors) * (this->numChannelGroups)
            * (this->width) * (this->height) * (this->maxScalarIndexInChannelGroup + 1);

    unsigned int numTBPerChannelGroup =
           1 + (1+maxScalarIndexInChannelGroup - 1) / ((1+maxScalarIndexInCluster) * (1+maxClusterIndexInTransferBlock));

    //Allocate space for the result tensor
    _fixedPointVector.resize(numElements,fpZero);

    //Copy the values
    unsigned int iFullVector = 0;
    for (unsigned int iTensor=0; iTensor<(this->num3DTensors); iTensor++)
    {
        unsigned int iAlignedVector = externalMemoryAddressStride * iTensor;

        for (unsigned int iGroup=0; iGroup<(this->numChannelGroups); iGroup++)
        {
            for (unsigned int iHeight=0; iHeight<(this->height); iHeight++)
            {
                for (unsigned int iWidth=0; iWidth<(this->width); iWidth++)
                {
                    //Overide the startign position in the aligned tensor if the tensor is
                    //an activation tensor
                    iAlignedVector = (!isKernel) ?
                                (iGroup*height*width + width*iHeight + iWidth) * externalMemoryAddressStride : iAlignedVector;

                    unsigned int iChannel = 0;
                    for (unsigned int iChannelTBInGroup=0;
                         iChannelTBInGroup < numTBPerChannelGroup;
                         iChannelTBInGroup++) {

                        t_transfer_block transferBlock =
                                valueVector.at(iAlignedVector++);

                        for (int iClusterInTB=0; iClusterInTB<=(this->maxClusterIndexInTransferBlock); iClusterInTB++)
                        {
                            for (int iScalarInCluster=0; iScalarInCluster<=(this->maxScalarIndexInCluster); iScalarInCluster++)
                            {
                                if (iChannel <= (this->maxScalarIndexInChannelGroup))
                                {
                                    signed char value = transferBlock.values[iClusterInTB*CLUSTER_SIZE + iScalarInCluster];
                                    _fixedPointVector.at(iFullVector) = fixedPointNumber((signed char) value, _fracWidth, _intWidth);

                                    iFullVector++;
                                    iChannel++;
                                }
                            }
                        } //over one TB

                    } // for channel in a channel group
                }
            }
        }
    } // for over 3D tensors
}

t_aligned_streamblock_address_vector& AlignedTensor::getTransferBlockCountVector()
{
    throw std::runtime_error("AlignedTensor::getTransferBlockCountVector() is not implemented");
}

//Constructor for initialize a sparse vector
FlexibleDirectCompressedTensor::FlexibleDirectCompressedTensor (
        unsigned short _num3DTensors,
        unsigned short _channel,
        unsigned short _width,
        unsigned short _height,
        unsigned short _maxScalarIndexInChannelGroup,
        unsigned char _maxClusterIndexInCompressionBlock,
        unsigned char _maxClusterIndexInTransferBlock,
        unsigned char _maxScalarIndexInCluster,
        bool _isKernel
        )
    : AlignedTensor(
          _num3DTensors,
          _channel,
          _width,
          _height,
          _maxScalarIndexInChannelGroup,
          _maxClusterIndexInTransferBlock,
          _maxScalarIndexInCluster,
          _isKernel
          )
{
    //Assign values to the member variables
    maxClusterIndexInCompressionBlock = _maxClusterIndexInCompressionBlock;
    maxClusterIndexInTransferBlock = _maxClusterIndexInTransferBlock;
    maxScalarIndexInCompressionBlock = (maxClusterIndexInCompressionBlock+1)*(maxScalarIndexInCluster+1) - 1;

    //Compute the number of transfer blocks in a compression block
    //Need an extra one at the end to account for the bitmask
    // which occupies its own transfer block
    unsigned int numTransferBlocksPerCompressionBlock
            =  (unsigned int) std::ceil ( (float) (1 + maxClusterIndexInCompressionBlock + TRANSFER_SIZE) / (float) (maxClusterIndexInTransferBlock + 1));

    //==========================================================
    //Compute the memory stride and allocate space in the vectors
    if (isKernel) {
        externalMemoryAddressStride = calculateExternalMemoryAddressStride(
                    _channel, //channelsPerGroup
                    1, //group
                    height, //height
                    width, //width
                    maxScalarIndexInCluster+1, //clusterSize
                    maxClusterIndexInTransferBlock+1, //transferBlockSize
                    maxClusterIndexInCompressionBlock+1, //compressionWindowSize, unused
                    WIDE_SIZE, //numTransferBlockPerDramBlock
                    true, //isKernel
                    false //isDense
                    );
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
        externalMemoryAddressStride = calculateExternalMemoryAddressStride(
                    maxScalarIndexInChannelGroup+1, //channelsPerGroup
                    numChannelGroups, //group
                    height, //height
                    width, //width
                    maxScalarIndexInCluster+1, //clusterSize
                    maxClusterIndexInTransferBlock+1, //transferBlockSize
                    maxClusterIndexInCompressionBlock+1, //compressionWindowSize, unused
                    WIDE_SIZE, //numTransferBlockPerDramBlock
                    false, //isKernel
                    false //isDense
                    );
        valueVector.resize( width * height * numChannelGroups * externalMemoryAddressStride);
        streamBlockAddressVector.resize(width * height * numChannelGroups);
    }
}

FlexibleDirectCompressedTensor::FlexibleDirectCompressedTensor (
                        std::vector<fixedPointNumber> & fixedPointVector,
                        unsigned short _num3DTensors,
                        unsigned short _channel,
                        unsigned short _width,
                        unsigned short _height,
                       // unsigned short _tilingSizeWidth,
                        unsigned short _maxScalarIndexInChannelGroup,
                        unsigned char _maxClusterIndexInCompressionBlock,
                        unsigned char _maxClusterIndexInTransferBlock,
                        unsigned char _maxScalarIndexInCluster,
                        bool _isKernel

                    ):
    //Constructor delegation
    FlexibleDirectCompressedTensor(
                 _num3DTensors,
                 _channel,
                 _width,
                 _height,
                 _maxScalarIndexInChannelGroup,
                 _maxClusterIndexInCompressionBlock,
                 _maxClusterIndexInTransferBlock,
                 _maxScalarIndexInCluster,
                 _isKernel
                 )
{
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
                           unsigned char bitmaskBytes[NUM_BITMASK_BYTES] = {0x0};
                           unsigned char numSurivingClustes = 0x0;
                           for (unsigned int i=0; i<=maxClusterIndexInCompressionBlock; i++) {
                               bool preserve = false;
                               for (unsigned int j=0; j<=maxScalarIndexInCluster; j++)
                               {
                                    signed char fpValue = compressionBlock.at(i*(maxScalarIndexInCluster+1) + j);
                                    preserve  = preserve || (fpValue != 0x0);
                               }
                               unsigned char bit = (preserve) ?
                                           0x01 : 0x00;
                               numSurivingClustes = (preserve) ? (numSurivingClustes) + 1 : numSurivingClustes;
                               unsigned char bitIndex = i % 8;
                               unsigned char byteIndex = i / 8;
                               bitmaskBytes[byteIndex] |= (bit << bitIndex);
                           } // for bitmask

                           //Populate the value vector
                           t_transfer_block transferBlock;
                           //First load the bitmask
                           //Assume it is 8 bits
                           for (int i=0; i<NUM_BITMASK_BYTES; i++)
                           {
                               transferBlock.values[i] = bitmaskBytes[i];
                           }
                           //transferBlock.values[SURVIVING_COUNT_TRANSFER_BLOCK_INDEX].cluster_values[SURVIVING_COUNT_CLUSTER_INDEX] = numSurivingClustes;

                           //std::cout <<"bitmask L: "<<(int)(bitmask & 0xFF)<<std::endl;

                           //Bitmask and count occupies its own transfer block
                           valueVector.at(iCompressVector++) = transferBlock;
                            //unsigned char iTransferBlock = 1; //account for the bitmask;
                           unsigned char iTransferBlock = 0;
                           streamBlockAddress++;

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
                                   for (int i=0; i<CLUSTER_SIZE; i++)
                                   {
                                       transferBlock.values[iTransferBlock*CLUSTER_SIZE+i] = cluster.cluster_values[i];
                                   }
                                   iTransferBlock++;
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

t_aligned_streamblock_address_vector& FlexibleDirectCompressedTensor::getTransferBlockCountVector()
{
    return (this->streamBlockAddressVector);
}

void FlexibleDirectCompressedTensor::decodeTensor(
        std::vector<fixedPointNumber> &_fixedPointVector
        ,signed char _fracWidth
        ,signed char _intWidth)
{
    //Dimension of the uncompressed, un-vectorized tensor
    unsigned short num3DTensors = this->num3DTensors;
    unsigned short channel = this->channel;
    unsigned short width = this->width;
    unsigned short height = this->height;
    //unsigned short tileSizeWidth = compTensor.tilingSizeWidth;
    bool isKernel = this->isKernel;

    //Compression parameters
    unsigned short maxScalarIndexInChannelGroup = this->maxScalarIndexInChannelGroup;
    unsigned char maxScalarIndexInCompressionBlock
        = this->maxScalarIndexInCompressionBlock;
    unsigned char maxClusterIndexInCompressionBlock
        = this->maxClusterIndexInCompressionBlock;
    unsigned char maxClusterIndexInTransferBlock
        = this->maxClusterIndexInTransferBlock;
    unsigned char maxScalarIndexInCluster =
            this->maxScalarIndexInCluster;
    unsigned short numChannelGroups = this->numChannelGroups;

    //Word stride between the start of adjacent rows in the external memory
    unsigned int externalMemoryAddressStride
        = this->externalMemoryAddressStride;

    //Allocate space for the dense tensor
    _fixedPointVector.resize(num3DTensors*channel*width*height, fixedPointNumber(0.0f, _fracWidth, _intWidth));

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


    //Read the stream blocks, one at a time
    for (unsigned int iStreamBlock=0;
        iStreamBlock < numStreamBlocksInTensor;
        iStreamBlock++)
    {
        //std::cout <<"flexibleDirectDecompression. iStreamBlock="<<iStreamBlock<<std::endl;
        //Need to know how many transfer blocks are there in the stream block that we
        //are about to read
        int numTransferBlockInStreamBlock = this->streamBlockAddressVector.at(iStreamBlock);

        //============Expands a compression block============
        // A assemble vector for the decompression
        std::vector<signed char> vectorCompressionBlock;
        unsigned char bitmaskBytes[NUM_BITMASK_BYTES] = {0};

        bool updateBitmask = true;
        unsigned char numNZClustersInCompressionBlock;
        unsigned char countNZClustersInCompressionBlock;

        //Decode each stream block
        for (int i=0; i<numTransferBlockInStreamBlock; i++) {
            t_transfer_block transferBlock =
               this->valueVector.at(iCompressVector++);

            if (updateBitmask) {
                for (int i=0; i<NUM_BITMASK_BYTES; i++)
                {
                    bitmaskBytes[i] = (unsigned char) transferBlock.values[i];
                }

                numNZClustersInCompressionBlock = countNumOnes(bitmaskBytes);
                //numNZClustersInCompressionBlock = transferBlock.values[SURVIVING_COUNT_TRANSFER_BLOCK_INDEX].cluster_values[SURVIVING_COUNT_CLUSTER_INDEX];
                countNZClustersInCompressionBlock = 0;
                vectorCompressionBlock.resize(0);
                //std::cout <<"bitmask R: "<<std::bitset<8>(bitmask & 0xFF)<<std::endl;
                //std::cout <<"numNZClustersInCompressionBlock: "<<(int)(numNZClustersInCompressionBlock)<<std::endl;
                updateBitmask = false;
//                for (int i=1; i<=maxClusterIndexInTransferBlock; i++)
//                {
//                    for (int j=0; j<=maxScalarIndexInCluster; j++)
//                    {
//                        vectorCompressionBlock.push_back(transferBlock.values[i].cluster_values[j]);
//                    }
//                }
                 //countNZClustersInCompressionBlock += (maxClusterIndexInTransferBlock);
            }
            else {
                for (int i=0; i<((maxClusterIndexInTransferBlock+1)*(maxScalarIndexInCluster+1)); i++)
                {
                    vectorCompressionBlock.push_back(transferBlock.values[i]);
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
                while (positionInCompressionBlock < COMPRESSION_WINDOW_SIZE)
                {
                    unsigned char indexInCompressionBlock = findFirstNonZero(bitmaskBytes, positionInCompressionBlock);

                    if (indexInCompressionBlock < COMPRESSION_WINDOW_SIZE)
                    {
                        for (int i=0; i<=maxScalarIndexInCluster; i++) {
                            int iChannelInGroup = iChannelInGroupBase + (int) indexInCompressionBlock*(maxScalarIndexInCluster+1) + i;

                            if (iChannelInGroup <= maxScalarIndexInChannelGroup) {
                                fixedPointNumber fpValue (
                                    (signed char) (vectorCompressionBlock.at(iVectorCompressionBlock++)), _fracWidth, _intWidth );
    //                            int iDenseVector =
    //                                iTensor * height * width * channel
    //                                + iHeight * width * channel
    //                                + iTileWidth * tileSizeWidth * channel
    //                                + iLocalWidth * channel
    //                                + iChannel;
//                                int iDenseVector =
//                                    iTensor * height * width * channel
//                                    + iHeight * width * channel
//                                    + iWidth * channel
//                                    + iGroup * (maxScalarIndexInChannelGroup + 1)
//                                    + iChannelInGroup;
                                int iDenseVector =
                                    iTensor * height * width * channel
                                        + iGroup * (maxScalarIndexInChannelGroup + 1) * height*width
                                        + iHeight * width * (maxScalarIndexInChannelGroup + 1)
                                        + iWidth * (maxScalarIndexInChannelGroup + 1)
                                        + iChannelInGroup;

                               // std::cout <<"Bitmask, iTensor, iGroup, iHeight, iWidth, iChannelInGroup: "
                               //    <<std::bitset<8> (bitmask)<<" "<<iTensor<<" "<<iGroup<<" "<<iHeight<<" "<<iWidth<<" "<<iChannelInGroup<<std::endl;

                                _fixedPointVector.at(iDenseVector) = fpValue;
                            } //if iChannelInGroup <= maxScalarIndexInChannelGroup
                        } // for from 0 to maxScalarIndexInCluster
                    }

                    //Update bitmask
                    positionInCompressionBlock = indexInCompressionBlock + 1;

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
}

unsigned int calculateExternalMemoryAddressStride(
        unsigned int channelsPerGroup,
        unsigned int group,
        unsigned int height,
        unsigned int width,
        unsigned int clusterSize,
        unsigned int transferBlockSize,
        unsigned int compressionWindowSize,
        unsigned int numTransferBlockPerDramBlock,
        bool isKernel,
        bool isDense)
{
    if (isDense == true)
    {
        unsigned int numScalarPerTransferBlock = clusterSize * transferBlockSize;
        unsigned int numTransferBlockPerChannelGroup = 1 + ( channelsPerGroup -1) / numScalarPerTransferBlock;
        unsigned int tempStride = 0;
        if (isKernel)
        {
            assert(group==1 && "The group parameter should be 1 if the tensor is a weight tensor.");
            tempStride = width*height*numTransferBlockPerChannelGroup;
        }
        else
        {
            tempStride = numTransferBlockPerChannelGroup;

        }
        return (1 + (tempStride - 1)/ numTransferBlockPerDramBlock) * numTransferBlockPerDramBlock;
    }
    else
    {
        //Compute the number of transfer blocks in a compression block
        //Need an extra one at the end to account for the bitmask
        // which occupies its own transfer block
        unsigned int numTransferBlocksPerCompressionBlock
                = 1 + (compressionWindowSize - 1 + transferBlockSize) / transferBlockSize;
        unsigned int tempStride = 0;
        if (isKernel)
        {
            assert(group==1 && "The group parameter should be 1 if the tensor is a weight tensor.");
            unsigned int numCompressionBlocksInChannel =
                    1 + (channelsPerGroup*group - 1) / (compressionWindowSize * clusterSize);
            tempStride = width*height*((unsigned int) numTransferBlocksPerCompressionBlock* (unsigned int) numCompressionBlocksInChannel);
        }
        else
        {
            unsigned int numCompressionBlocksInChannelGroup =
                    1 + (channelsPerGroup - 1) / (compressionWindowSize * clusterSize);
            tempStride = (numCompressionBlocksInChannelGroup * numTransferBlocksPerCompressionBlock);
        }
        return (1 + (tempStride - 1) / numTransferBlockPerDramBlock) * numTransferBlockPerDramBlock;
    }
}

