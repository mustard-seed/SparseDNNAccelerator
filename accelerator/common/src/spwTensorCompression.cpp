#include <iostream>
#include <cassert>
#include <cerrno>
#include <algorithm>

#include "spwTensorCompression.hpp"

#define DIVIDE_CEIL(x, y) (1 + (x-1) / (y) )

DeviceActivationTensor::DeviceActivationTensor (int _channel,
        int _width,
        int _height,
        int _stripStrideInExternalMemory)
{
    channel = _channel;
    width = _width;
    height = _height;
    stripStrideInExternalMemory = _stripStrideInExternalMemory;
    int tensorSize = width*height*stripStrideInExternalMemory;
    valueVector.resize(tensorSize, 0x0);
}

DeviceActivationTensor ::DeviceActivationTensor(std::vector<fixedPointNumber> & _vecFixedPoint,
           int _channel,
           int _width,
           int _height,
           int _stripStrideInExternalMemory)
    :DeviceActivationTensor(
            _channel,
            _width,
            _height,
            _stripStrideInExternalMemory
         )
{
    for (int iHeight=0; iHeight < height; iHeight++) {
        for (int iWidth=0; iWidth< width; iWidth++)
        {
            //Overide the startign position in the aligned tensor if the tensor is
            //an activation tensor
            int iDeviceTensor = (iHeight*width + iWidth) * stripStrideInExternalMemory;
            int iFullVector = (iHeight*width + iWidth) * channel;
            for (int iChannel=0; iChannel<channel; iChannel++) {
                valueVector.at(iDeviceTensor) = _vecFixedPoint.at(iFullVector).getBits();
                iDeviceTensor++;
                iFullVector++;
            } // for channel in a channel group
        }//iWidth
    } // for height
}

std::vector<fixedPointNumber>
DeviceActivationTensor::decodeTensor(
                signed char _fracWidth,
                signed char _intWidth
            ) const
{
    std::vector<fixedPointNumber> fullVector;
    fullVector.resize(channel*width*height);
    for (int iHeight=0; iHeight < height; iHeight++) {
        for (int iWidth=0; iWidth< width; iWidth++)
        {
            //Overide the startign position in the aligned tensor if the tensor is
            //an activation tensor
            int iDeviceTensor = (iHeight*width + iWidth) * stripStrideInExternalMemory;
            int iFullVector = (iHeight*width + iWidth) * channel;
            for (int iChannel=0; iChannel<channel; iChannel++) {
                fullVector.at(iFullVector) = fixedPointNumber
                        (valueVector.at(iDeviceTensor),
                         _fracWidth,
                         _intWidth);
                iDeviceTensor++;
                iFullVector++;
            } // for channel in a channel group
        }//iWidth
    } // for height

    return fullVector;
}

t_aligned_activation_vector &DeviceActivationTensor::getValueVector()
{
    return valueVector;
}

int
DeviceActivationTensor::getChannel() const
{
    return channel;
}
int
DeviceActivationTensor::getWidth() const
{
    return width;
}

int
DeviceActivationTensor::getHeight() const
{
    return height;
}

int
DeviceActivationTensor::getStripStride() const
{
    return stripStrideInExternalMemory;
}

DeviceWeightTensor::DeviceWeightTensor (
        int _outputChannel,
        int _inputChannel,
        int _width,
        int _height,
        int _peSimdSize,
        int _clusterSize
        )
{
    outputChannel = _outputChannel;
    inputChannel = _inputChannel;
    width = _width;
    height = _height;
    int sizeTB = _peSimdSize * _clusterSize;
    numTBPerStrip = DIVIDE_CEIL(_inputChannel, sizeTB);
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 0X1)
    numDramBlocksPerFilter = DIVIDE_CEIL(numTBPerStrip * width * height, WEIGHT_WIDE_SIZE);
#else
    numDramBlocksPerFilter = numTBPerStrip * width * height * WEIGHT_WIDE_SIZE;
#endif
    paddedInputChannel = numTBPerStrip* sizeTB;
    numWeightsInPaddedFilter = paddedInputChannel * width * height;
    t_weight_dram_block emptyBlock;
    for (int i=0; i<WEIGHT_BURST_SIZE_VALUE_BYTE; i++)
    {
        emptyBlock.values[i] = 0x0;
    }
    valueVector.resize(numDramBlocksPerFilter*outputChannel, emptyBlock);
}

DeviceWeightTensor:: DeviceWeightTensor(
        std::vector<fixedPointNumber> & _vecFixedPoint,
        int _outputChannel,
        int _inputChannel,
        int _width,
        int _height,
        int _peSimdSize,
        int _clusterSize
        ) : DeviceWeightTensor(
                _outputChannel,
                _inputChannel,
                _width,
                _height,
                _peSimdSize,
                _clusterSize
                )
{
   int sizeTB = _peSimdSize * _clusterSize;
   for (int iOC=0; iOC < outputChannel; iOC++){
       for (int iHeight=0; iHeight < height; iHeight++) {
           for (int iWidth=0; iWidth< width; iWidth++)
           {
               for (int iTB=0; iTB<numTBPerStrip; iTB++)
               {
                   //Find out the TB index that the value fits to
                   int idxTBInFilter = (iHeight*_width + iWidth) * numTBPerStrip + iTB;

                   for (int v=0; v<sizeTB; v++)
                   {
                       int idxPaddedChannel = iTB * sizeTB + v;
                       int idxFullVector =
                               iOC * height * width * inputChannel
                               + iHeight * width * inputChannel
                               + iWidth * inputChannel
                               + idxPaddedChannel;
                       if (idxPaddedChannel <inputChannel)
                       {
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 1)
                           int idxDeviceTensorDramBlock =
                                   iOC*numDramBlocksPerFilter
                                  + (idxTBInFilter / WEIGHT_WIDE_SIZE);
                           int idxInDramBlock =
                                   v + (idxTBInFilter % WEIGHT_WIDE_SIZE) * sizeTB;
#else
                           int idxDeviceTensorDramBlock =
                                   iOC*numDramBlocksPerFilter
                                  + (idxTBInFilter * WEIGHT_WIDE_SIZE)
                                  + v / WEIGHT_BURST_SIZE_VALUE_BYTE;
                           int idxInDramBlock = v % WEIGHT_BURST_SIZE_VALUE_BYTE;
#endif
                           valueVector.at(idxDeviceTensorDramBlock).values[idxInDramBlock]
                                   = _vecFixedPoint.at(idxFullVector).getBits();

                       }
                   } //for (int v=0; v<sizeTB; v++)
               } //for (int iTB=0; iTB<numTBPerStrip; iTB++)
           }//iWidth
       } // for height
    } //for outputChannel
}

std::vector<fixedPointNumber>
DeviceWeightTensor :: decodeTensor(
                signed char _fracWidth,
                signed char _intWidth
            )
{
    std::vector<fixedPointNumber> fullVector;
    fullVector.resize(outputChannel*height*width*inputChannel);
    int sizeTB = paddedInputChannel / numTBPerStrip;
    for (int iOC=0; iOC < outputChannel; iOC++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            for (int iWidth=0; iWidth< width; iWidth++)
            {
                for (int iTB=0; iTB<numTBPerStrip; iTB++)
                {
                    //Find out the TB index that the value fits to
                    int idxTBInFilter = (iHeight*width + iWidth) * numTBPerStrip + iTB;
                    for (int v=0; v<sizeTB; v++)
                    {
                        int idxPaddedChannel = iTB * sizeTB + v;
                        int idxFullVector =
                                iOC * height * width * inputChannel
                                + iHeight * width * inputChannel
                                + iWidth * inputChannel
                                + idxPaddedChannel;
                        if (idxPaddedChannel <inputChannel)
                        {
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 1)
                            int idxDeviceTensorDramBlock =
                                    iOC*numDramBlocksPerFilter
                                   + (idxTBInFilter / WEIGHT_WIDE_SIZE);
                            int idxInDramBlock =
                                    v + (idxTBInFilter % WEIGHT_WIDE_SIZE) * sizeTB;
#else
                            int idxDeviceTensorDramBlock =
                                    iOC*numDramBlocksPerFilter
                                   + (idxTBInFilter * WEIGHT_WIDE_SIZE)
                                   + v / WEIGHT_BURST_SIZE_VALUE_BYTE;
                            int idxInDramBlock = v % WEIGHT_BURST_SIZE_VALUE_BYTE;
#endif

                            signed char bits = valueVector.at(idxDeviceTensorDramBlock).values[idxInDramBlock];
                            fullVector.at(idxFullVector) = fixedPointNumber(
                                            bits,
                                            _fracWidth,
                                            _intWidth
                                        );
                        }
                    } //for (int v=0; v<sizeTB; v++)
                } //for (int iTB=0; iTB<numTBPerStrip; iTB++)
            }//iWidth
        } // for height
     } //for outputChannel

    return fullVector;
}

int
DeviceWeightTensor::getOutputChannel() const
{
    return outputChannel;
}

int
DeviceWeightTensor::getInputChannel() const
{
    return inputChannel;
}

int
DeviceWeightTensor::getWidth() const
{
    return width;
}

int
DeviceWeightTensor::getHeight() const
{
    return height;
}

int
DeviceWeightTensor::getPaddedInputChannel() const
{
    return paddedInputChannel;
}

int
DeviceWeightTensor::getNumWeightsInPaddedFilter() const
{
    return numDramBlocksPerFilter*WEIGHT_BURST_SIZE_VALUE_BYTE;
}

int
DeviceWeightTensor::getTBPerStrip() const
{
    return numTBPerStrip;
}

int
DeviceWeightTensor::getDramBlocksInFilter() const
{
    return numDramBlocksPerFilter;
}

int
DeviceWeightTensor::getStripStride() const
{
    return this->getPaddedInputChannel();
}

int
DeviceWeightTensor::getFilterStride() const
{
    return this->getNumWeightsInPaddedFilter();
}

t_aligned_weight_vector &DeviceWeightTensor::getValueVector()
{
    return valueVector;
}

#if defined(SPW_SYSTEM)
static bool absCompare(signed char a, signed char b)
{
    return (std::abs(a) < std::abs(b));
}

/*!
 * \brief getLInfNorm
 * \details Compute the L_inf norm (max absolute value) of a given vector
 * \param _group
 * \return The L_inf norm of the vector
 */
int getLInfNorm(
        const std::vector<signed char>& _group
        )
{
    auto iter = std::max_element(_group.begin(), _group.end(), absCompare);
        return std::abs(*iter);
}

int getL1Norm (
        const std::vector <signed char>& _group
        )
{
    int sum = 0;
    for (auto & val: _group) {
        sum += std::abs(val);
    }
    return sum;
}
DeviceSpWTensor::DeviceSpWTensor(
        int _outputChannel,
        int _inputChannel,
        int _width,
        int _height,
        int _peSimdSize,
        int _clusterSize,
        int _numClustersInPruningRange,
        int _numNZClustersPerPruningRange
        ) : DeviceWeightTensor(
                    _outputChannel,
                    _inputChannel,
                    _width,
                    _height,
                    _peSimdSize,
                    _clusterSize
                )
{
    outputChannel = _outputChannel;
    inputChannel = _inputChannel;
    width = _width;
    height = _height;
    numClustersInPruningRange = _numClustersInPruningRange;
    peSimdSize = _peSimdSize;
    clusterSize = _clusterSize;
    if (_numNZClustersPerPruningRange > numClustersInPruningRange)
    {
        std::cerr <<"Number of non-zero clusters in a pruning range should not exceed "
                    "the size of the pruning range in terms of cluster size"<<std::endl;
        throw;
    }
    if (_numNZClustersPerPruningRange <= 0)
    {
        numNZClustersInPruningRange = 1;
    }
    else
    {
        numNZClustersInPruningRange = _numNZClustersPerPruningRange;
    }
    int sizeTB = _peSimdSize * _clusterSize;
    numTBPerStrip =
            DIVIDE_CEIL(_inputChannel, sizeTB * numClustersInPruningRange)
            * numNZClustersInPruningRange;
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 1)
    numDramBlocksPerFilter = DIVIDE_CEIL(numTBPerStrip * height * width, WEIGHT_WIDE_SIZE);
#else
    numDramBlocksPerFilter = numTBPerStrip * height * width * WEIGHT_WIDE_SIZE;
#endif
    paddedInputChannel = numTBPerStrip* sizeTB;
    numWeightsInPaddedFilter = paddedInputChannel * width * height;
    t_weight_dram_block emptyBlock;
    for (int i=0; i<WEIGHT_BURST_SIZE_VALUE_BYTE; i++)
    {
        emptyBlock.values[i] = 0x0;
    }
    for (int j=0; j<WEIGHT_BURST_SIZE_INDEX_BYTE; j++)
    {
        emptyBlock.indices[j] = 0x0;
    }
    valueVector.resize(numDramBlocksPerFilter*outputChannel, emptyBlock);
}

DeviceSpWTensor::DeviceSpWTensor(
        std::vector<fixedPointNumber> & _vecFixedPoint,
        int _outputChannel,
        int _inputChannel,
        int _width,
        int _height,
        int _peSimdSize,
        int _clusterSize,
        int _numClustersInPruningRange,
        int _numNZClustersPerPruningRange
        ) : DeviceSpWTensor (
                _outputChannel,
                _inputChannel,
                _width,
                _height,
                _peSimdSize,
                _clusterSize,
                _numClustersInPruningRange,
                _numNZClustersPerPruningRange
                )
{
    int sizeTB = _peSimdSize * _clusterSize;
    int sizeCompressionWindow = sizeTB * numClustersInPruningRange;
    int numCompressionWindowPerStrip = DIVIDE_CEIL(inputChannel, sizeCompressionWindow);
    for (int iOC=0; iOC < outputChannel; iOC++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            for (int iWidth=0; iWidth< width; iWidth++)
            {
                for (int iCompressionWindow=0;
                     iCompressionWindow<numCompressionWindowPerStrip;
                     iCompressionWindow++)
                {
                    //Iterate through all the pruning ranges in the compression window
                    //Rank the clusters in each pruning range according to L1 Norm
                    //Scatter the top-ranked clusters to the weight dram blocks
                    for (int iPruneRange=0; iPruneRange < _peSimdSize; iPruneRange++)
                    {
                        std::vector<int> norms(numClustersInPruningRange, 0);
                        //Calculate the L1 norms of the clusters in a pruning range
                        for (int iCluster=0; iCluster<numClustersInPruningRange; iCluster++)
                        {
                            std::vector<signed char> vecCluster(_clusterSize, 0x0);

                            //Gather the data from the full vector from the dense vector
                            //into the cluster
                            for (int iV=0; iV<_clusterSize; iV++)
                            {
                                int idxChannel =
                                        iCompressionWindow * sizeCompressionWindow
                                        + iPruneRange * numClustersInPruningRange * _clusterSize
                                        + iCluster * _clusterSize
                                        + iV;
                                if (idxChannel < inputChannel)
                                {
                                    int idxFullVector =
                                            iOC * height * width * inputChannel
                                            + iHeight * width * inputChannel
                                            + iWidth * inputChannel
                                            + idxChannel;
                                    vecCluster.at(iV) = _vecFixedPoint.at(idxFullVector).getBits();
                                }
                            }

                            //int norm = getLInfNorm(vecCluster);
                            int norm = getL1Norm(vecCluster);
                            norms.at(iCluster) = norm;
                        }

                        /*
                         * Rank the cluster indices according to the Linf norms
                        */
                        //Generate a list of indices
                        std::vector<int> indices(numClustersInPruningRange, 0);
                        //Use iota to populate the indices, starting from 0
                        std::iota(indices.begin(), indices.end(), 0);
                        //Use the lambda expression to sort the indices based on the Linf norm non-ascending order
                        std::stable_sort(indices.begin(), indices.end(),
                                         [&norms](int i1, int i2) {return norms.at(i1) > norms.at(i2);});


                        /*!
                          Scatter the top ranked clusters into the compressed tensor
                        */
                        for (int iTopCluster=0; iTopCluster<numNZClustersInPruningRange; iTopCluster++)
                        {
                            int idxTBInCompressedFilter =
                                    (iHeight * width + iWidth) * numTBPerStrip
                                    + iCompressionWindow * numNZClustersInPruningRange + iTopCluster;
                            int idxClusterInPruneRange = indices.at(iTopCluster);
                            for (int v=0; v<_clusterSize; v++)
                            {
                                int idxChannelInDenseVector =
                                        iCompressionWindow * sizeCompressionWindow
                                        + iPruneRange * numClustersInPruningRange * _clusterSize
                                        + idxClusterInPruneRange * _clusterSize
                                        + v;
                                if (idxChannelInDenseVector < inputChannel)
                                {
                                    int idxFullVector =
                                            iOC * height * width * inputChannel
                                            + iHeight * width * inputChannel
                                            + iWidth * inputChannel
                                            + idxChannelInDenseVector;
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 1)
                                    int idxWeightDramBlock =
                                            iOC * numDramBlocksPerFilter
                                            + (idxTBInCompressedFilter / WEIGHT_WIDE_SIZE);
                                    int idxInDramBlock =
                                            (idxTBInCompressedFilter % WEIGHT_WIDE_SIZE) * sizeTB
                                            + iPruneRange*_clusterSize + v;
#else
                                    int idxWeightDramBlock =
                                            iOC * numDramBlocksPerFilter
                                            + (idxTBInCompressedFilter * WEIGHT_WIDE_SIZE);
                                    int idxInDramBlock =
                                          (iPruneRange*_clusterSize + v) % WEIGHT_BURST_SIZE_VALUE_BYTE;
#endif
                                    valueVector.at(idxWeightDramBlock).values[idxInDramBlock] = _vecFixedPoint.at(idxFullVector).getBits();
                                }
                            } //for. for (int v=0; v<_clusterSize; v++)

                            //Assign the bitmasks
                            int idxMaskInTB = iPruneRange / 2;
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 1)
                                int idxWeightDramBlock =
                                        iOC * numDramBlocksPerFilter
                                        + (idxTBInCompressedFilter / WEIGHT_WIDE_SIZE);
                                int idxMaskByteInCompressedBlock =
                                        (idxTBInCompressedFilter % WEIGHT_WIDE_SIZE) * INDEX_CHAR_ARRAY_SIZE
                                        + idxMaskInTB;
#else
                                int idxWeightDramBlock =
                                        iOC * numDramBlocksPerFilter
                                        + (idxTBInCompressedFilter * WEIGHT_WIDE_SIZE);
                                int idxMaskByteInCompressedBlock =
                                        idxMaskInTB % WEIGHT_BURST_SIZE_INDEX_BYTE;
#endif

                            if (iPruneRange % 2 == 0)
                            {
                                //Clear the lower 4 bits
                                valueVector.at(idxWeightDramBlock).indices[idxMaskByteInCompressedBlock]
                                        &= 0x0F0;
                                //Set the lower 4 bits
                                valueVector.at(idxWeightDramBlock).indices[idxMaskByteInCompressedBlock]
                                        |= idxClusterInPruneRange & CHAR_TO_SPW_INDEX_MASK;
                            }
                            else
                            {
                                //Clear the higher 4 bits
                                valueVector.at(idxWeightDramBlock).indices[idxMaskByteInCompressedBlock]
                                        &= 0x00F;
                                //Set the higher 4 bits
                                valueVector.at(idxWeightDramBlock).indices[idxMaskByteInCompressedBlock]
                                        |= (idxClusterInPruneRange & CHAR_TO_SPW_INDEX_MASK) << 0x4;
                            }
                        } //for. iTopCluster
                    } //for. iPruneRange
                } //iCompressionWindow
            }//iWidth
        } // for height
     } //for outputChannel
}

std::vector<fixedPointNumber>
DeviceSpWTensor :: decodeTensor(
                signed char _fracWidth,
                signed char _intWidth
            )
{
    int sizeTB = paddedInputChannel / numTBPerStrip;
    int sizeCompressionWindow = sizeTB * numClustersInPruningRange;
    int numCompressionWindowPerStrip = DIVIDE_CEIL(inputChannel, sizeCompressionWindow);
    std::vector<fixedPointNumber> fullVector;
    fullVector.resize(outputChannel*height*width*inputChannel);
    for (int iOC=0; iOC < outputChannel; iOC++){
        for (int iHeight=0; iHeight < height; iHeight++) {
            for (int iWidth=0; iWidth< width; iWidth++)
            {
                for (int iCompressionWindow=0;
                     iCompressionWindow<numCompressionWindowPerStrip;
                     iCompressionWindow++)
                {
                    //Iterate through all the pruning ranges in the compression window
                    //Inspect the sparse indice, and use them to scatter the NZ clusters
                    //back to the full vector.
                    for (int iPruneRange=0; iPruneRange < peSimdSize; iPruneRange++)
                    {
                        for (int iTopCluster=0; iTopCluster<numNZClustersInPruningRange; iTopCluster++)
                        {
                            int idxTBInCompressedFilter =
                                    (iHeight * width + iWidth) * numTBPerStrip
                                    + iCompressionWindow * numNZClustersInPruningRange + iTopCluster;
                            int idxMaskInTB = iPruneRange / 2;
#if (WEIGHT_BURST_SIZE_GEQ_PE_SIZE == 1)
                            int idxWeightDramBlock =
                                    iOC * numDramBlocksPerFilter
                                    + (idxTBInCompressedFilter / WEIGHT_WIDE_SIZE);
                            int idxMaskByteInCompressedBlock =
                                    (idxTBInCompressedFilter % WEIGHT_WIDE_SIZE) * INDEX_CHAR_ARRAY_SIZE
                                    + idxMaskInTB;
#else
                            int idxWeightDramBlock =
                                    iOC * numDramBlocksPerFilter
                                    + (idxTBInCompressedFilter * WEIGHT_WIDE_SIZE);
                            int idxMaskByteInCompressedBlock =
                                    idxMaskInTB % WEIGHT_BURST_SIZE_INDEX_BYTE;
#endif
                            int idxClusterInPruneRange;
                            if (iPruneRange % 2 == 0)
                            {

                                idxClusterInPruneRange
                                        = valueVector.at(idxWeightDramBlock).indices[idxMaskByteInCompressedBlock]
                                        & CHAR_TO_SPW_INDEX_MASK;
                            }
                            else
                            {
                                idxClusterInPruneRange
                                        = (valueVector.at(idxWeightDramBlock).indices[idxMaskByteInCompressedBlock] >> 0x04)
                                        & CHAR_TO_SPW_INDEX_MASK;
                            }

                            for (int v=0; v<clusterSize; v++)
                            {
                                int idxChannelInDenseVector =
                                        iCompressionWindow * sizeCompressionWindow
                                        + iPruneRange * numClustersInPruningRange * clusterSize
                                        + idxClusterInPruneRange * clusterSize
                                        + v;
                                if (idxChannelInDenseVector < inputChannel)
                                {
                                    int idxFullVector =
                                            iOC * height * width * inputChannel
                                            + iHeight * width * inputChannel
                                            + iWidth * inputChannel
                                            + idxChannelInDenseVector;
                                    int idxInDramBlock =
                                            (idxTBInCompressedFilter % WEIGHT_WIDE_SIZE) * sizeTB
                                            + iPruneRange*clusterSize + v;
                                    signed char bits = valueVector.at(idxWeightDramBlock).values[idxInDramBlock];
                                    fixedPointNumber fpValue(bits, _fracWidth, _intWidth);
                                    fullVector.at(idxFullVector) = fpValue;
                                }
                            } //for. for (int v=0; v<_clusterSize; v++)
                        } //for. iTopCluster
                    } //for. iPruneRange
                } //iCompressionWindow
            }//iWidth
        } // for height
     } //for outputChannel

    return fullVector;
}

#endif //SPW_SYSTEM
