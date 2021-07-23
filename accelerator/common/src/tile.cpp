#include "tile.hpp"
#include "params.hpp"


#define DIVIDE_CEIL(x, y) (1 + (x-1) / (y))

t_graph_output_tile_info deriveConvOutputTileShape(
        unsigned int outputHeight,
        unsigned int outputWidth,
        unsigned int sizeOutputFullTileHeight,
        unsigned int sizeOutputFullTileWidthPerCol,
        bool isConv
        )
{
    t_graph_output_tile_info tileInfo;
    unsigned int fullCols = (isConv == true) ? PE_COLS : 1;
    tileInfo.sizeOutputTileFullHeight = sizeOutputFullTileHeight;
    tileInfo.sizeOutputTileFullWidthPerCol = sizeOutputFullTileWidthPerCol;
    tileInfo.numOutputTileAlongHeight =
            1 + (outputHeight-1) / sizeOutputFullTileHeight;
    tileInfo.numFullOutputTileAlongHeight =
            outputHeight / sizeOutputFullTileHeight;

    unsigned int sizeOutputTileFullWidth =
            sizeOutputFullTileWidthPerCol * fullCols;
    tileInfo.numOutputTileAlongWidth =
            1 + (outputWidth-1) / sizeOutputTileFullWidth;
    tileInfo.numFullOutputTileAlongWidth =
            outputWidth / sizeOutputTileFullWidth;

    tileInfo.sizeOutputTilePartialHeight =
            outputHeight % sizeOutputFullTileHeight;

    unsigned int sizeOutputTilePartialWidth =
            outputWidth % sizeOutputTileFullWidth;

    if (sizeOutputTilePartialWidth == 0)
    {
        tileInfo.numActiveColsForPartialWidthTile = 0;
        tileInfo.sizeOutputTilePartialWidthPerCol =
                sizeOutputFullTileWidthPerCol;
    }
    else
    {
        unsigned int numColsForPartialWidthTile = fullCols;
        while (sizeOutputTilePartialWidth % numColsForPartialWidthTile != 0)
        {
            numColsForPartialWidthTile--;
        }
        tileInfo.sizeOutputTilePartialWidthPerCol =
                sizeOutputTilePartialWidth / numColsForPartialWidthTile;
        tileInfo.numActiveColsForPartialWidthTile =
                numColsForPartialWidthTile;
    }

    return tileInfo;
}


unsigned int deriveConvInputDimension1D(
        unsigned int outputDimension1D,
        unsigned int kernelSize,
        unsigned int kernelStride
        )
{
    unsigned int inputSPWithBorderPaddingDimension1D =
            (outputDimension1D - 1) * kernelStride + kernelSize;
    return inputSPWithBorderPaddingDimension1D;
}

unsigned int deriveNumActivationDramBlockPerStrip(
        unsigned int _numInputChannelsPerGroup
    )
{
    unsigned int numDramBlockPerStrip = DIVIDE_CEIL(_numInputChannelsPerGroup, ACTIVATION_BURST_SIZE_BYTE);
    return numDramBlockPerStrip;
}

/*!
 * Loop idle cycles.
 * Need to be verified against AOCL early estimation report
 */
#define NUM_IDLE_CYCLES_PER_FILTER_TRANSFER_FROM_W_MOVER 2
#define NUM_IDLE_CYCLES_PER_STRIP_TRANSFER_FROM_IA_MOVER 0
#define NUM_IDLE_CYCLES_PER_STRIP_TRANSFER_TO_OA_MOVER 0

unsigned int deriveDenseConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel
        )
{
    int latency;
    latency = deriveDenseComputeLatencyOneTile(
                    _numInputChannelsPerGroup,
                    CLUSTER_SIZE,
                    PE_SIMD_SIZE,
                    _sizeKernel,
                    _outputTileInfo.sizeOutputTileFullWidthPerCol,
                    _outputTileInfo.sizeOutputTileFullHeight,
                    _numOutputChannelsPerGroup,
                    PE_ROWS
                )
                * _outputTileInfo.numFullOutputTileAlongHeight * _outputTileInfo.numFullOutputTileAlongWidth;
    latency += deriveDenseComputeLatencyOneTile(
                    _numInputChannelsPerGroup,
                    CLUSTER_SIZE,
                    PE_SIMD_SIZE,
                    _sizeKernel,
                    _outputTileInfo.sizeOutputTilePartialWidthPerCol,
                    _outputTileInfo.sizeOutputTileFullHeight,
                    _numOutputChannelsPerGroup,
                    PE_ROWS
                )
                * _outputTileInfo.numFullOutputTileAlongHeight * (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth);
    latency += deriveDenseComputeLatencyOneTile(
                _numInputChannelsPerGroup,
                CLUSTER_SIZE,
                PE_SIMD_SIZE,
                _sizeKernel,
                _outputTileInfo.sizeOutputTileFullWidthPerCol,
                _outputTileInfo.sizeOutputTilePartialHeight,
                _numOutputChannelsPerGroup,
                PE_ROWS
            )
            * (_outputTileInfo.numOutputTileAlongHeight - _outputTileInfo.numFullOutputTileAlongHeight) * _outputTileInfo.numFullOutputTileAlongWidth;
    latency += deriveDenseComputeLatencyOneTile(
                _numInputChannelsPerGroup,
                CLUSTER_SIZE,
                PE_SIMD_SIZE,
                _sizeKernel,
                _outputTileInfo.sizeOutputTilePartialWidthPerCol,
                _outputTileInfo.sizeOutputTilePartialHeight,
                _numOutputChannelsPerGroup,
                PE_ROWS
            )
            * (_outputTileInfo.numOutputTileAlongHeight - _outputTileInfo.numFullOutputTileAlongHeight)
            * (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth);

    latency *= _numGroups;
    return latency;
}

unsigned int deriveSparseConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
         int _pruningRangeSizeActual
        )
{
    int latency;
    //Full height, full width
    latency = deriveSparseComputeLatencyOneTile(
                    _numInputChannelsPerGroup,
                    PE_SIMD_SIZE,
                    CLUSTER_SIZE,
                    PRUNE_RANGE_IN_CLUSTER,
                    _pruningRangeSizeActual,
                    _sizeKernel,
                    _outputTileInfo.sizeOutputTileFullWidthPerCol,
                    _outputTileInfo.sizeOutputTileFullHeight,
                    _numOutputChannelsPerGroup,
                    PE_ROWS
                )
                * _outputTileInfo.numFullOutputTileAlongHeight * _outputTileInfo.numFullOutputTileAlongWidth;
    //Full height, partial width
    latency += deriveSparseComputeLatencyOneTile(
                _numInputChannelsPerGroup,
                PE_SIMD_SIZE,
                CLUSTER_SIZE,
                PRUNE_RANGE_IN_CLUSTER,
                _pruningRangeSizeActual,
                _sizeKernel,
                _outputTileInfo.sizeOutputTilePartialWidthPerCol,
                _outputTileInfo.sizeOutputTileFullHeight,
                _numOutputChannelsPerGroup,
                PE_ROWS
            )
                * _outputTileInfo.numFullOutputTileAlongHeight * (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth);
    //Partial height, full width
    latency += deriveSparseComputeLatencyOneTile(
                _numInputChannelsPerGroup,
                PE_SIMD_SIZE,
                CLUSTER_SIZE,
                PRUNE_RANGE_IN_CLUSTER,
                _pruningRangeSizeActual,
                _sizeKernel,
                _outputTileInfo.sizeOutputTileFullWidthPerCol,
                _outputTileInfo.sizeOutputTilePartialHeight,
                _numOutputChannelsPerGroup,
                PE_ROWS
            )
            * (_outputTileInfo.numOutputTileAlongHeight - _outputTileInfo.numFullOutputTileAlongHeight) * _outputTileInfo.numFullOutputTileAlongWidth;
    latency += deriveSparseComputeLatencyOneTile(
                _numInputChannelsPerGroup,
                PE_SIMD_SIZE,
                CLUSTER_SIZE,
                PRUNE_RANGE_IN_CLUSTER,
                _pruningRangeSizeActual,
                _sizeKernel,
                _outputTileInfo.sizeOutputTilePartialWidthPerCol,
                _outputTileInfo.sizeOutputTilePartialHeight,
                _numOutputChannelsPerGroup,
                PE_ROWS
            )
            * (_outputTileInfo.numOutputTileAlongHeight - _outputTileInfo.numFullOutputTileAlongHeight)
            * (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth);

    latency *= _numGroups;
    return latency;
}

unsigned int deriveInputTransferLatency(t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        unsigned int _sizeStride,
        bool isConv,
        unsigned int _bw)
{
    unsigned int numFullOutputTileAlongHeight =
            _outputTileInfo.numFullOutputTileAlongHeight;
    unsigned int numPartialTileAlongHeight =
            _outputTileInfo.numOutputTileAlongHeight - numFullOutputTileAlongHeight;
    unsigned int numFullOutputTileAlongWidth =
            _outputTileInfo.numFullOutputTileAlongWidth;
    unsigned int numPartialTileAlongWidth =
            _outputTileInfo.numOutputTileAlongWidth - numFullOutputTileAlongWidth;
    unsigned int sizeFullTileInputHeight =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTileFullHeight,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    unsigned int sizePartialTileInputHeight =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTilePartialHeight,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    int cols = isConv ? PE_COLS : 1;
    unsigned int sizeFullTileInputWidth =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTileFullWidthPerCol * cols,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    unsigned int sizePartialTileInputWidth =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTilePartialWidthPerCol * _outputTileInfo.numActiveColsForPartialWidthTile,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    unsigned int latency = deriveInputTranserLatencyOneTile(
                   sizeFullTileInputWidth,
                   sizeFullTileInputHeight,
                   _numInputChannelsPerGroup,
                   _bw
                ) * numFullOutputTileAlongHeight*numFullOutputTileAlongWidth;
    latency += deriveInputTranserLatencyOneTile(
                sizePartialTileInputWidth,
                sizeFullTileInputHeight,
                _numInputChannelsPerGroup,
                _bw
             ) * numFullOutputTileAlongHeight*numPartialTileAlongWidth;
    latency += deriveInputTranserLatencyOneTile(
                sizeFullTileInputWidth,
                sizePartialTileInputHeight,
                _numInputChannelsPerGroup,
                _bw
             ) * numPartialTileAlongHeight*numFullOutputTileAlongWidth;
    latency += deriveInputTranserLatencyOneTile(
                sizePartialTileInputWidth,
                sizePartialTileInputHeight,
                _numInputChannelsPerGroup,
                _bw
             ) * numPartialTileAlongHeight*numPartialTileAlongWidth;
    latency *= _numGroups;
    return latency;
 }

unsigned int deriveOutputTransferLatency(t_graph_output_tile_info _outputTileInfo,
        unsigned int _sizeOutputHeight,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        bool isConv,
        unsigned int _bw)
{
    int cols = isConv ? PE_COLS : 1;
    unsigned int sizeOutputWidth =
            _outputTileInfo.numFullOutputTileAlongWidth * _outputTileInfo.sizeOutputTileFullWidthPerCol * cols
            + (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth)
                   * _outputTileInfo.sizeOutputTilePartialWidthPerCol * _outputTileInfo.numActiveColsForPartialWidthTile;
    unsigned int latency =
            DIVIDE_CEIL(_numOutputChannelsPerGroup, _bw)
            * _numGroups
            * _sizeOutputHeight
            * sizeOutputWidth;
    return latency;
}

unsigned int deriveDenseConvWeightTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        unsigned int _bw
        )
{
    unsigned int numTileAlongHeight = _outputTileInfo.numOutputTileAlongHeight;
    unsigned int numTileAlongWidth = _outputTileInfo.numOutputTileAlongWidth;
    unsigned int roundedChannels = DIVIDE_CEIL(_numInputChannelsPerGroup, PE_SIMD_SIZE * CLUSTER_SIZE) * PE_SIMD_SIZE * CLUSTER_SIZE;
    unsigned int latency =
            _numGroups * _numOutputChannelsPerGroup * numTileAlongHeight * numTileAlongWidth
            * DIVIDE_CEIL(_sizeKernel * _sizeKernel * roundedChannels, _bw);

    return latency;
}


unsigned int deriveSparseConvWeightTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        int _interPruningRangePara,
        int _clusteSize,
        int _pruningRangeSizeFull,
        int _pruningRangeSizeActual,
        int _bw
        )
{
    unsigned int effectiveICPerGroup = _numInputChannelsPerGroup < (_interPruningRangePara * _clusteSize)?
                _interPruningRangePara * _clusteSize :
            DIVIDE_CEIL(_numInputChannelsPerGroup, _interPruningRangePara * _clusteSize * _pruningRangeSizeFull)
                * _interPruningRangePara * _clusteSize * _pruningRangeSizeActual;

    return deriveDenseConvWeightTransferLatency(
                    _outputTileInfo,
                    effectiveICPerGroup,
                    _numOutputChannelsPerGroup,
                    _numGroups,
                    _sizeKernel,
                    _bw
                );
}

unsigned int deriveFirstTileConvInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _sizeKernel,
        unsigned _sizeStride
        )
{
    unsigned int sizeFirstTileOutputHeight =
        (_outputTileInfo.numFullOutputTileAlongHeight==0) ?
                 _outputTileInfo.sizeOutputTilePartialHeight
              : _outputTileInfo.sizeOutputTileFullHeight;

    unsigned int sizeFirstTileOuputWidth =
         (_outputTileInfo.numFullOutputTileAlongWidth==0) ?
                _outputTileInfo.sizeOutputTilePartialWidthPerCol
                    * _outputTileInfo.numActiveColsForPartialWidthTile
               : _outputTileInfo.sizeOutputTileFullWidthPerCol
                    * PE_COLS;

    //Convert to input height and width
    unsigned int sizeFirstTileInputHeight =
            deriveConvInputDimension1D(
                sizeFirstTileOutputHeight,
                _sizeKernel,
                _sizeStride
                );
    unsigned int sizeFirstTileInputWidth =
            deriveConvInputDimension1D(
                sizeFirstTileOuputWidth,
                _sizeKernel,
                _sizeStride
                );

    return deriveInputTranserLatencyOneTile(
                    sizeFirstTileInputWidth,
                    sizeFirstTileInputHeight,
                    _numInputChannelsPerGroup,
                    ACTIVATION_BURST_SIZE_BYTE
                );
}


unsigned int deriveLastTileOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup
        )
{
    unsigned int sizeLastTileOutputHeight =
            (_outputTileInfo.numFullOutputTileAlongHeight == _outputTileInfo.numOutputTileAlongHeight)?
                _outputTileInfo.sizeOutputTileFullHeight :
                _outputTileInfo.sizeOutputTilePartialHeight;

    unsigned int sizeLastTileOutputWidthPerCol =
            (_outputTileInfo.numFullOutputTileAlongWidth == _outputTileInfo.numOutputTileAlongWidth)?
                _outputTileInfo.sizeOutputTileFullWidthPerCol
              : _outputTileInfo.sizeOutputTilePartialWidthPerCol;

    unsigned int numActiveColsLastTile =
            (_outputTileInfo.numFullOutputTileAlongWidth == _outputTileInfo.numOutputTileAlongWidth)?
                PE_COLS : _outputTileInfo.numActiveColsForPartialWidthTile;

    return  deriveOutputTranserLatencyOneTile(
                sizeLastTileOutputWidthPerCol * numActiveColsLastTile,
                sizeLastTileOutputHeight,
                _numOutputChannelsPerGroup,
                ACTIVATION_BURST_SIZE_BYTE
            );
}

int deriveDenseComputeLatencyOneTile(int _numIC,
                                     int _clusteSize,
                                     int _interClusterPara,
                                     int _kernelSize,
                                     int _outputTileWidthPerCol,
                                     int _outputTileHeightPerCol,
                                     int _outputChannel,
                                     int _numPeRows)
{
    int numPERowFoldPerGroup =
            DIVIDE_CEIL(_outputChannel, _numPeRows);
    int numTranferBlockPerInputGroup =
            DIVIDE_CEIL(_numIC, _interClusterPara * _clusteSize);
    int numTransfersPerConvWindow = numTranferBlockPerInputGroup * _kernelSize * _kernelSize;

    // Total latency
    int latency = numTransfersPerConvWindow * _outputTileWidthPerCol * _outputTileHeightPerCol * numPERowFoldPerGroup;

    return latency;

}

int deriveSparseComputeLatencyOneTile(int _numIC,
                                      int _interPruningRangePara,
                                      int _clusteSize,
                                      int _pruningRangeSizeFull,
                                      int _pruningRangeSizeActual,
                                      int _kernelSize,
                                      int _outputTileWidthPerCol,
                                      int _outputTileHeightPerCol,
                                      int _outputChannel,
                                      int _numPeRows)
{
    unsigned int effectiveIC = _numIC < (_interPruningRangePara * _clusteSize)?
                _interPruningRangePara * _clusteSize :
            DIVIDE_CEIL(_numIC, _interPruningRangePara * _clusteSize * _pruningRangeSizeFull)
                * _interPruningRangePara * _clusteSize * _pruningRangeSizeActual;
    int numPERowFoldPerGroup =
            DIVIDE_CEIL(_outputChannel, _numPeRows);
    unsigned int numActivationTransferBlocksPerStrip =  
            DIVIDE_CEIL(_numIC, _interPruningRangePara * _clusteSize * _pruningRangeSizeFull);


    //The mininum number of cycles it takes the IB to stream the transfer blocks for one tile
    int inputBufferLatency = 
        _outputTileWidthPerCol * _outputTileHeightPerCol * numActivationTransferBlocksPerStrip * numPERowFoldPerGroup;

    inputBufferLatency = (_kernelSize == 1) ?
        inputBufferLatency + _outputTileHeightPerCol * numPERowFoldPerGroup
        : inputBufferLatency + _outputTileHeightPerCol * _outputTileWidthPerCol * _kernelSize * numPERowFoldPerGroup;
    int computeLatency =  deriveDenseComputeLatencyOneTile(
                    effectiveIC,
                    _clusteSize,
                    _interPruningRangePara,
                    _kernelSize,
                    _outputTileWidthPerCol,
                    _outputTileHeightPerCol,
                    _outputChannel,
                    _numPeRows
                );

    int latency = (computeLatency > inputBufferLatency) ? computeLatency : inputBufferLatency;
    return latency;
}

int deriveInputTranserLatencyOneTile(
            int _inputTileWidth,
            int _inputTileHeight,
            int _numIC,
            int _bw
        )
{
    return _inputTileHeight * _inputTileWidth * DIVIDE_CEIL(_numIC, _bw);
}

int deriveOutputTranserLatencyOneTile(int _outputTileWidth, int _outputTileHeight, int _numOC, int _bw)
{
    return _outputTileHeight * _outputTileWidth * DIVIDE_CEIL(_numOC, _bw);
}

int deriveDenseWeightTranserLatencyOneTile(
            int _kernelSize,
            int _numIC,
            int _numOC,
            int _bw
        )
{
    return _numOC * DIVIDE_CEIL(_kernelSize * _kernelSize * _numIC, _bw);
}

int deriveSparseWeightTranserLatencyOneTile(
            int _kernelSize,
            int _numIC,
            int _numOC,
            int _sizeFullPruneRange,
            int _sizeCluster,
            int _sizeSparsesPruneRange,
            int _interPruneRangePara,
            int _bw
        )
{
    int effectiveIC = _numIC < (_interPruneRangePara * _sizeCluster)?
                _interPruneRangePara * _sizeCluster :
                DIVIDE_CEIL(effectiveIC, _interPruneRangePara * _sizeCluster * _sizeFullPruneRange)
                    * _interPruneRangePara * _sizeCluster * _sizeSparsesPruneRange;

    return deriveDenseWeightTranserLatencyOneTile(_kernelSize, effectiveIC, _numOC, _bw);
}

int ia_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numDramBlockPerDenseStrip
        )
{
    int requirement = numDramBlockPerDenseStrip*heightTile*widthTile;
    return requirement;
}

int oa_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numChannels
        )
{
    int numBlocksPerStrip = DIVIDE_CEIL(numChannels, ACTIVATION_BURST_SIZE_BYTE);
    return (heightTile*widthTile* numBlocksPerStrip);
}

//TODO: CHANGE THE SIGNATURE TO CONSIDER NUMBER OF NZ BLOCKS
int filter_cache_boundary_check(
      int kernelSize,
      int inputChannelSize,
        int peBlockSize,
        int numClustersInPruningRange,
        int numNZClustersInPruningRange
        )
{
    //Nubmer of dram blocks required to store one filter
    int numTBPerStrip = DIVIDE_CEIL(inputChannelSize, peBlockSize * numClustersInPruningRange)
            * numNZClustersInPruningRange;
    int requirement = DIVIDE_CEIL(numTBPerStrip * kernelSize * kernelSize, WEIGHT_WIDE_SIZE);
    return requirement;
}

