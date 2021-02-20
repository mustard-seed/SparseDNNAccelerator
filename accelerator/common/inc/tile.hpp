#ifndef TILE_HPP
#define TILE_HPP
typedef struct {
    unsigned int sizeOutputTileFullHeight;
    unsigned int sizeOutputTilePartialHeight;
    unsigned int sizeOutputTileFullWidthPerCol;
    unsigned int sizeOutputTilePartialWidthPerCol;
    unsigned int numActiveColsForPartialWidthTile;
    unsigned int numFullOutputTileAlongWidth;
    unsigned int numOutputTileAlongWidth;
    unsigned int numFullOutputTileAlongHeight;
    unsigned int numOutputTileAlongHeight;
} t_graph_output_tile_info;

typedef struct {
    int     inputTransferLatency;
    int     weightTransferLatency;
    int     outputTransferLatency;
    int     computeLatency;
    int     computeLatencyWithOverhead;
    int     ddrLatency;
    unsigned int totalLatency;
    bool    isComputeBound;
} t_latency_info;

t_graph_output_tile_info deriveConvOutputTileShape(unsigned int outputHeight,
        unsigned int outputWidth,
        unsigned int sizeOutputFullTileHeight,
        unsigned int sizeOutputFullTileWidthPerCol
        , bool isConv);

unsigned int deriveConvInputDimension1D(
        unsigned int outputDimension1D,
        unsigned int kernelSize,
        unsigned int kernelStride
        );

unsigned int deriveDenseConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel
        );

unsigned int deriveSparseConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
         int _pruningRangeSizeActual
        );

unsigned int deriveInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        unsigned int _sizeStride,
        bool isConv,
        unsigned int _bw
        );

unsigned int deriveDenseConvWeightTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        unsigned int _bw
        );

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
        );

unsigned int deriveOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _sizeOutputHeight,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        bool isConv,
        unsigned int _bw);

unsigned int deriveNumActivationDramBlockPerStrip(
        unsigned int _numInputChannelsPerGroup
    );


int deriveSparseComputeLatencyOneTile(
           int _numIC,
           int _interPruningRangePara,
            int _clusteSize,
            int _pruningRangeSizeFull,
            int _pruningRangeSizeActual,
            int _kernelSize,
            int _outputTileWidthPerCol,
            int _outputTileHeightPerCol,
            int _outputChannel,
            int _numPeRows
        );

int deriveDenseComputeLatencyOneTile(
           int _numIC,
            int _clusteSize,
            int _interClusterPara,
            int _kernelSize,
            int _outputTileWidthPerCol,
            int _outputTileHeightPerCol,
            int _outputChannel,
            int _numPeRows
        );

int deriveInputTranserLatencyOneTile(
            int _inputTileWidth,
            int _inputTileHeight,
            int _numIC,
            int _bw
        );

int deriveOutputTranserLatencyOneTile(
            int _outputTileWidth,
            int _outputTileHeight,
            int _numOC,
            int _bw
        );

int deriveDenseWeightTranserLatencyOneTile(
            int _kernelSize,
            int _numIC,
            int _numOC,
            int _bw
        );

int deriveSparseWeightTranserLatencyOneTile(
            int _kernelSize,
            int _numIC,
            int _numOC,
            int _sizeFullPruneRange,
            int _sizeCluster,
            int _sizeSparsesPruneRange,
            int _interPruneRangePara,
            int _bw
        );

unsigned int deriveFirstTileConvInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _sizeKernel,
        unsigned _sizeStride
        );

unsigned int deriveLastTileOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup
        );

/*!
 * \brief ia_cache_boundary_check
 * \details Calculates the IA cache requirement in terms of dram block/activation burst block
 *  given the IA tile dimentions
 * \param heightTile Number of rows in a tile
 * \param widthTile Number of columns in a tile
 * \param numDramBlockPerDenseStrip Depth of the tile in terms of dram block
 * \return The cache size requirement in dram block
 */
int ia_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numDramBlockPerDenseStrip
        );

/*!
 * \brief oa_cache_boundary_check
 * \details Calculates the oa cache size requirements in terms of activation burst blocks
 * \param heightTile
 * \param widthTile
 * \param numChannels Number of channels in unsparsified output tensor group
 * \return The oa cache size
 */
int oa_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numChannels
        );

/*!
 * \brief filter_cache_boundary_check
 * \details Calculates the filter cache size requirements in term of dram blocks
 * \param kernelSize Kernel height/width
 * \param inputChannelSize Number of input channels
 * \param peBlockSize Number of weight words seen by a PE at a time
 * \param numClustersInPruningRange Number clusters in a pruning range.
 *    If the accelerator does not leverage SpW, then set this value to 1
 * \param numNZClustersInPruningRange
 *    If the accelerator does not leverage SpW, then set this value to 1
 * \return dram blocks
 */
int filter_cache_boundary_check(
      int kernelSize,
      int inputChannelSize,
      int peBlockSize,
      int numClustersInPruningRange,
      int numNZClustersInPruningRange
        );
#endif // TILE_HPP
