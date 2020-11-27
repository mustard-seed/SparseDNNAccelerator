#ifndef _LAYER_INSTRUCTION_GENERATOR_HPP_
#define _LAYER_INSTRUCTION_GENERATOR_HPP_
#include "vectorType.hpp"

enum OPERATION {CONVOLUTION, MAX_POOL, ELT_ADD, CONCATENATION, AVG_POOL};
void instruction_generator (//Type of the operation
        OPERATION op,

        //Instruction buffers
        t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
        t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
        t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
        t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
        t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,
        t_aligned_misc_instruction_vector & vecMiscInstruction,

        bool flagIA0ShiftLeft,
        unsigned int numIA0ShiftAmount,
        bool flagIA1ShiftLeft,
        unsigned int numIA1ShiftAmount,

        //Starting location of activation tensors in the OpenCL Buffer
        //IA and OA occupies the same OpenCL buffer

        //Starting location of the input tensros.
        //Support up to 2 input activation tensors
        signed int memIA0DramBlockStartIndex,
        signed int memIA1DramBlockStartIndex,

        //Starting location of the output tensor
        //Supports only one output tensor
        signed int memOADramBlockStartIndex,

        //Starting location of the weight tensor
        signed int memWeightDramBlockStartIndex,
        //Starting location of bias
        signed int memBiasStartIndex,

        //Input activation blob 0 strides in terms of DRAM block
        //Assuming GHWC layout
        signed int memIA0DramBlockColStride,
        signed int memIA0DramBlockRowStride,
        //If the operation is not CONVOLUTION or CONCATENATION, then _memIA0DramBlockGroupStride can be overwritten
        signed int _memIA0DramBlockGroupStride,

        //Input activation blob 1 strides in terms of DRAM block
        //Assuming GHWC layout
        signed int memIA1DramBlockColStride,
        signed int memIA1DramBlockRowStride,
        //If the operation is not CONVOLUTION or CONCATENATION, then _memIA0DramBlockGroupStride can be overwritten
        signed int _memIA1DramBlockGroupStride,

        //Output activation blob stride in terms of DRAM block
        //Assuming GHWC layout
        signed int memOADramBlockColStride,

        signed int memWeightDramBlockFilterStride,

        //TB count memory information. Only one input blob is supported for sparse operation
        #if defined(SPARSE_SYSTEM)
            signed int memIATB0CountStart,
            unsigned int memIATB0CountColStride,

            signed int memOATBCountStart,
            unsigned int memOATBCountColStride,

            signed int memWeightTBCountStart,
        #endif

        unsigned char flagSparseOutput,
        unsigned char flagSparseInput,
        //Whether the IA mover kernel should wait for the output from the previous tensor to commit before moving
        //on to the current tensor
        unsigned char flagTensorSync,
//        unsigned char flagOutputSync,
        unsigned char flagRelu,
        unsigned char outputShiftBits,
        unsigned char flagOutputShiftLeft,

        //Input stride-padded width and height, not including border padding
        unsigned short inputSPWidth,
        unsigned short inputSPHeight,
        //Input stride-padded unit sizes
        unsigned char inputSPWidthUnit,
        unsigned char inputSPHeightUnit,
        //Input border paddings
        unsigned char inputWidthPadding,
        unsigned char inputHeightPadding,

        unsigned char kernelSize,
        unsigned char kernelStride,

        //Only relevant for convolution
        unsigned char _sizeOutputTileFullHeight,
        //Only relevant for convolution
        unsigned char _sizeOutputTileFullWidthPerCol,
        //Only relevant for convolution
        unsigned char _numActiveColsPartialOutputTile,

        //Number of channels in input blobs 0 and 1
        //Only element-wise addition and pooling should use the second  blob
        unsigned short numInputChannels0,
        unsigned short numInputChannels1,

        //Number of groups in the current layer's output.
        //Only relevant for convolution
        //Other layers assumes the number of current layer's group is 1
        unsigned short numGroupsCurrentLayer,

        //Number of output channels
        //Only relevant for convolution
        unsigned short numOutputChannels,

        //Number of groups in the next layer

        unsigned short numGroupsNextLayer
        );

/*!
 * \brief ia_cache_boundary_check
 * \details Calculates the IA cache requirement in terms of dram block given the IA tile dimentions
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
 * \brief ia_tbcount_cache_boundary_check
 * \details Calculates the tb count cache size in terms of tb counts given the IA tile dimensions
 * \param heightTile
 * \param widthTile
 * \return The cache size in tb count
 */
int ia_tbcount_cache_boundary_check(
      int heightTile,
      int widthTile
        );

/*!
 * \brief oa_cache_boundary_check
 * \details Calculates the oa cache size requirements in terms of oa values
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
 * \return dram blocks
 */
int filter_cache_boundary_check(
      int kernelSize,
      int inputChannelSize
        );

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

unsigned int deriveConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel
        );

unsigned int deriveConvInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        unsigned _sizeStride
        );

unsigned int deriveConvWeightTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel
        );

unsigned int deriveOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _sizeOutputHeight,
        unsigned int _numOutputChannelsPerNextGroup,
        unsigned int _numNextGroups);

unsigned int deriveNumActivationDramBlockPerStrip(
        unsigned int _numInputChannelsPerGroup
    );

unsigned int deriveFirstTileConvInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _sizeKernel,
        unsigned _sizeStride
        );

unsigned int deriveFirstTileConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _sizeKernel
        );

unsigned int deriveLastTileOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerNextGroup
        );

#endif
