#ifndef STRUCTURES_HPP_DEF
#define STRUCTURES_HPP_DEF

#include "params.hpp"
#ifndef INTELFPGA_CL
#include "CL/cl.hpp"
#endif

#ifdef INTELFPGA_CL
#include "ihc_apint.h"
#endif

#define TRUE 0x1
#define FALSE 0X0

#ifdef INTELFPGA_CL
typedef unsigned char t_uchar;
typedef char t_char;
typedef unsigned short t_ushort;
typedef short t_short;
typedef unsigned int t_uint;
typedef int t_int;
    #if defined(C5SOC)
        #define VOLATILE volatile
    #else
        #define VOLATILE
    #endif
typedef uint1_t t_flag;
#else
typedef cl_uchar t_uchar;
typedef cl_char t_char;
typedef cl_ushort t_ushort;
typedef cl_short t_short;
typedef cl_uint t_uint;
typedef cl_int t_int;
#endif

#ifdef INTELFPGA_CL
//typedef short t_spValueAndZCount;
typedef unsigned short t_spOffset;

typedef char t_operand;
#else
//typedef cl_short t_spValueAndZCount;
typedef cl_ushort t_spOffset;

typedef cl_char t_operand;
#endif



#ifdef INTELFPGA_CL
typedef unsigned char t_simdblock_channel_offset; //Relative channel of a simdblock in a streaming block

typedef unsigned short t_streamblock_address; //Address of a streaming block in BRAM

#else
//typedef t_simdblock_host t_simdblock_value;

typedef cl_uchar t_simdblock_channel_offset;

typedef cl_ushort t_streamblock_address;
#endif
//#endif //Data structures used in direct compression SIMD

//=============================
/*
==================================================================
Types involved in operations
==================================================================
*/
#ifdef INTELFPGA_CL
#if defined(EMULATOR)
typedef signed int t_accumulator;
#elif (ACCUMULATOR_WIDTH == 32)
typedef signed int t_accumulator;
#elif (ACCUMULATOR_WIDTH == 28)
typedef int28_t t_accumulator;
#elif (ACCUMULATOR_WIDTH == 24)
typedef int24_t t_accumulator;
#elif (ACCUMULATOR_WIDTH == 20)
typedef int20_t t_accumulator;
#elif (ACCUMULATOR_WIDTH == 16)
typedef signed short t_accumulator;
#else
#error Accumulator width should be from 32-bit, 28-bit, 24-bit, 20-bit, and 16-bit
#endif

#if defined(EMULATOR)
typedef signed int t_misc_accum;
#elif (MISC_ACCUMULATOR_WIDTH == 32)
typedef signed int t_misc_accum;
#elif (MISC_ACCUMULATOR_WIDTH == 28)
typedef int28_t t_misc_accum;
#elif (MISC_ACCUMULATOR_WIDTH == 24)
typedef int24_t t_misc_accum;
#elif (MISC_ACCUMULATOR_WIDTH == 20)
typedef int20_t t_misc_accum;
#elif (MISC_ACCUMULATOR_WIDTH == 16)
typedef signed short t_misc_accum;
#else
#error MISC accumulator width should be from 32-bit, 28-bit, 24-bit, 20-bit, and 16-bit
#endif

typedef struct {
    signed char cluster_values [CLUSTER_SIZE];
} t_cluster;

// typedef signed short t_bias;
typedef signed short t_bias;
#else
// #if (ACCUMULATOR_WIDTH == 32)
// typedef signed int t_accumulator;
// #elif (ACCUMULATOR_WIDTH == 16)
// typedef signed short t_accumulator;
// #else
// #error ACCUMULATOR_WIDTH should either be 32 or 16!
// #endif

typedef struct {
    t_char cluster_values [CLUSTER_SIZE];
} t_cluster;

typedef cl_ushort t_streamblock_address;

// typedef signed short t_bias;
typedef signed short t_bias;
#endif


/**
 * =========================================
 * Data structures seen by the SpW PE
 * =========================================
 */
#if defined(SPW_SYSTEM)
    #if (PRUNE_RANGE_IN_CLUSTER == 2)
        #define CHAR_TO_SPW_INDEX_MASK 0x01
    #elif (PRUNE_RANGE_IN_CLUSTER == 4)
        #define CHAR_TO_SPW_INDEX_MASK 0x03
    #elif (PRUNE_RANGE_IN_CLUSTER == 8)
        #define CHAR_TO_SPW_INDEX_MASK 0x07
    #elif (PRUNE_RANGE_IN_CLUSTER == 16)
        #define CHAR_TO_SPW_INDEX_MASK 0x0F
    #else
        #error Pruning range in terms of clusters must be 2, 4, 8, or 16
    #endif
#endif
#if defined(INTELFPGA_CL)
    #if defined(SPW_SYSTEM)
        #if (PRUNE_RANGE_IN_CLUSTER == 2)
            typedef uint1_t t_spw_index;
        #elif (PRUNE_RANGE_IN_CLUSTER == 4)
            typedef uint2_t t_spw_index;
        #elif (PRUNE_RANGE_IN_CLUSTER == 8)
            typedef uint3_t t_spw_index;
        #elif (PRUNE_RANGE_IN_CLUSTER == 16)
            typedef uint4_t t_spw_index;
        #else
            #error Pruning range in terms of clusters must be 2, 4, 8, or 16
        #endif
    #endif
    typedef struct __attribute__((packed)) {
        char values[PE_SIMD_SIZE * CLUSTER_SIZE];
        #if defined(SPW_SYSTEM)
            t_spw_index indices[PE_SIMD_SIZE];
            t_flag  isLastInPruneRange; 
        #endif
        uint5_t maxTransportID;
        t_flag  isLastInFilter;
        t_bias  bias;
    } t_pe_w_block;

    typedef struct __attribute__((packed)) {
        char values[PE_ACTIVATION_BLOCK_SIZE_IN_WORD];
        uint5_t maxTransportID;
    } t_pe_a_block;


    // typedef struct {
    // #if defined(SPW_SYSTEM)
    //     t_char values[PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE];
    // #else
    //     t_char values[PE_SIMD_SIZE * CLUSTER_SIZE];
    // #endif
    // } t_activation_transfer_block;

    typedef struct {
        t_char values [ACTIVATION_BURST_SIZE_BYTE];
    } t_activation_dram_block;
#endif

// typedef struct {
//     t_char values [TRANSFER_SIZE*CLUSTER_SIZE];
// } t_transfer_block;

// typedef struct {
//     t_transfer_block transferBlocks[WIDE_SIZE];
// } t_dram_block;


// typedef struct {
//    t_char values[PE_SIMD_SIZE * CLUSTER_SIZE];
//     #if defined(SPW_SYSTEM)
//         t_uchar indices[INDEX_CHAR_ARRAY_SIZE];
//     #endif
// } t_weight_transfer_block;


typedef struct __attribute__((packed)) {
    //t_weight_transfer_block transferBlocks[WEIGHT_WIDE_SIZE];
    t_char values [WEIGHT_BURST_SIZE_VALUE_BYTE];
    #if defined(SPW_SYSTEM)
    t_uchar indices[WEIGHT_BURST_SIZE_INDEX_BYTE];
    #endif
} t_weight_dram_block;


/*!
   ==================================================
   Data mover and tile controller instructions
   ==================================================
*/
typedef struct __attribute__((packed)) __attribute__((aligned(64))) 
{
    //Concatenation of three signals
    //Bits [3:0] Number of active columns
    //Bit [4]: Flag for the compute engine. 0 for convolution, 1 for misc.
    //Bit [5]: Flag for sparse input. 0 for dense, 1 for sparse.
    //Bit [6]: Input arrangment mode.
    //  1'b0: One input tensor (e.g convolution, strided convolution)
    //  1'b1: Two input tensors, and interleave the two tensors per dramblock (e.g. eltwise addition)
    //Bit [7]: Flag indicating that the IA mover should wait for sync. token from the OA mover
    //in the beginninger
    t_uchar flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols;

    //Bit [3:0] Input 0 left shift amount
    //Bit [7:4] Input 1 left shift amount
    t_uchar inputShiftAmounts;

    //Arch parameter: Starting index of the input activation tensor in input 0's memory region
    //Address is counted in activation word (i.e.. signed char)
    t_int memBlockStart0;
    //Arch parameter: Starting index of the input dram block in input 1's memory region
    //Address is counted in activation word (i.e.. signed char)
    t_int memBlockStart1;

    //Important: we assume the two input tensors (if there are two) have the exact same input dimensions
    //Arch parameter: Column stride of input activation strips in dram block in both input memory regions
    //Address is counted in activation word (i.e.. signed char)
    t_int memBlockColStripStride;
    //Arch parameter: Row stride of input activation strips in activation word in both input memory regions
    t_int memBlockRowStripStride;


    //Problem parameter: Number of PE activation block along the strip
    t_ushort numTBPerStrip;

    //Problem parameter: memory input tile stretched padded height. Includes padding.
    t_uchar tileSPHeight;
    //Problem parameter: memory input tile stretched padded width. Includes padding.
    t_uchar tileSPWidth;

    //Problem parameter: input tile paddings
    //[1:0]: Left padding
    //[3:2]: Right padding
    //[5:4]: Top padding
    //[7:6]: Bottom padding
    t_uchar concatPadding;

    //SP-unit indicies of the upper-left planar position in the tile, not counting the padding
    //[3:0] Column index
    //[7:4]: Row index
    t_uchar concatInitSPIndices;


    //Problem parameter: memory input tile stretched unit size
    //[3:0] horizontal stretched unit size
    //[7:4]: vertical stretched unit size
    t_uchar concatSPSize;

    //Problem parameter: compute column input width stride
    t_uchar columnWidthStride;
    //Problem parameter: compute column strided padded input width
    t_uchar columnSPWidth;

    //Auxillary parameter: total number of strips to send in this transfer
    t_ushort tileSPWidthxTileSPHeight; 
} t_ia_mover_instruction;

typedef struct __attribute__((packed)) __attribute__((aligned(32)))
{
    //Arch. parameters.
    //[3:0]: Number of active compute columns
    //[4]: Sync Flag. 1 if there is a need to send sync. signal to the IA mover in the beginning
    t_uchar memSelectCatSparseFlagCatSyncFlagCatNumActiveCols;

    //Arch. parameter: Index of the first dram block of this transfer in memory
    //Address is counted in terms of the activation word (e.g. signed char)
    t_int memOAStart;
    //Arch. parameter: Group stride in terms of dram block in the output memory region
    //t_uint memOAGroupStride;
    //Arch. parameter: PE column stride in terms of activation word in the output memory region
    t_uint memOAPEColStride;
    //Arch. parameter: column stride in terms of activation word in the output memory region
    t_uint memOAColStride;
    //Arch. parameter: row stride in terms of activation word in the output memory region
    t_uint memOARowStride;

    //Problem parameter: Number of dram blocks per strip 
    //(not including the block used to transfer TB count when processing sparse data)
    t_ushort numNominalDramBlocksPerStrip;

    //Problem parameter: Output tile group to drain from   
    //t_uchar numOAGroup; 
    //Problem parameter: Output tile height per compute column 
    t_uchar tileHeight; 
    //Problem parameter: Output tile width per compute column
    t_uchar columnTileWidth; 

} t_oa_mover_instruction;

typedef struct __attribute__((packed)) __attribute__((aligned(32)))
{
    //Problem. Number of output channels in the group
    t_ushort numFiltersInGroup;
    //Arch. parameter: number of full folds
    t_ushort numFullFilterFold;
    //Arch. parameter: number of filter in the partial fold
    t_uchar numFiltersInPartialFold;
    //Arch. parameter: number of filter reuse
    t_ushort filterReuse;
    //Arch. Number of active pe cols
    t_uchar numActivePeCols;

    //Arch. parameter: Start of the transfer in the bias memory region
    t_int memBiasStart;
    //Arch. parameter: Start of the transfer in the weight dram_block region
    //Address is counted in weight dram block
    t_int memWeightStart;
    //Arch. parameter: filter stride in the weight dram block region.
    //Address is counted in weight dram block
    t_int memWeightFilterStride;

    t_uint numTBPerFilter;

    #if defined(SPW_SYSTEM)
    t_uchar numNZClustersPerPruneRange;
    #endif
} t_weight_mover_instruction;

//====================================================================
//Instructions for the input tile controller
typedef struct __attribute__((packed)) __attribute__((aligned(32)))
{
    //Input tile width per compute column
    t_uchar localTileWidth;
    //Input tile height per compute column
    t_uchar localTileHeight;
    //Filter planar stride
    t_uchar kernelStrideY;
    t_uchar kernelStrideX;
    //Filter planar kernel sizes
    t_uchar kernelSizeHeight;
    t_uchar kernelSizeWidth;
    //Number of streaming instruction for this tile
    t_uint numOutputInstructions;
    //Column stride of IA strip in IA cache in terms of DRAM BLOCK
    //Note: IA cache only sees ONE group. 
    //In other words, when the accelerator processes multiple groups
    //the strip seen by the IA buffer is shorter than the strip seen by the IA mover
    t_ushort cacheIAStripColStride;
    //Used to convert the col stride seen by the IA Buffer reader from
    //cacheIAStripColStride to cacheIAStripColStride * cacheIAStripColStrideMultiplier
    //Useful only for the 1x1 convolution optimization
    t_ushort cacheIAStripColStrideMultiplier;
    //Number of strips streamed by the IA Buffer Reader to the PE array per instruction
    t_uchar numStripsToPEPerInstruction;
    //Number of output channels in the output group
    t_ushort numOutputChannelsInGroup;
    //Bit[6:0] Number of active PE columns
    t_uchar flagPadBitmaskCatNumActiveCols;

} t_ia_tile_controller_instruction;

//Instructions for the output tile controller
typedef struct __attribute__((packed)) __attribute__((aligned(32)))
{
    //Number of planar indices in the output tile
    t_uchar numLocalTilePerColHxW;

    //Number of ROUNDED channels per group
    //rounded to a multiple of ACTIVATION_BURST_SIZE
    t_ushort numBurstAlignedChannelsPerCurrentGroup;

    //Number of compute drain instructions for a tile with
    //TileH x TileW x TileOCPerGroupCurrentLayer elements
    //Do NOT span over multiple layer!
    t_ushort numDrainInstructions;

    //Number of folds required per group to drain the current tile
    //One fold refers to a tile of the output tensor with number of channels 
    //equal or less to the number of PE rows
    t_ushort numFoldsInGroupCurrentLayer;
    //Number of full folds required to drian the current tile
    //Assertion: no smaller than numFoldsInGroupCurrentLayer - 1
    t_ushort numFullFoldsInCurrentLayer;
    //Number of elements per planar index to drain in the full fold
    //Assertion: it is divisible by PE_ROWS_IN_GROUP
    t_ushort numActiveElementsInFullFold;
    //Number of elements per planar index to drain in the partial fold
    t_ushort numActiveElementsInPartialFold;

    // //Number of channels per group in the next layer
    // t_ushort numLocalChannelsPerNextGroup;

    //Number of active compute columns
    t_uchar numActiveCols;

    //Number of dram blocks that belong to the same group at the same planar index
    //Only consider one group
    //This should be calculated assuming 100% density.
    //Also seen by the misc tee
    t_ushort numNominalDramBlocksPerStrip;

    //Concatenated signal
    //[3:0] number of bits to shift the convolution output
    //[4] Shift direction. 1 for left shift, 0 for right shift
    //[6] Source of the output. 0 for convolution engine, 1 for misc.
    //[7] Enable Relu. 1 for TRUE, 0 for false
    ////Also seen by the misc tee
    t_uchar flagSparseCatFlagReluCatFlagSourceCatShift;
    
} t_oa_tile_controller_instruction;

/**
 * Instruction for the misc module
 */
typedef struct __attribute__((aligned(16)))
{
    //Bit[3:0]: Number of active PE columns
    //Bit[5:4]: OpCode. 00: Add; 01: Max Pooling; 10: Stream
    t_uchar controlBits;

    //Number of dram blocks to reduce per output dram block
    t_ushort numDramBlocksToReduce;

    //Number of output dram blocks to be processed per misc unit
    //Seen by the misc units ONLY
    t_ushort numOutputBlocksPerUnit;

    // //Number of output dram blocks per group.
    // //Seen by the misc controller ONLY
    // t_ushort numOutputBlocksPerGroup;

    //Number of output dram blocks summmed along a strip across on groups
    //Seen by the misc controller ONLY
    // t_ushort numOutputBlocksPerStrip;

    /*
        Control bits
        Relu flag, shift direciton and the number of bits to shift the accumulator value 
        Bit 5: Enable relu
        Bit 4: shift direction. 0 for right, 1 for left. 
        Bit 3:0: shift amount
    */
    t_uchar outputModifierControl;

} t_misc_instruction;

#ifdef INTELFPGA_CL
/*
=======================================================================
Datatypes relevant to the filter transportation system
========================================================================
*/
//Raw data packet travelling on the channels that link the WeightTees.
typedef struct {
    t_weight_dram_block dramBlock;
    unsigned char destinationRow;
} t_dram_block_w_tagged;

//Control packet for the weight buffers
typedef struct __attribute__((packed)) {
    unsigned int numOutputsXNumTransferBlocks;
    unsigned short numTransferBlocks;
    t_bias bias; //short
    unsigned char maxPeCols; //Number of PE COLS that is activated
    t_flag flagIsReal; //Whether this filter row should stream zero padding
    #if defined(SPW_SYSTEM)
    unsigned char numNZClustersPerPruneRange;
    #endif

} t_filter_streamer_control;

// typedef struct __attribute__((packed)){
//     t_transfer_block values;

//     unsigned char isLastConcatMaxTransportID;
// } t_transferblock_tagged;

typedef struct __attribute__((packed)){
    t_accumulator value;
    //[7:1] row ID or the row that issued the packed
    //[0] Whether this is the last.
    unsigned char sourceRowIDCatIsLast;
} t_conv_drain_tagged;


typedef struct __attribute__((packed)){
    t_accumulator values[PE_ROWS_PER_GROUP];
    //[7:1] row ID or the row that issued the packed
    //[0] Whether this is the last.
    //unsigned char sourceRowIDCatIsLast;
    uint5_t sourceRowGroupID;
    t_flag  flagIsLast;
} t_conv_drain_multiple_tagged;
/*
===================================================================
Data structures that travel on the input activation bus system
===================================================================
*/
//Raw data packet travelling on the input activation buffer bus
typedef struct __attribute__((packed)){
    t_activation_dram_block dramBlock;

    //Bit[7]: Is last in strip
    //Bit[6]: Flag for going to misc engine
    //Bit[5:0] Destination col 
    unsigned char route;

    //Only read by the misc engine
    //Bit[3:0] Left shift amount
    unsigned char miscLeftShiftAmount;

    //Number of TB in this strip
    unsigned short numTB;

    //Used for steering the block to respective column
    unsigned char colSPWidth;
    unsigned char colSPStride;
    signed char iColInSPTile;
} t_dram_block_ia_tagged;

typedef struct __attribute__((packed)){
    t_activation_dram_block dramBlock;

    //Only read by the misc engine
    //Bit[3:0] Left shift amount
    unsigned char miscLeftShiftAmount;

    //bool toInputBuffer; 
} t_dram_block_ia_to_misc;

typedef struct __attribute__((packed)){
    t_activation_dram_block dramBlock;

    //Bit[7]: Is last in strip
    //Bit[6]: Flag for going to misc engine
    //Bit[5:0] Destination col 
    unsigned char route;

    //Number of TB in this strip
    unsigned short numTB;

} t_dram_block_ia_to_pe;



typedef struct __attribute__((packed))
{
    /**
     * Address information of the IA cache
     * counted in units of activation dram block
     * i.e. char [ACTIVATION_BURST_SIZE_BYTE]
     */
    unsigned short iaDramBlockAddressBase;
    unsigned short iaDramBlockColStride;

    unsigned char maxPeRowGroupID; //Only relevant for sending


    //Bit 0: Whether to access IA buffer 0 or 1
    //Bit 1: Whether to write in to the IA buffer (0), or to read from the IA buffer (1)
    //Bit 2: Only useful for sparse case. Flag for whether the tile require sparse bitmask padding. 1 for true, 0 for false
    //Bit 7:3: Max PE Cols to send to; 
    unsigned char controlBits; 

    //Whether the IA buffer reader is streaming the last row of a convolution window to the PEs
    //Only useful during streaming from the IA buffer
    unsigned char flagIsLastRow;

    //Number of columns in the transfer command
    unsigned char numStripsCol;
    
} t_input_buffer_tile_buffer_packet;


/**
 * Sees by the OA Buffer
 */
typedef struct __attribute__((packed))
{
    //Index of the output buffer at the start of the transaction
    //Assume the layout of the output is HWC
    unsigned short startOutputIndex; 

    //Number of output values in a strip that is to load into or to stream from the cache during this instruction cycle.
    unsigned short numOutputsPerStrip; 

    //Number of strips to access
    unsigned short numStripsToAccess;

    //Stride between successive strip
    //In terms of OA values
    //Assumption:
    //All the output values from a PE row-group land in the same output activation dram block
    //Specifically, indexOutput aligns with PE_ROW_GROUP blocks and ACTIVATION_BURST blocks
unsigned short iaStridePerCol;

    // //Number of memory transfer instructions, used for draining the cache only
    // unsigned char numGroupsCurrentLayer;

    // //Number of channels per group in the next layer, used for draining the cahce only
    // unsigned short numLocalChannelsPerNextGroup;

    //Number of dram blocks to send to the OA mover 
    //over tile W * tile H * num_groups_next_layer
    //Seen by the OA Tee only
    unsigned short numNominalDramBlocksPerOATee;

    //TODO: Add bit for output access bank
    /*
        Control bits
        Bit 4:0: Shift direciton and the number of bits to shift the accumulator value from the convolution PE array. Only relevant for loading
        Bit 5: Enable sparsification. Only relevant for the convolutional kernel during ending
        Bit 6: Drainage source. 1: from the MISC kernel. 0: from the convolution kernel.
        Bit 7: Enable Relu. Only relevant for loading
        Bit 8: Load from engine (0) or drain the buffer (1)
        Bit 9: Access bank
    */
    unsigned short controlBits;

} t_output_tile_buffer_packet;

typedef struct __attribute__((packed))
{
    t_output_tile_buffer_packet bufferPacket;
    unsigned char maxColID;
} t_output_tile_buffer_packet_tagged;

/**
 * Sees by the OA Tee
 */
typedef struct __attribute__((packed))
{
    //Number of dram blocks to send to the OA mover 
    //over tile W * tile H * num_groups_next_layer
    unsigned short numNominalDramBlocksPerOATee;


    //Bit [3:0] Maximum column ID
    //Bit [5] Flag for sparse draining sparse input (1 for true)
    //Bit [6] Drainage source. 1: from the MISC units. 0: from the convolution PE array
    unsigned char flagSourceCatFlagSparseFlagMaxColID;
} t_output_tile_tee_packet;

typedef struct __attribute__((packed))
{
    //Bit[3:0]: Number of active PE columns
    //Bit[5:4]: OpCode. 00: Add; 01: Max Pooling; 10: Stream
    t_uchar controlBits;

    //Number of dram blocks to reduce for eaach output dram block
    unsigned short numDramBlocksToReduce;

    //Number of output dram blocks to produce by a MISC engine in a tile plane
    unsigned short numOutputBlocks;

    /*
        Control bits
        Relu flag, shift direciton and the number of bits to shift the accumulator value 
        Bit 5: Enable relu
        Bit 4: shift direction. 0 for right, 1 for left. 
        Bit 3:0: shift amount
    */
    t_uchar outputModifierControl;

} t_misc_control_packet;

/*
===================================================================
Data structures that travel on the output activation bus system
===================================================================
*/

typedef struct __attribute__((packed)) {
    t_activation_dram_block dramBlock;
    t_flag isFromLastColumn;
} t_output_activation_dram_block_tagged;
// typedef struct __attribute__((packed)) {
//     //TODO: HANDLE MULTI-BYTE MASK
//     t_cluster cluster;
//     //unsigned char numSurvivingClusters;  //Number of surviving data cluster (not including the bitmask) in the window
//     //bool isLastWindowInStrip; //Whether this is the last window in a strip

//     //Status bits
//     //Bit 0: Enable sparsification
//     //Bit 1: Is last cluster in the strip
//     //Bit 2: Is last cluster in window
//     unsigned char statusBits;
// } t_cluster_to_compressor;
// //Used to send data to the tee
// typedef struct __attribute__((packed)) {
//     t_cluster cluster;
//     bool isLastInStrip;

// } t_output_cluster_tagged;

// //Output of the OA coalescer
// typedef struct {
//     t_cluster clusters[NUM_CLUSTER_IN_DRAM_SIZE];
// } t_output_dram_block;

// typedef struct __attribute__((packed)) {
//     t_output_dram_block block;

//     //Bit 0: Is issued by the last column
//     //Bit 1: Is valid block
//     //Bit 2: Read by sparse only. Is last valid dram block in strip.
//     unsigned char flags;
//     #if defined(SPARSE_SYSTEM)
//     unsigned short clusterCount;
//     #endif 
// } t_output_dram_block_tagged;

// typedef struct __attribute__((packed)) {
//     t_output_dram_block outputDramBlock;
//     #if defined(SPARSE_SYSTEM)
//     unsigned short numClustersInStrip;
//     #endif
//     bool isLastInStrip;
// } t_output_coalescer_packet;
#endif
//#endif

//=====================================

#endif //STRUCTURES_HPP_DEF
