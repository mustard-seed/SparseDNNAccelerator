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
#elif (ACCUMULATOR_WIDTH == 24)
typedef int24_t t_accumulator;
#elif (ACCUMULATOR_WIDTH == 20)
typedef int20_t t_accumulator;
#elif (ACCUMULATOR_WIDTH == 16)
typedef signed short t_accumulator;
#else
#error Accumulator width should be from 32-bit, 24-bit, 20-bit, and 16-bit
#endif

typedef struct {
    signed char cluster_values [CLUSTER_SIZE];
} t_cluster;

typedef short t_bias;
#else
// #if (ACCUMULATOR_WIDTH == 32)
// typedef signed int t_accumulator;
// #elif (ACCUMULATOR_WIDTH == 16)
// typedef signed short t_accumulator;
// #else
// #error ACCUMULATOR_WIDTH should either be 32 or 16!
// #endif

typedef struct {
    cl_char cluster_values [CLUSTER_SIZE];
} t_cluster;

typedef cl_ushort t_streamblock_address;

typedef signed short t_bias;
#endif

typedef struct {
    char values [TRANSFER_SIZE*CLUSTER_SIZE];
} t_transfer_block;

typedef struct {
    t_transfer_block transferBlocks[WIDE_SIZE];
} t_dram_block;

typedef struct {
    t_transfer_block transferBlocks[WEIGHT_WIDE_SIZE];
} t_weight_dram_block;

typedef struct {
        unsigned char bytes[NUM_BITMASK_BYTES];
} t_bitmask;

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
    } t_pe_w_block;

    typedef struct __attribute__((packed)) {
        #if defined(SPW_SYSTEM)
            char values[PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE];
        #else
            char values[PE_SIMD_SIZE * CLUSTER_SIZE];
        #endif
        uint5_t maxTransportID;
    } t_pe_a_block;
#endif

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

    //Arch parameter: Starting index of the input dram block in input 0's memory region
    t_int memBlockStart0;
    //Arch parameter: Starting index of the input dram block in input 1's memory region
    t_int memBlockStart1;

    //Important: we assume the two input tensors (if there are two) have the exact same input dimensions
    //Arch parameter: Column stride of input activation strips in dram block in both input memory regions
    t_ushort memBlockColStripStride;
    //Arch parameter: Row stride of input activation strips in dram block in both input memory regions
    t_ushort memBlockRowStripStride;

#if defined(SPARSE_SYSTEM)
    /*!
     * If TB memory is needed, then there is only one input
    */
    //Arch parameter: Starting index of the strip TB count in the memory
    t_int memTBCountStart;
    //Arch parameter: Column stride of input activation strip TB count in the memory
    t_ushort memTBCountColStride;
    //Arch parameter: Row stride of input activation strip TB count in the memory
    t_ushort memTBCountRowStride;
    //Problem parameter: 
    //When sending sparse data, These are the number of compression windows in an input group. Used for sending padding
    //When sending dense data, These are the number of valid TB in a strip
    //Important: we assume the two input tensors (if there are two) have the exact same input dimensions
    //Arch parameter: Column stride of input activation strips in dram block in both input memory regions
    t_ushort numCWOrTBInGroup;
#else
    //Important: we assume the two input tensors (if there are two) have the exact same input dimensions
    //Arch parameter: Column stride of input activation strips in dram block in both input memory regions
    t_ushort numTBPerStrip;
#endif


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
    //[6]: Flag for whether the output is sparse. 1 for YES, 0 for NO
    //[7]: Flag for selecting the memory region to write to
    t_uchar memSelectCatSparseFlagCatSyncFlagCatNumActiveCols;

    //Arch. parameter: Index of the first dram block of this transfer in memory
    t_int memOAStart;
    //Arch. parameter: Group stride in terms of dram block in the output memory region
    //t_uint memOAGroupStride;
    //Arch. parameter: PE column stride in terms of dram block in the output memory region
    t_uint memOAPEColStride;
    //Arch. parameter: column stride in terms of dram block in the output memory region
    t_ushort memOAColStride;
    //Arch. parameter: row stride in terms of dram block in the output memory region
    t_ushort memOARowStride;

    //Problem parameter: Number of dram blocks per strip 
    //(not including the block used to transfer TB count when processing sparse data)
    t_ushort numNominalDramBlocksPerStrip;

#if defined(SPARSE_SYSTEM)
    //Arch. parameter: Index of the first TB count element of this transfer in memory
    t_int memTBStart;
    //Arch. parameter: group stride in terms of TB count in the TB memory
    //t_ushort memTBGroupStride;
    //Arch. parameter: tile stride in terms of TB count in the TB memory.
    t_ushort memTBPEColStride;
    //Arch. parameter: column stride in terms of TB count in the TB memory.
    t_ushort memTBColStride;
    //Arch. parameter: row stride in terms of TB count in the TB moveremory.
    t_ushort memTBRowStride;
#endif

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
    t_int memWeightStart;
    //Arch. parameter: filter stride in the weight dram block region.
    t_int memWeightFilterStride;

#if defined(SPARSE_SYSTEM)
    //Arch. parameter: Start of the transfer in the weight TB count region
    t_int memTBCountStart;
#else
    t_uint numTBPerFilter;
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
    t_uchar kernelStride;
    //Filter planar kernel size
    t_uchar kernelSize;
    //Number of streaming instruction for this tile
    t_uint numOutputInstructions;
    //Column stride of IA strip in IA cache in terms of dram block
    t_ushort cacheIAStripColStride;
    //Number of output channels in the output group
    t_ushort numOutputChannelsInGroup;
    //Bit[6:0] Number of active PE columns
    //Bit[7] For sparse engine use only. Whether the input activation tensor is dense and hence need bitmask padding.
    t_uchar flagPadBitmaskCatNumActiveCols;

    #if defined(SPARSE_SYSTEM)
        //Partial bitmask for the last compression window in a strip
        //Only useful when the sparse accelerator is processing dense activation
        t_uchar partialBitmask[COMPRESSION_WINDOW_SIZE / 8];
    #endif
} t_ia_tile_controller_instruction;

//Instructions for the output tile controller
typedef struct __attribute__((packed)) __attribute__((aligned(32)))
{
    //Number of planar indices in the output tile
    t_uchar numLocalTilePerColHxW;
    //Number of channels in the tile, rounded up to a multiple of cluster size
    t_ushort numRoundedLocalChannels;
    //Number of compute drain instructions
    t_ushort numDrainInstructions;
    //Number of memory transfer instructions
    t_uchar numGroupsNextLayer;

    //TODO: Change data type to t_ushort
    //Number of folds required per group to drain the current tile
    t_ushort numFoldsInGroupCurrentLayer;
    //Number of full folds required to drian the current tile
    t_ushort numFullFoldsInCurrentLayer;
    //Number of elements per planar index to drain in the full fold
    t_ushort numActiveElementsInFullFold;
    //Number of elements per planar index to drain in the partial fold
    t_ushort numActiveElementsInPartialFold;

    //Number of channels per group in the next layer
    t_ushort numLocalChannelsPerCurrentGroup;

    //Number of channels per group in the next layer
    t_ushort numLocalChannelsPerNextGroup;

    //Number of active compute columns
    t_uchar numActiveCols;

    //Number of dram blocks that belong to the same group at the same planar index
    //This should be calculated assuming 100% density.
    t_ushort numNominalDramBlocksPerStrip;

    //Concatenated signal
    //[3:0] number of bits to shift the convolution output
    //[4] Shift direction. 1 for left shift, 0 for right shift
    //[5] Enable sparsification. 1 for TRUE, 0 for otherwise
    //[6] Source of the output. 0 for convolution engine, 1 for misc.
    //[7] Enable Relu. 1 for TRUE, 0 for false
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

    //Number of output dram blocks to be processed per col
    //Seen by the misc units ONLY
    t_ushort numOutputBlocksPerCol;

    // //Number of output dram blocks per group.
    // //Seen by the misc controller ONLY
    // t_ushort numOutputBlocksPerGroup;

    //Number of output dram blocks summmed along a strip across on groups
    //Seen by the misc controller ONLY
    t_ushort numOutputBlocksPerStrip;

    //Bit [2:0] Shift amount
    //Bit [3] Flag for left/right shift. 0 for right, 1 for left
    //t_uchar flagLeftShiftCatShiftAmount;

    ////Number of effective values in the final dram block in a group
    t_uchar numEffectiveValuesInLastOutputBlockInGroup;

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
    unsigned short numOutputs;
    unsigned short numTransferBlocks;
    t_bias bias; //short
    unsigned char maxPeCols; //Number of PE COLS that is activated

} t_filter_streamer_control;

typedef struct __attribute__((packed)){
    t_transfer_block values;

    unsigned char isLastConcatMaxTransportID;
} t_transferblock_tagged;

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
    unsigned char sourceRowIDCatIsLast;
} t_conv_drain_multiple_tagged;
/*
===================================================================
Data structures that travel on the input activation bus system
===================================================================
*/
//Raw data packet travelling on the input activation buffer bus
typedef struct __attribute__((packed)){
    t_dram_block dramBlock;

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
    t_dram_block dramBlock;

    //Only read by the misc engine
    //Bit[3:0] Left shift amount
    unsigned char miscLeftShiftAmount;

    //bool toInputBuffer; 
} t_dram_block_ia_to_misc;

typedef struct __attribute__((packed)){
    t_dram_block dramBlock;

    //Bit[7]: Is last in strip
    //Bit[6]: Flag for going to misc engine
    //Bit[5:0] Destination col 
    unsigned char route;

    //Number of TB in this strip
    unsigned short numTB;

} t_dram_block_ia_to_pe;



typedef struct __attribute__((packed))
{
    unsigned short iaDramBlockAddressBase;
    unsigned short iaDramBlockColStride;
    #if defined(SPARSE_SYSTEM)
        unsigned char tbAddressBase;
    #endif
    unsigned char maxPeRowID; //Only relevant for sending


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
    
    #if defined(SPARSE_SYSTEM)
        //Bit mask for the last compression window, which might be incomplete
        unsigned char partialBitmask[COMPRESSION_WINDOW_SIZE / 8];
    #endif
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
    unsigned short iaStridePerCol;

    //Number of memory transfer instructions, used for draining the cache only
    unsigned char numGroupsNextLayer;

    //Number of channels per group in the next layer, used for draining the cahce only
    unsigned short numLocalChannelsPerNextGroup;

    //Number of dram blocks to send to the OA mover 
    //over tile W * tile H * num_groups_next_layer
    //Seen by the OA Tee only
    unsigned short numNominalDramBlocksPerOATee;

    #if defined(SPARSE_SYSTEM)
    //Seen  by the OA Tee only
    unsigned short numNominalDramBlocksPerStrip;
    #endif


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

    #if defined(SPARSE_SYSTEM)
    unsigned short numNominalDramBlocksPerStrip;
    #endif

    //Bit [3:0] Maximum column ID
    //Bit [5] Flag for sparse draining sparse input (1 for true)
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

    ////Number of effective values per output block in the tile;
    t_uchar numEffectiveValuesPerOutputBlock;

} t_misc_control_packet;

/*
===================================================================
Data structures that travel on the output activation bus system
===================================================================
*/
typedef struct __attribute__((packed)) {
    //TODO: HANDLE MULTI-BYTE MASK
    t_cluster cluster;
    //unsigned char numSurvivingClusters;  //Number of surviving data cluster (not including the bitmask) in the window
    //bool isLastWindowInStrip; //Whether this is the last window in a strip

    //Status bits
    //Bit 0: Enable sparsification
    //Bit 1: Is last cluster in the strip
    //Bit 2: Is last cluster in window
    unsigned char statusBits;
} t_cluster_to_compressor;
//Used to send data to the tee
typedef struct __attribute__((packed)) {
    t_cluster cluster;
    bool isLastInStrip;

} t_output_cluster_tagged;

//Output of the OA coalescer
typedef struct {
    t_cluster clusters[NUM_CLUSTER_IN_DRAM_SIZE];
} t_output_dram_block;

typedef struct __attribute__((packed)) {
    t_output_dram_block block;

    //Bit 0: Is issued by the last column
    //Bit 1: Is valid block
    //Bit 2: Read by sparse only. Is last valid dram block in strip.
    unsigned char flags;
    #if defined(SPARSE_SYSTEM)
    unsigned short clusterCount;
    #endif 
} t_output_dram_block_tagged;

typedef struct __attribute__((packed)) {
    t_output_dram_block outputDramBlock;
    #if defined(SPARSE_SYSTEM)
    unsigned short numClustersInStrip;
    #endif
    bool isLastInStrip;
} t_output_coalescer_packet;
#endif
//#endif

//=====================================

#endif //STRUCTURES_HPP_DEF
