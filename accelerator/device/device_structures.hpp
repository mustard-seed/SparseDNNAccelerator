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
typedef short t_accumulator;

typedef struct {
    signed char cluster_values [CLUSTER_SIZE];
} t_cluster;
#else
typedef short t_accumulator;

typedef struct {
    cl_char cluster_values [CLUSTER_SIZE];
} t_cluster;

typedef cl_ushort t_streamblock_address;
#endif

typedef struct {
    char values [TRANSFER_SIZE*CLUSTER_SIZE];
} t_transfer_block;

typedef struct {
    t_transfer_block transferBlocks[WIDE_SIZE];
} t_dram_block;

typedef struct __attribute__((packed)){
    t_transfer_block values;

    unsigned char isLastConcatMaxTransportID;
} t_transferblock_tagged;

typedef struct __attribute__((packed)){
    t_accumulator value;
    unsigned char isLast;
} t_conv_drain_tagged;

typedef struct {
        unsigned char bytes[NUM_BITMASK_BYTES];
} t_bitmask;

/*!
   ==================================================
   Data mover and tile controller instructions
   ==================================================
*/
typedef struct __attribute__((packed)) __attribute__((aligned(32))) 
{
    //Concatenation of three signals
    //Bits [3:0] Number of active columns
    //Bit [4]: Flag for synchornization. 1 if there is a need to wait for the synchornization from OA
    //Bit [5] Flag for the compute engine. 0 for convolution, 1 for misc.
    //Bit [6] Flag for sparse input. 0 for dense, 1 for sparse.
    //Bit [7]: Flag for selecting the memory region to read from
    t_uchar memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols;

    //Arch parameter: Starting index of the input dram block in the input memory region
    t_int memBlockStart;
    //Arch parameter: Column stride of input activation strips in dram block in the input memory region
    t_ushort memBlockColStripStride;
    //Arch parameter: Row stride of input activation strips in dram block in the input memory region
    t_ushort memBlockRowStripStride;

#if defined(SPARSE_SYSTEM)
    //Arch parameter: Starting index of the strip TB count in the memory
    t_int memTBCountStart;
    //Arch parameter: Column stride of input activation strip TB count in the memory
    t_ushort memTBCountColStride;
    //Arch parameter: Row stride of input activation strip TB count in the memory
    t_ushort memTBCountRowStride;
    //Problem parameter: 
    //When sending sparse data, this is the number of compression windows in an input group. Used for sending padding
    //When sending dense data, this is the number of valid TB in a strip
    t_ushort numCWOrTBInGroup; 
#else
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
    //[4]: Sync Flag. 1 if there is a need to send sync. signal to the IA mover.
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
#else
    t_uint numDramBlockPerStrip;
#endif

    //Problem parameter: Output tile group to drain from   
    //t_uchar numOAGroup; 
    //Problem parameter: Output tile height per compute column 
    t_uchar tileHeight; 
    //Problem parameter: Output tile width per compute column
    t_uchar columnTileWidth; 
    //Auxillary parameter: Total number of strips to drain.
    t_ushort numColumnTileWidthxTileHeightxNumActiveCols;

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
typedef struct __attribute__((packed)) __attribute__((aligned(16)))
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
    t_ushort numOutputInstructions;
    //Column stride of IA strip in IA cache in terms of dram block
    t_ushort cacheIAStripColStride;
    //Number of output channels in the output group
    t_ushort numOutputChannelsInGroup;
    //Bit[6:0] Number of active PE columns
    //Bit[7] For sparse engine use only. Whether the input activation tensor is dense and hence need bitmask padding.
    t_uchar flagPadBitmaskCatNumActiveCols;
} t_ia_tile_controller_instruction;

//Instructions for the output tile controller
typedef struct __attribute__((packed)) __attribute__((aligned(16)))
{
    //Number of planar indices in the output tile
    t_uchar numLocalTilePerColHxW;
    //Number of channels in the tile
    t_uchar numLocalChannels;
    //Number of compute drain instructions
    t_ushort numDrainInstructions;
    //Number of memory transfer instructions
    t_ushort numMemInstructions;

    //Number of folds required per group to drain the current tile
    t_uchar numFoldsInGroupCurrentLayer;
    //Number of full folds required to drian the current tile
    t_uchar numFullFoldsInCurrentLayer;
    //Number of elements per planar index to drain in the full fold
    t_uchar numActiveElementsInFullFold;
    //Number of elements per planar index to drain in the partial fold
    t_uchar numActiveElementsInPartialFold;

    //Number of channels per group in the next layer
    t_ushort numLocalChannelsPerCurrentGroup;

    //Number of channels per group in the next layer
    t_ushort numLocalChannelsPerNextGroup;

    //Number of active compute columns
    t_uchar numActiveCols;

    //Concatenated signal
    //[2:0] number of bits to shift the convolution output
    //[3] Shift direction. 1 for left shift, 0 for right shift
    //[4] Enable sparsification. 1 for TRUE, 0 for otherwise
    //[5] Source of the output. 0 for convolution engine, 1 for misc.
    //[6] Enable Relu. 1 for TRUE, 0 for false
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

    //Number of dram blocks to reduce
    t_uchar numDramBlocksToReduce;

    //Bit [2:0] Shift amount
    //Bit [3] Flag for left/right shift. 0 for right, 1 for left
    //t_uchar flagLeftShiftCatShiftAmount;

    ////Number of effective values in the final dram block
    t_uchar numEffectiveValues;

} t_misc_instruction;

#ifdef INTELFPGA_CL
typedef uint1_t t_flag;
/*
=======================================================================
Datatypes relevant to the filter transportation system
========================================================================
*/
//Raw data packet travelling on the channels that link the WeightTees.
typedef struct {
    t_dram_block dramBlock;
    unsigned char destinationRow;
} t_dram_block_w_tagged;

//Control packet for the weight buffers
typedef struct __attribute__((packed)) {
    unsigned short numOutputs;
    unsigned short numTransferBlocks;
    t_accumulator bias; //short
    unsigned char maxPeCols; //Number of PE COLS that is activated

} t_filter_streamer_control;

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

    //bool toInputBuffer; 
} t_dram_block_ia_tagged;


typedef struct __attribute__((packed))
{
    unsigned short iaDramBlockAddressBase;
    unsigned short iaDramBlockColStride;
    unsigned short iaDramBlockRowStride;
    #if defined(SPARSE_SYSTEM)
        unsigned char tbAddressBase;
        unsigned char tbAddressRowStride;
    #endif
    unsigned char maxPeRowID; //Only relevant for sending


    //Bit 1:0: 
    // - 00: NOP for some fixed cycles. 
    // - 01: Update the buffer.
    // - 10: Stream from the buffer
    //Bit 2: Only useful for sparse case. Flag for whether the tile require sparse bitmask padding. 1 for true, 0 for false
    //Bit 7:3: Max PE Cols to send to; 
    unsigned char controlBits; 

    //Number of columns in the transfer command
    unsigned char numStripsCol;
    //Number of rows in the transfer command
    unsigned char numStripsRow;
} t_input_buffer_tile_buffer_packet;


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

    /*
        Control bits
        Bit 3:0: Shift direciton and the number of bits to shift the accumulator value from the convolution PE array. Only relevant for loading
        Bit 4: Enable sparsification. Only relevant for the convolutional kernel during ending
        Bit 5: Drainage source. 1: from the MISC kernel. 0: from the convolution kernel.
        Bit 6: Enable Relu. Only relevant for loading
        Bit 7: Load from engine (0) or drain the buffer (1)
    */
    unsigned char controlBits;

} t_output_tile_buffer_packet;

typedef struct __attribute__((packed))
{
    t_output_tile_buffer_packet bufferPacket;
    unsigned char maxColID;
} t_output_tile_buffer_packet_tagged;

typedef struct __attribute__((packed))
{
    unsigned short numLocalTileHeightxLocalTileWidth;

    //Bit [3:0] Maximum column ID
    //Bit [4] Flag for sparse draining sparse input (1 for true)
    //Bit [5] Flag for draining from the MISC kernel (1), or the conv engine (0)
    unsigned char flagSourceCatFlagSparseFlagMaxColID;
} t_output_tile_tee_packet;

typedef struct __attribute__((packed))
{
    //Bit[3:0]: Number of active PE columns
    //Bit[5:4]: OpCode. 00: Add; 01: Max Pooling; 10: Stream
    t_uchar controlBits;

    //Number of dram blocks to reduce
    t_uchar numDramBlocksToReduce;

    //Bit [2:0] Shift amount
    //Bit [3] Flag for left/right shift. 0 for right, 1 for left
    //t_uchar flagLeftShiftCatShiftAmount;

    //Number of effective values in the final dram block
    t_uchar numEffectiveValues;

} t_misc_control_packet;

/*
===================================================================
Data structures that travel on the output activation bus system
===================================================================
*/
typedef struct __attribute__((packed)) {
    //TODO: HANDLE MULTI-BYTE MASK
    t_bitmask bitmask;
    //unsigned char numSurvivingClusters;  //Number of surviving data cluster (not including the bitmask) in the window
    //bool isLastWindowInStrip; //Whether this is the last window in a strip

    //Status bits
    //Bit 5:0: Number of surviving clusters
    //Bit 6: isLastWindowInStrip
    //Bit 7: enableSparsification
    unsigned char statusBits;
} t_output_cluster_info;
//Used to send data to the tee
typedef struct __attribute__((packed)) {
    t_cluster cluster;
    bool isLastInStrip;

} t_output_cluster_tagged;

typedef struct {
    t_cluster clusters[NUM_CLUSTER_IN_DRAM_SIZE];
} t_output_dram_block;

typedef struct __attribute__((packed)) {
    t_output_dram_block block;
    unsigned char isLastFlag; //Bit 0: Is last dram block in a transfer. Bit 1: Is last column
} t_output_dram_block_tagged;
#endif
//#endif

//=====================================

#endif //STRUCTURES_HPP_DEF
