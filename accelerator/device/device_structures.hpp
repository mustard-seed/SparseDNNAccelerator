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
    t_short memBlockColStripStride;
    //Arch parameter: Row stride of input activation strips in dram block in the input memory region
    t_short memBlockRowStripStride;

#if defined(SPARSE_SYSTEM)
    //Arch parameter: Starting index of the strip TB count in the memory
    t_int memTBCountStart;
    //Arch parameter: Column stride of input activation strip TB count in the memory
    t_short memTBCountColStride;
    //Arch parameter: Row stride of input activation strip TB count in the memory
    t_short memTBCountRowStride;
#else
    t_int numTBPerStrip;
#endif


    //Problem parameter: memory input tile stretched padded height
    t_uchar tileSPHeight;
    //Problem parameter: memory input tile stretched padded width
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

    //Problem parameter: number of compression windows in an input group. Used for sending padding
    t_uchar numCWInGroup; 


    //Problem parameter: memory input tile stretched unit size
    //[3:0] horizontal stretched unit size
    //[7:4]: vertical stretched unit size
    t_uchar concatSPSize;

    //Problem parameter: compute column input width stride
    t_uchar columnWidthStride;
    //Problem parameter: compute column strided padded input width
    t_uchar columnSPWidth;

    //Auxillary parameter: total number of strips to send in this transfer
    t_ushort columnSPWidthxTileSPHeightxNumActiveCols; 
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
    t_int memOAGroupStride;
    //Arch. parameter: tile stride in terms of dram block in the output memory region
    t_short memOATileStride;
    //Arch. parameter: column stride in terms of dram block in the output memory region
    t_short memOAColStride;
    //Arch. parameter: row stride in terms of dram block in the output memory region
    t_short memOARowStride;

#if defined(SPARSE_SYSTEM)
    //Arch. parameter: Index of the first TB count element of this transfer in memory
    t_int memTBStart;
    //Arch. parameter: group stride in terms of TB count in the TB memory
    t_short memTBGroupStride;
    //Arch. parameter: tile stride in terms of TB count in the TB memory.
    t_short memTBTileStride;
    //Arch. parameter: column stride in terms of TB count in the TB memory.
    t_short memTBColStride;
    //Arch. parameter: row stride in terms of TB count in the TB memory.
    t_short memTBRowStride;
#else
    t_int numTBPerStrip;
#endif

    //Problem parameter: Output tile group to drain from   
    t_uchar numOAGroup; 
    //Problem parameter: Output tile height per compute column 
    t_uchar tileHeight; 
    //Problem parameter: Output tile width per compute column
    t_uchar columnTileWidth; 
    //Auxillary parameter: Total number of strips to drain.
    t_ushort numOAGroupxColumnTileWidthxTileHeightxNumActiveCols;

} t_oa_mover_instruction;

typedef struct __attribute__((packed)) __attribute__((aligned(32)))
{
    //Arch. parameter: number of folds
    t_ushort numFilterFold;
    //Arch. parameter: number of full folds
    t_ushort numFullFilterFold;
    //Arch. parameter: number of filter in the partial fold
    t_uchar numFiltersInPartialFold;
    //Arch. parameter: number of filter reuse
    t_ushort filterReuse;

    //Arch. parameter: Start of the transfer in the bias memory region
    t_int memBiasStart;
    //Arch. parameter: Start of the transfer in the weight dram_block region
    t_int memWeightStart;
    //Arch. parameter: filter stride in the weight dram block region.
    t_int memWeightFilterStride;

#if defined(SPARSE_SYSTEM)
    //Arch. parameter: Start of the transfer in the weight TB count region
    t_int memTBCountStart;
    //Arch. parameter: Filter stride in the weight TB count region.
    t_int memTBCountFilterStride;
#else
    t_int numTBPerFilter;
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
    //Column stride of strip in IA cache in terms of dram block
    t_ushort cacheStripStride;

    t_ushort numTBPerStrip;
} t_ia_tile_controller_instruction;

//Instructions for the output tile controller
typedef struct __attribute__((packed)) __attribute__((aligned(16)))
{
    //Number of planar indices in the output tile
    t_uchar numLocalTileHxW;
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
    //Number of elements per planar index to drain in the partial fold
    t_uchar numActiveElementsInPartialFold;
    //Number of elements par planar index to drain the full fold
    t_uchar numActiveElementsInFullFold;

    //Number of channels per group in the next layer
    t_ushort numLocalChannelsPerNextGroup;

    //Number of active compute columns
    t_uchar numActiveCols;

    //Concatenated signal
    //[3:0] Number of bits to right-shift the output
    //[4] Source of the output. 1 for convolution engine, 0 for misc.
    //[5] Enable Relu. 1 for TRUE, 0 for false
    //[6] Enable sparsification. 1 for TRUE, 0 for otherwise
    t_uchar flagSparseCatFlagReluCatFlagSourceCatRShift;
    
} t_oa_tile_controller_instruction;

#ifdef INTELFPGA_CL
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
    unsigned char destinationCol;
    //If true, then this packet is for an input buffer. 
    //Otherwise, it is fur all the output buffers less or equal to the detinationCol
    //bool toInputBuffer; 
} t_dram_block_ia_tagged;

//Input buffer control packet data structure
typedef struct __attribute__((packed)) 
{
    unsigned char inputTileWidth;
    unsigned char inputTileHeight;
    unsigned char strideConcatKernelSize; //Bit 7:4 stride, 3:0 KernelSize
    unsigned int numOutputInstructions;
    #if !defined (SPARSE_SYSTEM)
    unsigned short numTBCountPerStrip;
    #endif
    unsigned short numActivePeColsConcatNumOutputChannelsInGroup; //15:12: Number of Active PeCols, 11:0: Number of output channels in group
    unsigned short strideStripIACache; //Stride in terms of dram block
} t_input_buffer_tile_controller_packet;

typedef struct __attribute__((packed))
{
    unsigned short iActivationDramBlockAddressBase;
    unsigned short strideActivationDramBlock;
    #if defined(SPARSE_SYSTEM)
    unsigned char iAddressCache;
    #endif
    unsigned char maxPeRowID; //Only relevant for sending

    #if !defined (SPARSE_SYSTEM)
    unsigned short numTBCountPerStrip;
    #endif

    //Bit 1:0: 
    // - 00: NOP for some fixed cycles. 
    // - 01: Update the buffer.
    // - 10: Stream from the buffer, not the last strip. 
    // - 11: Stream from the buffer, is the last strip
    //Bit 7:2: Max PE Cols to send to; 
    unsigned char controlBits; 

    unsigned char numStripInRow; //Number of strips in the row concerned by the instruction.
} t_input_buffer_tile_buffer_packet;


typedef struct __attribute__((packed))
{
    unsigned char numOutputTileHeightxWidth;
    unsigned char numFoldsInGroupCurrentLayer;
    unsigned char numFullFoldsInGroupCurrentLayer;
    unsigned char numActiveRowsInPartialFolds;
    unsigned char numActivePeCols;

    unsigned short numGroupsNextLayer;
    unsigned short numChannelsInGroupCurrentLayer;
    unsigned short numChannelsInGroupNextLayer;
    //3:0: number of accumulator bits to right shift
    //4: enableRelu
    //5: enable sparsification 
    unsigned char outputModifierBits;


} t_output_tile_controller_packet;

typedef struct __attribute__((packed))
{
    //Index of the output buffer at the start of the transaction
    //Assume the layout of the output is HWC
    unsigned short startOutputIndex; 

    //Number of output values to load into or to stream from the cache during this instruction cycle.
    unsigned short numOutputToAccess; 

    /*
        Control bits
        Bit 3:0: Number of bits to right shift the accumulator value from the PE array. Only relevant for loading
        Bit 4: Enable Relu. Only relevant for loading
        Bit 5: Enable sparsification. Only relevant for sending
        Bit 6: Load from array (0) or send to drainer
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
    unsigned short numOutputGroupxTileHeightxTileWidth;
    unsigned char maxColID;
} t_output_tile_tee_packet;

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
