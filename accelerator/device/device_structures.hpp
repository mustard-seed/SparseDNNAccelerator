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
//typedef short t_spValueAndZCount;
typedef unsigned short t_spOffset;

typedef char t_operand;
#else
//typedef cl_short t_spValueAndZCount;
typedef cl_ushort t_spOffset;

typedef cl_char t_operand;
#endif



#ifdef INTELFPGA_CL
typedef struct {
    char values [SIMD_SIZE];
} t_simdblock_value; //Value in a simdblock

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
    t_cluster values [TRANSFER_SIZE];
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

// typedef struct __attribute__((packed)){
//     t_transfer_block values;
// #ifdef INTELFPGA_CL
//     uint1_t isLast;
// #else
//     bool isLast;
// #endif
// } t_transferblock_local;

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
    unsigned char bitmask;
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
