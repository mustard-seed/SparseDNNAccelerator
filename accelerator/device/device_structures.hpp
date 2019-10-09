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
#endif



#ifdef INTELFPGA_CL
typedef struct {
    char values [SIMD_SIZE];
} t_simdblock_value; //Value in a simdblock

typedef unsigned char t_simdblock_channel_offset; //Relative channel of a simdblock in a streaming block

typedef unsigned short t_streamblock_address; //Address of a streaming block in BRAM

#else
typedef t_simdblock_host t_simdblock_value;

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
    char cluster_values [CLUSTER_SIZE];
} t_cluster;
#else
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
    uint1_t isLast;
    char maxTransportID;
} t_transferblock_tagged;

typedef struct __attribute__((packed)){
    t_transfer_block values;
    uint1_t isLast;
} t_transferblock_local;

t_transfer_block bias2TransferBlcok (t_accumulator bias)
{
    t_transfer_block transferBlock;
    transferBlock.values[0].cluster_values[0] = bias & 0xFF;
    transferBlock.values[0].cluster_values[1] = (bias >> 8) & 0xFF;
    //transferBlock.values[1].cluster_values[0] = (bias >> 16) & 0xFF;
    //transferBlock.values[1].cluster_values[1] = (bias >> 24) & 0xFF;
    return transferBlock;

}

t_accumulator transferBlock2Bias (t_transfer_block block)
{
    t_accumulator bias =
        ( ((t_accumulator) block.values[0].cluster_values[0]) & 0xFF )
        | (( ((t_accumulator) block.values[0].cluster_values[1]) & 0xFF ) << 8);
        //| (( ((t_accumulator) block.values[1].cluster_values[0]) & 0xFF ) << 16)
        //| (( ((t_accumulator) block.values[1].cluster_values[1]) & 0xFF ) << 24);

    return bias;
}

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
    //nsigned char destinationRow; //f
    unsigned short numTransferBlocks;
    unsigned char maxPeCols; //Number of PE COLS that is activated

} t_filter_streamer_control;

t_filter_streamer_control dramBlock2FilterStreamerControl (t_dram_block block)
{
    t_filter_streamer_control control;
    control.numOutputs =
        ( ( ( (unsigned short) (block.transferBlocks[0].values[1].cluster_values[0]) ) & 0xFF )
            | ( (((unsigned short) (block.transferBlocks[0].values[1].cluster_values[1])) & 0xFF) << 8))  & 0xFFFF;

    //control.destinationRow 
    //    = block.transferBlocks[2].values[0].cluster_values[0];
    control.numTransferBlocks
        = ( ( ( (short) (block.transferBlocks[0].values[1].cluster_values[0]) ) & 0xFF )
            | ( (((short) (block.transferBlocks[0].values[1].cluster_values[1])) & 0xFF) << 8))  & 0xFFFF;

    control.maxPeCols = (unsigned char) block.transferBlocks[1].transferBlocks[0].values[0];
    
    return control;
}

t_dram_block filterStreamerControl2dramBlock (t_filter_streamer_control control)
{
    t_dram_block block;
    block.transferBlocks[0].values[0].cluster_values[0] = control.maxOutputHeightTileSize;
    block.transferBlocks[0].values[0].cluster_values[1] = control.maxOutputWidthTileSize;
    //block.transferBlocks[2].values[0].cluster_values[0] = control.destinationRow;
    block.transferBlocks[0].values[1].cluster_values[0] = control.numTransferBlocks & 0xFF;
    block.transferBlocks[0].values[1].cluster_values[1] = ((control.numTransferBlocks >> 8) & 0xFF);
    block.transferBlocks[1].transferBlocks[0].values[0] = (char) control.maxPeCols;

    return block;
}

/*
===================================================================
Data structures relevant for the input application buffer system
===================================================================
*/
//Raw data packet travelling on the input activation buffer bus
typedef struct {
    t_dram_block dramBlock;
    unsigned char destinationCol;
    //If true, then this packet is for an input buffer. 
    //Otherwise, it is fur all the output buffers less or equal to the detinationCol
    bool toInputBuffer; 
} t_dram_block_a_tagged;

// unsigned char dramBlock2FilterStreamerMaxPeCol (t_dram_block block) 
// {
//     unsigned char maxPeCol = block.transferBlocks[0].values[0].cluster_values[0] & 0xFF;
//     return maxPeCol;

// }

// t_dram_block filterStreamerMaxPeCol2DramBlock (unsigned char maxPeCol)
// {
//     t_dram_block block;
//     block.transferBlocks[0].values[0].cluster_values[0] = maxPeCol & 0xFF;
//     return block;
// }

#endif
//#endif

//=====================================

#endif //STRUCTURES_HPP_DEF
