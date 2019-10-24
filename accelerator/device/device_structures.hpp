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
    unsigned short numTransferBlocks;
    t_accumulator bias; //short
    unsigned char maxPeCols; //Number of PE COLS that is activated

} t_filter_streamer_control;

t_filter_streamer_control dramBlock2FilterStreamerControl (t_dram_block block)
{
    t_filter_streamer_control control;
    control.numOutputs =
        ( ( ( (unsigned short) (block.transferBlocks[0].values[0].cluster_values[0]) ) & 0xFF )
            | ( (((unsigned short) (block.transferBlocks[0].values[0].cluster_values[1])) & 0xFF) << 8));

    //control.destinationRow 
    //    = block.transferBlocks[2].values[0].cluster_values[0];
    control.numTransferBlocks
        = ( ( ( (short) (block.transferBlocks[0].values[1].cluster_values[0]) ) & 0xFF )
            | ( (((short) (block.transferBlocks[0].values[1].cluster_values[1])) & 0xFF) << 8));

    //Recover bias
    control.bias
        = ( ( ( (t_accumulator) (block.transferBlocks[1].values[0].cluster_values[0]) ) & 0xFF )
            | ( (((t_accumulator) (block.transferBlocks[1].values[0].cluster_values[1])) & 0xFF) << 8));

    control.maxPeCols = (unsigned char) block.transferBlocks[1].transferBlocks[1].values[0];
    
    return control;
}

t_dram_block filterStreamerControl2dramBlock (t_filter_streamer_control control)
{
    t_dram_block block;
    block.transferBlocks[0].values[0].cluster_values[0] = control.numOutputs & 0xFF;
    block.transferBlocks[0].values[0].cluster_values[1] = ((control.numOutputs >> 8) & 0xFF);

    block.transferBlocks[0].values[1].cluster_values[0] = control.numTransferBlocks & 0xFF;
    block.transferBlocks[0].values[1].cluster_values[1] = ((control.numTransferBlocks >> 8) & 0xFF);

    block.transferBlocks[1].values[0].cluster_values[0] = control.bias & 0xFF;
    block.transferBlocks[1].values[0].cluster_values[1] = ((control.bias >> 8) & 0xFF);
    //block.transferBlocks[2].values[0].cluster_values[0] = control.destinationRow;
    block.transferBlocks[1].transferBlocks[1].values[0] = (char) control.maxPeCols;

    return block;
}

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
    unsigned char topPadding;
    unsigned char bottomPadding;
    unsigned char leftPadding;
    unsigned char rightPadding;
    unsigned char inputTileWidth;
    unsigned char inputTileHeight;
    unsigned char stride;
    unsigned char kernelSize;
    unsigned short numOutputChannelsInGroup;
    unsigned short numInputChannelCompressionWindows;
} t_input_buffer_control;

t_input_buffer_control dramBlock2InputBufferControl (t_dram_block dramBlock)
{
    t_input_buffer_control controlBlock;
    char paddings = dramBlock.transferBlocks[0].values[0].cluster_values[0];
    controlBlock.topPadding = (unsigned char) ((paddings >> 0x0) & 0x3);
    controlBlock.bottomPadding = (unsigned char) ((paddings >> 0x2) & 0x3);
    controlBlock.leftPadding = (unsigned char) ((paddings >> 0x4) & 0x3);
    controlBlock.rightPadding = (unsigned char) ((paddings >> 0x6) & 0x3);

    controlBlock.inputTileWidth = dramBlock.transferBlocks[0].values[0].cluster_values[1];
    controlBlock.inputTileHeight = dramBlock.transferBlocks[0].values[1].cluster_values[0];
    controlBlock.stride = dramBlock.transferBlocks[0].values[1].cluster_values[1] & 0x0F;
    controlBlock.kernelSize = (dramBlock.transferBlocks[0].values[1].cluster_values[1] >> 0x4) & 0xF;
    
    char numOutputChannelsLow = dramBlock.transferBlocks[1].values[0].cluster_values[0];
    char numOutputChannelsHigh = dramBlock.transferBlocks[1].values[0].cluster_values[1];
    controlBlock.numOutputChannelsInGroup = 
        ( (((unsigned short) numOutputChannelsHigh) & 0xFF) << 0x8)
        | ( (((unsigned short) numOutputChannelsLow) & 0xFF) << 0x0);

    char numInputCompressionWindowsLow = dramBlock.transferBlocks[1].values[1].cluster_values[0];
    char numInputCompressionWindowsHigh = dramBlock.transferBlocks[1].values[1].cluster_values[1];
    controlBlock.numInputChannelCompressionWindows = 
        ( (((unsigned short) numInputCompressionWindowsHigh) & 0xFF) << 0x8)
        | ( (((unsigned short) numInputCompressionWindowsLow) & 0xFF) << 0x0);

    return controlBlock;
}

t_dram_block inputBufferControl2DramBlock(t_input_buffer_control controlBlock)
{
    t_dram_block dramBlock;
    char paddings = ((controlBlock.topPadding & 0x3) << 0x0)
        | ((controlBlock.bottomPadding & 0x3) << 0x2)
        | ((controlBlock.leftPadding & 0x3) << 0x4)
        | ((controlBlock.rightPadding & 0x3) << 0x6);
    dramBlock.transferBlocks[0].values[0].cluster_values[0] = paddings;
    dramBlock.transferBlocks[0].values[0].cluster_values[1] = controlBlock.inputTileWidth;
    dramBlock.transferBlocks[0].values[1].cluster_values[0] = controlBlock.inputTileHeight;
    dramBlock.transferBlocks[0].values[1].cluster_values[1] = 
        ((controlBlock.kernelSize & 0xF) << 0x4) | (controlBlock.stride & 0xF);

    dramBlock.transferBlocks[1].values[0].cluster_values[0] = controlBlock.numOutputChannelsInGroup & 0xFF;
    dramBlock.transferBlocks[1].values[1].cluster_values[0] = (controlBlock.numOutputChannelsInGroup >> 0x8) & 0xFF;
    dramBlock.transferBlocks[1].values[1].cluster_values[0] = controlBlock.numInputChannelCompressionWindows & 0xFF;
    dramBlock.transferBlocks[1].values[1].cluster_values[1] = (controlBlock.numInputChannelCompressionWindows >> 0x8) & 0xFF;


    return dramBlock;
}

typedef struct __attribute__((packed))
{
    unsigned char numOutputTileHeightxWidth;
    unsigned char numOutputGroupsCurrentLayer;
    unsigned short numChannelsInOutputGroupCurrentLayer;
    unsigned short numChannelsInInputGroupNextLayer;
    //5:2: number of accumulator bits to right shift
    //1: enableRelu
    //0: enable sparsification 
    unsigned char outputModifierBits;
    //unsigned char numBitsToRightShift;
    //unsigned char enableRelu;
    //unsigned char enableSparsification;
} t_output_buffer_control;

typedef struct __attribute__((packed))
{
    t_output_buffer_control control;
    unsigned char maxColID;
} t_output_buffer_control_tagged;

unsigned char outputModifer2RightShiftAmount (unsigned char outputModifier)
{
    return (outputModifier & 0xF);
}

unsigned char outputModifier2EnableRelu (unsigned char outputModifier)
{
    return (outputModifier >> 4) & 0x1;
}

unsigned char outputModifier2EnableSparsification (unsigned char enableSparsification)
{
    return (outputModifier >> 5) & 0x1;
}

unsigned char generateOutputModifier (unsigned char numBitsToRightShift, unsigned char enableRelu, unsigned char enableSparse)
{
    unsigned char bits =  
        ((enableSparse & 0x1) << 5)
        | ((enableRelu & 0x1) << 4)
        | (numBitsToRightShift & 0xF);

    return bits;
}

// t_output_buffer_control dram2Block2OutputBufferControl (t_dram_block dramBlock)
// {
//     t_output_buffer_control controlBlock;

//     controlBlock.numOutputTileHeightxWidth = dramBlock.transferBlocks[0].values[0].cluster_values[0];
//     controlBlock.numOutputGroupsCurrentLayer = dramBlock.transferBlocks[0].values[0].cluster_values[1];
//     char numChannelsInOutputGroupCurrentLayerLow = dramBlock.transferBlocks[0].values[1].cluster_values[0];
//     char numChannelsInOutputGroupCurrentLayerHigh = dramBlock.transferBlocks[0].values[1].cluster_values[1];
//     controlBlock.numOutputGroupsCurrentLayer = 
//         ((((unsigned short) numChannelsInOutputGroupCurrentLayerLow) & 0xFF) << 0x0)
//         | ((((unsigned short) numChannelsInOutputGroupCurrentLayerHigh) & 0xFF) << 0x8);
//     char numChannelsInOutputGroupNextLayerLow = dramBlock.transferBlocks[1].values[0].cluster_values[0];
//     char numChannelsInOutputGroupNextLayerHigh = dramBlock.transferBlocks[1].values[0].cluster_values[1];
//     controlBlock.numChannelsInInputGroupNextLayer = 
//         ((((unsigned short) numChannelsInOutputGroupNextLayerLow) & 0xFF) << 0x0)
//         | ((((unsigned short) numChannelsInOutputGroupNextLayerHigh) & 0xFF) << 0x8);
//     char outputControl = dramBlock.transferBlocks[1].values[1].cluster_values[0];

//     controlBlock.numBitsToRightShift = outputControl & 0xF;
//     controlBlock.enableRelu = (outputControl >> 4) & 0x1;
//     controlBlock.enableSparsification = (outputControl >> 5) & 0x1;

//     return controlBlock;
// }

// t_dram_block outputBufferControl2DramBlock (t_output_buffer_control controlBlock)
// {
//     t_dram_block dramBlock;

//     dramBlock.transferBlocks[0].values[0].cluster_values[0]= controlBlock.outputTileWidth;
//     dramBlock.transferBlocks[0].values[0].cluster_values[1]= controlBlock.outputTileHeight;
//     dramBlock.transferBlocks[0].values[1].cluster_values[0]= controlBlock.numOutputGroupsCurrentLayer;
//     dramBlock.transferBlocks[0].values[1].cluster_values[1] = 
//         (controlBlock.numChannelsInOutputGroupCurrentLayer & 0xFF);
//     dramBlock.transferBlocks[1].values[0].cluster_values[0] = 
//         ((controlBlock.numChannelsInOutputGroupCurrentLayer >> 8) & 0xFF);

//     dramBlock.transferBlocks[1].values[0].cluster_values[1] = 
//         ((controlBlock.numChannelsInInputGroupNextLayer) & 0xFF);
//     dramBlock.transferBlocks[1].values[1].cluster_values[0] = 
//         ((controlBlock.numChannelsInInputGroupNextLayer >> 8) & 0xFF);

//     dramBlock.transferBlocks[1].values[1].cluster_values[1] = 
//         controlBlock.numBitsToRightShift
//         | ((controlBlock.enableRelu & 0x1) << 4)
//         | ((controlBlock.enableSparsification & 0x1) << 5);
        

//     return dramBlock;
// }

t_dram_block transferBlockCount2DramBlock (t_streamblock_address transferBlockCount)
{
    t_dram_block dramBlock;
    dramBlock.transferBlocks[0].values[0].cluster_values[0] = (char) (transferBlockCount & 0xFF);
    dramBlock.transferBlocks[0].values[0].cluster_values[1] = (char) ((transferBlockCount >> 8) & 0xFF);
    return dramBlock;
}

t_streamblock_address dramBlock2TransferBlockCount (t_dram_block dramBlock)
{
    char countLow = dramBlock.transferBlocks[0].values[0].cluster_values[0];
    char countHigh = dramBlock.transferBlocks[0].values[0].cluster_values[1];

    t_streamblock_address count = 
        ((((t_streamblock_address) countHigh) & 0xFF) << 8)
        | ((((t_streamblock_address) countLow) & 0xFF));

    return count;
}

/*
===================================================================
Data structures that travel on the output activation bus system
===================================================================
*/
typedef struct __attribute__((packed)) {
    t_cluster cluster;
    bool isLastInStrip;
} t_output_cluster_tagged;

typedef struct {
    t_cluster clusters[NUM_CLUSTER_IN_DRAM_SIZE];
} t_output_dram_block;

typedef struct __attribute__((packed)) {
    t_output_dram_block block;
    bool isLast;
} t_output_dram_block_tagged;

t_output_dram_block transferBlockCount2OutputDramBlock (t_streamblock_address transferBlockCount)
{
    t_output_dram_block outputDramBlock;
    outputDramBlock.cluster[0].cluster_values[0] = (char) (transferBlockCount & 0xFF);
    outputDramBlock.cluster[0].cluster_values[1] = (char) ((transferBlockCount >> 8) & 0xFF);
    return outputDramBlock;
}

t_streamblock_address outputDramBlock2TransferBlockCount (t_output_dram_block outputDramBlock)
{
    char countLow = outputDramBlock.cluster[0].cluster_values[0];
    char countHigh = outputDramBlock.cluster[0].cluster_values[1];

    unsigned short count = 
        ((((t_streamblock_address) countHigh) & 0xFF) << 8)
        | ((((t_streamblock_address) countLow) & 0xFF));

    return count;
}
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
