#include "device_structures.hpp"

t_operand modifyOutput (
		t_accumulator accumulator,
		unsigned char rightShift,
		uint1_t enableRelu
		)
{
	t_accumulator comparedAccumulator;
	unsigned char rndRightShift = rightShift - 1;
	if (enableRelu == TRUE)
	{
		comparedAccumulator = (accumulator > 0x0) ? accumulator : 0x0;
	}
	else
	{
		comparedAccumulator = accumulator;
	}

	t_accumulator signExtensionMask = (comparedAccumulator>=0) ?
		0x00 : ~(0xFFFF >> rndRightShift);

	t_accumulator shiftedAccumulator = comparedAccumulator >> rndRightShift;

	t_accumulator accumulatorWithRndBit = signExtensionMask | shiftedAccumulator;

	t_accumulator accumulatorBiased;
	if(accumulatorWithRndBit >= ((t_accumulator) 256))
	{
		accumulatorBiased = 0x0FF; //=255
	}
	else if(accumulatorWithRndBit <((t_accumulator) -256))
	{
		accumulatorBiased = 0x0100; //=-256
	}
	else
	{
		accumulatorBiased = (t_accumulator) ((0x1FF & accumulatorWithRndBit)+ (t_accumulator) 0x01);
	}
	// final truncation
	t_operand result = 0xFF & (accumulatorBiased>>0x01);  // remove the last rounding bit
	return result;
}

#ifdef INTELFPGA_CL
t_transfer_block bias2TransferBlock (t_accumulator bias)
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
#endif

#ifdef INTELFPGA_CL
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

    control.maxPeCols = (unsigned char) block.transferBlocks[1].values[1].cluster_values[0];
    
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
    block.transferBlocks[1].values[1].cluster_values[0] = (char) control.maxPeCols;

    return block;
}

unsigned char outputModifier2RightShiftAmount (unsigned char outputModifier)
{
    return (outputModifier & 0xF);
}

unsigned char outputModifier2EnableRelu (unsigned char outputModifier)
{
    return (outputModifier >> 4) & 0x1;
}

unsigned char outputModifier2EnableSparsification (unsigned char outputModifier)
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

t_output_dram_block clusterCount2OutputDramBlock (unsigned short clusterCount)
{
    t_output_dram_block outputDramBlock;
    outputDramBlock.clusters[0].cluster_values[0] = (char) (clusterCount & 0xFF);
    outputDramBlock.clusters[0].cluster_values[1] = (char) ((clusterCount >> 8) & 0xFF);
    return outputDramBlock;
}

unsigned short outputDramBlock2ClusterCount (t_output_dram_block outputDramBlock)
{
    char countLow = outputDramBlock.clusters[0].cluster_values[0];
    char countHigh = outputDramBlock.clusters[0].cluster_values[1];

    unsigned short count = 
        ((((t_streamblock_address) countHigh) & 0xFF) << 8)
        | ((((t_streamblock_address) countLow) & 0xFF));

    return count;
}

#endif