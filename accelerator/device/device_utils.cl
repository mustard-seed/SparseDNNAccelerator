#include "device_structures.hpp"

//TODO: review this
t_operand modifyOutput (
		t_accumulator accumulator,
        //Bit [2:0] Shift amount
        //Bit [3] Flag for left/right shift. 0 for right, 1 for left
        unsigned char shiftDirectionCatShiftAmount,
		uint1_t enableRelu
		)
{
    uint1_t shiftLeft = (shiftDirectionCatShiftAmount >> 0x4) & 0x01;
    unsigned char shiftAmount = shiftDirectionCatShiftAmount & 0x0F;
    uint1_t preShiftIsNonNegative;
    t_accumulator ZERO = 0x0;


	t_accumulator comparedAccumulator;
	if (enableRelu == TRUE)
	{
		comparedAccumulator = (accumulator > ZERO) ? accumulator : 0x0;
        preShiftIsNonNegative = TRUE;
	}
	else
	{
		comparedAccumulator = accumulator;
        preShiftIsNonNegative = (accumulator >= ZERO) ? TRUE : FALSE;
	}

    //Handle the right shift case
    //Note the mask below makes sense if the accumulator is 16-bit
    // unsigned char rndRightShift = shiftAmount - 1;
    t_accumulator accumulatorMask = ACCUM_MASK;
	t_accumulator signExtensionMask = (preShiftIsNonNegative == TRUE) ?
		0x00 : ~(accumulatorMask >> shiftAmount);

    t_accumulator rightShiftBiasMask = 0x01 << shiftAmount;
    t_accumulator rightShiftLeadingBiasBit = (comparedAccumulator & rightShiftBiasMask) >> 1;
    t_accumulator rightShiftRemainingBiasBits = (rightShiftLeadingBiasBit > 0) ? 
        0x0 : (accumulatorMask & ((rightShiftBiasMask - 1) >> 1));
    t_accumulator rightShiftBias = rightShiftLeadingBiasBit + rightShiftRemainingBiasBits;

	t_accumulator rightShiftAccumulatorWithRndBit = signExtensionMask | ((t_accumulator) ((comparedAccumulator+rightShiftBias) >> shiftAmount));

	t_accumulator rightShiftAccumulatorSaturated;
    //Round toward positive infinity for half-point
	if(rightShiftAccumulatorWithRndBit >= ((t_accumulator) 128))
	{
		rightShiftAccumulatorSaturated = 0x07F; //=127
	}
	else if(rightShiftAccumulatorWithRndBit <((t_accumulator) -128))
	{
		rightShiftAccumulatorSaturated = 0x0080; //=-128
	}
	else
	{
		// rightShiftAccumulatorBiased = (t_accumulator) ((0x1FF & rightShiftAccumulatorWithRndBit)+ (t_accumulator) 0x01);
	    rightShiftAccumulatorSaturated = rightShiftAccumulatorWithRndBit;
    }
	// final truncation for the right shift
	//t_operand rightShiftResult = 0xFF & (rightShiftAccumulatorBiased>>0x01);  // remove the last rounding bit
	t_operand rightShiftResult = 0x0FF & rightShiftAccumulatorSaturated;

    //Handle the left shift case
    t_accumulator leftShiftTemp = comparedAccumulator << shiftAmount;
    t_accumulator leftShiftPreTrunc;
    if (preShiftIsNonNegative == TRUE)
    {   
        leftShiftPreTrunc = (leftShiftTemp < accumulator) ? 0x07F : (
                                    (leftShiftTemp <= ((t_accumulator) 127)) ? leftShiftTemp : 0x07F
                                );

    }
    else
    {
        leftShiftPreTrunc = (leftShiftTemp > accumulator) ? 0x080 : (
                                    (leftShiftTemp >= ((t_accumulator) -128)) ? leftShiftTemp : 0x080
                                );
    }

    t_operand leftShiftResult = 0xFF & leftShiftPreTrunc;

    t_operand result = (shiftLeft == TRUE) ? leftShiftResult : rightShiftResult;
    return result;
}

//TODO: review this
signed char modifyCharOutput (
        signed char input,
        //Bit [2:0] Shift amount
        //Bit [3] Flag for left/right shift. 0 for right, 1 for left
        unsigned char shiftDirectionCatShiftAmount
        )
{
    uint1_t shiftLeft = (shiftDirectionCatShiftAmount >> 0x4) & 0x01;
    unsigned char shiftAmount = shiftDirectionCatShiftAmount & 0x0F;
    uint1_t originalIsNonNegative = (input >= 0) ? TRUE : FALSE;

    //Handle the right shift
    //Use round to even, see
    //https://zipcpu.com/dsp/2017/07/22/rounding.html
    //unsigned char rndRightShift = shiftAmount - 1;
    signed short signExtensionMask = (originalIsNonNegative == TRUE) ?
        0x00 : ~( ((signed short) 0xFFFF) >> shiftAmount);
    signed short rightShiftBiasMask = 0x01 << shiftAmount;
    signed short rightShiftLeadingBiasBit = (((signed short) input) & rightShiftBiasMask) >> 1;
    signed short rightShiftRemainingBiasBits = (rightShiftLeadingBiasBit > 0) ? 
        0x0 : (0xFFFF & ((rightShiftBiasMask - 1) >> 1));
    signed short rightShiftBias = 
        rightShiftLeadingBiasBit + rightShiftRemainingBiasBits;

    signed short rightShiftAccumulatorWithRndBit = signExtensionMask | ((signed short) ((input+rightShiftBias) >> shiftAmount));

    signed short rightShiftAccumulatorSaturated;
    //Round toward positive infinity for half-point
    if(rightShiftAccumulatorWithRndBit >= ((signed short) 128))
    {
        rightShiftAccumulatorSaturated = 0x07F; //=127
    }
    else if(rightShiftAccumulatorWithRndBit <((signed short) -128))
    {
        rightShiftAccumulatorSaturated = 0x0080; //=-128
    }
    else
    {
        // rightShiftAccumulatorBiased = (t_accumulator) ((0x1FF & rightShiftAccumulatorWithRndBit)+ (t_accumulator) 0x01);
        rightShiftAccumulatorSaturated = rightShiftAccumulatorWithRndBit;
    }
    // final truncation for the right shift
    //t_operand rightShiftResult = 0xFF & (rightShiftAccumulatorBiased>>0x01);  // remove the last rounding bit
    signed short rightShiftResult = 0x0FF & rightShiftAccumulatorSaturated;

    //Handle the left shift
    signed short leftShiftTemp = ((signed short) input) << shiftAmount;
    signed short leftShiftFinal;
    if (originalIsNonNegative == TRUE)
    {
        leftShiftFinal = (leftShiftTemp < input) ? 0x7F : (
                                    (leftShiftTemp <= ((signed short) 127)) ? leftShiftTemp : 0x07F
                                );

    }
    else
    {
        leftShiftFinal = (leftShiftTemp > input) ? 0x80 : (
                                    (leftShiftTemp >= ((signed short) -128)) ? leftShiftTemp : 0x080
                                );
    }

    signed char output = (shiftLeft == TRUE) ? leftShiftFinal : rightShiftResult;

    return output;
}

#ifdef INTELFPGA_CL
t_transfer_block bias2TransferBlock (t_bias bias)
{
    t_transfer_block transferBlock;
    transferBlock.values[0] = bias & 0x0FF;
    transferBlock.values[1] = (bias >> 8) & 0x0FF;
    //transferBlock.values[1].cluster_values[0] = (bias >> 16) & 0xFF;
    //transferBlock.values[1].cluster_values[1] = (bias >> 24) & 0xFF;
    return transferBlock;

}

t_bias transferBlock2Bias (t_transfer_block block)
{
    t_bias bias =
        ( ((t_bias) block.values[0]) & 0xFF )
        | (( ((t_bias) block.values[1]) & 0xFF ) << 8);
        //| (( ((t_accumulator) block.values[1].cluster_values[0]) & 0xFF ) << 16)
        //| (( ((t_accumulator) block.values[1].cluster_values[1]) & 0xFF ) << 24);

    return (bias & 0xFFFF);
}
#endif

#ifdef INTELFPGA_CL
t_filter_streamer_control dramBlock2FilterStreamerControl (t_dram_block block)
{
    t_filter_streamer_control control;
    control.numOutputs =
        ( ( ( (unsigned short) (block.transferBlocks[0].values[0]) ) & 0xFF )
            | ( (((unsigned short) (block.transferBlocks[0].values[1])) & 0xFF) << 8));

    //control.destinationRow 
    //    = block.transferBlocks[2].values[0].cluster_values[0];
    control.numTransferBlocks
        = ( ( ( (short) (block.transferBlocks[0].values[2]) ) & 0xFF )
            | ( (((short) (block.transferBlocks[0].values[3])) & 0xFF) << 8));

    //Recover bias
    #if (NUM_SIMD_WORDS <= 4)
        control.bias
            = ( ( ( (t_bias) (block.transferBlocks[1].values[0]) ) & 0xFF )
                | ( (((t_bias) (block.transferBlocks[1].values[1])) & 0xFF) << 8));

        control.maxPeCols = (unsigned char) block.transferBlocks[1].values[2];
    #else
        control.bias
            = ( ( ( (t_bias) (block.transferBlocks[0].values[4]) ) & 0xFF )
                | ( (((t_bias) (block.transferBlocks[0].values[5])) & 0xFF) << 8));

        control.maxPeCols = (unsigned char) block.transferBlocks[0].values[6];
    #endif
    
    return control;
}

t_dram_block filterStreamerControl2dramBlock (t_filter_streamer_control control)
{
    t_dram_block block;
    block.transferBlocks[0].values[0] = control.numOutputs & 0xFF;
    block.transferBlocks[0].values[1] = ((control.numOutputs >> 8) & 0xFF);

    block.transferBlocks[0].values[2] = control.numTransferBlocks & 0xFF;
    block.transferBlocks[0].values[3] = ((control.numTransferBlocks >> 8) & 0xFF);

    #if (NUM_SIMD_WORDS <= 4)
        block.transferBlocks[1].values[0] = control.bias & 0xFF;
        block.transferBlocks[1].values[1] = ((control.bias >> 8) & 0xFF);
        //block.transferBlocks[2].values[0].cluster_values[0] = control.destinationRow;
        block.transferBlocks[1].values[2] = (char) control.maxPeCols;
    #else
        block.transferBlocks[0].values[4] = control.bias & 0xFF;
        block.transferBlocks[0].values[5] = ((control.bias >> 8) & 0xFF);
        //block.transferBlocks[2].values[0].cluster_values[0] = control.destinationRow;
        block.transferBlocks[0].values[6] = (char) control.maxPeCols;
    #endif

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
    dramBlock.transferBlocks[0].values[0] = (char) (transferBlockCount & 0xFF);
    dramBlock.transferBlocks[0].values[1] = (char) ((transferBlockCount >> 8) & 0xFF);

    return dramBlock;
}

t_streamblock_address dramBlock2TransferBlockCount (t_dram_block dramBlock)
{
    char countLow = dramBlock.transferBlocks[0].values[0];
    char countHigh = dramBlock.transferBlocks[0].values[1];

    t_streamblock_address count = 
        ((((t_streamblock_address) countHigh) & 0xFF) << 8)
        | ((((t_streamblock_address) countLow) & 0xFF));

    return count;
}

t_dram_block iaMetadata2DramBlock (unsigned short tbCount, unsigned char colSPWidth, unsigned char colSPStride, signed char iColInSPTile)
{
    t_dram_block dramBlock;
    dramBlock.transferBlocks[0].values[0] = (signed char) (tbCount & 0xFF);
    dramBlock.transferBlocks[0].values[1] = (signed char) ((tbCount >> 8) & 0x0FF);
    dramBlock.transferBlocks[0].values[2] = (signed char) colSPWidth;
    dramBlock.transferBlocks[0].values[3] = (signed char) colSPStride;
#if (WIDE_SIZE >= 2)
    dramBlock.transferBlocks[1].values[0] = (signed char) iColInSPTile;
#else
    dramBlock.transferBlocks[0].values[4] = (signed char) iColInSPTile;
#endif

    return dramBlock;
}

unsigned char getColSPWidth(t_dram_block block)
{
    return block.transferBlocks[0].values[2];
}

unsigned char getColSPStride(t_dram_block block)
{
    return block.transferBlocks[0].values[3];
}

unsigned short getTBCount(t_dram_block block)
{
    char countLow = block.transferBlocks[0].values[0];
    char countHigh = block.transferBlocks[0].values[1];

    t_streamblock_address count = 
        ((((unsigned short) countHigh) & 0xFF) << 8)
        | ((((unsigned short) countLow) & 0xFF));

    return count;
}

signed char getColSPIndex(t_dram_block block)
{
#if (WIDE_SIZE >= 2)
    return block.transferBlocks[1].values[0];
#else
    return block.transferBlocks[0].values[4];
#endif
    
}

t_output_dram_block clusterCount2OutputDramBlock (unsigned short clusterCount)
{
    t_output_dram_block outputDramBlock;
    outputDramBlock.clusters[0].cluster_values[0] = (char) (clusterCount & 0xFF);
    #if (CLUSTER_SIZE > 1)
        outputDramBlock.clusters[0].cluster_values[1] = (char) ((clusterCount >> 8) & 0xFF);
    #else
        outputDramBlock.clusters[1].cluster_values[0] = (char) ((clusterCount >> 8) & 0xFF);
    #endif
    return outputDramBlock;
}

unsigned short outputDramBlock2ClusterCount (t_output_dram_block outputDramBlock)
{
    char countLow = outputDramBlock.clusters[0].cluster_values[0];
    #if (CLUSTER_SIZE > 1)
        char countHigh = outputDramBlock.clusters[0].cluster_values[1];
    #else
        char countHigh = outputDramBlock.clusters[1].cluster_values[0];
    #endif

    unsigned short count = 
        ((((t_streamblock_address) countHigh) & 0xFF) << 8)
        | ((((t_streamblock_address) countLow) & 0xFF));

    return count;
}

unsigned char getIsLast(t_transferblock_tagged blockTagged)
{
    return (((blockTagged.isLastConcatMaxTransportID) >> 7) & 0x01);
}

unsigned char getMaxTransferID(t_transferblock_tagged blockTagged)
{
    return (blockTagged.isLastConcatMaxTransportID & 0x07F);
}

void setIsLast (t_transferblock_tagged* pBlockTagged, unsigned char isLast)
{
    pBlockTagged->isLastConcatMaxTransportID &= 0x07F;
    pBlockTagged->isLastConcatMaxTransportID |= ((unsigned char)((isLast << 7) & 0x080));
}

void setMaxTransferID (t_transferblock_tagged* pBlockTagged, unsigned char maxTransferID)
{
    pBlockTagged->isLastConcatMaxTransportID &= 0x080;
    pBlockTagged->isLastConcatMaxTransportID |= ((unsigned char)(maxTransferID & 0x07F));
}

void initialize_dramblock(t_dram_block* pDramBlock)
{
    #pragma unroll
    for (int itb=0; itb<WIDE_SIZE; itb++)
    {
        #pragma unroll
        for (int iVal=0; iVal<(TRANSFER_SIZE*CLUSTER_SIZE); iVal++)
        {
            pDramBlock->transferBlocks[itb].values[iVal] = 0x0;
        }
    }
}

#endif
