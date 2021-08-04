#include "device_structures.hpp"

//TODO: review this
t_operand modifyOutput (
		t_accumulator accumulator,
        //Bit [3:0] Shift amount
        //Bit [4] Flag for left/right shift. 0 for right, 1 for left
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
    //See round half to even in https://zipcpu.com/dsp/2017/07/22/rounding.html
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

t_operand modifyMiscOutput (
        t_misc_accum accumulator,
        //Bit [3:0] Shift amount
        //Bit [4] Flag for left/right shift. 0 for right, 1 for left
        unsigned char shiftDirectionCatShiftAmount,
        uint1_t enableRelu
        )
{
    uint1_t shiftLeft = (shiftDirectionCatShiftAmount >> 0x4) & 0x01;
    unsigned char shiftAmount = shiftDirectionCatShiftAmount & 0x0F;
    uint1_t preShiftIsNonNegative;
    t_misc_accum ZERO = 0x0;


    t_misc_accum comparedAccumulator;
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
    //See round half to even in https://zipcpu.com/dsp/2017/07/22/rounding.html
    //Note the mask below makes sense if the accumulator is 16-bit
    // unsigned char rndRightShift = shiftAmount - 1;
    t_misc_accum accumulatorMask = MISC_ACCUM_MASK;
    t_misc_accum signExtensionMask = (preShiftIsNonNegative == TRUE) ?
        0x00 : ~(accumulatorMask >> shiftAmount);

    t_misc_accum rightShiftBiasMask = 0x01 << shiftAmount;
    t_misc_accum rightShiftLeadingBiasBit = (comparedAccumulator & rightShiftBiasMask) >> 1;
    t_misc_accum rightShiftRemainingBiasBits = (rightShiftLeadingBiasBit > 0) ? 
        0x0 : (accumulatorMask & ((rightShiftBiasMask - 1) >> 1));
    t_misc_accum rightShiftBias = rightShiftLeadingBiasBit + rightShiftRemainingBiasBits;

    t_misc_accum rightShiftAccumulatorWithRndBit = signExtensionMask | ((t_accumulator) ((comparedAccumulator+rightShiftBias) >> shiftAmount));

    t_misc_accum rightShiftAccumulatorSaturated;
    //Round toward positive infinity for half-point
    if(rightShiftAccumulatorWithRndBit >= ((t_misc_accum) 128))
    {
        rightShiftAccumulatorSaturated = 0x07F; //=127
    }
    else if(rightShiftAccumulatorWithRndBit <((t_misc_accum) -128))
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
    t_misc_accum leftShiftTemp = comparedAccumulator << shiftAmount;
    t_misc_accum leftShiftPreTrunc;
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

// #ifdef INTELFPGA_CL
// t_transfer_block bias2TransferBlock (t_bias bias)
// {
//     t_transfer_block transferBlock;
//     transferBlock.values[0] = bias & 0x0FF;
//     transferBlock.values[1] = (bias >> 8) & 0x0FF;
//     //transferBlock.values[1].cluster_values[0] = (bias >> 16) & 0xFF;
//     //transferBlock.values[1].cluster_values[1] = (bias >> 24) & 0xFF;
//     return transferBlock;

// }

// t_bias transferBlock2Bias (t_transfer_block block)
// {
//     t_bias bias =
//         ( ((t_bias) block.values[0]) & 0xFF )
//         | (( ((t_bias) block.values[1]) & 0xFF ) << 8);
//         //| (( ((t_accumulator) block.values[1].cluster_values[0]) & 0xFF ) << 16)
//         //| (( ((t_accumulator) block.values[1].cluster_values[1]) & 0xFF ) << 24);

//     return (bias & 0xFFFF);
// }
// #endif

#ifdef INTELFPGA_CL
t_filter_streamer_control dramBlock2FilterStreamerControl (t_weight_dram_block block)
{
    #if (WEIGHT_DRAM_SIZE_VALUE_BYTE < 8)
        #error WEIGHT bus needs to be at least 8 words wide!
    #endif
    t_filter_streamer_control control;
    #if defined(SPW_SYSTEM)
        control.numNZClustersPerPruneRange = block.indices[0];
    #endif
    control.numOutputsXNumTransferBlocks =
        ( ( ( (unsigned int) (block.values[0]) ) & 0xFF )
            | ( (((unsigned int) (block.values[1])) & 0xFF) << 8)
            | ( (((unsigned int) (block.values[2])) & 0xFF) << 16)
            | ( (((unsigned int) (block.values[3])) & 0xFF) << 24)
            );

    //control.destinationRow 
    //    = block.transferBlocks[2].values[0].cluster_values[0];
    control.numTransferBlocks
        = ( ( ( (short) (block.values[4]) ) & 0xFF )
            | ( (((short) (block.values[5])) & 0xFF) << 8));

    control.maxPeCols = (unsigned char) block.values[6];
    control.flagIsReal = (unsigned char) block.values[7]; 

    return control;
}

t_weight_dram_block filterStreamerControl2dramBlock (t_filter_streamer_control control)
{
    #if (WEIGHT_DRAM_SIZE_VALUE_BYTE < 8)
        #error WEIGHT bus needs to be at least 8 words wide!
    #endif
    t_weight_dram_block block;
    #if defined(SPW_SYSTEM)
        block.indices[0] = control.numNZClustersPerPruneRange;
    #endif
    block.values[0] = control.numOutputsXNumTransferBlocks & 0xFF;
    block.values[1] = ((control.numOutputsXNumTransferBlocks >> 8) & 0xFF);
    block.values[2] = ((control.numOutputsXNumTransferBlocks >> 16) & 0xFF);
    block.values[3] = ((control.numOutputsXNumTransferBlocks >> 24) & 0xFF);

    block.values[4] = control.numTransferBlocks & 0xFF;
    block.values[5] = ((control.numTransferBlocks >> 8) & 0xFF);

    //block.transferBlocks[2].values[0].cluster_values[0] = control.destinationRow;
    block.values[6] = (char) control.maxPeCols;
    block.values[7] = (char) control.flagIsReal;

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




#endif
