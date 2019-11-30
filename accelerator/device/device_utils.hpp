#ifndef DEVICE_UTILS_HPP
#define DEVICE_UTILS_HPP
#include "device_structures.hpp"
//#define EMUPRINT
//#define HW_DEBUG
/*
printf enabled during SW emulation
*/
#if defined(EMULATOR) && defined(EMUPRINT)
	#define EMULATOR_PRINT(format) printf format
#else
	#define EMULATOR_PRINT(format)
#endif

/*
printf enabled on HW if -HW_DEBUG flag is set
*/
#if defined(HW_DEBUG) && defined(EMUPRINT)
	#define DEBUG_PRINT(format) printf format
#else
	#define DEBUG_PRINT(format)
#endif

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

#endif
