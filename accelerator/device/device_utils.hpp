#ifndef DEVICE_UTILS_HPP
#define DEVICE_UTILS_HPP
#include "device_structures.hpp"
#define EMUPRINT
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
		);

#ifdef INTELFPGA_CL
t_transfer_block bias2TransferBlock (t_accumulator bias);

t_accumulator transferBlock2Bias (t_transfer_block block);

t_filter_streamer_control dramBlock2FilterStreamerControl (t_dram_block block);

t_dram_block filterStreamerControl2dramBlock (t_filter_streamer_control control);

unsigned char outputModifier2RightShiftAmount (unsigned char outputModifier);

unsigned char outputModifier2EnableRelu (unsigned char outputModifier);

unsigned char outputModifier2EnableSparsification (unsigned char outputModifier);

unsigned char generateOutputModifier (unsigned char numBitsToRightShift, unsigned char enableRelu, unsigned char enableSparse);


t_dram_block transferBlockCount2DramBlock (t_streamblock_address transferBlockCount);

t_streamblock_address dramBlock2TransferBlockCount (t_dram_block dramBlock);

t_output_dram_block clusterCount2OutputDramBlock (unsigned short clusterCount);

unsigned short outputDramBlock2ClusterCount (t_output_dram_block outputDramBlock);

#endif

#endif
