#ifndef DEVICE_UTILS_HPP
#define DEVICE_UTILS_HPP
#include "device_structures.hpp"
#define EMUPRINT
#define HW_DEBUG
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
#elif defined (EMULATOR) && defined(EMUPRINT)
	#define DEBUG_PRINT(format) printf format
#else
	#define DEBUG_PRINT(format)
#endif

t_operand modifyOutput (
		t_accumulator accumulator,
		//Bit [6:0] Shift amount
		//Bit [7] Flag for left/right shift. 0 for right, 1 for left
		unsigned char shiftDirectionCatShiftAmount,
		uint1_t enableRelu
		);

signed char modifyCharOutput (
		signed char input,
		//Bit [6:0] Shift amount
		//Bit [7] Flag for left/right shift. 0 for right, 1 for left
		unsigned char shiftDirectionCatShiftAmount
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

t_dram_block iaMetadata2DramBlock (unsigned short tbCount, unsigned char colSPWidth, unsigned char colSPStride, signed char iColInSPTile);

unsigned char getColSPWidth(t_dram_block block);

unsigned char getColSPStride(t_dram_block block);

unsigned short getTBCount(t_dram_block block);

signed char getColSPIndex(t_dram_block block);

t_output_dram_block clusterCount2OutputDramBlock (unsigned short clusterCount);

unsigned short outputDramBlock2ClusterCount (t_output_dram_block outputDramBlock);

unsigned char getIsLast(t_transferblock_tagged blockTagged);

unsigned char getMaxTransferID(t_transferblock_tagged blockTagged);

void setIsLast (t_transferblock_tagged* pBlockTagged, unsigned char isLast);

void setMaxTransferID (t_transferblock_tagged* pBlockTagged, unsigned char maxTransferID);

#endif

#endif
