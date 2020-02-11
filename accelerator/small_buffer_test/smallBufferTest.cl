#define EMUPRINT
//#define HW_DEBUG
#include "rtl_lib.hpp"
#include "device_utils.hpp"
#include "small_buffer.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void smallBufferTest (
		__global t_smb_tb* restrict pTransferBlocks,
		__global t_smb_tb* restrict pFilteredBlocks,
		__global t_smb_tb* restrict pNextBuffer,
		__global unsigned char* pValid,

		unsigned char numTransferBlocks,
		unsigned char bitmask,
		unsigned char mutualBitmask
	)
{
	t_smb_tb operandBuffer;
	unsigned char maskFilterStartIndex = 0;
	unsigned char bufferSize = 0;

	//Initialize the buffers
	for (unsigned i=0; i<2; i++)
	{
		operandBuffer.values[i] = 0x0;
	}
	/*
	 Filter blocks
	*/
	for (unsigned char i=0; i<numTransferBlocks; i++)
	{
		//Read the blocks
		t_smb_tb newTB = pTransferBlocks[i];

		//Filter the mask
		unsigned short maskFilterOutput = smallBufferMaskFilter (
				bitmask, //bitmask
				mutualBitmask, //sparseInput
				maskFilterStartIndex
			);

		maskFilterStartIndex = maskFilterOutput & 0x0FF;
		unsigned char operandSelectMask = ((maskFilterOutput >> 8) & 0x3);

		ulong2 bufferUpdateBus = smallBufferMacBufferUpdate(
				operandSelectMask, //inputSelectBitmask
				
				newTB.values[0],
				newTB.values[1],
				newTB.values[2],
				newTB.values[3],

				operandBuffer.values[0],
				operandBuffer.values[1],
				operandBuffer.values[2],
				operandBuffer.values[3],

				bufferSize
			);

		t_smb_tb macOutput;
		t_smb_tb nextBuffer;
		for (unsigned char j=0; j<4; j++)
		{
			macOutput.values[j] = (bufferUpdateBus.x >> (j*8)) & 0x0FF;
			nextBuffer.values[j] = (bufferUpdateBus.x >> ((j*8) + 32)) & 0x0FF;
		}

		pFilteredBlocks[i] = macOutput;
		pNextBuffer[i] = nextBuffer;

		operandBuffer = nextBuffer;

		bufferSize = bufferUpdateBus.y & 0x03;
		pValid[i] = (bufferUpdateBus.y >> 8) & 0x01;

	}
}