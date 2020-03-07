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
		__global unsigned char* restrict pNumClusters,
		__global unsigned char* restrict pValid,

		unsigned short bitmask,
		unsigned short mutualBitmask
	)
{
	t_smb_tb operandBuffer;
	unsigned char maskFilterStartIndex = 0;
	unsigned char bufferSize = 0;
	unsigned char accumulatedMaskBytes[2];

	//Initialize the buffers
	for (unsigned i=0; i<SMB_BUFFER_SIZE; i++)
	{
		operandBuffer.values[i] = 0x0;
	}

	{
		ulong4 accumulateMask = smallBufferMaskAccumulator (
				bitmask & 0x0FF,
				((bitmask & 0xFF00) >> 8),
				0,
				0,
				0,
				0,
				0,
				0
			);

		accumulatedMaskBytes[0] = (unsigned char) accumulateMask.s0;
		accumulatedMaskBytes[1] = (unsigned char) (accumulateMask.s0 >> 8);
	}

	unsigned char numClusters = smallBufferPopCounter (
				bitmask & 0x0FF,
				((bitmask & 0xFF00) >> 8),
				0,
				0,
				0,
				0,
				0,
				0
	);

	*pNumClusters = numClusters;
	unsigned char numTransferBlocks = (numClusters == 0) ? 0 : (numClusters - 1) / SMB_TRANSFER_SIZE + 1;
	/*
	 Filter blocks
	*/
	for (unsigned char i=0; i<numTransferBlocks; i++)
	{
		//Read the blocks
		t_smb_tb newTB = pTransferBlocks[i];

		//Filter the mask
		unsigned short maskFilterOutput = smallBufferMaskFilter (
				mutualBitmask & 0x0FF, //unsigned char mutualBitmask0,
				mutualBitmask >> 8, //unsigned char mutualBitmask1,
				0, //unsigned char mutualBitmask2,
				0, //unsigned char mutualBitmask3,
				0, //unsigned char mutualBitmask4,
				0, //unsigned char mutualBitmask5,
				0, //unsigned char mutualBitmask6,
				0, //unsigned char mutualBitmask7,


				//Bytes of the accumulated bitmask
				//Might not need all of them
				accumulatedMaskBytes[0], //unsigned char	accumulatedBitmask0,
				accumulatedMaskBytes[1], //unsigned char	accumulatedBitmask1,
				0, //unsigned char	accumulatedBitmask2,
				0, //unsigned char	accumulatedBitmask3,
				0, //unsigned char	accumulatedBitmask4,
				0, //unsigned char	accumulatedBitmask5,
				0, //unsigned char	accumulatedBitmask6,
				0, //unsigned char	accumulatedBitmask7,
				0, //unsigned char	accumulatedBitmask8,
				0, //unsigned char	accumulatedBitmask9,
				0, //unsigned char	accumulatedBitmask10,
				0, //unsigned char	accumulatedBitmask11,
				0, //unsigned char	accumulatedBitmask12,
				0, //unsigned char	accumulatedBitmask13,
				0, //unsigned char	accumulatedBitmask14,
				0, //unsigned char	accumulatedBitmask15,
				0, //unsigned char	accumulatedBitmask16,
				0, //unsigned char	accumulatedBitmask17,
				0, //unsigned char	accumulatedBitmask18,
				0, //unsigned char	accumulatedBitmask19,
				0, //unsigned char	accumulatedBitmask20,
				0, //unsigned char	accumulatedBitmask21,
				0, //unsigned char	accumulatedBitmask22,
				0, //unsigned char	accumulatedBitmask23,
				0, //unsigned char	accumulatedBitmask24,
				0, //unsigned char	accumulatedBitmask25,
				0, //unsigned char	accumulatedBitmask26,
				0, //unsigned char	accumulatedBitmask27,
				0, //unsigned char	accumulatedBitmask28,
				0, //unsigned char	accumulatedBitmask29,
				0, //unsigned char	accumulatedBitmask30,
				0, //unsigned char	accumulatedBitmask31,
				maskFilterStartIndex
			);

		maskFilterStartIndex = (maskFilterOutput >> 8) & 0x0FF;
		unsigned char operandSelectMask = (maskFilterOutput & 0x3);

		ulong4 bufferUpdateBus = smallBufferMacBufferUpdate(
				operandSelectMask, //inputSelectBitmask
				
				newTB.values[0],
				newTB.values[1],
				newTB.values[2],
				newTB.values[3],
				0,
				0,
				0,
				0,

				operandBuffer.values[0],
				operandBuffer.values[1],
				operandBuffer.values[2],
				operandBuffer.values[3],
				0,
				0,
				0,
				0,

				bufferSize
			);

		t_smb_tb macOutput;
		t_smb_tb nextBuffer;
		for (int j=0; j<SMB_BUFFER_SIZE; j++)
		{
			macOutput.values[j] = (bufferUpdateBus.s0 >> (j*8)) & 0x0FF;
			nextBuffer.values[j] = (bufferUpdateBus.s1 >> (j*8)) & 0x0FF;
		}

		pFilteredBlocks[i] = macOutput;
		pNextBuffer[i] = nextBuffer;

		operandBuffer = nextBuffer;

		bufferSize = bufferUpdateBus.s2 & 0x01;
		pValid[i] = (bufferUpdateBus.s2 >> 8) & 0x01;

	}
}