#include "rtl_lib.hpp"
#include "params.hpp"

#define TRUE 0x1
#define FALSE 0X0

int findFirstOccurance (unsigned char accumulatedIndices [], int startIndex, int bitsPerIndex, unsigned char target)
{
	int lowIndex = startIndex*bitsPerIndex;
	int highIndex = lowIndex+bitsPerIndex-1;
	int index = startIndex;
	while (lowIndex < (bitsPerIndex)*COMPRESSION_WINDOW_SIZE)
	{
		unsigned char subString = 0;
		getBitsInBus(&subString, accumulatedIndices, lowIndex, highIndex);
		if (subString == target)
		{
			break;
		}
		index += 1;
		lowIndex += bitsPerIndex;
		highIndex += bitsPerIndex;
	}

	return index;
}

unsigned char getBitsInByte (unsigned char source, int nLow, int nHigh)
{
	unsigned int mask = (((unsigned int) (0x1 << (nHigh - nLow + 1))) - 1) << nLow;
	unsigned char result = (unsigned char) ((((unsigned int) source) & ((unsigned int) mask)) >> nLow);

	return result;
}

void setBitsInByte (unsigned char * pDestination, unsigned char source, int nLow, int nHigh)
{
	unsigned int zeroMask = ~((((unsigned int) (0x1 << (nHigh - nLow + 1))) - 1) << nLow);
	unsigned int selectMask = (unsigned int) (0x1 << (nHigh - nLow + 1)) - 1;

	unsigned char result = *pDestination;
	result &= zeroMask;

	result |= ((unsigned char) ((source & selectMask) << nLow)) | result;

	*pDestination = result;

}

void getBitsInBus (unsigned char pDestination[], unsigned char pSource[], int nLow, int nHigh)
{
	int numBitsToGet = nHigh - nLow + 1;
	int iDestination;
	for (iDestination=0; iDestination<numBitsToGet; iDestination++) {
		int iBitInSourceBus = nLow + iDestination;
		int iByteInSource = iBitInSourceBus / 8;
		int iBitInSourceByte = iBitInSourceBus % 8;
		unsigned char bit = getBitsInByte(pSource[iByteInSource], iBitInSourceByte, iBitInSourceByte);

		int iByteInDestination = iDestination / 8;
		int iBitInDestinationByte = iDestination % 8;
		setBitsInByte(pDestination+iByteInDestination, bit, iBitInDestinationByte, iBitInDestinationByte);
	}
}

void setBitsInBus (unsigned char pDestination[], unsigned char pSource[], int nLow, int nHigh)
{
	int numBitsToGet = nHigh - nLow + 1;
	int iSource;
	for (iSource=0; iSource<numBitsToGet; iSource++) {
		int iBitInDestinationBus = nLow + iSource;
		int iByteInDestination = iBitInDestinationBus / 8;
		int iBitInDestinationByte = iBitInDestinationBus % 8;

		int iByteInSource = iSource / 8;
		int iBitInSourceByte = iSource % 8;


		unsigned char bit = getBitsInByte(pSource[iByteInSource], iBitInSourceByte, iBitInSourceByte);

		setBitsInByte(pDestination+iByteInDestination, bit, iBitInDestinationByte, iBitInDestinationByte);
	}
}


unsigned char getNthOnePosition (unsigned short bitmask, unsigned char n, unsigned char startIndex)
{
	unsigned char result = 16;
	unsigned char count = 0;
	for (unsigned char i=0; i<16; i++)
	{
		unsigned short selectMask = (unsigned short) ((0x01) << i);
		unsigned short bit = (bitmask & selectMask) >> i;
		if ((bit == 0x01) && (i >= startIndex))
		{
			if (result == 16) {
				count++;
				if (count == n) {
					result = i;
				}
			}
		}
	}
	return result;
}

unsigned char smallBufferPopCounter (
		unsigned char bitmask0,
		unsigned char bitmask1,
		unsigned char bitmask2,
		unsigned char bitmask3,
		unsigned char bitmask4,
		unsigned char bitmask5,
		unsigned char bitmask6,
		unsigned char bitmask7
	)
{
	unsigned char input[8] = {bitmask0, bitmask1, bitmask2, bitmask3, bitmask4, bitmask5, bitmask6, bitmask7};

	unsigned char count = 0;

	for (int i=0; i<64; i++)
	{
		int iByte = i / 8;
		int iBit = i % 8;

		unsigned char bit = getBitsInByte(input[iByte], iBit, iBit);

		if ((bit & 0x01) == 0x01)
		{
			count++;
		}
	}

	return count;
}

ulong4 smallBufferMaskAccumulator (
		unsigned char bitmask0,
		unsigned char bitmask1,
		unsigned char bitmask2,
		unsigned char bitmask3,
		unsigned char bitmask4,
		unsigned char bitmask5,
		unsigned char bitmask6,
		unsigned char bitmask7
	)
{
	unsigned char bitmaskBytes[8] = {bitmask0, bitmask1, bitmask2, bitmask3, bitmask4, bitmask5, bitmask6, bitmask7};
	unsigned char accumulatedBytes[32] = {0};
	ulong4 result= (ulong4) (0,0,0,0);

	unsigned char currentAccum = 0;
	for (int i=0; i<COMPRESSION_WINDOW_SIZE; i++)
	{

		currentAccum = (currentAccum == TRANSFER_SIZE) ? 0 : currentAccum;
		unsigned char inputBit = 0;
		getBitsInBus(&inputBit, bitmaskBytes, i, i);
		currentAccum = currentAccum + (inputBit & 0x01);

		int nLow = i*BITMASK_ACCUM_COUNT_BITWIDTH;
		int nHigh = nLow + BITMASK_ACCUM_COUNT_BITWIDTH - 1;

		setBitsInBus(accumulatedBytes, &currentAccum, nLow, nHigh);
	}


	//Assemble the output;
	result.s0 = (((ulong) accumulatedBytes[0]) << 0) 
		| (((ulong) accumulatedBytes[1]) << 8) 
		| (((ulong) accumulatedBytes[2]) << 16)
		| (((ulong) accumulatedBytes[3]) << 24)
		| (((ulong) accumulatedBytes[4]) << 32) 
		| (((ulong) accumulatedBytes[5]) << 40)
		| (((ulong) accumulatedBytes[6]) << 48)
		| (((ulong) accumulatedBytes[7]) << 56);

	result.s1 = (((ulong) accumulatedBytes[8]) << 0) 
		| (((ulong) accumulatedBytes[9]) << 8) 
		| (((ulong) accumulatedBytes[10]) << 16)
		| (((ulong) accumulatedBytes[11]) << 24)
		| (((ulong) accumulatedBytes[12]) << 32) 
		| (((ulong) accumulatedBytes[13]) << 40)
		| (((ulong) accumulatedBytes[14]) << 48)
		| (((ulong) accumulatedBytes[15]) << 56);

	result.s2 = (((ulong) accumulatedBytes[16]) << 0) 
		| (((ulong) accumulatedBytes[17]) << 8) 
		| (((ulong) accumulatedBytes[18]) << 16)
		| (((ulong) accumulatedBytes[19]) << 24)
		| (((ulong) accumulatedBytes[20]) << 32) 
		| (((ulong) accumulatedBytes[21]) << 40)
		| (((ulong) accumulatedBytes[22]) << 48)
		| (((ulong) accumulatedBytes[23]) << 56);

	result.s3 = (((ulong) accumulatedBytes[24]) << 0) 
		| (((ulong) accumulatedBytes[25]) << 8) 
		| (((ulong) accumulatedBytes[26]) << 16)
		| (((ulong) accumulatedBytes[27]) << 24)
		| (((ulong) accumulatedBytes[28]) << 32) 
		| (((ulong) accumulatedBytes[29]) << 40)
		| (((ulong) accumulatedBytes[30]) << 48)
		| (((ulong) accumulatedBytes[31]) << 56);

	return result;
}


unsigned short smallBufferMaskFilter (
	//Bytes of the mutual mask
		unsigned char mutualBitmask0,
		unsigned char mutualBitmask1,
		unsigned char mutualBitmask2,
		unsigned char mutualBitmask3,
		unsigned char mutualBitmask4,
		unsigned char mutualBitmask5,
		unsigned char mutualBitmask6,
		unsigned char mutualBitmask7,


		//Bytes of the accumulated bitmask
		//Might not need all of them
		unsigned char	accumulatedBitmask0,
		unsigned char	accumulatedBitmask1,
		unsigned char	accumulatedBitmask2,
		unsigned char	accumulatedBitmask3,
		unsigned char	accumulatedBitmask4,
		unsigned char	accumulatedBitmask5,
		unsigned char	accumulatedBitmask6,
		unsigned char	accumulatedBitmask7,
		unsigned char	accumulatedBitmask8,
		unsigned char	accumulatedBitmask9,
		unsigned char	accumulatedBitmask10,
		unsigned char	accumulatedBitmask11,
		unsigned char	accumulatedBitmask12,
		unsigned char	accumulatedBitmask13,
		unsigned char	accumulatedBitmask14,
		unsigned char	accumulatedBitmask15,
		unsigned char	accumulatedBitmask16,
		unsigned char	accumulatedBitmask17,
		unsigned char	accumulatedBitmask18,
		unsigned char	accumulatedBitmask19,
		unsigned char	accumulatedBitmask20,
		unsigned char	accumulatedBitmask21,
		unsigned char	accumulatedBitmask22,
		unsigned char	accumulatedBitmask23,
		unsigned char	accumulatedBitmask24,
		unsigned char	accumulatedBitmask25,
		unsigned char	accumulatedBitmask26,
		unsigned char	accumulatedBitmask27,
		unsigned char	accumulatedBitmask28,
		unsigned char	accumulatedBitmask29,
		unsigned char	accumulatedBitmask30,
		unsigned char	accumulatedBitmask31,

		unsigned char startIndex
)
{
	unsigned char denseOutput = 0x0;
	unsigned char nextStartIndex = startIndex;

	unsigned char mutualBitmask[8] = {mutualBitmask0, mutualBitmask1, mutualBitmask2, mutualBitmask3, mutualBitmask4, mutualBitmask5, mutualBitmask5, mutualBitmask7};
	unsigned char accumulatedBitmask[32] = {
		 	accumulatedBitmask0,
		 	accumulatedBitmask1,
		 	accumulatedBitmask2,
		 	accumulatedBitmask3,
		 	accumulatedBitmask4,
		 	accumulatedBitmask5,
		 	accumulatedBitmask6,
		 	accumulatedBitmask7,
		 	accumulatedBitmask8,
		 	accumulatedBitmask9,
		 	accumulatedBitmask10,
		 	accumulatedBitmask11,
		 	accumulatedBitmask12,
		 	accumulatedBitmask13,
		 	accumulatedBitmask14,
		 	accumulatedBitmask15,
		 	accumulatedBitmask16,
		 	accumulatedBitmask17,
		 	accumulatedBitmask18,
		 	accumulatedBitmask19,
		 	accumulatedBitmask20,
		 	accumulatedBitmask21,
		 	accumulatedBitmask22,
		 	accumulatedBitmask23,
		 	accumulatedBitmask24,
		 	accumulatedBitmask25,
		 	accumulatedBitmask26,
		 	accumulatedBitmask27,
		 	accumulatedBitmask28,
		 	accumulatedBitmask29,
		 	accumulatedBitmask30,
		 	accumulatedBitmask31
	};

	for (int i=0; i<TRANSFER_SIZE; i++)
	{
		int position = findFirstOccurance(accumulatedBitmask, nextStartIndex, BITMASK_ACCUM_COUNT_BITWIDTH, (unsigned char)(i+1));
		if (position < COMPRESSION_WINDOW_SIZE)
		{
			unsigned char denseBits = 0x0;
			getBitsInBus(&denseBits, mutualBitmask, position, position);
			setBitsInByte(&denseOutput, denseBits, i, i);
			nextStartIndex = position + 1;
		}
		else
		{
			break;
		}
	}

	return ( ( ((unsigned short) denseOutput) & 0x00FF) | ( ((unsigned short) nextStartIndex << 8) & 0xFF00));
}

ulong4 smallBufferMacBufferUpdate (
		unsigned char inputSelectBitmask,

		//Bytes of the input buffer
		unsigned char inputTransferBlock0,
		unsigned char inputTransferBlock1,
		unsigned char inputTransferBlock2,
		unsigned char inputTransferBlock3,
		unsigned char inputTransferBlock4,
		unsigned char inputTransferBlock5,
		unsigned char inputTransferBlock6,
		unsigned char inputTransferBlock7,

		//Bytes of the buffer
		unsigned char currentBuffer0,
		unsigned char currentBuffer1,
		unsigned char currentBuffer2,
		unsigned char currentBuffer3,
		unsigned char currentBuffer4,
		unsigned char currentBuffer5,
		unsigned char currentBuffer6,
		unsigned char currentBuffer7,

		unsigned char currentBufferSize
	)
{
	ulong4 result;
	result.s0123 = (ulong4) (0x0, 0x0, 0x0, 0x0);
	unsigned char currentBuffer[8] = {currentBuffer0, currentBuffer1, currentBuffer2, currentBuffer3, currentBuffer4, currentBuffer5, currentBuffer6, currentBuffer7};
	unsigned char inputTransferBlock[8] = {
		inputTransferBlock0,
		inputTransferBlock1,
		inputTransferBlock2,
		inputTransferBlock3,
		inputTransferBlock4,
		inputTransferBlock5,
		inputTransferBlock6,
		inputTransferBlock7
	};

	unsigned char denseInput[CLUSTER_SIZE*TRANSFER_SIZE] = {0};
	unsigned char concatenatedBuffer[CLUSTER_SIZE*TRANSFER_SIZE*2] = {0};

	//Get the dense output, and count the number of dense clusters 
	unsigned char numClusterValid = 0;
	for (int i=0; i<TRANSFER_SIZE; i++)
	{
		int position = getNthOnePosition((unsigned short)inputSelectBitmask, i+1, 0);
		if (position < TRANSFER_SIZE)
		{
			int nLow = position*CLUSTER_SIZE*8; //Low bit position
			int nHigh = nLow + CLUSTER_SIZE*8-1;
			unsigned char selectedBytes[CLUSTER_SIZE] = {0};
			getBitsInBus(selectedBytes, inputTransferBlock, nLow, nHigh);

			int denseLow = i*CLUSTER_SIZE*8;
			int denseHigh = denseLow + CLUSTER_SIZE*8;
			setBitsInBus (denseInput, selectedBytes, denseLow, denseHigh);
			numClusterValid++;
		}
	}

	//Concatenate the buffer
	for (int i=0; i<(2*TRANSFER_SIZE); i++) {
		unsigned char operandBytes[CLUSTER_SIZE] = {0};
		if (i < currentBufferSize)
		{
			getBitsInBus(operandBytes, currentBuffer, i*CLUSTER_SIZE*8, (i+1)*CLUSTER_SIZE*8-1);
		}
		else if (i< (currentBufferSize + numClusterValid)) {
			getBitsInBus(operandBytes, denseInput, (i-currentBufferSize)*CLUSTER_SIZE*8, (i-currentBufferSize+1)*CLUSTER_SIZE*8-1);
		}
		
		setBitsInBus (concatenatedBuffer, operandBytes, i*CLUSTER_SIZE*8, (i+1)*CLUSTER_SIZE*8);
	}

	//Select from the buffer
	unsigned char macValid = ((currentBufferSize + numClusterValid) >= TRANSFER_SIZE) ? TRUE : FALSE;
	unsigned char newBufferSize = (currentBufferSize + numClusterValid) % TRANSFER_SIZE;

	unsigned char newBuffer[8] = {0};
	unsigned char macOperands[8] = {0};

	getBitsInBus(macOperands, concatenatedBuffer, 0, TRANSFER_SIZE*CLUSTER_SIZE*8-1);
	int nConcatenatedBufferLow = (macValid == TRUE) ? (TRANSFER_SIZE*CLUSTER_SIZE*8) : 0;
	int nConcatenatedBufferHigh = (macValid == TRUE) ? (2*TRANSFER_SIZE*CLUSTER_SIZE*8 - 1) : (TRANSFER_SIZE*CLUSTER_SIZE*8 - 1);


	getBitsInBus(newBuffer, concatenatedBuffer, nConcatenatedBufferLow, nConcatenatedBufferHigh);

	//[63:0] macOutput
	//[127:64] nextBuffer
	//[128 + BUFFER_COUNT_WIDTH -1	-:	BUFFER_COUNT_WIDTH]  Next buffer size
	//[136] macOutputIsValid
	result.s0 = ( (((ulong) macOperands[0]) & 0x0FF) << 0)
			| ( ((ulong) macOperands[1]) << 8)
			| (((ulong) macOperands[2]) << 16)
			| (((ulong) macOperands[3]) << 24)
			| (((ulong) macOperands[4]) << 32)
			| (((ulong) macOperands[5]) << 40)
			| (((ulong) macOperands[6]) << 48)
			| (((ulong) macOperands[7]) << 56);

	result.s1 = (((ulong) newBuffer[0]) << 0)
			| (((ulong) newBuffer[1]) << 8)
			| (((ulong) newBuffer[2]) << 16)
			| (((ulong) newBuffer[3]) << 24)
			| (((ulong) newBuffer[4]) << 32)
			| (((ulong) newBuffer[5]) << 40)
			| (((ulong) newBuffer[6]) << 48)
			| (((ulong) newBuffer[7]) << 56);

	result.s2 = (((ulong) newBufferSize) & 0x0FF ) | ((((ulong) macValid) & 0x0FF) << 8);

	return result;
}