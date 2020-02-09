#include "rtl_lib.hpp"

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

unsigned short smallBufferMaskFilter (
	unsigned short bitmask, 
	unsigned short sparseInput, 
	unsigned char startIndex)
{
	unsigned char denseOutput = 0x0;
	unsigned char nextStartIndex = 0;

	for (unsigned char n=0; n<2; n++)
	{
		unsigned char pos = getNthOnePosition(bitmask, n+1, startIndex);
		if (pos < 16) {
			unsigned char bit = (sparseInput >> pos) & 0x01;
			denseOutput = denseOutput | (bit << n);
		}
		nextStartIndex = pos+1;
	}

	return (((denseOutput << 8) & 0xFF00) | (nextStartIndex & 0x00FF));
}

ulong2 smallBufferMacBufferUpdate (
		unsigned char inputSelectBitmask,

		unsigned char inputTransferBlockA0,
		unsigned char inputTransferBlockA1,
		unsigned char inputTransferBlockB0,
		unsigned char inputTransferBlockB1,

		unsigned char currentBufferA0,
		unsigned char currentBufferA1,
		unsigned char currentBufferB0,
		unsigned char currentBufferB1,

		unsigned char currentBufferSize
	)
{
	unsigned char currentBuffer[4];

	currentBuffer[3] = currentBufferB1;
	currentBuffer[2] = currentBufferB0;
	currentBuffer[1] = currentBufferA1;
	currentBuffer[0] = currentBufferA0;

	unsigned char inputTransferBlock[4];
	unsigned char filteredTransferBlock[4];
	inputTransferBlock[3] = inputTransferBlockB1;
	inputTransferBlock[2] = inputTransferBlockB0;
	inputTransferBlock[1] = inputTransferBlockA1;
	inputTransferBlock[0] = inputTransferBlockA0;

	unsigned char concatenatedBuffer[8];

	unsigned char numFiltered = 0;

	for (unsigned char n=0; n<2; n++)
	{
		unsigned char pos = getNthOnePosition((inputSelectBitmask & 0x03), n+1, 0);
		unsigned char block0 = 0;
		usnigned char block1 = 0;
		if (pos < 2) {
			block0 = inputTransferBlock[pos*2];
			block1 = inputTransferBlock[pos*2+1];
			numFiltered++;
		}
		filteredTransferBlock[n*2] = block0;
		filteredTransferBlock[n*2+1] = block1;
	}

	for (unsigned char i=0; i<4; i++)
	{
		unsigned char block0 = 0;
		unsigned char block1 = 0;
		if (i < currentBufferSize)
		{
			block0 = currentBuffer[i*2];
			block1 = currentBuffer[i*2+1];
		}
		else if (i < (currentBufferSize + numFiltered))
		{
			block0 = filteredTransferBlock[(i-currentBufferSize)*2];
			block0 = filteredTransferBlock[(i-currentBufferSize)*2+1];
		}

		concatenatedBuffer[i*2] = block0;
		concatenatedBuffer[i*2+1] = block1;
	}

	unsigned char newSize = (currentBufferSize + numFiltered) % 2;
	unsigned char macValid = (currentBufferSize + numFiltered) / 2;

	unsigned char macOutput[4];
	unsigned char nextBuffer[4];

	for (unsigned char i=0; i<4; i++)
	{
		macOutput[i] = concatenatedBuffer[i];
		if (macValid == 1)
		{
			nextBuffer[i] = concatenatedBuffer[i+4];
		}
		else
		{
			nextBuffer[i] = concatenatedBuffer[i];
		}
	}

	ulong2 result;
	result.x = 
		((nextBuffer[3] & 0x0FF) << 56)
		| ((nextBuffer[2] & 0x0FF) << 48)
		| ((nextBuffer[1] & 0x0FF) << 40)
		| ((nextBuffer[0] & 0x0FF) << 32)
		| ((macOutput[3] & 0x0FF) << 24)
		| ((macOutput[2] & 0x0FF) << 16)
		| ((macOutput[1] & 0x0FF) << 8)
		| ((macOutput[0] & 0x0FF) << 0);

	result.y = ((macValid & 0x01) << 8) | ((newSize & 0x11));

	return result;
}