#ifndef RTL_LIB_HPP
#define RTL_LIB_HPP

#if defined(ARRIA10)
	int a10_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int a10_mac_8bitx4_input_registered (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int a10_mac_8bitx2 (char a0, char b0, char a1, char b1);
#elif defined (C5SOC)
	int c5_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int c5_mac_8bitx4_input_registered (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int c5_mac_8bitx2 (char a0, char b0, char a1, char b1);
#else 
#error Unsupported FPGA!
#endif

unsigned int accumulate8 (
		unsigned char bitmask
	);

unsigned int bubbleCollapse8 (
		unsigned int shiftAccum,
		unsigned int accumWithGap
	);

unsigned int maskAccum (
		unsigned char bitmask,
		unsigned int accum
	);

unsigned char popCounter (
		unsigned char bitmask
	);

//  [23:0] Packed indices of A; 
//  [47:24] Packed indices of W; 
//  [51:48] Number of pairs; 
//  [63:52]: Padding
unsigned long operandMatcher8 (
		unsigned char bitmaskW,
		unsigned char bitmaskA
	);

unsigned char leadingZeroCounter (unsigned char bitmask);

//clMaskFilter16c2_1bit
//Filter up to 2 "1" bit from the mask
unsigned short smallBufferMaskFilter (unsigned short bitmask, unsigned short sparseInput, unsigned char startIndex);

//clSparseMacBufferUpdate
ulong2 smallBufferMacBufferUpdate (
		unsigned char inputSelectBitmask,

		unsigned char inputTransferBlockA0,
		unsigned char inputTransferBlockA1,
		unsigned char inputTransferBlockB0,
		unsigned char inputTransferBlockB1,

		unsigned char currentBufferA0,
		unsigned char currentBufferA1,
		unsigned char currentBufferB0,
		unsigned char currentbufferB1,

		unsigned char currentBufferSize
	);

unsigned char getNthOnePosition (unsigned short bitmask, unsigned char n, unsigned char startIndex);

#endif  