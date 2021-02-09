#ifndef RTL_LIB_HPP
#define RTL_LIB_HPP

#if defined(ARRIA10)
	int a10_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int a10_mac_8bitx4_input_registered (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int a10_mac_8bitx2 (char a0, char b0, char a1, char b1);
	int a10_chain_madd_8bitx8(
		char a0,
		char b0,
		char a1,
		char b1,
		char a2,
		char b2,
		char a3,
		char b3,
		char a4,
		char b4,
		char a5,
		char b5,
		char a6,
		char b6,
		char a7,
		char b7
	);
#elif defined (C5SOC)
	int c5_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int c5_mac_8bitx4_input_registered (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int c5_mac_8bitx2 (char a0, char b0, char a1, char b1);
#else 
#error Unsupported FPGA!
#endif

// unsigned int accumulate8 (
// 		unsigned char bitmask
// 	);

// unsigned int bubbleCollapse8 (
// 		unsigned int shiftAccum,
// 		unsigned int accumWithGap
// 	);

// unsigned int maskAccum (
// 		unsigned char bitmask,
// 		unsigned int accum
// 	);

// unsigned char popCounter (
// 		unsigned char bitmask
// 	);

// //  [23:0] Packed indices of A; 
// //  [47:24] Packed indices of W; 
// //  [51:48] Number of pairs; 
// //  [63:52]: Padding
// unsigned long operandMatcher8 (
// 		unsigned char bitmaskW,
// 		unsigned char bitmaskA
// 	);

// unsigned char leadingZeroCounter (unsigned char bitmask);

// /**
//  * Small buffer library
//  */
// /**
//  * @brief      Counts the number of 1s inside the bitmask
//  *
//  * @param[in]  bitmask0  Bitmask byte 0
//  * @param[in]  bitmask1  Bitmask byte 1
//  * @param[in]  bitmask2  Bitmask byte 2
//  * @param[in]  bitmask3  Bitmask byte 3
//  * @param[in]  bitmask4  Bitmask byte 4
//  * @param[in]  bitmask5  Bitmask byte 5
//  * @param[in]  bitmask6  Bitmask byte 6
//  * @param[in]  bitmask7  Bitmask byte 7
//  *
//  * @return     The number of 1s inside the bitmask
//  */
// unsigned char smallBufferPopCounter (
// 		unsigned char bitmask0,
// 		unsigned char bitmask1,
// 		unsigned char bitmask2,
// 		unsigned char bitmask3,
// 		unsigned char bitmask4,
// 		unsigned char bitmask5,
// 		unsigned char bitmask6,
// 		unsigned char bitmask7
// 	);

// /**
//  * @brief      For every bit N in the bitmask, Finds the number of 1s that appeared in bits [N-1:0];
//  *
//  * @param[in]  bitmask0  Bitmask byte 0
//  * @param[in]  bitmask1  Bitmask byte 1
//  * @param[in]  bitmask2  Bitmask byte 2
//  * @param[in]  bitmask3  Bitmask byte 3
//  * @param[in]  bitmask4  Bitmask byte 4
//  * @param[in]  bitmask5  Bitmask byte 5
//  * @param[in]  bitmask6  Bitmask byte 6
//  * @param[in]  bitmask7  Bitmask byte 7
//  *
//  * @return     The accumulated bitmask
//  */
// ulong4 smallBufferMaskAccumulator (
// 		unsigned char bitmask0,
// 		unsigned char bitmask1,
// 		unsigned char bitmask2,
// 		unsigned char bitmask3,
// 		unsigned char bitmask4,
// 		unsigned char bitmask5,
// 		unsigned char bitmask6,
// 		unsigned char bitmask7
// 	);

// //Filters the relevant mutual bitmask bits
// //[7:0] Packed mutual bitmask. Only [TRANSFER_SIZE-1 : 0] are meaningful
// //[15:8] Next start index. Only [8 + INDEX_BITWIDTH - 1 -: INDEX_BITWIDTH] are meanintful
// unsigned short smallBufferMaskFilter (
// 		//Bytes of the mutual mask
// 		unsigned char mutualBitmask0,
// 		unsigned char mutualBitmask1,
// 		unsigned char mutualBitmask2,
// 		unsigned char mutualBitmask3,
// 		unsigned char mutualBitmask4,
// 		unsigned char mutualBitmask5,
// 		unsigned char mutualBitmask6,
// 		unsigned char mutualBitmask7,


// 		//Bytes of the accumulated bitmask
// 		//Might not need all of them
// 		unsigned char	accumulatedBitmask0,
// 		unsigned char	accumulatedBitmask1,
// 		unsigned char	accumulatedBitmask2,
// 		unsigned char	accumulatedBitmask3,
// 		unsigned char	accumulatedBitmask4,
// 		unsigned char	accumulatedBitmask5,
// 		unsigned char	accumulatedBitmask6,
// 		unsigned char	accumulatedBitmask7,
// 		unsigned char	accumulatedBitmask8,
// 		unsigned char	accumulatedBitmask9,
// 		unsigned char	accumulatedBitmask10,
// 		unsigned char	accumulatedBitmask11,
// 		unsigned char	accumulatedBitmask12,
// 		unsigned char	accumulatedBitmask13,
// 		unsigned char	accumulatedBitmask14,
// 		unsigned char	accumulatedBitmask15,
// 		unsigned char	accumulatedBitmask16,
// 		unsigned char	accumulatedBitmask17,
// 		unsigned char	accumulatedBitmask18,
// 		unsigned char	accumulatedBitmask19,
// 		unsigned char	accumulatedBitmask20,
// 		unsigned char	accumulatedBitmask21,
// 		unsigned char	accumulatedBitmask22,
// 		unsigned char	accumulatedBitmask23,
// 		unsigned char	accumulatedBitmask24,
// 		unsigned char	accumulatedBitmask25,
// 		unsigned char	accumulatedBitmask26,
// 		unsigned char	accumulatedBitmask27,
// 		unsigned char	accumulatedBitmask28,
// 		unsigned char	accumulatedBitmask29,
// 		unsigned char	accumulatedBitmask30,
// 		unsigned char	accumulatedBitmask31,

// 		unsigned char startIndex
// 	);

// //clSparseMacBufferUpdate
// //[63:0] macOutput
// //[127:64] nextBuffer
// //[128 + BUFFER_COUNT_WIDTH -2	-:	(BUFFER_COUNT_WIDTH-1)]  Next buffer size
// //[136] macOutputIsValid
// ulong4 smallBufferMacBufferUpdate (
// 		unsigned char inputSelectBitmask,

// 		//Bytes of the input buffer
// 		unsigned char inputTransferBlock0,
// 		unsigned char inputTransferBlock1,
// 		unsigned char inputTransferBlock2,
// 		unsigned char inputTransferBlock3,
// 		unsigned char inputTransferBlock4,
// 		unsigned char inputTransferBlock5,
// 		unsigned char inputTransferBlock6,
// 		unsigned char inputTransferBlock7,

// 		//Bytes of the buffer
// 		unsigned char currentBuffer0,
// 		unsigned char currentBuffer1,
// 		unsigned char currentBuffer2,
// 		unsigned char currentBuffer3,
// 		unsigned char currentBuffer4,
// 		unsigned char currentBuffer5,
// 		unsigned char currentBuffer6,
// 		unsigned char currentBuffer7,

// 		unsigned char currentBufferSize
// 	);

// //N starts counting from 1.
// unsigned char getNthOnePosition (unsigned short bitmask, unsigned char n, unsigned char startIndex);

// int findFirstOccurance (unsigned char accumulatedIndices [], int startIndex, int bitsPerIndex, unsigned char target);

// //nLow and nHigh are positions in the destination
// void setBitsInByte (unsigned char * pDestination, unsigned char source, int nLow, int nHigh);

// //nlOw and nHigh are positions in the source
// unsigned char getBitsInByte (unsigned char source, int nLow, int nHigh);

// //nLow and nHigh are positions in the destination
// void setBitsInBus (unsigned char pDestination[], unsigned char pSource[], int nLow, int nHigh);

// //nlOw and nHigh are positions in the source
// void getBitsInBus (unsigned char pDestination[], unsigned char pSource[], int nLow, int nHigh);

#endif  