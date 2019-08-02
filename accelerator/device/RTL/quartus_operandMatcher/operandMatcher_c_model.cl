#include "rtl_lib.hpp"

unsigned long operandMatcher8 (
		unsigned char bitmaskW,
		unsigned char bitmaskA
	) {

	//Compute the mutual bitmask
	unsigned char mutualBitmask = bitmaskA & bitmaskW;

	unsigned char negateMutualBitmask = ~mutualBitmask;

	unsigned int shiftAccum = accumulate8(negateMutualBitmask);

	//Compute the dense indices of the activations
	unsigned int activationAccum = accumulate8(bitmaskA);

	unsigned int activationAccumWithGap = maskAccum (mutualBitmask, activationAccum);

	unsigned int activationAccumDense = 
		bubbleCollapse8 (shiftAccum, activationAccumwithGap);

	//Compute the dense indices of the weights
	unsigned int weightAccum = accumulate8(bitmaskW);

	unsigned int weightAccumWithGap = maskAccum (mutualBitmask, weightAccum);

	unsigned int weightAccumDense = 
		bubbleCollapse8 (shiftAccum, weightAccumwithGap);

	//Count the pairs of operands
	unsigned char pairCount = popCounter(mutualBitmask);

	//Assume the output
	unsigned long result = 
		(((unsigned long) activationAccumDense) & 0xFFFFFF)
		| ((((unsigned long) weightAccumDense) & 0xFFFFFF) << 24)
		| ((((unsigned long) pairCount) & 0xFF) << 48);

	return result;
}

unsigned int accumulate8 (unsigned char bitmask) {
	unsigned char accumulators [8];

	accumulators [0] = 0;

	for (unsigned int i=1; i<8; i++) {
		accumulators [i] =
			( bitmask 
				& ( ((unsigned char) 0x1) <<  (i-1) ) ) 
			>> (i-1);
	} // for

	//Assemble the output
	unsigned int output = 0;
	for (unsigned j=0; j<8; j++) {
		output |= (((unsigned int) accumulators[j]) & 0x7 )<< (j*3);
	}

	output &= 0xFFFFFF;

	return output;
}

unsigned int bubbleCollapse8 (
		unsigned int shiftAccum,
		unsigned int accumWithGap
	) {
	
	unsigned char accumulators [8];

	for (unsigned int i=0; i<8; i++) {
		for (unsigned int j=i, q=0; j<8; j++, q++) {
			unsigned char shiftAmount
				= (shiftAccum >> (j*3)) & 0x7;
			if (shiftAmount == ((unsigned char) q)) {
				accumulators [i] = (accumWithGap >> (j*3)) & 0x7;
			}
		}
	} // for

	//Assemble the output
	unsigned int output = 0;
	for (unsigned j=0; j<8; j++) {
		output |= (((unsigned int) accumulators[j]) & 0x7 )<< (j*3);
	}

	output &= 0xFFFFFF;

	return output;
}

unsigned int maskAccum (
		unsigned char bitmask,
		unsigned int accum
	) {
	unsigned int expandedMask = 0;
	for (unsigned int i=0; i<8; i++) {
		unsigned char bit = (bitmask >> i) & 0x1;
		if (bit == 0x1) {
			expandedMask |= ((unsigned int) 0x7) << (i*3);
		}
	}

	unsigned int result = (accum & expandedMask) & 0xFFFFFF;

	return result;
}

unsigned char popCount (
		unsigned char bitmask
	) {
	unsigned char count = 0;

	for (unsigned int i=0; i<8; i++) {
		unsigned char bit = (bitmask >> i) & 0x1;
		if (bit == 0x1) {
			count += 1;
		}
	}

	return count;
}