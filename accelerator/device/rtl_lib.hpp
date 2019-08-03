#ifndef RTL_LIB_HPP
#define RTL_LIB_HPP

#if defined(ARRIA10)
	int a10_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
	int a10_mac_8bitx2 (char a0, char b0, char a1, char b1);
#elif defined (C5SOC)
	int c5_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
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

unsigned long operandMatcher8 (
		unsigned char bitmaskW,
		unsigned char bitmaskA
	);

unsigned char leadingZeroCounter (unsigned char bitmask);
#endif  