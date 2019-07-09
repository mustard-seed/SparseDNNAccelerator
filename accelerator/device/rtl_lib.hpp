#ifndef RTL_LIB_HPP
#define RTL_LIB_HPP

#if defined(ARRIA10)
	int a10_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
#elif defined (C5SOC)
	int c5_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3);
#else 
#error Unsupported FPGA!
#endif

#endif  