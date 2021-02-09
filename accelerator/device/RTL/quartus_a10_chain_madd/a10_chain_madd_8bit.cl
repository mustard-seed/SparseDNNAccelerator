#include "rtl_lib.hpp"

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
	)
{
	return (a0*b0 + (int) a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7);
}