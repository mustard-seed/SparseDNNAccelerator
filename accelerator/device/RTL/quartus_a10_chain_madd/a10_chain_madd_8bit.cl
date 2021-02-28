#include "rtl_lib.hpp"

signed int a10_chain_madd_8bitx8(
		signed char a0,
		signed char b0,
		signed char a1,
		signed char b1,
		signed char a2,
		signed char b2,
		signed char a3,
		signed char b3,
		signed char a4,
		signed char b4,
		signed char a5,
		signed char b5,
		signed char a6,
		signed char b6,
		signed char a7,
		signed char b7
	)
{
	return ((signed int) a0* (signed int) b0 
			+ (signed int) a1* (signed int) b1
			+ (signed int) a2* (signed int) b2
			+ (signed int) a3* (signed int) b3
			+ (signed int) a4* (signed int) b4
			+ (signed int) a5* (signed int) b5
			+ (signed int) a6* (signed int) b6
			+ (signed int) a7* (signed int) b7
			);
}