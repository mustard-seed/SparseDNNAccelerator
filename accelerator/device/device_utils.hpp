#ifndef DEVICE_UTILS_HPP
#define DEVICE_UTILS_HPP
//#define EMUPRINT
/*
printf enabled during SW emulation
*/
#if defined(EMULATOR) && defined(EMUPRINT)
	#define EMULATOR_PRINT(format) printf format
#else
	#define EMULATOR_PRINT(format)
#endif

/*
printf enabled on HW if -HW_DEBUG flag is set
*/
#if defined(HW_DEBUG) && defined(EMUPRINT)
	#define DEBUG_PRINT(format) printf format
#else
	#define DEBUG_PRINT(format)
#endif

#endif
