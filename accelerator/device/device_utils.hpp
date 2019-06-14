#ifndef DEVICE_UTILS_HPP
#define DEVICE_UTILS_HPP

/*
printf enabled during SW emulation
*/
#ifdef EMULATOR
	#define EMULATOR_PRINT(format) printf format
#else
	#define EMULATOR_PRINT(format)
#endif

/*
printf enabled on HW if -HW_DEBUG flag is set
*/
#ifdef HW_DEBUG
	#define DEBUG_PRINT(format) printf format
#else
	#define DEBUG_PRINT(format)
#endif

#endif
