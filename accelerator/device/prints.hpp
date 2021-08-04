#ifndef PRINTS_HPP
#define PRINTS_HPP
//#define HW_DEBUG
//#define EMUPRINT

//MACRO Stringification
#define _IN_QUOTES(x) #x
#define IN_QUOTES(x) _IN_QUOTES(x)
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
#elif defined (EMULATOR) && defined(EMUPRINT)
	#define DEBUG_PRINT(format) printf format
#else
	#define DEBUG_PRINT(format)
#endif

#endif
