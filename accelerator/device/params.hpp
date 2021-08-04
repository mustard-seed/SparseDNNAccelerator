#ifndef PARAMS_DEFINED
#define PARAMS_DEFINED
#include "prints.hpp"

//#define HOST_DEBUG
#define SPW_SYSTEM
//#define DENSE_SYSTEM
#define OA_PING_PONG
//#define WMOVER_STREAM_CACHE
//#define OAMOVER_TB_STREAM_CACHE
//#define WMOVER_WEIGHT_COALESCE_CACHE
//#define HW_SYNC

#define NOOP
#if defined(SPW_TEST)
#undef DENSE_SYSTEM
#define SPW_SYSTEM
#endif
/**
 * Global memory settings
 */
#define MAX_DRAM_BYTE_INPUT_ACTIVATION (1 << 29)
#define MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT (1 << 22)
#define MAX_DRAM_BYTE_INPUT_WEIGHT (1 << 28)
#define MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT 0x40000
#define MAX_DRAM_BYTE_INPUT_BIAS 0x20000
//#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION 0x400000
//#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT 0x40000
#define MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION (1 << 25)
#define MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION (1 << 23)

// #define PACKET_SIZE 1

//Assume on Arria 10 Dev Kit, the memory bandwidth is on
#define DDR_BANDWIDTH_GBS_INT 17
#define FMAX_MHZ 221
#define DDR_BYTES_PER_CYCLE ((DDR_BANDWIDTH_GBS_INT * 1000 - 1) / FMAX_MHZ + 1)

#if defined(FULL_SYSTEM)
	// #define PE_COLS 2
	// #define PE_ROWS_PER_GROUP 4
	// #define PE_ROW_GROUPS 2
	// #define MISC_COLS 1
	// #define PE_COLS 7
	// #define PE_ROWS_PER_GROUP 8
	// #define PE_ROW_GROUPS 3
	// #define MISC_COLS 1
	#define PE_COLS 2
	#define PE_ROWS_PER_GROUP 2
	#define PE_ROW_GROUPS 1
	#define MISC_COLS 1
	// #define PE_COLS 7
	// #define PE_ROWS_PER_GROUP 8
	// #define PE_ROW_GROUPS 6
	// #define MISC_COLS 1
#else
	#define PE_COLS 1
	#define PE_ROWS_PER_GROUP 16
	#define PE_ROW_GROUPS 1
	#define MISC_COLS 1
#endif
#define MISC_UNROLL 16

#define PE_ROWS (PE_ROWS_PER_GROUP * PE_ROW_GROUPS)
//Derived parameter
#if (PE_ROWS_PER_GROUP == 1)
#define DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT 0
#define DIVIDE_BY_PE_ROWS_PER_GROUP_REMAINDER_MASK 0x0
#elif (PE_ROWS_PER_GROUP == 2)
#define DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT 1
#define DIVIDE_BY_PE_ROWS_PER_GROUP_REMAINDER_MASK 0x1
#elif (PE_ROWS_PER_GROUP == 4)
#define DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT 2
#define DIVIDE_BY_PE_ROWS_PER_GROUP_REMAINDER_MASK 0x3
#elif (PE_ROWS_PER_GROUP == 8)
#define DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT 3
#define DIVIDE_BY_PE_ROWS_PER_GROUP_REMAINDER_MASK 0x7
#elif (PE_ROWS_PER_GROUP == 16)
#define DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT 4
#define DIVIDE_BY_PE_ROWS_PER_GROUP_REMAINDER_MASK 0xF
#else
#error DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT should be chosen from {1, 2, 4, 8, 16}
#pragma message "DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT is" IN_QUOTES(DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT)
#endif

#if (MISC_COLS > PE_COLS)
#error Configuration MISC_COLS should be less or equal to PE_COLS
#endif

#define CHANNEL_DEPTH 1
#define OA_DRAIN_CHANNEL_DEPTH 1

#define SIMD_SIZE 2
#define SYNC_SIZE 8
#define MAX_SIMD_BLOCK_INDEX 0x0FF

//Activation memory region offsets
//in terms of DRAM blocks
//TB count memory region offsets
//In terms of TB counts (shorts)
#define NUM_ACTIVATION_REGIONS 3
#define MEM_ACTIVATION_REGION_SIZE_PER_SLICE (1 << 20)
#define MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE (1 << 18)


#define CLUSTER_TO_WINDOW_SHIFT 0X3
#define CLUSTER_TO_WINDOW_REMAINDER_MASK 0x07
#define CLUSTER_SIZE 2 //cluster size in terms of values

#if (CLUSTER_SIZE == 1)
#define VALUE_TO_CLUSTER_SHIFT 0x00
#elif (CLUSTER_SIZE == 2)
#define VALUE_TO_CLUSTER_SHIFT 0x01
#elif (CLUSTER_SIZE == 4)
#define VALUE_TO_CLUSTER_SHIFT 0x02
#elif (CLUSTER_SIZE == 8)
#define VALUE_TO_CLUSTER_SHIFT 0x03
#elif (CLUSTER_SIZE == 16)
#define VALUE_TO_CLUSTER_SHIFT 0x04
#else
#error CLUSTER_SIZE should be chosen from {1, 2, 4, 8, 16}
#endif 
#define VALUE_DIVIDED_BY_CLUSTER_SIZE_REMAINDER_MASK ((1 << VALUE_TO_CLUSTER_SHIFT) - 1)


#define VALUE_DIVIDED_BY_SIMD_SIZE_REMAINDER_MASK ((1 << (VALUE_TO_CLUSTER_SHIFT + CLUSTER_TO_TRANSFER_SIZE_SHIFT)) - 1)
#define CLUSTER_TO_TRANSFER_BLOCK_SHIFT CLUSTER_TO_TRANSFER_SIZE_SHIFT //amount of right shift required to convert a cluster count into transfer block count

/**
 * Parameters relevant for balanced-block sparsity
 */
//Prune range of balanced-sparsity
#define PRUNE_RANGE_IN_CLUSTER 4
//Number of prune range processed in parallel.
//Equal to SIMD size (in terms of cluster) in a SpW PE
#define PE_SIMD_SIZE 4

#if (PE_SIMD_SIZE == 1)
#define PE_SIMD_SIZE_CLUSTER_OFFSET 0x0
#elif (PE_SIMD_SIZE == 2)
#define PE_SIMD_SIZE_CLUSTER_OFFSET 0x01
#elif (PE_SIMD_SIZE == 4)
#define PE_SIMD_SIZE_CLUSTER_OFFSET 0x02
#elif (PE_SIMD_SIZE == 8)
#define PE_SIMD_SIZE_CLUSTER_OFFSET 0x03
#elif (PE_SIMD_SIZE == 16)
#define PE_SIMD_SIZE_CLUSTER_OFFSET 0x04
#else
#error PE_SIMD_SIZE should be chosen from {1, 2, 4, 8, 16}
#endif
#define PE_SIMD_SIZE_CLUSTER_MASK ((1 << (PE_SIMD_SIZE_CLUSTER_OFFSET)) - 1)

#if (PRUNE_RANGE_IN_CLUSTER == 1)
#define PRUNE_RNAGE_IN_CLUSTER_OFFSET 0x0
#elif (PRUNE_RANGE_IN_CLUSTER == 2)
#define PRUNE_RNAGE_IN_CLUSTER_OFFSET 0x01
#elif (PRUNE_RANGE_IN_CLUSTER == 4)
#define PRUNE_RNAGE_IN_CLUSTER_OFFSET 0x02
#elif (PRUNE_RANGE_IN_CLUSTER == 8)
#define PRUNE_RNAGE_IN_CLUSTER_OFFSET 0x03
#elif (PRUNE_RANGE_IN_CLUSTER == 16)
#define PRUNE_RNAGE_IN_CLUSTER_OFFSET 0x04
#else
#error PRUNE_RANGE_IN_CLUSTER should be chosen from {1, 2, 4, 8, 16}
#endif
#define PRUNE_RANGE_IN_CLUSTER_MASK ((1 << (PRUNE_RNAGE_IN_CLUSTER_OFFSET)) - 1)

//Size of the char array in the host dram weight blocks
//used to contain the indices of the NZ clusters
//Each char is split into two 4-bit halfs.
//Each half corresponds to an index
#if defined(PE_SIMD_SIZE) && defined(SPW_SYSTEM)
#if (PE_SIMD_SIZE <= 2)
#define INDEX_CHAR_ARRAY_SIZE 1
#define INDEX_CHAR_ARRAY_SIZE_OFFSET 0
#elif (PE_SIMD_SIZE <= 4)
#define INDEX_CHAR_ARRAY_SIZE 2
#define INDEX_CHAR_ARRAY_SIZE_OFFSET 1
#elif (PE_SIMD_SIZE <= 8)
#define INDEX_CHAR_ARRAY_SIZE 4
#define INDEX_CHAR_ARRAY_SIZE_OFFSET 2
#elif (PE_SIMD_SIZE <= 16)
#define INDEX_CHAR_ARRAY_SIZE 8
#define INDEX_CHAR_ARRAY_SIZE_OFFSET 3
#else
#error PE_SIMD_SIZE should be chosen from {1, 2, 4, 8, 16}
#endif
#endif

#if defined(SPW_SYSTEM)
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD (CLUSTER_SIZE * PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER)
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET (VALUE_TO_CLUSTER_SHIFT + PE_SIMD_SIZE_CLUSTER_OFFSET + PRUNE_RNAGE_IN_CLUSTER_OFFSET)
#else
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD (CLUSTER_SIZE * PE_SIMD_SIZE)
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET (VALUE_TO_CLUSTER_SHIFT + PE_SIMD_SIZE_CLUSTER_OFFSET)
#endif
#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD_MASK ((1 << PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET) - 1)

#define PE_WEIGHT_BLOCK_SIZE_IN_WORD (CLUSTER_SIZE * PE_SIMD_SIZE)
#define PE_WEIGHT_BLOCK_SIZE_IN_WORD_OFFSET (VALUE_TO_CLUSTER_SHIFT + PE_SIMD_SIZE_CLUSTER_OFFSET)


//=========================================

#define SURVIVING_COUNT_CLUSTER_INDEX 0X1
#define SURVIVING_COUNT_TRANSFER_BLOCK_INDEX 0x1


#define ACTIVATION_DRAM_SIZE_GEQ_PE_SIZE 0
#define ACTIVATION_WIDE_SIZE 4
//==========Derived MARCOS for activation============
#if (ACTIVATION_WIDE_SIZE == 1)
	#define ACTIVATION_WIDE_SIZE_OFFSET 0x0 //Number of arithmetic shift that correspond to the activation conversion factor
	#define ACTIVATION_WIDE_SIZE_REMAINDER_MASK 0x0 //Mask that correpsond to division by the conversation factor
#elif (ACTIVATION_WIDE_SIZE == 2)
	#define ACTIVATION_WIDE_SIZE_OFFSET 0x1 
	#define ACTIVATION_WIDE_SIZE_REMAINDER_MASK 0x1
#elif (ACTIVATION_WIDE_SIZE == 4)
	#define ACTIVATION_WIDE_SIZE_OFFSET 0x2 
	#define ACTIVATION_WIDE_SIZE_REMAINDER_MASK 0x3
#elif (ACTIVATION_WIDE_SIZE == 8)
	#define ACTIVATION_WIDE_SIZE_OFFSET 0x3 
	#define ACTIVATION_WIDE_SIZE_REMAINDER_MASK 0x7
#elif (ACTIVATION_WIDE_SIZE == 16)
	#define ACTIVATION_WIDE_SIZE_OFFSET 0x4 
	#define ACTIVATION_WIDE_SIZE_REMAINDER_MASK 0xF
#else
	#error ACTIVATION_WIDE_SIZE should be chosen from {1, 2, 4, 8, 16}
#endif
#if (ACTIVATION_DRAM_SIZE_GEQ_PE_SIZE == 1) //Activation dram size in word is greater than or equal to activation PE block size
	#define ACTIVATION_DRAM_SIZE_BYTE (PE_ACTIVATION_BLOCK_SIZE_IN_WORD * ACTIVATION_WIDE_SIZE)
	#define ACTIVATION_DRAM_SIZE_BYTE_OFFSET (ACTIVATION_WIDE_SIZE_OFFSET + PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET)
	#define ACTIVATION_WIDE_SIZE_BYTE_MASK ((1 << ACTIVATION_DRAM_SIZE_BYTE_OFFSET) - 1)
#elif (ACTIVATION_DRAM_SIZE_GEQ_PE_SIZE == 0) 
	#if (PE_ACTIVATION_BLOCK_SIZE_IN_WORD < ACTIVATION_WIDE_SIZE)
		#error ACTIVATION_DRAM_SIZE_GEQ_PE_SIZE is 0, but PE_ACTIVATION_BLOCK_SIZE_IN_WORD < PE_ACTIVATION_BLOCK_SIZE_IN_WORD
	#else
		#define ACTIVATION_DRAM_SIZE_BYTE (PE_ACTIVATION_BLOCK_SIZE_IN_WORD / ACTIVATION_WIDE_SIZE)
		#define ACTIVATION_DRAM_SIZE_BYTE_OFFSET (PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET - ACTIVATION_WIDE_SIZE_OFFSET)
		#define ACTIVATION_WIDE_SIZE_BYTE_MASK ((1 << ACTIVATION_DRAM_SIZE_BYTE_OFFSET) - 1)
	#endif
#else
	#error ACTIVATION_DRAM_SIZE_GREATER_THAN_PE_SIZE should be either 0 (false) or 1 (true)
#endif
#if (ACTIVATION_DRAM_SIZE_BYTE < PE_ROWS_PER_GROUP)
	#error ACTIVATION_DRAM_SIZE_BYTE should be greater than or equal to PE_ROWS_PER_GROUP
	#pragma message "ACTIVATION_DRAM_SIZE_BYTE is " IN_QUOTES(ACTIVATION_DRAM_SIZE_BYTE)
#else
	#define ACTIVATION_DRAM_SIZE_IN_PE_ROW_GROUP (ACTIVATION_DRAM_SIZE_BYTE / PE_ROWS_PER_GROUP)
	#define ACTIVATION_DRAM_SIZE_IN_PE_ROW_GROUP_OFFSET (ACTIVATION_DRAM_SIZE_BYTE_OFFSET - DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT)
	#define ACTIVATION_WIDE_SIZE_IN_PE_ROW_GROUP_MASK ((1 << ACTIVATION_DRAM_SIZE_IN_PE_ROW_GROUP_OFFSET) - 1)
#endif
//========End of derived MARCOS for activation=============

//Flag. 
//Whether the number of weights in a DRAM block is greater
//than the number of weights in a PE block
#define WEIGHT_DRAM_SIZE_GEQ_PE_SIZE 0
//Conversion factor between the number of weights in a DRAM block
//and the the number of weights in a PE block
#define WEIGHT_WIDE_SIZE 1
//============Derived Marcos for weights===
#if (WEIGHT_WIDE_SIZE == 1)
	//Numnber of bit-shift that correspond to the conversion factor
	#define WEIGHT_WIDE_SIZE_OFFSET 0x0 
	//Mask bit required for calculating the remainder of the conversion factor
	#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0x0 
#elif (WEIGHT_WIDE_SIZE == 2)
	#define WEIGHT_WIDE_SIZE_OFFSET 0x1
	#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0x1
#elif (WEIGHT_WIDE_SIZE == 4)
	#define WEIGHT_WIDE_SIZE_OFFSET 0x2
	#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0x3
#elif (WEIGHT_WIDE_SIZE == 8)
	#define WEIGHT_WIDE_SIZE_OFFSET 0x3
	#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0x7
#elif (WEIGHT_WIDE_SIZE == 16)
	#define WEIGHT_WIDE_SIZE_OFFSET 0x4
	#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0xF
#else
	#error WEIGHT_WISE_SIZE should be picked from {1, 2, 4, 8, 16}
#endif
//If the number of weights in a DRAM block
//is greater or equal to the number of weights in a PE block
#if (WEIGHT_DRAM_SIZE_GEQ_PE_SIZE == 1)
	// Number of weights in a DRAM block
	#define 	WEIGHT_DRAM_SIZE_VALUE_BYTE (WEIGHT_WIDE_SIZE * PE_WEIGHT_BLOCK_SIZE_IN_WORD)
	// Log_2 (Number of weights in a DRAM block)
	#define WEIGHT_DRAM_SIZE_VALUE_BYTE_OFFSET (PE_WEIGHT_BLOCK_SIZE_IN_WORD_OFFSET + WEIGHT_WIDE_SIZE_OFFSET)
	#if defined(SPW_SYSTEM)
		// Number of indices bytes in a weight DRAM block
		#define WEIGHT_DRAM_SIZE_INDEX_BYTE (WEIGHT_WIDE_SIZE*INDEX_CHAR_ARRAY_SIZE)
		#define WEIGHT_DRAM_SIZE_INDEX_BYTE_OFFSET (INDEX_CHAR_ARRAY_SIZE_OFFSET + WEIGHT_WIDE_SIZE_OFFSET)
	#endif
#elif (WEIGHT_DRAM_SIZE_GEQ_PE_SIZE == 0)
	#if (PE_WEIGHT_BLOCK_SIZE_IN_WORD < WEIGHT_WIDE_SIZE)
		#error WEIGHT_DRAM_SIZE_GEQ_PE_SIZE is 0 but WEIGHT_WIDE_SIZE cannot divide PE_WEIGHT_BLOCK_SIZE_IN_WORD
	#endif
	#if defined(SPW_SYSTEM)
		#if (INDEX_CHAR_ARRAY_SIZE < WEIGHT_WIDE_SIZE)
			#error WEIGHT_DRAM_SIZE_GEQ_PE_SIZE is 0 but WEIGHT_WIDE_SIZE cannot divide INDEX_CHAR_ARRAY_SIZE
		#endif
	#endif
	#define WEIGHT_DRAM_SIZE_VALUE_BYTE (PE_WEIGHT_BLOCK_SIZE_IN_WORD / WEIGHT_WIDE_SIZE)
	#define WEIGHT_DRAM_SIZE_VALUE_BYTE_OFFSET (PE_WEIGHT_BLOCK_SIZE_IN_WORD_OFFSET - WEIGHT_WIDE_SIZE_OFFSET)
	#if defined(SPW_SYSTEM)
		#define WEIGHT_DRAM_SIZE_INDEX_BYTE (INDEX_CHAR_ARRAY_SIZE / WEIGHT_WIDE_SIZE)
		#define WEIGHT_DRAM_SIZE_INDEX_BYTE_OFFSET (INDEX_CHAR_ARRAY_SIZE_OFFSET - WEIGHT_WIDE_SIZE_OFFSET)
	#endif
	#else
	#error WEIGHT_DRAM_SIZE_GEQ_PE_SIZE must be either 0 (false) or 1 (true)
#endif
//==============End of derived marcos for weights

#define WMOVER_FILTER_DRAM_BLOCK_ACCESS_UNROLL_FACTOR 4
#define KERNEL_CACHE_SIZE_VALUE_BYTE 32768
#define KERNEL_CACHE_DEPTH (KERNEL_CACHE_SIZE_VALUE_BYTE / WEIGHT_DRAM_SIZE_VALUE_BYTE)

#define MAX_OUTPUT_TILE_WIDTH_PER_COL 8 
#define MAX_OUTPUT_TILE_HEIGHT 31
#define MAX_INPUT_TILE_WIDTH_PER_COL 8
#define MAX_INPUT_TILE_HEIGHT 31

//Cylone V SoC
#define IA_CACHE_SIZE_BYTE 16384
//Arria 10GX1150
//#define IA_CACHE_SIZE_BYTE 32768
#define IA_CACHE_DEPTH (IA_CACHE_SIZE_BYTE/ACTIVATION_DRAM_SIZE_BYTE)

#define WEIGHT_MOVER_BIAS_CACHE_SIZE 2048

//TODO: Change this back to the commented line
#define OA_CACHE_SIZE_BYTE 16384
//Arria 10 GX1150
//#define OA_CACHE_SIZE_BYTE 32768
#define OA_CACHE_DEPTH (OA_CACHE_SIZE_BYTE/ACTIVATION_DRAM_SIZE_BYTE)

//Accumulator width
#define ACCUMULATOR_WIDTH 28
#if defined(EMULATOR)
#pragma message("WARNING: IN EMULATOR MODE, ACCUMULATOR_WIDTH IS FIXED TO 32")
#define ACCUM_MASK 0x0FFFFFFFF
#define MULT_MASK 0x0FFFFFFFF
#define ACCUM_MIN 0x80000000
#elif (ACCUMULATOR_WIDTH == 32)
#define ACCUM_MASK 0x0FFFFFFFF
#define MULT_MASK 0x0FFFFFFFF
#define ACCUM_MIN 0x80000000
#elif (ACCUMULATOR_WIDTH == 28)
#define ACCUM_MASK 0x00FFFFFFF
#define MULT_MASK 0x00FFFFFFF
#define ACCUM_MIN 0x08000000
#elif (ACCUMULATOR_WIDTH == 24)
#define ACCUM_MASK 0x00FFFFFF
#define MULT_MASK 0x00FFFFFF
#define ACCUM_MIN 0x00800000
#elif (ACCUMULATOR_WIDTH == 20)
#define ACCUM_MASK 0x000FFFFF
#define MULT_MASK 0x000FFFFF
#define ACCUM_MIN 0x00080000
#elif (ACCUMULATOR_WIDTH == 16)
#define ACCUM_MASK 0x0FFFF
#define MULT_MASK 0x0FFFF
#define ACCUM_MIN 0x00008000
#else
#error Accumulator width should be from 32-bit, 28-bit, 24-bit, 20-bit, and 16-bit
#endif

#define MISC_ACCUMULATOR_WIDTH 16
#if defined(EMULATOR)
#pragma message("WARNING: IN EMULATOR MODE, MISC_ACCUMULATOR_WIDTH IS FIXED TO 32")
#define MISC_ACCUM_MASK 0x0FFFFFFFF
#define MISC_MULT_MASK 0x0FFFFFFFF
#define MISC_ACCUM_MIN 0x80000000
#elif (MISC_ACCUMULATOR_WIDTH == 32)
#define MISC_ACCUM_MASK 0x0FFFFFFFF
#define MISC_MULT_MASK 0x0FFFFFFFF
#define MISC_ACCUM_MIN 0x80000000
#elif (MISC_ACCUMULATOR_WIDTH == 28)
#define MISC_ACCUM_MASK 0x00FFFFFFF
#define MISC_MULT_MASK 0x00FFFFFFF
#define MISC_ACCUM_MIN 0x08000000
#elif (MISC_ACCUMULATOR_WIDTH == 24)
#define MISC_ACCUM_MASK 0x00FFFFFF
#define MISC_MULT_MASK 0x00FFFFFF
#define MISC_ACCUM_MIN 0x00800000
#elif (MISC_ACCUMULATOR_WIDTH == 20)
#define MISC_ACCUM_MASK 0x000FFFFF
#define MISC_MULT_MASK 0x000FFFFF
#define MISC_ACCUM_MIN 0x00080000
#elif (MISC_ACCUMULATOR_WIDTH == 16)
#define MISC_ACCUM_MASK 0x0FFFF
#define MISC_MULT_MASK 0x0FFFF
#define MISC_ACCUM_MIN 0x00008000
#else
#error Misc accumulator width should be from 32-bit, 28-bit,  24-bit, 20-bit, and 16-bit
#endif


#define TIMEOUT 0X1FFFFFF
#endif

