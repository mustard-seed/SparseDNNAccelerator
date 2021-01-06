#ifndef PARAMS_DEFINED
#define PARAMS_DEFINED

//#define HOST_DEBUG
#define SPW_SYSTEM
#define OA_PING_PONG
//#define WMOVER_STREAM_CACHE
//#define OAMOVER_TB_STREAM_CACHE
//#define WMOVER_WEIGHT_COALESCE_CACHE

#define NOOP
/**
 * Global memory settings
 */
#define MAX_DRAM_BYTE_INPUT_ACTIVATION (1 << 27)
#define MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT (1 << 22)
#define MAX_DRAM_BYTE_INPUT_WEIGHT (1 << 26)
#define MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT 0x40000
#define MAX_DRAM_BYTE_INPUT_BIAS 0x20000
//#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION 0x400000
//#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT 0x40000
#define MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION (1 << 23)
#define MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION (1 << 23)

#define PACKET_SIZE 1

#if defined(FULL_SYSTEM)
	#define PE_COLS 1
	#define PE_ROWS_PER_GROUP 1
	#define PE_ROW_GROUPS 1
	#define MISC_COLS 1
#else
	#define PE_COLS 1
	#define PE_ROWS_PER_GROUP 4
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
#else
#error DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT should be chosen from {1, 2, 4, 8}
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


//TODO: Change COMPRESSION_WINDOW_SIZE, TRANSFER_SIZE, CLUSTER_SIZE, and related offsets and masks if compression configuration changes
#define COMPRESSION_WINDOW_SIZE 8 //compression window size in terms of clusters
#define CLUSTER_TO_WINDOW_SHIFT 0X3
#define CLUSTER_TO_WINDOW_REMAINDER_MASK 0x07
#define TRANSFER_SIZE 2 //transfer block size in terms of clusters
#define CLUSTER_TO_TRANSFER_SIZE_SHIFT 0X1
#define CLUSTER_TO_TRANSEFER_SIZE_REMAINDER 0X1
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

//Size of the char array in the host weight blocks
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
#else
#error "Parameter PE_SIMD_SIZE is not been defined."
#endif

#if defined(SPW_SYSTEM)
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD (CLUSTER_SIZE * PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER)
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET (VALUE_TO_CLUSTER_SHIFT + PE_SIMD_SIZE_CLUSTER_OFFSET + PRUNE_RNAGE_IN_CLUSTER_OFFSET)
#else
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD (CLUSTER_SIZE * PE_SIMD_SIZE)
	#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET (VALUE_TO_CLUSTER_SHIFT + PE_SIMD_SIZE_CLUSTER_OFFSET)
#endif
#define PE_ACTIVATION_BLOCK_SIZE_IN_WORD_MASK ((1 << PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET) - 1)

/**
 * Small buffer operand filterign related
 */
#define BITMASK_LENGTH COMPRESSION_WINDOW_SIZE
#define MAX_NUM_OUTPUT TRANSFER_SIZE
#define BITMASK_ACCUM_COUNT_BITWIDTH 2 //$rtoi($clog2(MAX_NUM_OUTPUT) + 1.0)
#define BITMASK_INDEX_BITWIDTH 3 //$rtoi($ceil($clog2(COMPRESSION_WINDOW_SIZE)))
#define NUM_BITMASK_BYTES 1
#define NUM_ACCUM_BITMASK_BYTES 2

//=========================================

#define SURVIVING_COUNT_CLUSTER_INDEX 0X1
#define SURVIVING_COUNT_TRANSFER_BLOCK_INDEX 0x1

#define BURST_SIZE_BYTE 32
//TODO: Change WIDE_SIZE and related offsets when compression configuration changes
#define WIDE_SIZE (BURST_SIZE_BYTE/CLUSTER_SIZE/TRANSFER_SIZE)  //Each transfer block takes 4 bytes, so need 8 transfer blocks to populate 256 bits
#define WIDE_SIZE_OFFSET 0x2 //Numnber of bits to shift the transfer block index to the right in order to recover the wide offset
#define WIDE_SIZE_REMAINDER_MASK 0x3

#define NUM_CLUSTER_IN_DRAM_SIZE BURST_SIZE_BYTE/CLUSTER_SIZE

#define ACTIVATION_WIDE_SIZE 1
#define ACTIVATION_WIDE_SIZE_OFFSET 0x0 //Numnber of bits to shift the transfer block index to the right in order to recover the wide offset
#define ACTIVATION_WIDE_SIZE_REMAINDER_MASK 0x0
#define ACTIVATION_BURST_SIZE_BYTE (PE_ACTIVATION_BLOCK_SIZE_IN_WORD * ACTIVATION_WIDE_SIZE)
#define ACTIVATION_BURST_SIZE_BYTE_OFFSET (ACTIVATION_WIDE_SIZE_OFFSET + PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET)
#define ACTIVATION_WIDE_SIZE_BYTE_MASK ((1 << ACTIVATION_BURST_SIZE_BYTE_OFFSET) - 1)

#define WEIGHT_WIDE_SIZE 4
#define WEIGHT_WIDE_SIZE_OFFSET 0x2 //Numnber of bits to shift the transfer block index to the right in order to recover the wide offset
#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0x3
#define WEIGHT_BURST_SIZE_VALUE_BYTE (WEIGHT_WIDE_SIZE * PE_SIMD_SIZE * CLUSTER_SIZE)
#if defined(SPW_SYSTEM)
#define WEIGHT_BURST_SIZE_INDEX_BYTE (WEIGHT_WIDE_SIZE*INDEX_CHAR_ARRAY_SIZE)
#endif


#define WMOVER_FILTER_DRAM_BLOCK_ACCESS_UNROLL_FACTOR 4
#define KERNEL_CACHE_SIZE_VALUE_BYTE 8192
#define KERNEL_CACHE_DEPTH (KERNEL_CACHE_SIZE_VALUE_BYTE / WEIGHT_BURST_SIZE_VALUE_BYTE)

#define MAX_OUTPUT_TILE_WIDTH_PER_COL 7 
#define MAX_OUTPUT_TILE_HEIGHT 32
#define MAX_INPUT_TILE_WIDTH_PER_COL 7
#define MAX_INPUT_TILE_HEIGHT 32

// #define IA_CACHE_SIZE_BYTE 16384
#define IA_CACHE_SIZE_BYTE 32768
#define IA_CACHE_DEPTH (IA_CACHE_SIZE_BYTE/ACTIVATION_BURST_SIZE_BYTE)

#define WEIGHT_MOVER_BIAS_CACHE_SIZE 2048

//TODO: Change this back to the commented line
//#define OA_CACHE_SIZE_BYTE 8192
#define OA_CACHE_SIZE_BYTE 32768
#define OA_CACHE_DEPTH (OA_CACHE_SIZE_BYTE/ACTIVATION_BURST_SIZE_BYTE)

//Accumulator width
#define ACCUMULATOR_WIDTH 20
#if defined(EMULATOR)
#pragma message("WARNING: IN EMULATOR MODE, ACCUMULATOR_WIDTH IS FIXED TO 32")
#define ACCUM_MASK 0x0FFFFFFFF
#define MULT_MASK 0x0FFFFFFFF
#define ACCUM_MIN 0x80000000
#elif (ACCUMULATOR_WIDTH == 32)
#define ACCUM_MASK 0x0FFFFFFFF
#define MULT_MASK 0x0FFFFFFFF
#define ACCUM_MIN 0x80000000
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
#error Accumulator width should be from 32-bit, 24-bit, 20-bit, and 16-bit
#endif


#define TIMEOUT 0X1FFFFFF
#endif

