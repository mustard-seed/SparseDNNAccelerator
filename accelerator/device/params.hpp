#ifndef PARAMS_DEFINED
#define PARAMS_DEFINED

//#define HOST_DEBUG
//#define SPARSE_SYSTEM
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


#ifdef WEIGHT_MEMORY_TEST
//Instruction IDs
#define OPCODE_FILL_WEIGHT_BUFFER 0x1
#define OPCODE_DRAIN_WEIGHT_BUFFER 0X2
#define OPCODE_SWAP_WEIGHT_BUFFER 0x3
#define OPCODE_COLLECT_WEIGHT 0X4

#define NUM_TRANSPORTS 0x4

//Transport IDs
#define TRANSPORT_ID_FILL_WEIGHT_BUFFER 0X0
#define TRANSPORT_ID_DRAIN_WEIGHT_BUFFER 0X1
#define TRANSPORT_ID_SWAP_WEIGHT_BUFFER 0x2
#define TRANSPORT_ID_COLLECT_WEIGHT 0X3

//Instruction size bytes
#define INSTRUCTION_SIZE_BYTE_FILL_WEIGHT_BUFFER 21
#define INSTRUCTION_SIZE_BYTE_DRAIN_WEIGHT_BUFFER 6
#define INSTRUCTION_SIZE_BYTE_SWAP_WEIGHT_BUFFER 0x01
#define INSTRUCTION_SIZE_BYTE_COLLECT_WEIGHT 11

#endif //WEIGHT_MEMORY_TEST

//=======Number of bits allocated to the zero count and the value
#define WEIGHT_VALID_BITWIDTH 1
#define WEIGHT_VALID_BITOFFSET 15
#define WEIGHT_VALID_MASK 0x01
#define WEIGHT_ZCOUNT_BITWIDTH 4
#define WEIGHT_ZCOUNT_BITOFFSET 8
#define WEIGHT_ZCOUNT_MASK 0xF
#define WEIGHT_ZCOUNT_MAX 15
#define WEIGHT_BITWIDTH 8
#define WEIGHT_BITOFFSET 0
#define WEIGHT_MASK 0x0FF

//Must be in HEX
#define WEIGHT_MAX 255
#define WEIGHT_MIN -256
//================================================================

//=======Mask used for the new encoding scheme (2019/07)=========
//channel offset information
#define CHANNEL_OFFSET_MASK 0x07F
#define CHANNEL_OFFSET_BITOFFSET 0
#define IS_LAST_BLOCK_MASK 0x01
#define IS_LAST_BLOCK_BITOFFSET	7
//===============================
//===============================================================


#define PACKET_SIZE 1

#if defined(FULL_SYSTEM)
	#define PE_COLS 4
	#define PE_ROWS_PER_GROUP 4
	#define PE_ROW_GROUPS 1
	#define PE_ROWS (PE_ROWS_PER_GROUP * PE_ROW_GROUPS)
	#define MISC_COLS 1
#else
	#define PE_COLS 1
	#define PE_ROWS_PER_GROUP 4
	#define PE_ROW_GROUPS 1
	#define PE_ROWS (PE_ROWS_PER_GROUP * PE_ROW_GROUPS)
	#define MISC_COLS 1
#endif
#define MISC_UNROLL 16

#if (MISC_COLS > PE_COLS)
#error Configuration MISC_COLS should be less or equal to PE_COLS
#endif

#define CHANNEL_DEPTH 1
#define OA_DRAIN_CHANNEL_DEPTH 1

//Encoding weight length
#define ENCODING_LENGTH 64
//Number of encoded values to be transfered together
//#define COMPRESSION_VEC_SIZE 4

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
#define VALUE_TO_CLUSTER_SHIFT 1 //amount of right shift required to convert a value index into cluster index
#define VALUE_DIVIDED_BY_CLUSTER_SIZE_REMAINDER_MASK 0x1
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

/**
 * Small buffer operand filterign related
 */
#define BITMASK_LENGTH COMPRESSION_WINDOW_SIZE
#define MAX_NUM_OUTPUT TRANSFER_SIZE
#define BITMASK_ACCUM_COUNT_BITWIDTH 2 //$rtoi($clog2(MAX_NUM_OUTPUT) + 1.0)
#define BITMASK_INDEX_BITWIDTH 3 //$rtoi($ceil($clog2(COMPRESSION_WINDOW_SIZE)))
#define NUM_BITMASK_BYTES 1
#define NUM_ACCUM_BITMASK_BYTES 2
#define NUM_SIMD_WORDS (CLUSTER_SIZE*TRANSFER_SIZE)

//=========================================

#define SURVIVING_COUNT_CLUSTER_INDEX 0X1
#define SURVIVING_COUNT_TRANSFER_BLOCK_INDEX 0x1

#define BURST_SIZE_BYTE 32
//TODO: Change WIDE_SIZE and related offsets when compression configuration changes
#define WIDE_SIZE (BURST_SIZE_BYTE/CLUSTER_SIZE/TRANSFER_SIZE)  //Each transfer block takes 4 bytes, so need 8 transfer blocks to populate 256 bits
#define WIDE_SIZE_OFFSET 0x2 //Numnber of bits to shift the transfer block index to the right in order to recover the wide offset
#define WIDE_SIZE_REMAINDER_MASK 0x3

#define NUM_CLUSTER_IN_DRAM_SIZE BURST_SIZE_BYTE/CLUSTER_SIZE

#define WEIGHT_BURST_SIZE_BYTE 32
#define WEIGHT_WIDE_SIZE (WEIGHT_BURST_SIZE_BYTE/CLUSTER_SIZE/TRANSFER_SIZE)  //Each transfer block takes 4 bytes, so need 8 transfer blocks to populate 256 bits
#define WEIGHT_WIDE_SIZE_OFFSET 0x2 //Numnber of bits to shift the transfer block index to the right in order to recover the wide offset
#define WEIGHT_WIDE_SIZE_REMAINDER_MASK 0x3

#define WMOVER_FILTER_DRAM_BLOCK_ACCESS_UNROLL_FACTOR 4
#define KERNEL_CACHE_LANES PE_ROWS
#define KERNEL_CACHE_LANE_MASK 0x7
#define KERNEL_CACHE_SIZE_BYTE 8192
#define KERNEL_CACHE_DEPTH (KERNEL_CACHE_SIZE_BYTE/WEIGHT_BURST_SIZE_BYTE)
#define KERNEL_CACHE_DEPTH_MASK 0x03FF

#define MAX_OUTPUT_TILE_WIDTH_PER_COL 7 
#define MAX_OUTPUT_TILE_HEIGHT 32
#define MAX_INPUT_TILE_WIDTH_PER_COL 7
#define MAX_INPUT_TILE_HEIGHT 32

// #define IA_CACHE_SIZE_BYTE 16384
#define IA_CACHE_SIZE_BYTE 32768
#define IA_CACHE_DEPTH (IA_CACHE_SIZE_BYTE/BURST_SIZE_BYTE)

#define IA_BUFFER_TBCOUNT_CACHE_SIZE 256

#define IA_MOVER_TBCOUNT_CACHE_SIZE 2048
#define OA_MOVER_TBCOUNT_CACHE_SIZE 2048
#define WEIGHT_MOVER_TBCOUNT_CACHE_SIZE 2048
#define WEIGHT_MOVER_BIAS_CACHE_SIZE WEIGHT_MOVER_TBCOUNT_CACHE_SIZE

//TODO: Change this back to the commented line
//#define OA_CACHE_SIZE_BYTE 8192
#define OA_CACHE_SIZE_BYTE 32768
#define OA_CACHE_DEPTH (OA_CACHE_SIZE_BYTE/CLUSTER_SIZE)

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

#define KERNEL_INDEX_CACHE_DEPTH 512
#define KERNEL_INDEX_CACHE_DEPTH_MASK 0x1FF
#define KERNEL_INDEX_CACHE_LANES KERNEL_CACHE_LANES



#define TIMEOUT 0X1FFFFFF
#endif

