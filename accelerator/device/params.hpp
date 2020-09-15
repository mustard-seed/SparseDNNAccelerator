#ifndef PARAMS_DEFINED
#define PARAMS_DEFINED

//#define HOST_DEBUG
#define SPARSE_SYSTEM
#define OA_PING_PONG
/**
 * Global memory settings
 */
#define MAX_DRAM_BYTE_INPUT_ACTIVATION (1 << 27)
#define MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT (1 << 22)
#define MAX_DRAM_BYTE_INPUT_WEIGHT (1 << 26)
#define MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT 0x800
#define MAX_DRAM_BYTE_INPUT_BIAS 0x800
//#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION 0x400000
//#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT 0x40000
#define MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION (1 << 20)


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
	#define PE_ROWS 1
	#define PE_COLS 1
#else
	#define PE_ROWS 2
	#define PE_COLS 2
#endif

#define CHANNEL_DEPTH 2

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
#define CLUSTER_SIZE 4 //cluster size in terms of values
#define VALUE_TO_CLUSTER_SHIFT 2 //amount of right shift required to convert a value index into cluster index
#define VALUE_DIVIDED_BY_CLUSTER_SIZE_REMAINDER_MASK 0x3
#define VALUE_DIVIDED_BY_SIMD_SIZE_REMAINDER_MASK ((1 << (VALUE_TO_CLUSTER_SHIFT + CLUSTER_TO_TRANSFER_SIZE_SHIFT)) - 1)
#define CLUSTER_TO_TRANSFER_BLOCK_SHIFT CLUSTER_TO_TRANSFER_SIZE_SHIFT //amount of right shift required to convert a cluster count into transfer block count

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

#define BURST_SIZE_BYTE 8

#define KERNEL_CACHE_LANES PE_ROWS
#define KERNEL_CACHE_LANE_MASK 0x7
#define KERNEL_CACHE_SIZE_BYTE 8192
#define KERNEL_CACHE_DEPTH (KERNEL_CACHE_SIZE_BYTE/BURST_SIZE_BYTE)
#define KERNEL_CACHE_DEPTH_MASK 0x03FF

#define IA_CACHE_SIZE_BYTE 16384
#define IA_CACHE_DEPTH (IA_CACHE_SIZE_BYTE/BURST_SIZE_BYTE)

#define IA_TBCOUNT_CACHE_SIZE 256

#define OA_CACHE_SIZE_BYTE 32768
#define OA_CACHE_DEPTH (OA_CACHE_SIZE_BYTE/CLUSTER_SIZE)


//TODO: Change WIDE_SIZE and related offsets when compression configuration changes
#define WIDE_SIZE (BURST_SIZE_BYTE/CLUSTER_SIZE/TRANSFER_SIZE)  //Each transfer block takes 4 bytes, so need 8 transfer blocks to populate 256 bits
#define WIDE_SIZE_OFFSET 0x0 //Numnber of bits to shift the transfer block index to the right in order to recover the wide offset
#define WIDE_SIZE_REMAINDER_MASK 0x0

#define NUM_CLUSTER_IN_DRAM_SIZE BURST_SIZE_BYTE/CLUSTER_SIZE

//Accumulator width
#define ACCUMULATOR_WIDTH 32
#if (ACCUMULATOR_WIDTH == 32)
#define ACCUM_MASK 0x0FFFFFFFF
#define MULT_MASK 0x0FFFFFFFF
#elif (ACCUMULATOR_WIDTH == 16)
#define ACCUM_MASK 0x0FFFF
#define MULT_MASK 0x0FFFF
#else
#error Accumulator width should either be 32 bit or 16 bit
#endif

#define KERNEL_INDEX_CACHE_DEPTH 512
#define KERNEL_INDEX_CACHE_DEPTH_MASK 0x1FF
#define KERNEL_INDEX_CACHE_LANES KERNEL_CACHE_LANES

//PE operaton modes
#define PE_MODE_LOAD_BIAS 0X0
#define PE_MODE_LOAD_ACTIVATION 0x01
#define PE_MODE_ELTWISE_ADD 0x02
#define PE_MODE_DOT_PRODUCT 0x03
#define PE_MODE_MAX_POOL 0x04
#define PE_MODE_DRAIN_PSUM 0x05

//PE FIFO parameters
#define PE_VEC_FIFO_SIZE (COMPRESSION_WINDOW_SIZE / TRANSFER_SIZE)  //Need to set this to avoid deadlock
//#define PE_VEC_FIFO_SIZE 1
//#define PE_NUM_MULT COMPRESSION_VEC_SIZE

//PE datawidth parameters
#define REG_FF_FRAC 16 //16 bit fraction width, make  sure it is wider than all possible frac_width used on the short data format
#define REG_FF_WIDTH 32 //32 bit FF, int



#define TIMEOUT 0X1FFFFFF
#endif

