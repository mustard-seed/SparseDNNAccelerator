#ifndef PARAMS_DEFINED
#define PARAMS_DEFINED

#define INSTRUCTION_SIZE_BYTE 24
#define DEPENDENCY_LIST_SIZE_BYTE 2
#define MAX_INSTRUCTION_IN_FLIGHT_COUNT_PER_TYPE 3
#define MAX_INSTRUCTION_TYPE_BITS 16


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


#define PACKET_SIZE 1

#define PE_ROWS 4
#define PE_COLS 2

//Encoding weight length
#define ENCODING_LENGTH 64
//Number of encoded values to be transfered together
#define COMPRESSION_VEC_SIZE 4

#define KERNEL_CACHE_LANES PE_ROWS
#define KERNEL_CACHE_LANE_MASK 0x7
#define KERNEL_CACHE_DEPTH 1024
#define KERNEL_CACHE_DEPTH_MASK 0x03FF

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
#define PE_VEC_FIFO_SIZE 16
#define PE_NUM_MULT COMPRESSION_VEC_SIZE

//PE datawidth parameters
#define REG_FF_FRAC 16 //16 bit fraction width, make  sure it is wider than all possible frac_width used on the short data format
#define REG_FF_WIDTH 32 //32 bit FF, int

#define IDX 0
#define IDY 0



#define TIMEOUT 0X1FFFFFF
#endif

