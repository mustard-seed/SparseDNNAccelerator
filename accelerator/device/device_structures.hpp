#ifndef STRUCTURES_HPP_DEF
#define STRUCTURES_HPP_DEF

#include "params.hpp"
#ifndef INTELFPGA_CL
#include "CL/cl.hpp"
#endif

#ifdef INTELFPGA_CL
#include "ihc_apint.h"
#endif

/*! t_instruction
 * \brief VLIW instructions for controlling the accelerator
 */
#ifdef INTELFPGA_CL
typedef struct __attribute__((aligned(32))) __attribute__((packed)) {
    unsigned char header;
    unsigned char instructionSizeBytes;
    unsigned char dependencyList[DEPENDENCY_LIST_SIZE_BYTE];
    unsigned char words[INSTRUCTION_SIZE_BYTE];
} t_instruction;
#else
typedef struct __attribute__((aligned(32))) __attribute__((packed)) {
    cl_uchar header;
    cl_uchar instructionSizeBytes;
    cl_uchar dependencyList[DEPENDENCY_LIST_SIZE_BYTE];
    cl_uchar words[INSTRUCTION_SIZE_BYTE];
} t_instruction;
#endif

#ifdef INTELFPGA_CL
typedef short t_spValueAndZCount;
typedef unsigned short t_spOffset;
#else
typedef cl_short t_spValueAndZCount;
typedef cl_ushort t_spOffset;
#endif



typedef struct  __attribute__((aligned(8))) __attribute__((packed)) {
    t_spValueAndZCount vec[COMPRESSION_VEC_SIZE];
} t_vecSpValueAndZCount;

typedef struct __attribute((aligned(32))) __attribute__((packed)){
    short nzValues [COMPRESSION_VEC_SIZE];
    unsigned char validMasks [COMPRESSION_VEC_SIZE];
    unsigned short indices [COMPRESSION_VEC_SIZE];
} t_vecUnpackedHost;

#ifdef INTELFPGA_CL
typedef struct  __attribute__((aligned(32))) __attribute__((packed)) {
    short nzValue;
    unsigned short indexInStreamingBlock;
} t_spValueAndZCountUnpackedHost;
#else
typedef struct  __attribute__((aligned(32))) __attribute__((packed)) {
    cl_short nzValue;
    cl_ushort indexInStreamingBlock;
} t_spValueAndZCountUnpackedHost;
#endif

//Data structures used in direct compression SIMD
//#ifdef DIRECT_COMPRESSION_SIMD
#ifdef INTELFPGA_CL
typedef struct __attribute__((packed)) {
    char values [SIMD_SIZE];
    unsigned char runLength;
} t_simdblock;

typedef struct __attribute__((aligned(8))) __attribute__((packed)) {
    char values [SIMD_SIZE];
    //unsigned char runLength;
} t_simdblock_host;
#else
typedef struct __attribute__((aligned(8))) __attribute__((packed)) {
    cl_char values [SIMD_SIZE];
    //cl_uchar runLength;
} t_simdblock_host;
#endif

#ifdef INTELFPGA_CL
typedef struct {
    char values [SIMD_SIZE];
} t_simdblock_value; //Value in a simdblock

typedef unsigned char t_simdblock_channel_offset; //Relative channel of a simdblock in a streaming block

typedef unsigned short t_streamblock_address; //Address of a streaming block in BRAM

#else
//typedef struct {
//    cl_char values [SIMD_SIZE];
//} t_simdblock_value;
typedef t_simdblock_host t_simdblock_value;

typedef cl_uchar t_simdblock_channel_offset;

typedef cl_ushort t_streamblock_address;
#endif
//#endif //Data structures used in direct compression SIMD

//=============================
//Structs used in the flexible bitmask compression test
//#ifdef FLEXIBLE_BITMASK_COMPRESSION
#ifdef INTELFPGA_CL
typedef struct {
    char cluster_values [CLUSTER_SIZE];
} t_cluster;
#else
typedef struct {
    cl_char cluster_values [CLUSTER_SIZE];
} t_cluster;

typedef cl_ushort t_streamblock_address;
#endif

typedef struct {
    t_cluster values [TRANSFER_SIZE];
} t_transfer_block;

typedef struct {
    t_transfer_block transferBlocks[WIDE_SIZE];
} t_dram_block;

typedef struct {
    t_dram_block dramBlock;
    unsigned char destinationRow;
} t_dram_block_tagged;

#ifdef INTELFPGA_CL
typedef struct {
    unsigned char maxOutputHeightTileSize; //maxTP
    unsigned char maxOutputWidthTileSize; //maxTQ
    //nsigned char destinationRow; //f
    unsigned short numTransferBlocks;

} t_filter_streamer_control;

t_filter_streamer_control dramBlock2FilterStreamerControl (t_dram_block block)
{
    t_filter_streamer_control control;
    control.maxOutputHeightTileSize
        = block.transferBlocks[0].values[0].cluster_values[0];
    control.maxOutputWidthTileSize
        = block.transferBlocks[0].values[0].cluster_values[1];
    //control.destinationRow 
    //    = block.transferBlocks[2].values[0].cluster_values[0];
    control.numTransferBlocks
        = ( ( ( (short) (block.transferBlocks[0].values[1].cluster_values[0]) ) & 0xFF )
            | ( (((short) (block.transferBlocks[0].values[1].cluster_values[1])) & 0xFF) << 8))  & 0xFFFF;

    return control;
}

t_dram_block filterStreamerControl2dramBlock (t_filter_streamer_control control)
{
    t_dram_block block;
    block.transferBlocks[0].values[0].cluster_values[0] = control.maxOutputHeightTileSize;
    block.transferBlocks[0].values[0].cluster_values[1] = control.maxOutputWidthTileSize;
    //block.transferBlocks[2].values[0].cluster_values[0] = control.destinationRow;
    block.transferBlocks[0].values[1].cluster_values[0] = control.numTransferBlocks & 0xFF;
    block.transferBlocks[0].values[1].cluster_values[1] = ((control.numTransferBlocks >> 8) & 0xFF);

    return block;
}

typedef struct __attribute__((packed)){
    t_transfer_block values;
    uint1_t isLast;
    char maxTransportID;
} t_transferblock_tagged;

typedef struct __attribute__((packed)){
    t_transfer_block values;
    uint1_t isLast;
} t_transferblock_local;
#endif
//#endif

//=====================================


#ifdef INTELFPGA_CL
#include "ihc_apint.h"
typedef uint4_t t_zCount;
typedef uint12_t t_weight;

//Needs to be signed!!!!
typedef char t_operand;
typedef int t_accumulator;

#ifdef DIRECT_COMPRESSION_SIMD
//With the max transport length and last bit annotation
typedef struct __attribute__((packed)){
    char values [SIMD_SIZE];
    unsigned char streamingBlockIndex;
    bool isLast;
    char maxTransportID;

} t_simdblock_di_tagged;

typedef struct __attribute__((packed)){
    char values [SIMD_SIZE];
    bool isLast;
    char maxTransportID;

} t_simdblock_bitmask_tagged;
#endif


#ifdef PE_PROTOTYPE_TEST
typedef short t_value;
#else
typedef int12_t t_value;
#endif


typedef struct  __attribute__((packed)) {
    unsigned char metaInformation; //lsLast, indexInStreamingBlock
    //uint1_t isLast;
    //uint6_t indexInStreamingBlock;
    t_operand nzValue;
} t_spValueAndZCountUnpacked;


//typedef unsigned short t_spValueAndZountUnpacked;
#define UNPACKED_INDEX_MASK 0x3F
#define UNPACKED_INDEX_BITOFFSET 0
#define UNPACKED_ISLAST_MASK 0x01
#define UNPACKED_ISLAST_BITOFFSET 6

/*! t_tokenFillWeightCache
    Token used to command filling of the sparse weight cache
*/
typedef struct __attribute__((packed)){
    unsigned int ddrKernelIndexStartOffset; //Word offset of the indices of the kernel relative to the start of the global memory
    unsigned int ddrKernelWeightStartOffset; //Word offset of the weights of the kernel relative to the start of the global memory
    unsigned short filterStart; //Index of the first filter to be streamed into the cache
    unsigned char numFiltersToStream;
    unsigned short cbStart; //The first encoded block to be streamed. Index 0 corresponds to the beginning of the row
    unsigned short cbEnd; //The last encoded block to be streamed. Index 0 corresponds to the beginning of the row

    unsigned short numEncodingBlocksInFilter; //Number of encoding blocks in a filter. R*S*CB
    unsigned int numWeightsInFilter; //Number of weights in a filter, if no compression is applied. R*S*C

    //uint1_t fillSetNumber; //Which bank to fill. Either 0 or 1;
} t_tokenFillWeightCache;


typedef union {
    t_spValueAndZCount weightAndOffset;
    t_spOffset offset;
} u_index_data;

/*! t_packetDMAToWeightFeeder
    Structure encapsulating the data from the Sparse Weight DMA to the Sparse Weight feeders
*/
typedef struct __attribute__((packed)){
    u_index_data packet;
    unsigned short laneNumber;
    unsigned short depth;
    uint1_t isIndex;
} t_packetDMAToWeightFeeder;


/*! t_tokenDrainWeightCache
    Token used to command draining of the sparse weight cache
*/
typedef struct __attribute__((packed)){
    unsigned char laneStart; //First lane to be streamed
    unsigned char laneEnd; //One plus the last lane to be streamed
    /*
    Index of the first encoder block inside each lane's index cache line to be streamed.
    The block at the start of the cache line has index 0
    */
    unsigned short cbStart;
    /*
    Index of the last encoder block plus one inside each lane's index cache line to be streamed
    */
    unsigned short cbEnd; //Index of the last encoder block insider each filter to be streamed

    //uint1_t drainSetNumber; //Which bank to drain. Either 0 or 1;
} t_tokenDrainWeightCache;

#ifdef WEIGHT_MEMORY_TEST
typedef struct __attribute__((packed)){
  unsigned int ddrKernelWeightStartOffset; //Offset of the start of the tensor in DDR
  unsigned short filterStart; //The first filter (i.e. matrix row) to be collected
  unsigned short numFiltersToCollect; //Number of filters (i.e. matrix row to collect)
  uint24_t numWeightsInFilter; //Number of weights in the filter if it were uncompressed 

  } t_weightCollectToken;
#endif

#endif //INTELFPGA_CL

#endif //STRUCTURES_HPP_DEF
