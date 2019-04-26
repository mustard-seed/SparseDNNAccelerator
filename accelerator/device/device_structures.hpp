#ifndef STRUCTURES_HPP_DEF
#define STRUCTURES_HPP_DEF
#include "ihc_apint.h"
#include "params.hpp"

typedef short t_spWeightAndOffset;
typedef unsigned short t_spOffset;
typedef uint4_t t_zCount;
typedef uint12_t t_weight;

/*! t_tokenFillWeightCache
    Token used to command filling of the sparse weight cache
*/
typedef struct {
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
    t_spWeightAndOffset weightAndOffset;
    t_spOffset offset;
} u_index_data;

/*! t_packetDMAToWeightFeeder
    Structure encapsulating the data from the Sparse Weight DMA to the Sparse Weight feeders
*/
typedef struct {
    u_index_data packet;
    short laneNumber;
    short depth;
    uint1_t isIndex;
} t_packetDMAToWeightFeeder;


/*! t_tokenDrainWeightCache
    Token used to command draining of the sparse weight cache
*/
typedef struct {
    unsigned char laneStart; //First lane to be streamed
    unsigned char laneEnd; //Last lane to be streamed
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

/*! t_instruction
 * \brief VLIW instructions for controlling the accelerator
 */
typedef struct {
    unsigned char header;
    unsigned char instructionSizeBytes;
    unsigned char dependencyList[DEPENDENCY_LIST_SIZE_BYTE];
    unsigned char words[INSTRUCTION_SIZE_BYTE];
} t_instruction;

#ifdef WEIGHT_MEMORY_TEST
typedef struct {
  unsigned int ddrKernelWeightStartOffset; //Offset of the start of the tensor in DDR
  unsigned short filterStart; //The first filter (i.e. matrix row) to be collected
  unsigned short numFiltersToCollect; //Number of filters (i.e. matrix row to collect)
  uint18_t numWeightsInFilter; //Number of weights in the filter if it were uncompressed 

  } t_weightCollectToken;
#endif
#endif
