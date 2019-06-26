#ifndef PE_COMPONENT_HPP
#define PE_COMPONENT_HPP
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"

typedef struct __attribute__((packed)){
    t_operand nzValues [COMPRESSION_VEC_SIZE];
    uint1_t validMasks [COMPRESSION_VEC_SIZE];
    unsigned short indices [COMPRESSION_VEC_SIZE];
} t_vecUnpacked;

typedef struct __attribute__((packed)){
    t_operand weightNzValues [COMPRESSION_VEC_SIZE];
    uint1_t weightValidMasks [COMPRESSION_VEC_SIZE];
    unsigned short weightIndices [COMPRESSION_VEC_SIZE];

    t_operand activationNzValues [COMPRESSION_VEC_SIZE];
    uint1_t activationValidMasks [COMPRESSION_VEC_SIZE];
    unsigned short activationIndices [COMPRESSION_VEC_SIZE];
} t_vecMultData;

typedef struct __attribute__((packed)) {
	unsigned short lastIndexWeight;
	unsigned short lastIndexActivation;
} t_dpInstruction;


//The width given to the FiFO counters need to
//match the size of the fifo closely
typedef struct {
	uint4_t pReadNow;
	uint4_t pWriteNext;
	t_vecSpValueAndZCount regs[PE_VEC_FIFO_SIZE];
} t_vec_fifo;

bool checkVecFIFOFull (t_vec_fifo * pFifo);

bool checkVecFIFOEmpty (t_vec_fifo * pFifo);

t_vecSpValueAndZCount peekVecFIFO (t_vec_fifo *pFifo);

void popVecFIFO (t_vec_fifo *pFifo);

void pushVecFIFO (t_vec_fifo *pFifo, t_vecSpValueAndZCount data);

bool writeNonBlockVecFIFO (t_vec_fifo *pFifo, t_vecSpValueAndZCount data);

bool readNonBlockVecFIFO (t_vec_fifo *pFifo, t_vecSpValueAndZCount *pData);

typedef struct {
	uint4_t pReadNow;
	uint4_t pWriteNext;
	t_spValueAndZCount regs[PE_VEC_FIFO_SIZE];
} t_fifo;

bool checkFIFOFull (t_fifo * pFifo);

bool checkFIFOEmpty (t_fifo * pFifo);

t_spValueAndZCount peekFIFO (t_fifo *pFifo);


void popFIFO (t_fifo *pFifo);

void pushFIFO (t_fifo * pFifo, t_spValueAndZCount data);

bool writeNonBlockFIFO (t_fifo *pFifo, t_spValueAndZCount data);

/*!
 * \brief readNonBlockFIFO
 * \param pFifo
 * \param pData
 * \return
 */
bool readNonBlockFIFO (t_fifo *pFifo, t_spValueAndZCount *pData);

/*!
 * \brief decodeRunLength
 * \param pCompressionBlock
 * \param pUnpacked
 * \param startIndex
 * \details Decode a compressed block of NZ values
 * The runlength array is decoded into absolute indices.
 * After the conversion, pUnpacked->indices[i] equals to the index of corresponding non-zero element plus 1.
 */
void decodeRunLength (t_vecSpValueAndZCount* pCompressionBlock
    , t_vecUnpacked *pUnpacked
    , unsigned short startIndex);

/*!
 * \brief findMatchInUnpackedBlock
 * \param pUnpacked
 * \param targetIndex
 * \param pValueHolder
 * \return True if match if found, false otherwise
 */
bool findMatchInUnpackedBlock (
        t_vecUnpacked *pUnpacked,
        unsigned short targetIndex,
        t_operand *pValueHolder
        );

int convertSignedFixedPointToAccumulator(
        t_operand fixedPointValue,
        unsigned char fracWidthFixedPointValue
        );

t_operand convertAccumulatorToSignedFixedPoint(
		int accumulator,
		unsigned char fracWidthFixedPointValue
	);

#endif 
