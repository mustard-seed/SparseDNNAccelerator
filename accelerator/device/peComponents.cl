#include "peComponents.hpp"


bool checkVecFIFOFull (t_vec_fifo * pFifo) {
	return (pFifo->pWriteNext + 0x01 == pFifo->pReadNow);
}

bool checkVecFIFOEmpty (t_vec_fifo * pFifo) {
	return (pFifo->pWriteNext == pFifo->pReadNow);
}

t_vecSpValueAndZCount peekVecFIFO (t_vec_fifo *pFifo) {
	//Doesn't perform empty check. User should do it first

	return pFifo->regs[pFifo->pReadNow];
}

void popVecFIFO (t_vec_fifo *pFifo) {
	pFifo->pReadNow += 1;
}

void pushVecFIFO (t_vec_fifo *pFifo, t_vecSpValueAndZCount data) {
	pFifo->regs[pFifo->pWriteNext] = data;
	pFifo->pWriteNext += 1;
}

bool writeNonBlockVecFIFO (t_vec_fifo *pFifo, t_vecSpValueAndZCount data) {
	if (checkVecFIFOFull(pFifo)) {
		return false;
	}
	else {
		pushVecFIFO(pFifo, data);
		return true;
	}
}

bool readNonBlockVecFIFO (t_vec_fifo *pFifo, t_vecSpValueAndZCount *pData) {
	if (checkVecFIFOEmpty(pFifo)) {
		return false;
	} 
	else {
		*pData = peekVecFIFO(pFifo);
		popVecFIFO(pFifo);
		return true;
	}
}


bool checkFIFOFull (t_fifo * pFifo) {
	//FIFO is full if pWriteNext + 1 == pReadNow
	//Wrap around is handled by the match between the bit width and number of elements
	//i.e. 4'b1111 + 1 = 4'b0000

	return (pFifo->pWriteNext + 0x01 == pFifo->pReadNow);
}

bool checkFIFOEmpty (t_fifo * pFifo) {

	return (pFifo->pWriteNext == pFifo->pReadNow);
}

t_spValueAndZCount peekFIFO (t_fifo *pFifo) {
	//Doesn't perform empty check. User should do it first

	return pFifo->regs[pFifo->pReadNow];
}


void popFIFO (t_fifo *pFifo) {
	//Doesn't perform empty check. User should do it first
	pFifo->pReadNow += 1;
}

void pushFIFO (t_fifo * pFifo, t_spValueAndZCount data) {
	pFifo->regs[pFifo->pWriteNext] = data;
	pFifo->pWriteNext += 1;
}

bool writeNonBlockFIFO (t_fifo *pFifo, t_spValueAndZCount data) {
	if (checkFIFOFull(pFifo)) {
		return false;
	}
	else {
		pushFIFO(pFifo, data);
		return true;
	}

}

bool readNonBlockFIFO (t_fifo *pFifo, t_spValueAndZCount *pData) {
	if (checkFIFOEmpty(pFifo)) {
		return false;
	} 
	else {
		*pData = peekFIFO(pFifo);
		popFIFO(pFifo);
		return true;
	}
}

void decodeRunLength (t_vecSpValueAndZCount* pCompressionBlock
    ,t_vecUnpacked * pUnpacked
    ,unsigned short startIndex) {

    pUnpacked->indices[0] = pUnpacked->validMasks[0] ?
            1 + startIndex + (unsigned short) ( (pCompressionBlock->vec[0] & WEIGHT_ZCOUNT_MASK) >> WEIGHT_ZCOUNT_BITOFFSET ) :
            startIndex;

    //Transfer the values
    #pragma unroll
    for (unsigned char i=0; i<COMPRESSION_VEC_SIZE; i++) {
        pUnpacked->nzValues[i & 0x3] = (t_operand) ( (pCompressionBlock->vec[i & 0x3]) & WEIGHT_MASK);
        pUnpacked->validMasks[i & 0x3] = (uint1_t) ((pCompressionBlock->vec[i & 0x3] & WEIGHT_VALID_MASK) >> WEIGHT_VALID_BITOFFSET);
        //pUnpacked->indices[i & 0x3] = (pCompressionBlock->vec[i & 0x3] & WEIGHT_VALID_MASK) > 0 ? 1 : 0;
        //pUnpacked->indices[i] = pUnpacked->validMasks[i] ? 1 : 0;
        
        if (i>0) {
        	pUnpacked->indices[i & 0x3] = pUnpacked->validMasks[i & 0x3] ?
                        1 + pUnpacked->indices[(i-1) & 0x3] + (unsigned short) ( (pCompressionBlock->vec[i & 0x3] & WEIGHT_ZCOUNT_MASK) >> WEIGHT_ZCOUNT_BITOFFSET ) :
                        pUnpacked->indices[(i-1) & 0x3];            
        }
    }

}

bool findMatchInUnpackedBlock (
        t_vecUnpacked *pUnpacked,
        unsigned short targetIndex,
        t_operand *pValueHolder
        ) {
    bool returnVal;

    if ( ( pUnpacked->indices[0] <= (0x1FFFF & ((0xFFFF & targetIndex) + (0xFFFF & 1)) ) ) 
    	&& ( pUnpacked->indices[COMPRESSION_VEC_SIZE-1 ]  >= (0x1FFFF & ((0xFFFF & targetIndex) + (0xFFFF & 1)) ) ) ) {
    	returnVal = true;
    }
    else {
    	returnVal = false;
    }

	t_operand value = 0x0;
	#pragma unroll
	for (uint4_t i=0; i<COMPRESSION_VEC_SIZE; i++) {
        bool localMatch = (pUnpacked->indices[i] == (unsigned short) (targetIndex+ (unsigned short) 1) );
        if (localMatch) {
        	value = pUnpacked->nzValues[i];
        }
	}
	*pValueHolder = value;

    return returnVal;

}

int convertSignedFixedPointToAccumulator(
        t_operand fixedPointValue,
        unsigned char fracWidthFixedPointValue
        ) {
    //int temp = (int) ( fixedPointValue & WEIGHT_MASK);
    int tempSignExtended = (int) (fixedPointValue);
    int returnVal = tempSignExtended
            << (unsigned char)(REG_FF_FRAC - fracWidthFixedPointValue);
    return returnVal;
}

t_operand convertAccumulatorToSignedFixedPoint(
		int accumulator,
		unsigned char fracWidthFixedPointValue
	) {
	//See if sign extension is required
	int signExtensionMask = (accumulator >= 0) ?
		0x0 :
		~(0xFFFFFFFF >> ( (unsigned char)( REG_FF_FRAC - fracWidthFixedPointValue)) );

	//Match the binary point
	int fpValueWide = signExtensionMask | ( accumulator >> (unsigned char)( (unsigned char) REG_FF_FRAC - fracWidthFixedPointValue) );

	//Clip from above and below
	int fpValueClipped;
	if (fpValueWide > (unsigned char) WEIGHT_MAX) {
		fpValueClipped = WEIGHT_MAX;
	}
	else if (fpValueWide < (unsigned char) WEIGHT_MIN) {
		fpValueClipped = WEIGHT_MIN;
	}
	else {
		fpValueClipped = fpValueWide;
	}

	//Finally, return the value
	return ((t_operand) (fpValueClipped & WEIGHT_MASK));
}