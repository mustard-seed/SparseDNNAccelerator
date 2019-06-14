#include "peComponents.hpp"

bool checkFIFOFull (t_fifo * pFifo) {
	//FIFO is full if pWriteNext + 1 == pReadNow
	//Wrap around is handled by the match between the bit width and number of elements
	//i.e. 4'b1111 + 1 = 4'b0000

	return (pFifo->pWriteNext + 0x01 == pFifo->pReadNow);
}

bool checkFIFOEmpty (t_fifo * pFifo) {

	return (pFifo->pWriteNext == pFifo->pReadNow);
}

t_spWeightAndOffset peekFIFO (t_fifo *pFifo) {
	//Doesn't perform empty check. User should do it first

	return pFifo->regs[pFifo->pReadNow];
}


void popFIFO (t_fifo *pFifo) {
	//Doesn't perform empty check. User should do it first
	pFifo->pReadNow += 1;
}

void pushFIFO (t_fifo * pFifo, t_spWeightAndOffset data) {
	pFifo->regs[pFifo->pWriteNext] = data;
	pFifo->pWriteNext += 1;
}
