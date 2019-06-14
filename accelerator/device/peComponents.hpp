#ifndef PE_COMPONENT_HPP
#define PE_COMPONENT_HPP
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"

//The width given to the FiFO counters need to
//match the size of the fifo closely
typedef struct {
	uint4_t pReadNow;
	uint4_t pWriteNext;
	t_spWeightAndOffset regs[16];
} t_fifo;

bool checkFIFOFull (t_fifo * pFifo);

bool checkFIFOEmpty (t_fifo * pFifo);

t_spWeightAndOffset peekFIFO (t_fifo *pFifo);

void popFIFO (t_fifo *pFifo);

void pushFIFO (t_fifo *pFifo, t_spWeightAndOffset data);

#endif 