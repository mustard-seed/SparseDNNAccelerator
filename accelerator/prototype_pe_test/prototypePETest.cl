#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"


__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelPrototypePE (
		__global t_spValueAndZCount* restrict pActivationInput,
		__global t_spValueAndZCount* restrict pActivationOutput,
		__global t_spValueAndZCount* restrict pWeightInput,
		__global t_spValueAndZCount* restrict pWeightOutput,
		__global short * restrict pBiasIn,
		__global short * restrict pBiasOut,
		__global t_pe_prototype_instruction* restrict pInstruction,
		unsigned int numInstructions,
		int idx,
		int idy

	)
{
	//Declare varibles and regisetrs that will be used
	//State register
	uint4_t state = PE_STATE_IDLE;
	uint4_t next_state = PE_STATE_IDLE;
	
	//Dense index tracker used in MAC reduction
	unsigned short weightIndex = 0;
	unsigned short activationIndex = 0;

	//Memory counters used for this test
	int counterActivationInput=0;
	int counterActivationOutput=0;
	int counterWeightInput=0;
	int counterWeightOutput=0;
	int counterBiasInput=0;
	int counterBiasOutput=0;
	int counterInstruction=0;

	//Register that holds the partial sum
	//Should be interpreted as a signed fixed-point number
	//Fraction width = REG_FF_FRAC. Total width = PE_FF_WIDTH
	int pSumFF;

	//Fifos
	t_fifo fifoWeight = {.pReadNow=0x0, .pWriteNext=0x0};
	t_fifo fifoActivation = {.pReadNow=0x0, .pWriteNext=0x0};

	checkFIFOFull(&fifoActivation);

	//Continuation flag used for this test
	uint1_t run = 0x1;

/*
	while (run) {
		//Generate enable signals based on states

		//Perform actions

		//State transitions
	}
*/

}