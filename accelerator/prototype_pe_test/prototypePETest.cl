#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"
#include "rtl_lib.hpp"


#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS

/*
Used to control whether the weight/activation transport should forward the block received
*/
typedef struct __attribute__((packed)) {

  	//Whether to forward teh activation block to the compute unit / pSum manager
  	uint2_t forwardEnable;
} t_convBlockTransportInstruction;

typedef struct __attribute__((packed)) {
	uint2_t biasForwardEnable;
	unsigned char numPSumToSend;
	unsigned char accumulatorFracWidth;
	unsigned char biasFracWidth;
	unsigned char outputFracWidth;
} t_drainTransportInstruction;


typedef struct __attribute__((packed)) {
	unsigned char numPSumToProcess;
	unsigned char numBitsToRightShift;
	uint1_t enableRelu;
} t_outputInstruction;


#ifdef DIRECT_COMPRESSION_SIMD
//With the final array
typedef struct __attribute__((packed)){
	char values [SIMD_SIZE];
	uint6_t streamingBlockIndex;
	uint1_t isLast;
} t_simdblock_di;

typedef struct __attribute__((packed)){
	t_simdblock_value values;
	uint1_t isLast;
} t_simdblock_bitmask;
#endif

#ifdef FLEXIBLE_BITMASK_COMPRESSION
typedef struct __attribute__((packed)) {
    char values [COMPRESSION_WINDOW_SIZE];
} t_compression_window;

typedef struct __attribute__((packed)){
    t_compression_window weightWindow;
    t_compression_window activationWindow;
    unsigned char bitmaskW;
    unsigned char bitmaskA;
    bool isLast;
} t_alignment_input;

typedef struct __attribute__((packed)){
    t_compression_window weightWindow;
    t_compression_window activationWindow;
    unsigned long alignmentData;
    bool isLast;
} t_alignment_output;
#endif //FLEXIBLE_BITMASK_COMPRESSION

//MAC Operands
typedef struct __attribute__((packed)) {
	char values [SIMD_SIZE*CLUSTER_SIZE];
} t_simd_operand;

typedef struct __attribute__((packed)){
	t_simd_operand weights;
	t_simd_operand activations;
	uint1_t isLast;
} t_mac_operands;

t_accumulator madd (t_simd_operand activations, t_simd_operand weights) {
	t_accumulator output = 0x0;

	//#ifdef DIRECT_COMPRESSION_SIMD
		#pragma unroll
		for(int i=0; i<SIMD_SIZE*CLUSTER_SIZE/4; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += a10_mac_8bitx4(
					activations.values[i*4],
					weights.values[i*4],
					activations.values[i*4+1],
					weights.values[i*4+1],
					activations.values[i*4+2],
					weights.values[i*4+2],
					activations.values[i*4+3],
					weights.values[i*4+3]
					);
			#elif defined (C5SOC)
				output += c5_mac_8bitx4(
						activations.values[i*4],
						weights.values[i*4],
						activations.values[i*4+1],
						weights.values[i*4+1],
						activations.values[i*4+2],
						weights.values[i*4+2],
						activations.values[i*4+3],
						weights.values[i*4+3]
						);
			#else
			#error Unsupported FPGA type!
			#endif
		}
	//#endif
	//#ifdef FLEXIBLE_BITMASK_COMPRESSION
	/*
		#pragma unroll
		for(int i=0; i<SIMD_SIZE/2; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += a10_mac_8bitx2(
					activations.values[i*2],
					weights.values[i*2],
					activations.values[i*2+1],
					weights.values[i*2+1]
					);
			#elif defined (C5SOC)
				output += c5_mac_8bitx2(
						activations.values[i*2],
						weights.values[i*2],
						activations.values[i*2+1],
						weights.values[i*2+1]
					);
			#else
			#error Unsupported FPGA type!
			#endif
		}
		*/
	//#endif

	return output;
}


#ifdef DIRECT_COMPRESSION_SIMD
channel t_simdblock_bitmask_tagged channel_activationInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_activationOutput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightOutput __attribute__((depth(1)));
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
channel t_transferblock_tagged channel_activationInput __attribute__((depth(1)));
channel t_transferblock_tagged channel_activationOutput __attribute__((depth(1)));
channel t_transferblock_tagged channel_weightInput __attribute__((depth(1)));
channel t_transferblock_tagged channel_weightOutput __attribute__((depth(1)));
#endif
//channel t_operand channel_biasInput __attribute__((depth(1)));
//channel t_operand channel_biasOutput __attribute__((depth(1)));
channel t_accumulator channel_drainInput __attribute__((depth(1)));
channel t_accumulator channel_drainOutput __attribute__((depth(1)));
channel t_operand channel_processedDrain __attribute__((depth(0)));


channel t_outputInstruction channel_outputInstruction __attribute__((depth(0)));

#ifdef DIRECT_COMPRESSION_SIMD
channel t_simdblock_bitmask channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_simdblock_bitmask channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
#endif

#ifdef FLEXIBLE_BITMASK_COMPRESSION
channel t_transferblock_local channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_transferblock_local channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));

channel t_alignment_input channel_alignmentInput __attribute__((depth(0)));
channel t_alignment_output channel_alignmentOutput __attribute__((depth(0)));

#endif

channel t_accumulator channel_peDrainOutput __attribute__((depth(0)));



__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelOutputModifier (
	) 
{

	while (true)
	{
		//Decode the instruction
		t_outputInstruction instruction = read_channel_intel(channel_outputInstruction);
		unsigned char numPSum = instruction.numPSumToProcess;
		unsigned char rndRightShift = instruction.numBitsToRightShift - 1;
		uint1_t enableRelu = instruction.enableRelu;

		unsigned char countPSum = 0;
		while (countPSum < numPSum)
		{
			t_accumulator accumulator = read_channel_intel(channel_drainOutput);
			
			t_operand result = modifyOutput(accumulator, rndRightShift, enableRelu);
			
			write_channel_intel(channel_processedDrain, result);
			
			countPSum++;		
		}
	}
}


__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelWeightTransport (
	)
{
	

	int idx = IDX;
	int idy = IDY;

	//t_simdblock_di_tagged block = read_channel_intel(channel_weightInput);
	//t_simdblock_di peBlock;
	#ifdef DIRECT_COMPRESSION_SIMD
	t_simdblock_bitmask_tagged block;
	t_simdblock_bitmask peBlock;
	#endif

	#ifdef FLEXIBLE_BITMASK_COMPRESSION
	t_transferblock_tagged block;
	t_transferblock_local peBlock;
	#endif

	block = read_channel_intel(channel_weightInput);
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		#ifdef DIRECT_COMPRESSION_SIMD
			peBlock.values.values[i] = block.values[i];
		#endif
		#ifdef FLEXIBLE_BITMASK_COMPRESSION
			peBlock.values.values[i] = block.values.values[i];
		#endif
	}
	//peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idx < (PE_COLS - 1)){
		if ( idx < block.maxTransportID ) {
			//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
			write_channel_intel(channel_weightOutput, block);
		}
	}

	write_channel_intel(channel_dpWeightInput, peBlock); 
}

#define STATE_ACTIVATION_TRANSPORT_READ 0X0
#define STATE_ACTIVATION_TRANSPORT_DRAIN_SELF 0x1
#define STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS 0x2

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelActivationTransport (
	)
{
	typedef uint2_t t_state;

	int idx = IDX;
	int idy = IDY;

	t_state state = STATE_ACTIVATION_TRANSPORT_READ;
	unsigned char numOtherPSumToDrain;
	unsigned char countOtherPSum;

	while (true)
	{
		t_state nextState = state;
		t_accumulator pSum;
		if (state == STATE_ACTIVATION_TRANSPORT_READ)
		{
			#ifdef DIRECT_COMPRESSION_SIMD
			t_simdblock_bitmask_tagged block;
			t_simdblock_bitmask peBlock;
			#endif

			#ifdef FLEXIBLE_BITMASK_COMPRESSION
			t_transferblock_tagged block;
			t_transferblock_local peBlock;
			#endif

			block = read_channel_intel(channel_activationInput);
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				#ifdef DIRECT_COMPRESSION_SIMD
					peBlock.values.values[i] = block.values[i];
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					peBlock.values.values[i] = block.values.values[i];
				#endif
			}
			//peBlock.streamingBlockIndex = block.streamingBlockIndex;
			peBlock.isLast = block.isLast;

			if (idy < (PE_ROWS - 1)){
				if ( idy < block.maxTransportID ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass an activation block to the output\n") );
					write_channel_intel(channel_activationOutput, block);
				}
			}

			if (block.isLast == TRUE)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_DRAIN_SELF;	
				numOtherPSumToDrain = block.maxTransportID - idy;
				countOtherPSum = 0;
			}

			write_channel_intel(channel_dpActivationInput, peBlock); 

		} //STATE_ACTIVATION_TRANSPORT_READ
		else if (state == STATE_ACTIVATION_TRANSPORT_DRAIN_SELF)
		{
			pSum = read_channel_intel(channel_peDrainOutput);
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT] Drain from PE\n"));
			if (countOtherPSum == numOtherPSumToDrain)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_READ;
			}
			else
			{
				nextState = STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS;
			}
		} //STATE_ACTIVATION_TRANSPORT_DRAIN_SELF
		else if (state == STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS)
		{
			//TODO: change the following in deply
			pSum = read_channel_intel(channel_drainInput);
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT] Drain from Others\n"));
			countOtherPSum++;
			if (countOtherPSum == numOtherPSumToDrain)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_READ;
			} 
		} //STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS

		if ((state == STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS) 
			|| 
			(state == STATE_ACTIVATION_TRANSPORT_DRAIN_SELF))
		{
			write_channel_intel(channel_drainOutput, pSum);
		}

		state = nextState;
	}
}



__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelTestInterface (
		#ifdef DIRECT_COMPRESSION_SIMD
			volatile __global t_simdblock_host* restrict pActivationInput,
			volatile __global t_simdblock_host* restrict pActivationOutput,
			volatile __global t_simdblock_host* restrict pWeightInput,
			volatile __global t_simdblock_host* restrict pWeightOutput,
		#endif
		#ifdef FLEXIBLE_BITMASK_COMPRESSION
			volatile __global t_transfer_block* restrict pActivationInput,
			volatile __global t_transfer_block* restrict pActivationOutput,
			volatile __global t_transfer_block* restrict pWeightInput,
			volatile __global t_transfer_block* restrict pWeightOutput,
		#endif
		volatile __global int * restrict pDrainIn, 
		volatile __global char * restrict pDrainOut,
		int bias,
		volatile __global t_output_instruction_host* restrict pOutputInstruction,
		unsigned short numInputActivationBlocks,
		unsigned short numOutputActivationBlocks,
		unsigned short numInputWeightBlocks,
		unsigned short numOutputWeightBlocks,
		unsigned short numInputDrain,
		unsigned short numOutputDrain,

		unsigned char maxIDX,
		unsigned char maxIDY
	)
{
	unsigned short countInputActivationBlocks = 0,
				   countOutputActivationBlocks = 0,
				   countInputWeightBlocks = 0,
				   countOutputWeightBlocks = 0,
				   countInputDrain = 0,
				   countOutputDrain = 0,
				   countOutputInstruction = 0;

	//uint5_t		   indexActivationTracker = 0;
	//uint5_t		   indexWeightTracker = 0;
	int idx = IDX;
	int idy = IDY;

	while (
			(countOutputActivationBlocks < numOutputActivationBlocks)
			||
			(countOutputWeightBlocks < numOutputWeightBlocks)
			||
			(countOutputDrain < numOutputDrain)
		) {

		if (countInputActivationBlocks < numInputActivationBlocks) {
			bool valid;
			#ifdef DIRECT_COMPRESSION_SIMD
				t_simdblock_host block;
				t_simdblock_bitmask_tagged taggedBlock;
			#endif
			#ifdef FLEXIBLE_BITMASK_COMPRESSION
				t_transfer_block block;
				t_transferblock_tagged taggedBlock;
			#endif
			
			block = pActivationInput[countInputActivationBlocks];

			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values.values[i] = block.values[i];
			}

			//uint6_t indexInStreamingBlock = indexActivationTracker + (uint6_t) block.runLength; 
			uint1_t isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
				TRUE : FALSE;
            //taggedBlock.streamingBlockIndex = indexInStreamingBlock;
            taggedBlock.isLast = isLast;;
            taggedBlock.maxTransportID = maxIDY;

			valid = write_channel_nb_intel (channel_activationInput, taggedBlock);
			if (valid) {
				countInputActivationBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				//indexActivationTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			if (countOutputActivationBlocks < numOutputActivationBlocks) {
				bool valid;
				#ifdef DIRECT_COMPRESSION_SIMD
					t_simdblock_host block;
					t_simdblock_bitmask_tagged taggedBlock;
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					t_transfer_block block;
					t_transferblock_tagged taggedBlock;
				#endif
				taggedBlock = read_channel_nb_intel(channel_activationOutput, &valid);
				if (valid) {
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						block.values[i] = taggedBlock.values.values[i];
					}
					//hostBlock.runLength = block.streamingBlockIndex;
					//pActivationOutput[countOutputActivationBlocks++] = block;
					countOutputActivationBlocks++;
					EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
					//DEBUG_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
				}
			}
		}

		//Need an extra one for bias
		if (countInputWeightBlocks < (numInputWeightBlocks+1)) {
			bool valid;

			#ifdef DIRECT_COMPRESSION_SIMD
				t_simdblock_host block;
				t_simdblock_bitmask_tagged taggedBlock;
			#endif
			#ifdef FLEXIBLE_BITMASK_COMPRESSION
				t_transfer_block block;
				t_transferblock_tagged taggedBlock;
			#endif

			block = (countInputWeightBlocks == 0) ?
				bias2TransferBlcok( (t_accumulator) bias)
				: pWeightInput[countInputWeightBlocks-1];

			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values.values[i] = block.values[i];
			}

			//Need an extra one for bias
			uint1_t isLast = (countInputWeightBlocks == (numInputWeightBlocks)) ? 
				TRUE : FALSE;
            //taggedBlock.streamingBlockIndex = indexInStreamingBlock;
            taggedBlock.isLast = isLast;;
            taggedBlock.maxTransportID = maxIDX;

			valid = write_channel_nb_intel (channel_weightInput, taggedBlock);
			if (valid) {
				countInputWeightBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				//indexWeightTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		//Need an extra one for bias
		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			if (countOutputWeightBlocks < numOutputWeightBlocks+1) {
				bool valid;
				#ifdef DIRECT_COMPRESSION_SIMD
					t_simdblock_host block;
					t_simdblock_bitmask_tagged taggedBlock;
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					t_transfer_block block;
					t_transferblock_tagged taggedBlock;
				#endif

				taggedBlock = read_channel_nb_intel(channel_weightOutput, &valid);

				if (valid) {
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						block.values[i] = taggedBlock.values.values[i];
					}
					//hostBlock.runLength = block.streamingBlockIndex;
					//pWeightOutput[countOutputWeightBlocks++] = block;
					countOutputWeightBlocks++;
					EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
					//DEBUG_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
				}
			}
		}

		if (countInputDrain < numInputDrain) {
			bool valid;
			int value = pDrainIn [countInputDrain];
			valid = write_channel_nb_intel (channel_drainInput, value);
			if (valid) {
				countInputDrain++;
				EMULATOR_PRINT ( ("[kernelTestInferace]: Sent %d out of %d pSums\n", countInputDrain, numInputDrain) );
			}
		}

		if (countOutputDrain < numOutputDrain) {
			bool valid;
			char value = read_channel_nb_intel(channel_processedDrain, &valid);
			if (valid) {
				pDrainOut [countOutputDrain] = (char) value;
				countOutputDrain++;
			}
		}

		if (countOutputInstruction < 1)
		{
			t_outputInstruction instruction;
			t_output_instruction_host hostOutputInstruction = pOutputInstruction[0];
			instruction.numPSumToProcess = hostOutputInstruction.numPSumToProcess;
			instruction.numBitsToRightShift = hostOutputInstruction.numBitsToRightShift;
			instruction.enableRelu = hostOutputInstruction.enableRelu ? TRUE : FALSE;
			bool valid = write_channel_nb_intel(channel_outputInstruction, instruction);
			if (valid)
			{
				countOutputInstruction++;
			}
		}
		
	}
}

#ifdef FLEXIBLE_BITMASK_COMPRESSION
#define ASSEMBLER_STATE_LOAD_BITMASK 0X0
#define ASSEMBLER_STATE_LOAD_VALUE 0X1
//#define ASSEMBLER_STATE_ALIGN 0x2
#define ASSEMBLER_STATE_WAIT 0x2
#define ASSEMBLER_STATE_LOAD_BIAS 0x3

#define BITWIDTH_COMPRESSION_WINDOW_INDEX 3
#define MASK_COMPRESSION_WINDOW_INDEX 0x7

#define MAC_STATE_WAIT 0x0
#define MAC_STATE_ALIGN 0x1
#define MAC_STATE_PROCESS_WINDOW 0x2
#define MAC_STATE_WRITE_PSUM 0x3
#define MAC_STATE_LOAD_BIAS 0x4

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void kernelPE ()
{
	//================Ping-ponged registers========================
	//BRAM for storing the compression windows
	t_cluster activationWindow[COMPRESSION_WINDOW_SIZE+1][2]; 
	t_cluster weightWindow[COMPRESSION_WINDOW_SIZE+1][2]; 

	//Flags that indicates whether we are at the last window
	uint1_t isLast[2] = {FALSE, TRUE};
	unsigned char bitmaskA[2];
	unsigned char bitmaskW[2];
	t_accumulator bias[2];

	uint1_t regLoadSide = 0x0;

	//========Assembler side registers====================
	unsigned char countActivation;
	unsigned char countWeight;
	unsigned char numActivation;
	unsigned char numWeight;
	uint2_t stateActivation = ASSEMBLER_STATE_LOAD_BIAS;
	uint2_t stateWeight = ASSEMBLER_STATE_LOAD_BIAS;
	//unsigned long alignmentData;


	//=========MAC side logic========================
	uint3_t stateMac = MAC_STATE_WAIT;
	t_accumulator pSum = 0;
	unsigned char countOperands;
	unsigned char numOperands;
	unsigned int indicesW;
	unsigned int indicesA;

	//================Debug====================
	//unsigned short debugCount = 0;

	#pragma ivdep array(activationWindow)
	#pragma ivdep array(weightWindow)
	//#pragma ivdep safelen(7)
	//#pragma ivdep
	while (true)
	{

		//================ACTIVATION========================
		
		uint2_t nextStateActivation = stateActivation;
		{ 
			if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateActivation == ASSEMBLER_STATE_LOAD_VALUE)
			{
				t_transferblock_local activationTransferBlock;
				bool activationReadSuccess;

				activationTransferBlock = read_channel_nb_intel (
							channel_dpActivationInput,
							&activationReadSuccess
						);

				if (activationReadSuccess)
				{
					//isLastActivation = activationTransferBlock.isLast;
					//DEBUG_PRINT(("[Assembler] Activation read!\n"));

					if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK)
					{
						unsigned char bitmask = activationTransferBlock.values.values[0].cluster_values[0];
						bitmaskA[regLoadSide & 0x01] = bitmask;
						numActivation = popCounter(bitmask);
						countActivation = 0;
						EMULATOR_PRINT(("[assembler] bitmaskA: %#04x \n", bitmask));
					}
					//else
					//{

						//uint3_t offset = (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK) ?
						//	0X1 : 0X0; 

						#pragma unroll
						for (uint3_t i=0; i<TRANSFER_SIZE; i++)
						{
							//if (i >= offset)
							//{
								activationWindow[countActivation+i][regLoadSide & 0x01]
									= activationTransferBlock.values.values[i];
								//EMULATOR_PRINT(("[assembler] activation value: %#04x %#04x \n"
								//	, activationTransferBlock.values.values[i].cluster_values[0] & 0xFF
								//	, activationTransferBlock.values.values[i].cluster_values[1] & 0xFF));
								//EMULATOR_PRINT(("[assembler] activation offset, countActivation: %#04x %#04x\n"
								//	, offset, countActivation));
							//}
						} // for. Transfer the values in the transfer block to the compression window

						//if (debugCount < maxDebugCount)
						//{
							DEBUG_PRINT(("[PE] ActivationTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
								activationTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
								activationTransferBlock.values.values[0].cluster_values[1] & 0xFF,
								activationTransferBlock.values.values[1].cluster_values[0] & 0xFF,
								activationTransferBlock.values.values[1].cluster_values[1] & 0xFF));
						//}

						countActivation += (unsigned char)(TRANSFER_SIZE);
					//}

					//State update
					if (countActivation > numActivation) //countActivation needs to be strictly larger than numActivation
					{
						nextStateActivation = ASSEMBLER_STATE_WAIT;
					}
					else {
						nextStateActivation = ASSEMBLER_STATE_LOAD_VALUE;
					}

				} // if activationReadSuccess
			} // ASSEMBLER_STATE_LOAD_BITMASK || ASSEMBLER_STATE_LOAD_VALUE 
			else if (stateActivation == ASSEMBLER_STATE_LOAD_BIAS)
			{
				EMULATOR_PRINT(("[ACTIVATION ASSEMBLER] Wait for bias\n"));
				nextStateActivation = ASSEMBLER_STATE_WAIT;
			}
		}
		//===================================================

		//================WEIGHT========================
		
		uint2_t nextStateWeight = stateWeight;
		{
			bool weightReadSuccess;
			t_transferblock_local weightTransferBlock;

			if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE
				|| stateWeight == ASSEMBLER_STATE_LOAD_BIAS) 
			{
				weightTransferBlock = read_channel_nb_intel (
							channel_dpWeightInput,
							&weightReadSuccess
						);
			}

			if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE)
			{
				if (weightReadSuccess)
				{
					isLast[regLoadSide & 0x01] = weightTransferBlock.isLast;
					//DEBUG_PRINT(("[Assembler] Weight read!\n"));

					if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK)
					{
						unsigned char bitmask =  weightTransferBlock.values.values[0].cluster_values[0];
						bitmaskW[regLoadSide & 0x01] = bitmask; 
						numWeight = popCounter(bitmask);
						countWeight = 0;
						EMULATOR_PRINT(("[assembler] bitmaskW: %#04x \n", bitmask));
					}
					//else
					//{

						//uint3_t offset = (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK) ?
						//	0X1 : 0X0; 

						#pragma unroll
						for (uint3_t i=0; i<TRANSFER_SIZE; i++)
						{
							//if (i >= offset)
							//{
								weightWindow[countWeight+i][regLoadSide & 0x01]
									= weightTransferBlock.values.values[i];
								//EMULATOR_PRINT(("[assembler] weight value: %#04x %#04x \n"
								//	, weightTransferBlock.values.values[i].cluster_values[0] & 0xFF
								//	, weightTransferBlock.values.values[i].cluster_values[1] & 0xFF));
							//}
						} // for. Transfer the values in the transfer block to the compression window

						//if (debugCount < maxDebugCount)
						//{
							DEBUG_PRINT(("[PE] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
								weightTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
								weightTransferBlock.values.values[0].cluster_values[1] & 0xFF,
								weightTransferBlock.values.values[1].cluster_values[0] & 0xFF,
								weightTransferBlock.values.values[1].cluster_values[1] & 0xFF));
						//}

						countWeight += (unsigned char)(TRANSFER_SIZE);
					//}

					//State update
					if (countWeight > numWeight) //countWeight needs to be strictly larger than numWeight
					{
						nextStateWeight = ASSEMBLER_STATE_WAIT;
					}
					else 
					{
						nextStateWeight = ASSEMBLER_STATE_LOAD_VALUE;
					}

				} // if weightReadSuccess
			} //ASSEMBLER_STATE_LOAD_BITMASK || ASSEMBLER_STATE_LOAD_VALUE
			else if (stateWeight == ASSEMBLER_STATE_LOAD_BIAS)
			{
				if (weightReadSuccess)
				{
					EMULATOR_PRINT(("[WEIGHT ASSEMBLER] Wait for bias\n"));
					bias[regLoadSide & 0x01] = transferBlock2Bias(weightTransferBlock.values);
					nextStateWeight = ASSEMBLER_STATE_WAIT;
				}
			}
		}
		//===================================================

		//==================MAC states===================
		uint3_t nextStateMac = stateMac;

		if (stateMac == MAC_STATE_ALIGN)
		{
			unsigned long alignmentData = operandMatcher8(
				bitmaskW [(~regLoadSide) & 0x1],
				bitmaskA [(~regLoadSide) & 0x1]
			);
			numOperands = (alignmentData >> 48) & 0xFF;
			indicesW = (alignmentData >> 24) & 0xFFFFFF;
			indicesA = (alignmentData) & 0xFFFFFF;
			countOperands = 0; 
			//EMULATOR_PRINT ( ("[aligner]: indicesW: %#06x indicesA: %#06x numOperands: %#04x \n"
			//		, indicesW, indicesA,  numOperands) );

			/*
			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1])
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
			else
			{
				nextStateMac = MAC_STATE_PROCESS_WINDOW;
			}
			*/
			nextStateMac = MAC_STATE_PROCESS_WINDOW;
		}
		else if (stateMac == MAC_STATE_PROCESS_WINDOW)
		{

			t_simd_operand simdActivations;
			t_simd_operand simdWeights;
			t_cluster zeros;
			#pragma unroll
			for (int i=0; i<CLUSTER_SIZE; i++)
			{
				zeros.cluster_values[i] = 0x0;
			}


			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++)
			{
				unsigned char indexW = 
					(indicesW >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster w = ((countOperands + i) < numOperands) ?
					weightWindow[indexW+1][(~regLoadSide) & 0x01] : zeros;
				//char w = weightWindow[i][(~regLoadSide) & 0x1];
				//simdWeights.values[i] = w;

				unsigned char indexA = 
					(indicesA >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster a = ((countOperands + i) < numOperands) ?
					activationWindow[indexA+1][(~regLoadSide) & 0x01] : zeros;
				//char a = activationWindow[i][(~regLoadSide) & 0x1];

				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = a.cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = w.cluster_values[j];
				}

				//EMULATOR_PRINT ( ("[dispatcher]: w0: %#04x w1: %#04x a0: %#04x a1: %#04x \n"
				//	, w.cluster_values[0] & 0xFF, w.cluster_values[1] & 0xFF,  a.cluster_values[0] & 0xFF, a.cluster_values[1] & 0xFF) );
				//EMULATOR_PRINT ( ("[dispatcher]: wIndex: %u aIndex :%u \n", (indexW) & 0xFF, (indexA) & 0xFF));
			}


			t_accumulator tempPSum = madd(simdActivations, simdWeights);
			pSum += tempPSum;
			//if (debugCount < maxDebugCount)
			//	{
			//		DEBUG_PRINT(("[PE Dispatcher] a0, a1, a1, a2: %#04x %#04x %#04x %#04x\n",
			//			simdActivations.values[0] & 0xFF, 
			//			simdActivations.values[1] & 0xFF,
			//			simdActivations.values[2] & 0xFF,
			//			simdActivations.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Dispatcher] w0, w1, w2, w3: %#04x %#04x %#04x %#04x\n",
			//			simdWeights.values[0] & 0xFF, 
			//			simdWeights.values[1] & 0xFF,
			//			simdWeights.values[2] & 0xFF,
			//			simdWeights.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Madd] Psum %#04x\n", pSum));

			//	}		
			countOperands += SIMD_SIZE;
			indicesW = indicesW >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);
			indicesA = indicesA >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);

			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1] == TRUE)
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
		} // if state == MAC_STATE_PROCESS_WINDOW
		else if (stateMac == MAC_STATE_WRITE_PSUM)
		{
			bool writeSuccess;
			writeSuccess = write_channel_nb_intel(channel_peDrainOutput, pSum);

			//write_channel_intel(channel_peDrainOutput, pSum);
			if (writeSuccess)
			{
				//DEBUG_PRINT(("[MAC] Sending!\n"));
				EMULATOR_PRINT(("[MAC] Commit. pSum value: %#04x \n", pSum));
				//DEBUG_PRINT(("[PE Psum] Commit. %#04x\n", pSum));
				//pSum = 0;
				nextStateMac = MAC_STATE_WAIT;
				//pSum = 0;
			}
		}
		else if (stateMac == MAC_STATE_LOAD_BIAS)
		{
			EMULATOR_PRINT(("[MAC] Load Bias\n"));
			pSum = bias[(~regLoadSide) & 0x1];
			nextStateMac = MAC_STATE_WAIT;
		}


	//===================SWAP===========================
	//Take an extra iteration for swapping, otherwise Fmax is low
		if ( (stateActivation == ASSEMBLER_STATE_WAIT)
			&& (stateWeight == ASSEMBLER_STATE_WAIT)
			&& (stateMac == MAC_STATE_WAIT) )
		{
			nextStateWeight = (isLast[(regLoadSide) & 0x1] == TRUE) ? 
				ASSEMBLER_STATE_LOAD_BIAS : ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateActivation = (isLast[(regLoadSide) & 0x1] == TRUE) ? 
				ASSEMBLER_STATE_LOAD_BIAS : ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateMac = (isLast[(~regLoadSide) & 0x1] == TRUE) ?
				MAC_STATE_LOAD_BIAS: MAC_STATE_ALIGN;

			regLoadSide = ~regLoadSide;
			//countActivation = 0;
			//countWeight = 0;

		}

		//================DEBUG==============================
		//if (debugCount < maxDebugCount)
		//{
		//	DEBUG_PRINT(("[PE] countWeight, %#03x\n", countWeight));
		//	DEBUG_PRINT(("[PE] countActivation: %#03x\n", countActivation));
		//	DEBUG_PRINT(("[PE] countOperands: %#03x\n", countOperands));
		//	DEBUG_PRINT(("[PE] indicesW: %#03x\n", indicesW));
		//	DEBUG_PRINT(("[PE] indicesA: %#03x\n", indicesA));
		//	debugCount++;
		//}
		
		//===================================================

		//================Next state update==================
		stateWeight = nextStateWeight;
		stateActivation = nextStateActivation;
		stateMac = nextStateMac;
		//===================================================
	} // while true
} // end of kernel
#endif