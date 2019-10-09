#include "params.hpp"
#include "device_structures.hpp"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "channels.cl"


#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS

/*
Used to control whether the weight/activation transport should forward the block received
*/
/*
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
*/

/*
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
*/
#ifdef FLEXIBLE_BITMASK_COMPRESSION
//typedef struct __attribute__((packed)) {
//    char values [COMPRESSION_WINDOW_SIZE];
//} t_compression_window;

/*
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
*/
#endif //FLEXIBLE_BITMASK_COMPRESSION

/*
typedef struct __attribute__((packed)){
	t_simd_operand weights;
	t_simd_operand activations;
	uint1_t isLast;
} t_mac_operands;
*/


/*
#ifdef DIRECT_COMPRESSION_SIMD
channel t_simdblock_bitmask_tagged channel_activationInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_activationOutput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightOutput __attribute__((depth(1)));
#endif
*/

typedef struct __attribute__((packed)) {
	unsigned char numPSumToProcess;
	unsigned char numBitsToRightShift;
	uint1_t enableRelu;
} t_outputInstruction;

#ifdef FLEXIBLE_BITMASK_COMPRESSION
//channel t_transferblock_tagged channel_activationInput __attribute__((depth(1)));
//channel t_transferblock_tagged channel_activationOutput __attribute__((depth(1)));
//channel t_transferblock_tagged channel_weightInput __attribute__((depth(1)));
//channel t_transferblock_tagged channel_weightOutput __attribute__((depth(1)));
#endif
//channel t_operand channel_biasInput __attribute__((depth(1)));
//channel t_operand channel_biasOutput __attribute__((depth(1)));
//channel t_accumulator channel_drainInput __attribute__((depth(1)));
//channel t_accumulator channel_drainOutput __attribute__((depth(1)));
channel t_operand channel_processedDrain __attribute__((depth(0)));


channel t_outputInstruction channel_outputInstruction __attribute__((depth(0)));

/*
#ifdef DIRECT_COMPRESSION_SIMD
channel t_simdblock_bitmask channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_simdblock_bitmask channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
#endif
*/

#ifdef FLEXIBLE_BITMASK_COMPRESSION
//channel t_transferblock_local channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
//channel t_transferblock_local channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));

//channel t_alignment_input channel_alignmentInput __attribute__((depth(0)));
//channel t_alignment_output channel_alignmentOutput __attribute__((depth(0)));

#endif

//channel t_accumulator channel_peDrainOutput __attribute__((depth(0)));



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
			t_accumulator accumulator = read_channel_intel(channel_drain[0][0]);
			
			t_operand result = modifyOutput(accumulator, rndRightShift, enableRelu);
			
			write_channel_intel(channel_processedDrain, result);
			
			countPSum++;		
		}
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
		volatile __global short * restrict pDrainIn, 
		volatile __global char * restrict pDrainOut,
		short bias,
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

			valid = write_channel_nb_intel (channel_activation[0][0], taggedBlock);
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
				taggedBlock = read_channel_nb_intel(channel_activation[0][1], &valid);
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

			valid = write_channel_nb_intel (channel_weight[0][0], taggedBlock);
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

				taggedBlock = read_channel_nb_intel(channel_weight[0][1], &valid);

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
			short value = pDrainIn [countInputDrain];
			valid = write_channel_nb_intel (channel_drain[1][0], value);
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