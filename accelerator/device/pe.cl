#if defined(SPW_TEST)
#include "spw_pe_test_types.hpp"
#include "spw_pe_test_channels.hpp"
#else
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.hpp"
#endif
#include "prints.hpp"

#include "ihc_apint.h"
#include "rtl_lib.hpp"


typedef struct __attribute__((packed)) {
	char values [PE_SIMD_SIZE * CLUSTER_SIZE];
} t_simd_operand;

t_accumulator madd (t_simd_operand activations, t_simd_operand weights) {
	t_accumulator output = 0x00 & MULT_MASK;

	//#ifdef DIRECT_COMPRESSION_SIMD
		#pragma unroll
		for(int i=0; i<PE_SIMD_SIZE*CLUSTER_SIZE/4; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += MULT_MASK & ((t_accumulator) a10_mac_8bitx4_input_registered(
					activations.values[i*4],
					weights.values[i*4],
					activations.values[i*4+1],
					weights.values[i*4+1],
					activations.values[i*4+2],
					weights.values[i*4+2],
					activations.values[i*4+3],
					weights.values[i*4+3]
					));
			#elif defined (C5SOC)
				output += MULT_MASK & ((t_accumulator) c5_mac_8bitx4_input_registered(
						activations.values[i*4],
						weights.values[i*4],
						activations.values[i*4+1],
						weights.values[i*4+1],
						activations.values[i*4+2],
						weights.values[i*4+2],
						activations.values[i*4+3],
						weights.values[i*4+3]
						));
			#else
			#error Unsupported FPGA type!
			#endif
		}

	return output;
}

t_accumulator chain_maddx8 (t_simd_operand activations, t_simd_operand weights) {	
	t_accumulator output = 0x00 & MULT_MASK;

	#if defined (ARRIA10)
		#pragma unroll
		for(int i=0; i<PE_SIMD_SIZE*CLUSTER_SIZE/8; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
				output += MULT_MASK & ((t_accumulator) a10_chain_madd_8bitx8(
						activations.values[i*8+0],
						weights.values[i*8+0],
						activations.values[i*8+1],
						weights.values[i*8+1],
						activations.values[i*8+2],
						weights.values[i*8+2],
						activations.values[i*8+3],
						weights.values[i*8+3],
						activations.values[i*8+4],
						weights.values[i*8+4],
						activations.values[i*8+5],
						weights.values[i*8+5],
						activations.values[i*8+6],
						weights.values[i*8+6],
						activations.values[i*8+7],
						weights.values[i*8+7]
					));
		}
		#else
	#error Unsupported FPGA type!
	#endif

	return output;
}


//__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
__attribute__ ((autorun))
__kernel void kernelWeightTransport (
	)
{
	
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);

	while (true)
	{

			EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Waiting weight/bias transfer block.\n", idy, idx));

			t_pe_w_block block;
			block = read_channel_intel(channel_weight[idy][idx]);

			EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Read weight/bias transfer block. IsLast: %#04x, maxTransportID: %#04x\n", 
				idy, idx, (unsigned int) block.isLastInFilter, (unsigned int) block.maxTransportID));

			uint5_t maxTransportID = block.maxTransportID;
			#if defined(FULL_SYSTEM)
			if (idx < (PE_COLS - 1))
			#endif
			{
				if ( ((uint5_t) idx) < maxTransportID ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
					write_channel_intel(channel_weight[idy][idx+1], block);

					EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Passed on weight/bias transfer block.\n", idy, idx));
				}
			}
			write_channel_intel(channel_weight_local[idy][idx], block); 
	}
}

#if defined(SPW_SYSTEM)
typedef uint1_t t_spw_pe_state;
#define SPW_PE_INSTRUCTION_READ_BIAS 0x0
#define SPW_PE_INSTRUCTION_MAC 0x1
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(PE_ROW_GROUPS, PE_COLS)))
__attribute__((autorun))
__kernel void kernelSpWPE ()
{
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
	#if defined(EMULATOR)
		int weightBlockCount = 0;
		int activationBlockCount = 0;
	#endif

	/**
	 * Data registers
	 */
	t_accumulator  __attribute__((register)) regPSums[PE_ROWS_PER_GROUP];
	t_char  __attribute__((register)) regActivations [PE_SIMD_SIZE][PRUNE_RANGE_IN_CLUSTER][CLUSTER_SIZE];
	t_flag regIsLastRowGroup = FALSE;

	/**
	 * Control registers
	 */
	t_spw_pe_state regState = SPW_PE_INSTRUCTION_READ_BIAS;
	t_flag regLoadActivation = TRUE;
	#pragma ii 1
	#pragma speculated_iterations 0
	while (1)
	{
		t_spw_pe_state sigState = regState;
		t_flag sigIsLastRowGroup = regIsLastRowGroup;
		// Need the attribute
		t_char __attribute__((register))  nextActivations [PE_SIMD_SIZE][PRUNE_RANGE_IN_CLUSTER][CLUSTER_SIZE];
		t_flag nextLoadActivation = regLoadActivation;

		#pragma unroll
		for (int s=0; s<PE_SIMD_SIZE; s++)
		{
			#pragma unroll
			for (int c=0; c<PRUNE_RANGE_IN_CLUSTER; c++)
			{
				#pragma unroll
				for (int v=0; v<CLUSTER_SIZE; v++)
				{
					nextActivations[s][c][v] = regActivations[s][c][v];
				}
			}
		}

		/**
		 * Handle reading from the weight channels
		 */
		t_flag sigIsLastWBlockInFilter = FALSE;
		t_flag sigIsLastWBlockInPruneRange = FALSE;
		t_pe_w_block sigWBlocks[PE_ROWS_PER_GROUP];
		{
			// The compiler refuses to unroll the loop 
				// when it is given the unroll pragma
				// so we have to unroll manually
			sigWBlocks[0] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+0][idx]
					);
			#if (PE_ROWS_PER_GROUP>1)
			sigWBlocks[1] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+1][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>2)
			sigWBlocks[2] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+2][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>3)
			sigWBlocks[3] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+3][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>4)
			sigWBlocks[4] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+4][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>5)
			sigWBlocks[5] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+5][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>6)
			sigWBlocks[6] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+6][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>7)
			sigWBlocks[7] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+7][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>8)
			sigWBlocks[8] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+8][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>9)
			sigWBlocks[9] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+9][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>10)
			sigWBlocks[10] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+10][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>11)
			sigWBlocks[11] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+11][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>12)
			sigWBlocks[12] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+12][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>13)
			sigWBlocks[13] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+13][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>14)
			sigWBlocks[14] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+14][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>15)
			sigWBlocks[15] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+15][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>16)
				#error "PE_ROWS_PER_GROUP should be between 1 and 16"
			#endif


			sigIsLastWBlockInFilter = sigWBlocks[0].isLastInFilter;
			sigIsLastWBlockInPruneRange = sigWBlocks[0].isLastInPruneRange;
			// uint5_t maxTransportID = sigWBlocks[0].maxTransportID;
			// #if defined(FULL_SYSTEM)
			// if (idx < (PE_COLS - 1))
			// #endif
			// {
			// 	if ( idx < maxTransportID ) {
			// 		// The compiler refuses to unroll the loop 
			// 		// when it is given the unroll pragma
			// 		// so we have to unroll manually
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+0][idx+1], 
			// 					sigWBlocks[0]
			// 				);
			// 		#if (PE_ROWS_PER_GROUP>1)
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+1][idx+1], 
			// 					sigWBlocks[1]
			// 				);
			// 		#endif
			// 		#if (PE_ROWS_PER_GROUP>2)
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+2][idx+1], 
			// 					sigWBlocks[2]
			// 				);
			// 		#endif
			// 		#if (PE_ROWS_PER_GROUP>3)
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+3][idx+1], 
			// 					sigWBlocks[3]
			// 				);
			// 		#endif
			// 		#if (PE_ROWS_PER_GROUP>4)
			// 			#error "PE_ROWS_PER_GROUP should be between 1 and 4"
			// 		#endif
			// 	}
			// } //end (optional) if
			#if defined(EMULATOR)
				weightBlockCount++;
				EMULATOR_PRINT((
						"[SpW PE (%d, %d)]: "
						"Read %d weight blocks.\n",
						idy, idx, weightBlockCount
					));
			#endif
		} //weight channel read and transfer

		/**
		 * Handle the activation read
		 */
		t_pe_a_block sigABlocks;
		if (regLoadActivation == TRUE)
		{
			sigABlocks = read_channel_intel(channel_activation[idy][idx]);

			sigIsLastRowGroup = 
				(sigABlocks.maxTransportID == (uint5_t) idy) ? TRUE : FALSE;

			//Pass on the activation
			#if defined(FULL_SYSTEM)
			if (idy < (PE_ROW_GROUPS - 1))
			#endif
			{
				if ( sigIsLastRowGroup == FALSE ) {
					write_channel_intel(
							channel_activation[idy+1][idx],
							sigABlocks
						);
				}
			} //end (optional) if
			#if defined(EMULATOR)
				activationBlockCount++;
				EMULATOR_PRINT((
						"[SpW PE (%d, %d)]: "
						"Read and passed on %d activation blocks.\n",
						idy, idx, activationBlockCount
					));
			#endif

			// mem_fence(CLK_CHANNEL_MEM_FENCE);
			//Distribute the clusters into the local register
			#pragma unroll
			for (int s=0; s<PE_SIMD_SIZE; s++)
			{
				#pragma unroll
				for (int c=0; c<PRUNE_RANGE_IN_CLUSTER; c++)
				{
					#pragma unroll
					for (int v=0; v<CLUSTER_SIZE; v++)
					{
						char val = sigABlocks.values[
								s*PRUNE_RANGE_IN_CLUSTER*CLUSTER_SIZE 
								+ c*CLUSTER_SIZE
								+ v];
						nextActivations[s][c][v] = val;

						//regActivations[s][c][v] = val;
					}
				}
			}
		} //if (regState == SPW_PE_INSTRUCTION_LOAD_ACTIVATION)

		/**
		 * Handle pSum update
		 */
		// if (regState == SPW_PE_INSTRUCTION_READ_BIAS)
		// {
		// 	#pragma unroll
		// 	for (int i=0; i<PE_ROWS_PER_GROUP; i++)
		// 	{
		// 		t_bias tempBias = 
		// 			(((t_bias) sigWBlocks[i].values[0]) & 0x0FF)
		// 			| ((((t_bias) sigWBlocks[i].values[1]) & 0x0FF) << 8);
		// 		regPSums[i] = ((t_bias) ACCUM_MASK) & ((t_bias) tempBias);
		// 	}
		// }
		// else
		{
			#pragma unroll
			for (int row = 0; row < PE_ROWS_PER_GROUP; row++)
			{
				t_simd_operand activations, weights;
				#pragma unroll
				for (int v=0; v < PE_SIMD_SIZE * CLUSTER_SIZE; v++)
				{
					weights.values[v] = sigWBlocks[row].values[v];
				} //unroll-for PE_SIMD_SIZE * CLUSTER_SIZE

				//Select the activations
				//THIS IS where the cross bar is located
				#pragma unroll
				for (int simd=0; simd < PE_SIMD_SIZE; simd++)
				{
					t_spw_index index = sigWBlocks[row].indices[simd];
					#pragma unroll
					for (int v=0; v < CLUSTER_SIZE; v++)
					{
						activations.values[simd * CLUSTER_SIZE + v] = 
							nextActivations[simd][index][v];
					} //unroll-for CLUSTER_SIZE
				} //unroll-for PE_SIMD_SIZE

				t_accumulator tempPSum = chain_maddx8(activations, weights);
				if (regState == SPW_PE_INSTRUCTION_READ_BIAS) {
					regPSums[row] = 
						(((t_accumulator) ACCUM_MASK) & ((t_accumulator) sigWBlocks[row].bias)) + tempPSum;
				}
				else {
					regPSums[row] += tempPSum;
				}
				// t_accumulator localAddTerm;
				// if (regState == SPW_PE_INSTRUCTION_READ_BIAS) {
				// 	localAddTerm = (t_accumulator) sigWBlocks[row].bias;
				// }
				// else {
				// 	localAddTerm = regPSums[row];
				// }

				// regPSums[row] = chain_maddx8(activations, weights, localAddTerm);
			} //unroll-for PE_ROWS_PER_GROUP


			//Handle draining
			if (sigIsLastWBlockInFilter == TRUE)
			{
				t_conv_drain_multiple_tagged drainBlock;
				drainBlock.sourceRowGroupID = idy;
				drainBlock.flagIsLast = sigIsLastRowGroup;
				
				#pragma unroll
				for (int row = 0; row < PE_ROWS_PER_GROUP; row++)
				{
					EMULATOR_PRINT((
							"[SpW PE DRAIN (%d, %d)] "
							"Sending a PSum on channel %d: %#06x \n", 
							idy, idx, 
							row,
							(unsigned int) regPSums[row]
							));
					drainBlock.values[row] = regPSums[row];
				}
				write_channel_intel(channel_drain_conv_local[idy][idx], drainBlock);
			}
		} //Handle PSum update

		//Whether the next cycle should load activation
		if ((sigIsLastWBlockInPruneRange == TRUE) || (sigIsLastWBlockInFilter == TRUE)) {
			nextLoadActivation = TRUE;
		}
		else {
			nextLoadActivation = FALSE;
		}

		//State update
		switch (regState) {
			case SPW_PE_INSTRUCTION_READ_BIAS: {
				sigState = SPW_PE_INSTRUCTION_MAC;
				if (sigIsLastWBlockInPruneRange == TRUE)
				{
					EMULATOR_PRINT((
						"[SpW PE (%d, %d)]: "
						"Detected the last SpW block in prune ranges.\n",
						idy, idx, weightBlockCount
					));
					if (sigIsLastWBlockInFilter == TRUE)
					{
						sigState = SPW_PE_INSTRUCTION_READ_BIAS;
					}
				}
			} //case SPW_PE_INSTRUCTION_READ_BIAS
			break;
			case SPW_PE_INSTRUCTION_MAC: {
				if (sigIsLastWBlockInPruneRange == TRUE)
				{
					EMULATOR_PRINT((
						"[SpW PE (%d, %d)]: "
						"Detected the last SpW block in prune ranges.\n",
						idy, idx, weightBlockCount
					));
					if (sigIsLastWBlockInFilter == TRUE)
					{
						sigState = SPW_PE_INSTRUCTION_READ_BIAS;
					}
				}
			} //case SPW_PE_INSTRUCTION_LOAD_ACTIVATION
			break;
			default:
			break;
		}


		regState = sigState;
		regIsLastRowGroup = sigIsLastRowGroup;
		regLoadActivation = nextLoadActivation;
		#pragma unroll
		for (int s=0; s<PE_SIMD_SIZE; s++)
		{
			#pragma unroll
			for (int c=0; c<PRUNE_RANGE_IN_CLUSTER; c++)
			{
				#pragma unroll
				for (int v=0; v<CLUSTER_SIZE; v++)
				{
					regActivations[s][c][v] = nextActivations[s][c][v];
				}
			}
		}
	} //while
} //kernelSpWPE
#endif //SPW_SYSTEM

#if defined(DENSE_SYSTEM)
#define DENSE_PE_INSTRUCTION_READ_BIAS 0X0
#define DENSE_PE_INSTRUCTION_MAC 0X1

typedef uint2_t t_drain_instruction;
typedef uint1_t t_dense_pe_instruction;
typedef uint1_t t_dense_pe_flag;


__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(PE_ROW_GROUPS, PE_COLS)))
__attribute__((autorun))
__kernel void kernelDensePE ()
{
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);

	#if defined(EMULATOR)
		int weightBlockCount = 0;
		int activationBlockCount = 0;
	#endif
	//====================registers===============
	//Psum and drain parameters
	t_accumulator regPSums[PE_ROWS_PER_GROUP];
	//unsigned char regMaxTransportID[2];


	//===============Control registers===================
	t_dense_pe_instruction regInstruction = DENSE_PE_INSTRUCTION_READ_BIAS;

	#pragma ii 1
	#pragma speculated_iterations 0
	while (1) {
		//t_accumulator sigDrainPSum = pSum;
		t_flag sigIsLastRowGroup = FALSE;

		t_dense_pe_instruction sigNextInstruction = regInstruction;

		//Access the activation block
		
		t_pe_a_block sigActivationTB;
		// if (regInstruction == DENSE_PE_INSTRUCTION_MAC)
		{
			sigActivationTB = read_channel_intel(
				channel_activation[idy][idx]);


            sigIsLastRowGroup = 
            	(sigActivationTB.maxTransportID == (uint5_t) idy) ? TRUE : FALSE;

            EMULATOR_PRINT(("[DENSE PE ACTIVATION(%d, %d)] Read new activation block. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx,
						sigActivationTB.values[0],
						sigActivationTB.values[1],
						sigActivationTB.values[2],
						sigActivationTB.values[3],
						(unsigned char) regInstruction));

            //When the PE is placed in the test harness
            //it always passes along the activation block it receives
            #if defined(FULL_SYSTEM)
			if (idy < (PE_ROW_GROUPS - 1))
			#endif
			{
				if ( sigIsLastRowGroup == FALSE ) {
					write_channel_intel(
							channel_activation[idy+1][idx],
							sigActivationTB
						);
				}
			} //end (optional) if
			#if defined(EMULATOR)
				activationBlockCount++;
				EMULATOR_PRINT((
						"[DENSE PE ACTIVATION (%d, %d)]: "
						"Read and passed on %d activation blocks.\n",
						idy, idx, activationBlockCount
					));
			#endif
		} //(regInstruction == DENSE_PE_INSTRUCTION_MAC)

		//Access the weight block
		t_pe_w_block sigWeightTB[PE_ROWS_PER_GROUP];
		t_flag sigIsLastWBlockInFilter = FALSE;
		{
			// The compiler refuses to unroll the loop 
				// when it is given the unroll pragma
				// so we have to unroll manually
			sigWeightTB[0] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+0][idx]
					);
			EMULATOR_PRINT(("[DENSE PE WEIGHT (%d, %d)] Read new weight block from channel 0. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, 
						sigWeightTB[0].values[0],
						sigWeightTB[0].values[1],
						sigWeightTB[0].values[2],
						sigWeightTB[0].values[3],
						(unsigned char) regInstruction));
			#if (PE_ROWS_PER_GROUP>1)
			sigWeightTB[1] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+1][idx]
					);
			EMULATOR_PRINT(("[DENSE PE WEIGHT (%d, %d)] Read new weight block from channel 1. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, 
						sigWeightTB[1].values[0],
						sigWeightTB[1].values[1],
						sigWeightTB[1].values[2],
						sigWeightTB[1].values[3],
						(unsigned char) regInstruction));
			#endif
			#if (PE_ROWS_PER_GROUP>2)
			sigWeightTB[2] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+2][idx]
					);
			EMULATOR_PRINT(("[DENSE PE WEIGHT (%d, %d)] Read new weight block from channel 2. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, 
						sigWeightTB[2].values[0],
						sigWeightTB[2].values[1],
						sigWeightTB[2].values[2],
						sigWeightTB[2].values[3],
						(unsigned char) regInstruction));
			#endif
			#if (PE_ROWS_PER_GROUP>3)
			sigWeightTB[3] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+3][idx]
					);
			EMULATOR_PRINT(("[DENSE PE WEIGHT (%d, %d)] Read new weight block from channel 3. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, 
						sigWeightTB[3].values[0],
						sigWeightTB[3].values[1],
						sigWeightTB[3].values[2],
						sigWeightTB[3].values[3],
						(unsigned char) regInstruction));
			#endif
			#if (PE_ROWS_PER_GROUP>4)
			sigWeightTB[4] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+4][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>5)
			sigWeightTB[5] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+5][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>6)
			sigWeightTB[6] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+6][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>7)
			sigWeightTB[7] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+7][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>8)
			sigWeightTB[8] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+8][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>9)
			sigWeightTB[9] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+9][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>10)
			sigWeightTB[10] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+10][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>11)
			sigWeightTB[11] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+11][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>12)
			sigWeightTB[12] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+12][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>13)
			sigWeightTB[13] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+13][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>14)
			sigWeightTB[14] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+14][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>15)
			sigWeightTB[15] = read_channel_intel(
						channel_weight_local[idy*PE_ROWS_PER_GROUP+15][idx]
					);
			#endif
			#if (PE_ROWS_PER_GROUP>16)
				#error "PE_ROWS_PER_GROUP should be between 1 and 16"
			#endif

			sigIsLastWBlockInFilter = sigWeightTB[0].isLastInFilter;
			// uint5_t maxTransportID = sigWeightTB[0].maxTransportID;
			// #if defined(FULL_SYSTEM)
			// if (idx < (PE_COLS - 1))
			// #endif
			// {
			// 	if ( idx < maxTransportID ) {
			// 		// The compiler refuses to unroll the loop 
			// 		// when it is given the unroll pragma
			// 		// so we have to unroll manually
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+0][idx+1], 
			// 					sigWeightTB[0]
			// 				);
			// 		#if (PE_ROWS_PER_GROUP>1)
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+1][idx+1], 
			// 					sigWeightTB[1]
			// 				);
			// 		#endif
			// 		#if (PE_ROWS_PER_GROUP>2)
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+2][idx+1], 
			// 					sigWeightTB[2]
			// 				);
			// 		#endif
			// 		#if (PE_ROWS_PER_GROUP>3)
			// 		write_channel_intel(
			// 					channel_weight[idy*PE_ROWS_PER_GROUP+3][idx+1], 
			// 					sigWeightTB[3]
			// 				);
			// 		#endif
			// 		#if (PE_ROWS_PER_GROUP>4)
			// 			#error "PE_ROWS_PER_GROUP should be between 1 and 4"
			// 		#endif
			// 	}
			// } //end (optional) if
			#if defined(EMULATOR)
				weightBlockCount++;
				EMULATOR_PRINT((
						"[Dense PE (%d, %d)]: "
						"Read %d group of weight blocks.\n",
						idy, idx, weightBlockCount
					));
			#endif
		} //weight channel read and transfer

		/**
		 * Handle PSum update
		 */
		// if (regInstruction == DENSE_PE_INSTRUCTION_READ_BIAS)
		// {
		// 	#pragma unroll
		// 	for (int i=0; i<PE_ROWS_PER_GROUP; i++)
		// 	{
		// 		t_bias tempBias = 
		// 			(((t_bias) sigWeightTB[i].values[0]) & 0x0FF)
		// 			| ((((t_bias) sigWeightTB[i].values[1]) & 0x0FF) << 8);
		// 		regPSums[i] = ((t_bias) ACCUM_MASK) & ((t_bias) tempBias);
		// 	}

		// 	sigNextInstruction = DENSE_PE_INSTRUCTION_MAC;
		// }
		//else //DENSE_PE_INSTRUCTION_MAC
		{
			#pragma unroll
			for (int row = 0; row < PE_ROWS_PER_GROUP; row++)
			{
				t_simd_operand activations, weights;
				#pragma unroll
				for (int v=0; v < PE_SIMD_SIZE * CLUSTER_SIZE; v++)
				{
					weights.values[v] = sigWeightTB[row].values[v];
					activations.values[v] = sigActivationTB.values[v];
				} //unroll-for PE_SIMD_SIZE * CLUSTER_SIZE

				t_accumulator tempPSum = chain_maddx8(activations, weights);
				if (regInstruction == DENSE_PE_INSTRUCTION_READ_BIAS) {
					regPSums[row] = 
						(((t_accumulator) ACCUM_MASK) & ((t_accumulator) sigWeightTB[row].bias)) + tempPSum;
				}
				else {
					regPSums[row] += tempPSum;
				}

				// t_accumulator tempPSum = madd(activations, weights);
				// regPSums[row] += tempPSum;
			} //unroll-for PE_ROWS_PER_GROUP

			if (sigIsLastWBlockInFilter == TRUE)
			{
				t_conv_drain_multiple_tagged drainBlock;
				drainBlock.sourceRowGroupID = idy;
				drainBlock.flagIsLast = sigIsLastRowGroup;
				#pragma unroll
				for (int row = 0; row < PE_ROWS_PER_GROUP; row++)
				{
					EMULATOR_PRINT((
						"[DENSE PE DRAIN (%d, %d)] "
						"Sending a PSum on channel %d: %#06x \n", 
						idy, idx, 
						row,
						(unsigned int) regPSums[row]
						));
					drainBlock.values[row] = regPSums[row];
				}
				write_channel_intel(channel_drain_conv_local[idy][idx], drainBlock);
			}
		}

		//State update 
		switch (regInstruction) {
			case DENSE_PE_INSTRUCTION_READ_BIAS: {
				sigNextInstruction = DENSE_PE_INSTRUCTION_MAC;
				if (sigIsLastWBlockInFilter == TRUE) {
					sigNextInstruction = DENSE_PE_INSTRUCTION_READ_BIAS;
				}
			}
			break;
			case DENSE_PE_INSTRUCTION_MAC: {
				if (sigIsLastWBlockInFilter == TRUE) {
					sigNextInstruction = DENSE_PE_INSTRUCTION_READ_BIAS;
				}
			}
		}

		regInstruction = sigNextInstruction;

	} // while-loop

}
#endif //DENSE_SYSTEM