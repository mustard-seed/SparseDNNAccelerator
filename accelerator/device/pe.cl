#if defined(SPW_TEST)
#include "spw_pe_test_types.hpp"
#include "spw_pe_test_channels.hpp"
#else
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.hpp"
#endif

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

#if defined(SPW_SYSTEM)
typedef uint2_t t_spw_pe_state;
#define SPW_PE_INSTRUCTION_READ_BIAS 0x0
#define SPW_PE_INSTRUCTION_LOAD_ACTIVATION 0x1
#define SPW_PE_INSTRUCTION_HOLD_ACTIVATION 0x2
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(PE_ROW_GROUPS, PE_COLS)))
__attribute__((autorun))
__kernel void kernelSpWPE ()
{
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);

	/**
	 * Data registers
	 */
	t_accumulator regPSums[PE_ROWS_PER_GROUP];
	char regActivations[PE_SIMD_SIZE][PRUNE_RANGE_IN_CLUSTER][CLUSTER_SIZE];
	t_flag regIsLastRowGroup = FALSE;

	/**
	 * Control registers
	 */
	t_spw_pe_state regState = SPW_PE_INSTRUCTION_READ_BIAS;

	#pragma ii 1
	#pragma speculated_iterations 0
	while (1)
	{
		t_spw_pe_state sigState = regState;
		t_flag sigIsLastRowGroup = regIsLastRowGroup;

		/**
		 * Handle reading from the weight channels
		 */
		t_flag sigIsLastWBlockInFilter = FALSE;
		t_flag sigIsLastWBlockInPruneRange = FALSE;
		t_pe_w_block sigWBlocks[PE_ROWS_PER_GROUP];
		{
			#pragma unroll
			for (int i=0; i<PE_ROWS_PER_GROUP; i++)
			{
				sigWBlocks[i] = read_channel_intel(
						channel_weight[idy*PE_ROWS_PER_GROUP+i][idx]
					);
			}

			sigIsLastWBlockInFilter = sigWBlocks[0].isLastInFilter;
			sigIsLastWBlockInPruneRange = sigWBlocks[0].isLastInPruneRange;
			maxTransportID = sigWBlocks[0].maxTransportID;
			#if defined(FULL_SYSTEM)
			if (idx < (PE_COLS - 1))
			#endif
			{
				if ( idx < maxTransportID ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
					#pragma unroll
					for (int i=0; i<PE_ROWS_PER_GROUP; i++)
					{
						write_channel_intel(
								channel_weight[idy*PE_ROWS_PER_GROUP+i][idx+1],
								sigWBlocks[i]
							)
					}

				}
			} //end (optional) if
		} //weight channel read and transfer

		/**
		 * Handle the activation read
		 */
		t_pe_a_block sigABlocks;
		if (regState == SPW_PE_INSTRUCTION_LOAD_ACTIVATION)
		{
			sigABlocks = read_channel_intel(channel_activation[idy][idx]);

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
						regActivations[s][c][v] = 
							sigABlocks.values[
								s*PRUNE_RANGE_IN_CLUSTER*CLUSTER_SIZE 
								+ c*CLUSTER_SIZE
								+ v];
					}
				}
			}
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
							sigWBlocks[i]
						)
				}
			} //end (optional) if
		} //if (regState == SPW_PE_INSTRUCTION_LOAD_ACTIVATION)

		/**
		 * Handle pSum update
		 */
		if (regState == SPW_PE_INSTRUCTION_READ_BIAS)
		{
			#pragma unroll
			for (int i=0; i<PE_ROWS_PER_GROUP; i++)
			{
				t_bias tempBias = 
					(((t_bias) sigWBlocks[i].values[0]) & 0x0FF)
					| ((((t_bias) sigWBlocks[i].values[1]) & 0x0FF) << 8);
				regPSums[i] = ((t_bias) ACCUM_MASK) & ((t_bias) tempBias);
			}
		}
		else
		{
			#pragma unroll
			for (int row = 0; row < PE_ROWS_PER_GROUP; row++)
			{
				t_simd_operand activations, weights;
				#pragma unroll
				for (int v=0; v < PE_SIMD_SIZE * CLUSTER_SIZE; v++)
				{
					weights.values[v] = sigWBlocks.values[v];
				} //unroll-for PE_SIMD_SIZE * CLUSTER_SIZE

				//Select the activations
				//THIS IS where the cross bar is located
				#pragma unroll
				for (int simd=0; simd < PE_SIMD_SIZE; simd++)
				{
					t_spw_index index = sigWBlocks.indices[simd];
					#pragma unroll
					for (int v=0; v < CLUSTER_SIZE; v++)
					{
						activations.values[simd * CLUSTER_SIZE + v] = 
							regActivations[simd][index][v];
					} //unroll-for CLUSTER_SIZE
				} //unroll-for PE_SIMD_SIZE

				t_accumulator tempPSum = madd(activations, weights);
				regPSums[row] += tempPSum;
			} //unroll-for PE_ROWS_PER_GROUP
		}

		//Handle draining
		if (sigIsLastWBlockInFilter == TRUE)
		{
			t_conv_drain_multiple_tagged drainBlock;
			drainBlock.sourceRowIDCatIsLast = 
				(unsigned char) ( ((((unsigned char) idy) & 0x7F) << 0x01) 
									| (sigIsLastRowGroup & 0x01));
			#pragma unroll
			for (int row = 0; row < PE_ROWS_PER_GROUP; row++)
			{
				drainBlock.values[row] = regPSums[row]
			}
			write_channel_intel(channel_drain_conv_local[idy][idx], drainBlock);
		}

		//State update
		switch (regState) {
			case SPW_PE_INSTRUCTION_READ_BIAS: {
				sigState = SPW_PE_INSTRUCTION_LOAD_ACTIVATION
			} //case SPW_PE_INSTRUCTION_READ_BIAS
			break;
			case SPW_PE_INSTRUCTION_LOAD_ACTIVATION: {
				sigState = SPW_PE_INSTRUCTION_HOLD_ACTIVATION;
				if (sigIsLastWBlockInPruneRange == TRUE)
				{
					sigState = SPW_PE_INSTRUCTION_LOAD_ACTIVATION;
					if (sigIsLastWBlockInFilter == TRUE)
					{
						sigState = SPW_PE_INSTRUCTION_READ_BIAS;
					}
				}
			} //case SPW_PE_INSTRUCTION_LOAD_ACTIVATION
			break;
			case SPW_PE_INSTRUCTION_HOLD_ACTIVATION: {
				sigState = SPW_PE_INSTRUCTION_HOLD_ACTIVATION;
				if (sigIsLastWBlockInPruneRange == TRUE)
				{
					sigState = SPW_PE_INSTRUCTION_LOAD_ACTIVATION;
					if (sigIsLastWBlockInFilter == TRUE)
					{
						sigState = SPW_PE_INSTRUCTION_READ_BIAS;
					}
				}
			} //case: SPW_PE_INSTRUCTION_HOLD_ACTIVATION
			break;
			default:
			break;
		}

		regState = sigState;
		regIsLastRowGroup = sigIsLastRowGroup;
	} //while
} //kernelSpWPE
#endif //SPW_SYSTEM