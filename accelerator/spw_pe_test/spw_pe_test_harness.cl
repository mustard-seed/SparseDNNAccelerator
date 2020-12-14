#include "spw_pe_test_types.hpp"
#include "spw_pe_test_channels.hpp"
#include "ihc_apint.h"
#include "prints.hpp"

#if !(defined(SPW_TEST) && defined(SPW_SYSTEM))
#error "For the SpW PE test, SPW_TEST needs to be defined as a compiler marco, and SPW_SYSTEM should be defined in params.hpp."
#endif


__attribute__((max_global_work_dim(0)))
__kernel void kernelActivationFeeder (
		__global const t_test_activation_host_block* restrict pActivation,
		unsigned int numActivationBlocks
	)
{
	#pragma speculated_iterations 0
	for (unsigned int iBlock=0; iBlock<numActivationBlocks; iBlock++)
	{
		t_test_activation_host_block block = pActivation[iBlock];

		/**
		 * Convert the host block into the PE block
		 */
		t_pe_a_block peBlock;
		#pragma unroll
		for (int v=0; v< PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE; v++)
		{
			peBlock.values[v] = block.values[v];
		}
		//Set the max transport ID to 1 so that the DUT will pass it to the drainer
		peBlock.maxTransportID = 0x01;
		write_channel_intel(channel_activation[0][0], peBlock);
	}
}

__kernel void kernelActivationDrainer (
		__global t_test_activation_host_block* restrict pActivation,
		unsigned int numActivationBlocks
	)
{
	for (unsigned int iBlock=0; iBlock<numActivationBlocks; iBlock++)
	{
		t_pe_a_block peBlock = read_channel_intel(channel_activation[1][0]);
		t_test_activation_host_block block;
		#pragma unroll
		for (int v=0; v< PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE; v++)
		{
			block.values[v] = peBlock.values[v];
		}
		pActivation[iBlock] = block;
	}
}

__kernel void kernelFilterFeeder (
		__global const t_test_weight_host_block* restrict pWeight,
		__global const t_bias* restrict pBias,
		unsigned int numWeightBlocks,
		unsigned int numNZClustersPerPruneRange
	)
{
	//Total number of loops is ONE PLUs the number of weight blocks
	//One extra iteration a the beginning to transfer the bias
	unsigned int numIterations = numWeightBlocks + 1;
	unsigned int iBlock = 0;
	unsigned int iterNZClustersPerPruneRange = 0;
	#pragma speculated_iterations 0
	for (unsigned int iter=0; iter<numIterations; iter++)
	{
		t_pe_w_block peBlock;
		//Transfer the bias
		if (iter == 0)
		{
			t_bias bias = pBias[0];

			//Use the PE weight block to encode the bias
			peBlock.values[0] = bias & 0x0FF;
			peBlock.values[1] = (bias >> 0x08) & 0x0FF;

			peBlock.isLastInPruneRange = FALSE;
			peBlock.maxTransportID = 0x01;
			peBlock.isLastInFilter = FALSE;
		}
		//Transfer weights
		else
		{
			bool isLastInPruneRange = 
				(iterNZClustersPerPruneRange+1) == numNZClustersPerPruneRange;

			t_test_weight_host_block block = pWeight[iBlock];
			//Bridge the weight values
			#pragma unroll 
			for (int v=0; v<PE_SIMD_SIZE*CLUSTER_SIZE; v++)
			{
				peBlock.values[v] = block.values[v];
			}

			//Brdige the indices
			#pragma unroll
			for (unsigned char iChar=0; iChar<INDEX_CHAR_ARRAY_SIZE; iChar++)
			{
				unsigned char index0 = iChar << 1; //*2
				unsigned char index1 = (iChar << 1) + 1; //*2, +1
				t_spw_index val0 = block.indices[iChar] & CHAR_TO_SPW_INDEX_MASK;
				t_spw_index val1 = (block.indices[iChar] >> 0x04) & CHAR_TO_SPW_INDEX_MASK;
				if (index0 < PE_SIMD_SIZE)
				{
					peBlock.indices[index0] = val0;
				}
				if (index1 < PE_SIMD_SIZE)
				{
					peBlock.indices[index1] = val1;
				}
			}

			peBlock.isLastInPruneRange = isLastInPruneRange ? TRUE : FALSE;
			peBlock.maxTransportID = 0x01;
			peBlock.isLastInFilter = ((iBlock+1) == numWeightBlocks) ?
				TRUE : FALSE;
			
			iBlock++;
			if (isLastInPruneRange == true)
			{
				iterNZClustersPerPruneRange = 0x0;
			}
			else
			{
				iterNZClustersPerPruneRange += 0x01;
			}

		}

		//Duplicate the weight block and broadcast them
		// #pragma unroll
		// for (int i=0; i<PE_ROWS_PER_GROUP; i++)
		// {
		// 	write_channel_intel(channel_weight[i][0], peBlock);
		// }
		write_channel_intel(channel_weight[0][0], peBlock);
		#if (PE_ROWS_PER_GROUP>1)
		write_channel_intel(channel_weight[1][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>2)
		write_channel_intel(channel_weight[2][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>3)
		write_channel_intel(channel_weight[3][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>4)
		#error "PE_ROWS_PER_GROUP should be between 1 and 4"
		#endif
	}
}

__kernel void kernelFilterDrainer (
		__global t_test_weight_host_block* restrict pWeight,
		__global t_bias* restrict pBias,
		unsigned int numWeightBlocks
	)
{
	//Total number of loops is ONE PLUs the number of weight blocks
	//One extra iteration a the beginning to transfer the bias
	unsigned int numIterations = numWeightBlocks + 1;
	unsigned int iBlock = 0;
	#pragma speculated_iterations 0
	for (unsigned int iter=0; iter<numIterations; iter++)
	{
		t_pe_w_block peBlock = read_channel_intel(channel_weight[0][1]);
		#if (PE_ROWS_PER_GROUP>1)
		read_channel_intel(channel_weight[1][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>2)
		read_channel_intel(channel_weight[2][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>3)
		read_channel_intel(channel_weight[3][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>4)
		#error "PE_ROWS_PER_GROUP should be between 1 and 4"
		#endif
		//Transfer the bias
		if (iter == 0)
		{
			t_bias bias = 
				((t_bias) peBlock.values[0])
				| (((t_bias) peBlock.values[1]) << 0x08);

			pBias[0] = bias;
		}
		//Transfer weights
		else
		{
			
			t_test_weight_host_block block;
			//Bridge the weight values
			#pragma unroll 
			for (int v=0; v<PE_SIMD_SIZE*CLUSTER_SIZE; v++)
			{
				block.values[v] = peBlock.values[v];
			}

			//Brdige the indices
			#pragma unroll
			for (unsigned char iChar=0; iChar<INDEX_CHAR_ARRAY_SIZE; iChar++)
			{
				unsigned char index0 = iChar << 1; //*2
				unsigned char index1 = (iChar << 1) + 1; //*2, +1
				t_spw_index val0 = 0;
				t_spw_index val1 = 0;
				if (index0 < PE_SIMD_SIZE)
				{
					val0 = peBlock.indices[index0];
				}
				if (index1 < PE_SIMD_SIZE)
				{
					val1 = peBlock.indices[index1];
				}
				block.indices[iChar] = 
					(((unsigned char) val0) & 0x0F)
					| ((((unsigned char) val1) & 0x0F) << 0x04);
			}
			
			pWeight[iBlock] = block;

			iBlock++;
		}
	} //for over iter
}

__kernel void kernelResultDrainer (
		__global t_wide_psum* restrict pOutputs
	)
{
	t_conv_drain_multiple_tagged peOutputs = 
		read_channel_intel(channel_drain_conv_local[0][0]);

	t_wide_psum outputs;
	#pragma unroll
	for (int i=0; i<PE_ROWS_PER_GROUP; i++)
	{
		//Naively cast from t_accumulator to int without explicit sign extension
		outputs.psums[i] = peOutputs.values[i];
	}

	pOutputs[0] = outputs;
}