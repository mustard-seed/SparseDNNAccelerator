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
	EMULATOR_PRINT(("[Kernel activation feeder. Start \n."));
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
	unsigned int numIterations = numWeightBlocks;
	unsigned int iBlock = 0;
	unsigned int iterNZClustersPerPruneRange = 0;
	t_bias bias = pBias[0];
	#pragma speculated_iterations 0
	for (unsigned int iter=0; iter<numIterations; iter++)
	{
		t_pe_w_block peBlock;
		//Transfer weights
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

		peBlock.bias = bias;
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
		write_channel_intel(channel_weight[4][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>5)
		write_channel_intel(channel_weight[5][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>6)
		write_channel_intel(channel_weight[6][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>7)
		write_channel_intel(channel_weight[7][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>8)
		write_channel_intel(channel_weight[8][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>9)
		write_channel_intel(channel_weight[9][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>10)
		write_channel_intel(channel_weight[10][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>11)
		write_channel_intel(channel_weight[11][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>12)
		write_channel_intel(channel_weight[12][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>13)
		write_channel_intel(channel_weight[13][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>14)
		write_channel_intel(channel_weight[14][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>15)
		write_channel_intel(channel_weight[15][0], peBlock);
		#endif
		#if (PE_ROWS_PER_GROUP>16)
		#error "PE_ROWS_PER_GROUP should be between 1 and 16"
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
	unsigned int numIterations = numWeightBlocks;
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
		read_channel_intel(channel_weight[4][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>5)
		read_channel_intel(channel_weight[5][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>6)
		read_channel_intel(channel_weight[6][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>7)
		read_channel_intel(channel_weight[7][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>8)
		read_channel_intel(channel_weight[8][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>9)
		read_channel_intel(channel_weight[9][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>10)
		read_channel_intel(channel_weight[10][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>11)
		read_channel_intel(channel_weight[11][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>12)
		read_channel_intel(channel_weight[12][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>13)
		read_channel_intel(channel_weight[13][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>14)
		read_channel_intel(channel_weight[14][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>15)
		read_channel_intel(channel_weight[15][1]);
		#endif
		#if (PE_ROWS_PER_GROUP>16)
		#error "PE_ROWS_PER_GROUP should be between 1 and 16"
		#endif
		//Transfer weights
		if (iter == 0) {
			pBias[0] = peBlock.bias;
		}	
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
	} //for over iter
}

__kernel void kernelResultDrainer (
		__global t_wide_psum* restrict pOutputs
	)
{
	EMULATOR_PRINT(("[Kernel result drainer. Start \n."));
	t_conv_drain_multiple_tagged peOutputs = 
		read_channel_intel(channel_drain_conv_local[0][0]);

	t_wide_psum outputs;
	#pragma unroll
	for (int i=0; i<PE_ROWS_PER_GROUP; i++)
	{

		//Naively cast from t_accumulator to int without explicit sign extension
		outputs.psums[i] = peOutputs.values[i];

		EMULATOR_PRINT(("[Kernel result drainer. Sent the pSum: %#04x \n.",
						outputs.psums[i]));
	}

	pOutputs[0] = outputs;
	EMULATOR_PRINT(("[Kernel result drainer. Done \n."));
}