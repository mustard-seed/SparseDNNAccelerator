#include "spw_pe_test_types.hpp"
#include "ihc_apint.h"

#if !(defined(SPW_TEST) && defined(SPW_SYSTEM))
#error "For the SpW PE test, SPW_TEST needs to be defined as a compiler marco, and SPW_SYSTEM should be defined in params.hpp."
#endif


__attribute__((max_global_work_dim(0)))
__kernel void kernelActivationFeeder (
		__global const t_test_activation_host_block* restrict pActivation,
		unsigned int numActivationBlocks
	)
{

}

__kernel void kernelActivationDrainer (
		__global t_test_activation_host_block* restrict pActivation
	)
{

}

__kernel void kernelFilterFeeder (
		__global const t_test_weight_host_block* restrict pWeight,
		__global const t_bias* restrict pBias,
		unsigned int numWeightBlocks,
		unsigned int numNZClustersPerPruneRange
	)
{

}

__kernel void kernelFilterDrainer (
		__global t_test_weight_host_block* restrict pWeight,
		__global t_bias* restrict pBias
	)
{

}

__kernel void kernelResultDrainer (
		__global t_wide_psum* restrict pOutputs
	)
{

}