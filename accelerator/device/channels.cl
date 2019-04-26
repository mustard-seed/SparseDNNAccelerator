#ifndef CHANNELS_CL_DEF
#define CHANNELS_CL_DEF
#include "device_structures.hpp"
#include "params.hpp"
//Must include the following line in order to use channel
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel t_tokenFillWeightCache channel_spWeightDMA __attribute__((depth(0)));
channel uint1_t channel_spWeightDMACommit __attribute__((depth(9)));

channel t_packetDMAToWeightFeeder channel_packetDMAToWeightFeeder [KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_packetDMAToWeightFeederLoopBack __attribute__((depth(0)));

channel t_tokenDrainWeightCache channel_tokenDrainWeightCacheControl[KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_drainWeightCacheInternalCommit[KERNEL_CACHE_LANES-1] __attribute__((depth(0)));
channel uint1_t channel_drainWeightCacheCommit __attribute__((depth(0)));

channel uint1_t channel_spWeightFeederDrainSelect [KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_spWeightFeederDrainSelectCommit __attribute__((depth(0)));

#ifdef INCLUDE_COMPUTE_CORE
channel t_spWeightAndOffset channel_sparseWeights[PE_ROWS][PE_COLS] __attribute__((depth(0)));
#else
channel t_spWeightAndOffset channel_sparseWeights[PE_ROWS] __attribute__((depth(0)));
#endif

#ifdef WEIGHT_MEMORY_TEST
// In the weight memory test, there are 4 types of instructions
channel unsigned short channel_instructions[4] __attribute__((depth(16)));

channel t_weightCollectToken channel_weightCollectControl[KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_weightCollectControlCommitInternal[KERNEL_CACHE_LANES-1] __attribute__((depth(0))); 
channel uint1_t channel_weightCollectControlCommit __attribute__((depth(0)));
#endif

#endif