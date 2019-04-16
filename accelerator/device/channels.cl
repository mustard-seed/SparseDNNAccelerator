#ifndef CHANNELS_CL_DEF
#define CHANNELS_CL_DEF
#include "device_structures.hpp"
#include "params.hpp"
//Must include the following line in order to use channel
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel t_tokenFillWeightCache channel_spWeightDMA __attribute__((depth(0)));

channel t_packetDMAToWeightFeeder channel_packetDMAToWeightFeeder [KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_packetDMAToWeightFeederLoopBack __attribute__((depth(0)));

channel t_tokenDrainWeightCache channel_tokenDrainWeightCacheControl[KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel bool channel_tokenDrainWeightCacheFinish[KERNEL_CACHE_LANES] __attribute__((depth(0)));

channel uint1_t channel_spWeightFeederDrainSelect [KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_spWeightFeederDrainSelectLoopBack __attribute__((depth(0)));

channel t_spWeight channel_sparseWeights[PE_ROWS][PE_COLS] __attribute__((depth(0)));
#endif
