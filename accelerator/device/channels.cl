#ifndef CHANNELS_CL_DEF
#define CHANNELS_CL_DEF
#include "device_structures.hpp"
#include "params.hpp"
//Must include the following line in order to use channel
#pragma OPENCL EXTENSION cl_intel_channels : enable

//#ifdef PE_PROTOTYPE_TEST
//channel t_transferblock_tagged channel_activationInput __attribute__((depth(1)));
//channel t_transferblock_tagged channel_activationOutput __attribute__((depth(1)));
//channel t_transferblock_tagged channel_weightInput __attribute__((depth(1)));
//channel t_transferblock_tagged channel_weightOutput __attribute__((depth(1)));

channel t_transferblock_tagged channel_activation[PE_ROWS][PE_COLS]  __attribute__((depth(0)));
channel t_transferblock_tagged channel_weight[PE_ROWS][PE_COLS]  __attribute__((depth(0)));

//channel t_accumulator channel_drainInput __attribute__((depth(1)));
//channel t_accumulator channel_drainOutput __attribute__((depth(1)));

channel t_accumulator channel_drain[PE_ROWS][PE_COLS] __attribute__((depth(0)));

channel t_operand channel_processedDrain __attribute__((depth(0)));
//#endif
#ifdef PE_SYSTEM


channel t_transferblock_local channel_dpWeightInput[PE_ROWS][PE_COLS] __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_transferblock_local channel_dpActivationInput[PE_ROWS][PE_COLS] __attribute__((depth(PE_VEC_FIFO_SIZE)));

channel t_accumulator channel_peDrainOutput[PE_ROWS][PE_COLS] __attribute__((depth(0)));
#endif

#ifdef MEMORY_READER
//channel t_transfer_block channel_weightLanes[PE_ROWS][PE_COLS] __attribute__((depth(0)));
//channel t_transferblock_tagged channel_weightLanes[PE_ROWS][PE_COLS] __attribute__((depth(0)));
channel t_dram_block_tagged channel_filter_transport[PE_ROWS] __attribute__((depth(0))); //communication between filter streamer kernels
channel t_dram_block channel_filter_local[PE_ROWS] __attribute__((depth(0))); //communication between filter tee and streamer
#endif

#endif