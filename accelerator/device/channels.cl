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

channel t_accumulator channel_drain[PE_ROWS][PE_COLS] __attribute__((depth(1)));

channel t_operand channel_processedDrain __attribute__((depth(0)));
//#endif
#ifdef PE_SYSTEM


channel t_transferblock_local channel_dpWeightInput[PE_ROWS][PE_COLS] __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_transferblock_local channel_dpActivationInput[PE_ROWS][PE_COLS] __attribute__((depth(PE_VEC_FIFO_SIZE)));

channel t_accumulator channel_peDrainOutput[PE_ROWS][PE_COLS] __attribute__((depth(0)));
#endif

#ifdef WEIGHT_MEMORY_INTERCONNECT
//channel t_transfer_block channel_weightLanes[PE_ROWS][PE_COLS] __attribute__((depth(0)));
//channel t_transferblock_tagged channel_weightLanes[PE_ROWS][PE_COLS] __attribute__((depth(0)));

/*
===========================================================
Weight transporation channels linking the memory reader to the buffers
===========================================================
*/
channel t_dram_block_w_tagged channel_weight_wide[PE_ROWS] __attribute__((depth(0))); //communication between filter streamer kernels
channel t_dram_block channel_weight_wide_local[PE_ROWS] __attribute__((depth(0))); //communication between filter tee and streamer

#endif //WEIGHT_MEMORY_INTERCONNECT

#ifdef ACTIVATION_MEMORY_INTERCONNECT
/*
=========================================================================
Activation transporation channels linking the memory reader to the buffers
=========================================================================
*/
channel t_dram_block_ia_tagged channel_ia_wide[PE_COLS] __attribute__((depth(0)));
channel t_dram_block channel_ia_wide_local[PE_COLS] __attribute__((depth(0)));

/*
=======================================================================================
Channels that connected the tees on the input activation bus to the input activation buffers
=======================================================================================
*/
channel t_dram_block channel_to_input_buffer_local[PE_COLS] __attribute__((depth(0)));

/*
===================================================================
Channels for passing output controls
===================================================================
*/
channel t_output_buffer_control_tagged channel_output_buffer_control[PE_COLS]__attribute__((depth(0)));
channel t_output_buffer_control channel_output_buffer_local[PE_COLS] __attribute__((depth(0)));
/*
=========================================================================
Output activation channels coming out of the output buffers
======================================================================
*/
//channel t_cluster channel_output_buffer_to_compressor[PE_COLS] __attribute__((depth(0)));
channel t_output_cluster_tagged channel_output_buffer_to_tee[PE_COLS] __attribute__((depth(0)));
channel t_output_dram_block_tagged channel_output_wide[PE_COLS] __attribute__((depth(0)));
#endif //ACTIVATION_MEMORY_INTERCONNECT

#endif