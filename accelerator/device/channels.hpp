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

channel t_pe_a_block channel_activation[PE_ROW_GROUPS][PE_COLS]  __attribute__((depth(CHANNEL_DEPTH)));
channel t_pe_w_block channel_weight[PE_ROWS][PE_COLS]  __attribute__((depth(1)));
channel t_pe_w_block channel_weight_local[PE_ROWS][PE_COLS] __attribute__((depth(CHANNEL_DEPTH)));

//channel t_accumulator channel_drainInput __attribute__((depth(1)));
//channel t_accumulator channel_drainOutput __attribute__((depth(1)));

channel t_conv_drain_multiple_tagged channel_drain_conv[PE_ROW_GROUPS][PE_COLS] __attribute__((depth(1)));
channel t_conv_drain_multiple_tagged channel_drain_conv_local[PE_ROW_GROUPS][PE_COLS] __attribute__((depth(1)));
channel unsigned char channel_drain_token[PE_ROWS][PE_COLS] __attribute__((depth(1)));

#if defined(MISC_ENGINE)
channel t_output_activation_dram_block_tagged channel_misc_to_oa_tee[MISC_COLS] __attribute__((depth(0)));
channel t_dram_block_ia_to_misc channel_ia_wide_misc[MISC_COLS] __attribute__((depth(0))); 
channel t_misc_control_packet channel_misc_instruction[MISC_COLS]  __attribute__((depth(0)));
channel t_misc_control_packet channel_misc_instruction_local[MISC_COLS] __attribute__((depth(0)));
#endif

//#endif
#ifdef PE_SYSTEM


//channel t_transferblock_tagged channel_dpWeightInput[PE_ROWS][PE_COLS] __attribute__((depth(CHANNEL_DEPTH)));
//channel t_transferblock_tagged channel_dpActivationInput[PE_ROWS][PE_COLS] __attribute__((depth(CHANNEL_DEPTH)));

channel t_accumulator channel_peDrainOutput[PE_ROWS][PE_COLS] __attribute__((depth(0)));
#endif //PE_SYSTEM

#ifdef WEIGHT_MEMORY_INTERCONNECT
//channel t_transfer_block channel_weightLanes[PE_ROWS][PE_COLS] __attribute__((depth(0)));
//channel t_transferblock_tagged channel_weightLanes[PE_ROWS][PE_COLS] __attribute__((depth(0)));

/*
===========================================================
Weight transporation channels linking the memory reader to the buffers
===========================================================
*/
channel t_dram_block_w_tagged channel_weight_wide[PE_ROWS] __attribute__((depth(0))); //communication between filter streamer kernels
channel t_weight_dram_block channel_weight_wide_local[PE_ROWS] __attribute__((depth(0))); //communication between filter tee and streamer

#endif //WEIGHT_MEMORY_INTERCONNECT

#ifdef ACTIVATION_MEMORY_INTERCONNECT
/*
=========================================================================
Activation transporation channels linking the memory reader to the buffers
=========================================================================
*/
channel t_dram_block_ia_tagged channel_ia_wide[PE_COLS] __attribute__((depth(0)));
channel t_dram_block_ia_to_pe channel_ia_wide_local[PE_COLS] __attribute__((depth(0)));
channel t_input_buffer_tile_buffer_packet channel_control_to_ia_buffer [PE_COLS] __attribute__((depth(0)));
channel t_input_buffer_tile_buffer_packet channel_control_to_ia_buffer_local [PE_COLS] __attribute__((depth(0)));

/*
=======================================================================================
Channels that connected the tees on the input activation bus to the input activation buffers
=======================================================================================
*/
// channel t_dram_block channel_to_input_buffer_local[PE_COLS] __attribute__((depth(0)));

/*
===================================================================
Channels for passing output controls
===================================================================
*/
channel t_output_tile_tee_packet channel_oa_tee_local [PE_COLS] __attribute__((depth(0)));
channel t_output_tile_buffer_packet_tagged channel_oa_noc_control [PE_COLS] __attribute__((depth(0)));
channel t_output_tile_buffer_packet channel_control_to_oa_buffer_local [PE_COLS] __attribute__((depth(0)));
/*
=========================================================================
Output activation channels coming out of the output buffers
======================================================================
*/
//channel t_cluster channel_output_buffer_to_compressor[PE_COLS] __attribute__((depth(0)));
//channel t_output_cluster_tagged channel_output_buffer_to_tee[PE_COLS] __attribute__((depth(0)));
//channel t_output_cluster_tagged channel_output_buffer_to_tee[PE_COLS] __attribute__((depth(0)));
// #if defined(SPARSE_SYSTEM)
// channel t_cluster_to_compressor channel_output_buffer_to_compressor_data[PE_COLS] __attribute__((depth(0)));
// channel t_output_cluster_tagged channel_compressor_to_coalescer[PE_COLS] __attribute__((depth(0)));
// #else
// channel t_output_cluster_tagged channel_oa_buffer_to_coalescer[PE_COLS] __attribute__((depth(0)));
// #endif
// channel t_output_coalescer_packet channel_coalescer_to_oa_tee[PE_COLS] __attribute__((depth(1)));
channel t_output_activation_dram_block_tagged channel_oa_buffer_to_oa_tee[PE_COLS] __attribute__((depth(1)));
channel t_output_activation_dram_block_tagged channel_output_wide[PE_COLS] __attribute__((depth(OA_DRAIN_CHANNEL_DEPTH)));

/*
 *=========================================================
  Synchornization channel from the OA mover to IA mover
  =========================================================
*/
channel unsigned char channel_activation_sync __attribute__((depth(1)));
#endif //ACTIVATION_MEMORY_INTERCONNECT

#endif