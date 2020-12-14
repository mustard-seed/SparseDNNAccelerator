#ifndef SPW_PE_TEST_CHANNELS
#define SPW_PE_TEST_CHANNELS
#include "ihc_apint.h"
#include "spw_pe_test_types.hpp"

#pragma OPENCL EXTENSION cl_intel_channels : enable

//Sets the channels' depths more than 1 to avoid potential deadlock
channel t_pe_w_block channel_weight[PE_ROWS_PER_GROUP][2] __attribute__((depth(2))); 

channel t_pe_a_block channel_activation[PE_ROW_GROUPS+1][1] __attribute__((depth(2))); 

channel t_conv_drain_multiple_tagged channel_drain_conv_local[1][1] __attribute__((depth(2)));
#endif