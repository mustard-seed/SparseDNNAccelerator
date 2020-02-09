#define EMUPRINT
//#define HW_DEBUG
#include "rtl_lib.hpp"
#include "device_utils.hpp"
#include "small_buffer.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void smallBufferTest (
		__global t_smb_tb* restrict pTransferBlocks,
		__global t_smb_tb* restrict pFilteredBlocks,
		__global t_smb_tb* restrict pNextBuffer,
		__global unsigned char* pValid,

		unsigned char numTransferBlocks,
		unsigned short bitmask,
		unsigned short mutualBitmask
	)
{
	
}