#ifndef PARAMS_DEFINED
#define PARAMS_DEFINED

#define KERNEL_CACHE_LANES 8
#define KERNEL_CACHE_LANE_MASK 0x7
#define KERNEL_CACHE_DEPTH 4096
#define KERNEL_CACHE_DEPTH_MASK 0xFFF

#define KERNEL_INDEX_CACHE_DEPTH 512
#define KERNEL_INDEX_CACHE_DEPTH_MASK 0x1FF
#define KERNEL_INDEX_CACHE_LANES KERNEL_CACHE_LANES

#endif

