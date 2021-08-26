# Accelerator Configuration

This file is a detailed explanation of the adjustable parameters. Most of these are found in [accelerator/device/params.hpp](../accelerator/device/params.hpp). 

Important: make sure the host-side software and the accelerator are compiled with the same settings!

General parameters:

- If MCBBS support is required, then define **SPW_SYSTEM**. Otherwise, define **DENSE_SYSTEM**. Do NOT define both.
- DDR_BANDWIDTH_GBS_INT: Estimation of the off-chip memory access bandwidth in GB/s. Used by the instruction generator to generate the best tile configuration.
- FMAX_MHZ: Clock frequency of the accelerator in MHz. Used by the instruction generator to generate the best tile configuration.
- PE_COLS: Number of columns of the convolution systolic array.
- PE_ROWS: Number of rows of the convolution systolic array.
- PE_ROW_GROUPS: PE column coalescing size, equivalent to the "G" parameter mentioned in the paper and the thesis.
- MISC_ACCUMULATOR_WIDTH: Number of bits allocated for each temporary value accumulator inside the MISC engine. Must be chosen from 32, 28, 24, 20, or 16.
- KERNEL_CACHE_SIZE_VALUE_BYTE: Number of bytes that each side of a weight buffer inside the CONV engine can hold.
- IA_CACHE_SIZE_BYTE: Number of bytes that each side of an input neuron buffer inside the CONV engine can hold.
- OA_CACHE_SIZE_BYTE: Number of bytes that each side of an output neuron buffer inside the CONV engine can hold.

Parameters that affect the PEs:

- CLUSTER_SIZE: MCBBS cluster size (C). 
- PRUNE_RANGE_IN_CLUSTER: Pruning range size in clusters (R).
- PE_SIMD_SIZE: Number of clusters that are multiplied in parallel (P). If SPW_SYSTEM is enabled, this is the processing window size.
- ACCUMULATOR_WIDTH: Number of bits allocated for the partial-sum accumulator in each PE. Must be chosen from 32, 28, 24, 20, or 16.

Parameters that access the off-chip data access block size:

- ACTIVATION_WIDE_SIZE: Number of neuron "fetch blocks" inside a neuron DRAM data block. If MCBBS is enabled, then each neuron fetch block contains P * C * R neurons. Otherwise, each neuron fetch block contains P * C neurons. This value also controls how many values are processed in parallel inside the MISC engine. **It must be an interger power of 2**.
- ACTIVATION_WIDE_SIZE_OFFSET: Set this to log2(ACTIVATION_WIDE_SIZE).
- ACTIVATION_WIDE_SIZE_REMAINDER_MASK: Set this to ACTIVATION_WIDE_SIZE - 1.
- WEIGHT_WIDE_SIZE: Number of weight "fetch blocks" inside a weight DRAM data block. The number of sparse/dense weights inside a weight fetch block is P * C. **It must be an interger power of 2**.
- WEIGHT_WIDE_SIZE_OFFSET: Set this to log2(WEIGHT_WIDE_SIZE).
- WEIGHT_WIDE_SIZE_REMAINDER_MASK: Set this to WEIGHT_WIDE_SIZE - 1.

During software emulation, it might be helpful to see some print statement outputs. To do so, please use the EMULATOR_PRINT macro inside [accelerator/device/prints.hpp](../accelerator/device/prints.hpp), and define the EMUPRINT flag.  