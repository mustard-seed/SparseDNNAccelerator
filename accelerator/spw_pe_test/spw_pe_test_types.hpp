#ifndef SPW_PE_TEST_TYPES_HPP
#define SPW_PE_TEST_TYPES_HPP

#include "params.hpp"
#include "device_structures.hpp"

/**
 * Structures seen by both the device and the host
 */
typedef struct __attribute__((packed)) {
	t_char values[PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE];
} t_test_activation_host_block;


typedef struct __attribute__((packed)) {
	t_char values[PE_SIMD_SIZE * CLUSTER_SIZE];
	t_uchar indices[INDEX_CHAR_ARRAY_SIZE];
} t_test_weight_host_block;

/**
 * Type used to transfer multiple psums from the psum drainer 
 * to the host
 */
typedef struct __attribute__((packed)) {
	t_int psums[PE_ROWS_PER_GROUP];
} t_wide_psum;

#endif