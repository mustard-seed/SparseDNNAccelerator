#ifndef SPW_PE_TEST_TYPES_HPP
#define SPW_PE_TEST_TYPES_HPP

#include "params.hpp"
#include "device_structure.hpp"

/**
 * Structures seen by both the device and the host
 */
//Size of the char array in the host weight blocks
//used to contain the indices of the NZ clusters
//Each char is split into two 4-bit halfs.
//Each half corresponds to an index
#if defined(PE_SIMD_SIZE)
#if (PE_SIMD_SIZE <= 2)
#define INDEX_CHAR_ARRAY_SIZE 1
#elif (PE_SIMD_SIZE <= 4)
#define INDEX_CHAR_ARRAY_SIZE 2
#elif (PE_SIMD_SIZE <= 8)
#define INDEX_CHAR_ARRAY_SIZE 4
#elif (PE_SIMD_SIZE <= 16)
#define INDEX_CHAR_ARRAY_SIZE 8
#else
#error "PE_SIMD_SIZE needs to be between 1 and 16"
#endif
#else
#error "Parameter PE_SIMD_SIZE is not been defined."
#endif
typedef struct __attribute__((packed)) {
	char values[PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE];
} t_test_activation_host_block;


typedef struct __attribute__((packed)) {
	char values[PE_SIMD_SIZE * CLUSTER_SIZE];
	char indices[INDEX_CHAR_ARRAY_SIZE];
} t_test_weight_host_block;

/**
 * Type used to transfer multiple psums from the psum drainer 
 * to the host
 */
typedef struct __attribute__((packed)) {
	int psums[PE_ROWS_PER_GROUP];
} t_wide_psum;

#endif