#ifndef _VECTOR_TYPE_HPP_DEF_
#define _VECTOR_TYPE_HPP_DEF_
#include "boost/align/aligned_allocator.hpp"
#include "device_structures.hpp"
#include <vector>
#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"

typedef
std::vector<t_simdblock_value, boost::alignment::aligned_allocator<t_simdblock_value, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_simd_value_vector;

typedef
std::vector<t_simdblock_channel_offset, boost::alignment::aligned_allocator<t_simdblock_channel_offset, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_channel_offset_vector;

typedef
std::vector<t_streamblock_address, boost::alignment::aligned_allocator<t_streamblock_address, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_streamblock_address_vector;
#endif
