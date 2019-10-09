#ifndef _VECTOR_TYPE_HPP_DEF_
#define _VECTOR_TYPE_HPP_DEF_
#include "boost/align/aligned_allocator.hpp"
#include "device_structures.hpp"
#include <vector>
#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"

typedef
std::vector<t_streamblock_address, boost::alignment::aligned_allocator<t_streamblock_address, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_streamblock_address_vector;

typedef
std::vector<t_transfer_block, boost::alignment::aligned_allocator<t_transfer_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_transfer_block_vector;

typedef
std::vector<t_dram_block, boost::alignment::aligned_allocator<t_dram_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_dram_block_vector;
#endif
