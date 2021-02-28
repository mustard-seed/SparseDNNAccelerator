#ifndef _VECTOR_TYPE_HPP_DEF_
#define _VECTOR_TYPE_HPP_DEF_
#include "boost/align/aligned_allocator.hpp"
#include "device_structures.hpp"
#include <vector>
#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"

//typedef
//std::vector<t_streamblock_address, boost::alignment::aligned_allocator<t_streamblock_address, aocl_utils_cpp::AOCL_ALIGNMENT>>
//t_aligned_streamblock_address_vector;

//typedef
//std::vector<t_transfer_block, boost::alignment::aligned_allocator<t_transfer_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
//t_aligned_transfer_block_vector;

//typedef
//std::vector<t_dram_block, boost::alignment::aligned_allocator<t_dram_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
//t_aligned_dram_block_vector;

typedef
std::vector<t_char, boost::alignment::aligned_allocator<t_char, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_activation_vector;

typedef
std::vector<t_weight_dram_block, boost::alignment::aligned_allocator<t_weight_dram_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_weight_vector;


typedef
std::vector<t_ia_mover_instruction, boost::alignment::aligned_allocator<t_ia_mover_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_ia_mover_instruction_vector;

typedef
std::vector<t_oa_mover_instruction, boost::alignment::aligned_allocator<t_oa_mover_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_oa_mover_instruction_vector;

typedef
std::vector<t_weight_mover_instruction, boost::alignment::aligned_allocator<t_weight_mover_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_weight_mover_instruction_vector;

typedef
std::vector<t_ia_tile_controller_instruction, boost::alignment::aligned_allocator<t_ia_tile_controller_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_ia_tile_controller_instruction_vector;

typedef
std::vector<t_oa_tile_controller_instruction, boost::alignment::aligned_allocator<t_oa_tile_controller_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_oa_tile_controller_instruction_vector;

typedef
std::vector<t_misc_instruction, boost::alignment::aligned_allocator<t_misc_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_misc_instruction_vector;

/*typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_short_vector*/

std::vector<cl_int, boost::alignment::aligned_allocator<cl_int, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_bias_vector;
#endif
