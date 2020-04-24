#ifndef _LAYER_INSTRUCTION_GENERATOR_HPP_
#define _LAYER_INSTRUCTION_GENERATOR_HPP_
#include "vectorType.hpp"

enum OPERATION {CONVOLUTION, MAX_POOL, ELT_ADD, CONCATENATION};
void instruction_generator(//Type of the operation
        OPERATION op,

        //Instruction buffers
        t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
        t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
        t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
        t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
        t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,
        t_aligned_misc_instruction_vector & vecMiscInstruction,

        //Starting location of activation tensors in the OpenCL Buffer
        //IA and OA occupies the same OpenCL buffer

        //Starting location of the input tensros.
        //Support up to 2 input activation tensors
        signed int memIA0DramBlockStartIndex,
        signed int memIA1DramBlockStartIndex,

        //Starting location of the output tensor
        //Supports only one output tensor
        signed int memOADramBlockStartIndex,

        //Starting location of the weight tensor
        signed int memWeightDramBlockStartIndex,
        //Starting location of bias
        signed int memBiasStartIndex,

        //Input activation blob 0 strides in terms of DRAM block
        //Assuming GHWC layout
        signed int memIA0DramBlockColStride,
        signed int memIA0DramBlockRowStride,
        //If the operation is not CONVOLUTION or CONCATENATION, then _memIA0DramBlockGroupStride can be overwritten
        signed int _memIA0DramBlockGroupStride,

        //Input activation blob 1 strides in terms of DRAM block
        //Assuming GHWC layout
        signed int memIA1DramBlockColStride,
        signed int memIA1DramBlockRowStride,
        //If the operation is not CONVOLUTION or CONCATENATION, then _memIA0DramBlockGroupStride can be overwritten
        signed int _memIA1DramBlockGroupStride,

        //Output activation blob stride in terms of DRAM block
        //Assuming GHWC layout
        signed int memOADramBlockColStride,

        signed int memWeightDramBlockFilterStride,

        //TB count memory information. Only one input blob is supported for sparse operation
        #if defined(SPARSE_SYSTEM)
            signed int memIATB0CountStart,
            unsigned int memIATB0CountColStride,

            signed int memOATBCountStart,
            unsigned int memOATBCountColStride,

            signed int memWeightTBCountStart,
        #else
            unsigned int numTBPerOAStrip,
            unsigned int numTBPerWeightFilter,
        #endif

        unsigned char flagSparseOutput,
        unsigned char flagSparseInput,
        unsigned char flagInputSync,
        unsigned char flagOutputSync,
        unsigned char flagRelu,
        unsigned char outputShiftBits,
        unsigned char flagOutputShiftLeft,

        //Input stride-padded width and height, not including border padding
        unsigned short inputSPWidth,
        unsigned short inputSPHeight,
        //Input stride-padded unit sizes
        unsigned char inputSPWidthUnit,
        unsigned char inputSPHeightUnit,
        //Input border paddings
        unsigned char inputWidthPadding,
        unsigned char inputHeightPadding,

        unsigned char kernelSize,
        unsigned char kernelStride,

        //Only relevant for convolution
        unsigned char _sizeOutputTileFullHeight,
        //Only relevant for convolution
        unsigned char _sizeOutputTileFullWidthPerCol,
        //Only relevant for convolution
        unsigned char _numActiveColsPartialOutputTile,

        //Number of channels in input blobs 0 and 1
        //Only element-wise addition and pooling should use the second  blob
        unsigned short numInputChannels0,
        unsigned short numInputChannels1,

        //Number of groups in the current layer's output.
        //Only relevant for convolution
        //Other layers assumes the number of current layer's group is 1
        unsigned short numGroupsCurrentLayer,

        //Number of output channels
        //Only relevant for convolution
        unsigned short numOutputChannels,

        //Number of groups in the next layer

        unsigned short numGroupsNextLayer

        );

#endif
