#ifndef _LAYER_INSTRUCTION_GENERATOR_HPP_
#define _LAYER_INSTRUCTION_GENERATOR_HPP_
#include "vectorType.hpp"

enum OPERATION {CONVOLUTION, MAX_POOL, ELT_ADD, CONCATENATION, AVG_POOL};
void instruction_generator (//Type of the operation
        OPERATION op,

        //Instruction buffers
        t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
        t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
        t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
        t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
        t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,
        t_aligned_misc_instruction_vector & vecMiscInstruction,

        //Pre-shift control to be seen by the IA Mover
        bool flagIA0ShiftLeft,
        unsigned int numIA0ShiftAmount,
        bool flagIA1ShiftLeft,
        unsigned int numIA1ShiftAmount,

        //Starting location of activation tensors in the OpenCL Buffer
        //IA and OA occupies the same OpenCL buffer

        //Starting location of the input tensros.
        //In terms of activation dram block
        //Support up to 2 input activation tensors
        signed int memIA0DramBlockStartIndex,
        signed int memIA1DramBlockStartIndex,

        //Starting location of the output tensor
        //In terms of activation dram block
        //Supports only one output tensor
        signed int memOADramBlockStartIndex,

        //Starting location of the weight tensor
        signed int memWeightDramBlockStartIndex,
        //Starting location of bias
        signed int memBiasStartIndex,

        //Input activation blob 0 strides in terms of activation words
        //Assuming GHWC layout
        signed int memIA0ColStride,
        signed int memIA0RowStride,
        //If the operation is not CONVOLUTION or CONCATENATION,
        //then _memIA0GroupStride can be overwritten
//        signed int memIA0GroupStride,

        //Input activation blob 1 strides in terms of activation words
        //Assuming GHWC layout
        signed int memIA1ColStride,
        signed int memIA1RowStride,
        //If the operation is not CONVOLUTION or CONCATENATION,
        //then memIAGroupStride can be overwritten
//        signed int memIA1GroupStride,

        //Output activation blob stride in terms of activation words
        //Assuming GHWC layout
        signed int memOAColStride,

        //Filter stride in terms of weight dram block
        signed int memWeightDramBlockFilterStride,

        //TB count memory information. Only one input blob is supported for sparse operation
//        #if defined(SPARSE_SYSTEM)
//            signed int memIATB0CountStart,
//            unsigned int memIATB0CountColStride,

//            signed int memOATBCountStart,
//            unsigned int memOATBCountColStride,

//            signed int memWeightTBCountStart,
//        #endif

//        unsigned char flagSparseOutput,
//        unsigned char flagSparseInput,
        //Whether the IA mover kernel should wait for the output from the previous tensor to commit before moving
        //on to the current tensor
        unsigned char flagTensorSync,
//        unsigned char flagOutputSync,
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

        #if defined(SPW_SYSTEM)
        //Number of NZ clusters in a pruning range. Only useful for convolution
        unsigned int numNZClustersInPruningRange,
        #endif

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
        unsigned short numOutputChannels

        //Number of groups in the next layer
//        unsigned short numGroupsNextLayer
        );

#endif
