#ifndef _LAYER_INSTRUCTION_GENERATOR_HPP_
#define _LAYER_INSTRUCTION_GENERATOR_HPP_
#include "vectorType.hpp"

void convolution_instruction_generator(
        t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
        t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
        t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
        t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
        t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,


        signed int memIADramBlockStartIndex,
        signed int memOADramBlockStartIndex,
        signed int memWeightDramBlockStartIndex,
        signed int memBiasStartIndex,

        signed int memIADramBlockStripStride,

        signed int memOADramBlockStripStride,

        signed int memWeightDramBlockFilterStride,

        #if defined(SPARSE_SYSTEM)
            signed int memIATBCountStart,
            unsigned int memIATBCountStripStride,

            signed int memOATBCountStart,
            unsigned int memOATBCountStripStride,

            signed int memWeightTBCountStart,
        #else
            unsigned int numTBPerIAStrip,
            unsigned int numTBPerOAStrip,
            unsigned int numTBPerWeightFilter,
        #endif

        unsigned char pingPongRegionIA,
        unsigned char pingPongRegionOA,
        unsigned char flagSparseOutput,
        unsigned char flagSparseInput,
        unsigned char flagInputSync,
        unsigned char flagOutputSync,
        unsigned char flagRelu,
        unsigned char inputFracBits,
        unsigned char weightFracBits,
        unsigned char outputFracBits,

        unsigned short inputSPWidth,
        unsigned short inputSPHeight,
        unsigned char inputSPWidthUnit,
        unsigned char inputSPHeightUnit,
        unsigned char inputWidthPadding,
        unsigned char inputHeightPadding,
        unsigned short numGroupsCurrentLayer,
        unsigned short inputChannels,

        unsigned short outputChannels,
        unsigned short numGroupsNextLayer,

        unsigned char kernelSize,
        unsigned char kernelStride,

        unsigned char sizeOutputTileFullHeight,
        unsigned char sizeOutputTileFullWidthPerCol,
        unsigned char numActiveColsPartialOutputTile
        );

#endif
