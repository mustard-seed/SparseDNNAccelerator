#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"
#include "ihc_apint.h"
#include "rtl_lib.hpp"

#ifdef MEMORY_READER
/*! kernelSimpleWeightStreamer
Important assumption: The address cache only stores the BRAM address of the first streaming block in each strip.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelMemoryReader (
	/*
	Pointers to external memory regions
	*/

	//Pointer to the filter weights in external memory
	 volatile __global t_dram_block* restrict pDramWeights,
	 //Pointer to filter transfer block count
	 volatile __global t_streamblock_address* restrict pFilterStreamBlockAddress,
	 //Pointer to input activations
	 volatile __global t_dram_block* restrict pInputActivation,
	 //Pointer to input activation transfer block count
	 volatile __global t_streamblock_address* restrict pIAStreamBlockAddress,
	 //Pointer to bias
	 volatile __global t_accumulator* restrict pBias,

	 /*
	 //Distance between the start of successive X-Y strip/filters in DRAM in terms of transfer blocks
	 */
	unsigned int strideExternalMemoryWeights,
	unsigned int strideExternalMemoryIA,


	/*
	Output width tiling parameters
	*/
	unsigned short outputWidth, //Q
	unsigned char sizeOutputTilePerColumnWidth, //TQ_A
	unsigned char sizeOutputTileWidthPerColumnPartial, //Output tile width per column for the final few columns
	//Ad-hoc parameters
	//They are employed to deal with the irregularities in output width tiling caused by 
	//imperfect division between number of PE columns and the overall activation width
	//if 0 <= iterQ < maxRegularQ, then all the PE columns receive tile of width TQ_A
	//else, PE columns 0 to APartial-1 each receives a tile of width TQ_APartial, the other columns are idle.
	unsigned short maxRegularQ, 
	unsigned char sizePartialColumns, //APartial 

	/*
	Output height tiling parameters
	*/
	unsigned short outputHeight, //P
	unsigned char sizeOutputHeightTile, //TP

	/*
	Input X-Y dimensions
	*/
	unsigned short inputWidth,
	unsigned short inputHeight,

	/*
	Paddings. Assume padding is symmetrical
	*/
	unsigned char horizontalPadding,
	unsigned char verticalPadding,

	/*
	Stride and kernel sizes
	*/
	unsigned char kernelSize,
	unsigned char stride,

	/*
	Input and output channels
	*/
	unsigned int numFiltersInKernel, //L

	unsigned char numGroups, // L / G
	unsigned short numFiltersInGroup, // G
	//unsigned short numFoldInGroup, // ceil (G / F)
	unsigned short numInputChannelCompressionWindows


	) 
{
	//typedef uint3_t t_state;
	//Cache: NzBlocks blocks
	//t_dram_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH][PE_ROWS] __attribute__ ((numbanks(PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE*WIDE_SIZE)));
	//t_transfer_block cacheNzBlocks [2][PE_ROWS][KERNEL_CACHE_DEPTH] __attribute__ ((numbanks(BURST_SIZE_TRANSFER_BLOCK * PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE)));

	//Cache: Cache address of the streaming blocks
	t_streamblock_address cacheFilterStreamBlockAddress [4096] __attribute__((numbanks(1)));
	t_streamblock_address cacheIAStreamBlockAddress [4096] __attribute__((numbanks(1)));
	t_accumulator cacheBias [4096] __attribute__((numbanks(1)));

	//=============================================================
	//Read all the streaming block address in to BRAM as soon as possible
	//==============================================================
	EMULATOR_PRINT(("[kernelMemoryReader] Reading the filter biases and stream block addresses\n"));
	{
		for (unsigned int countFilters=0; countFilters < numFiltersInKernel; countFilters++) {
			cacheFilterStreamBlockAddress[countFilters] = pFilterStreamBlockAddress[countFilters];
		}

		for (unsigned int countFilters=0; countFilters < numFiltersInKernel; countFilters++) {
			cacheBias[countFilters] = pBias[countFilters];
		}
	}


	unsigned short iterP = 0; // iterP	

	//Loops
	while (iterP < outputHeight)
	{
		//Calculate the effectual output tile height
		unsigned char maxTP = (((unsigned short) sizeOutputHeightTile) < (outputHeight - iterP) ) ?
			sizeOutputHeightTile :  (unsigned char) (outputHeight - iterP);


		//Calculate the vertical input paddings
		unsigned char tileTopPadding = (iterP==0) ? verticalPadding : 0;
		unsigned char tileBottomPadding = ( (outputHeight - iterP) <= (unsigned short) (sizeOutputHeightTile) )
			? verticalPadding : 0;

		//Input activation height index that corresponds to iterP
		unsigned short startM = (iterP*(unsigned short) stride) < (unsigned short) verticalPadding ?
				0 : iterP*(unsigned short) stride - (unsigned short) verticalPadding;

		unsigned short iterQ = 0; //countQCovertedByTQ

		while (iterQ < outputWidth) //tq
		{
			/*
			Input activation tile parameters
			*/
			unsigned char maxTQ_A = (iterQ < maxRegularQ) 
				? sizeOutputTilePerColumnWidth : sizeOutputTileWidthPerColumnPartial;

			unsigned char maxPeCols =  (iterQ < maxRegularQ)
				? PE_COLS : sizePartialColumns;

			unsigned short maxTQ = (unsigned short) maxTQ_A * (unsigned short) maxPeCols;

			unsigned char tileLeftPadding = (iterQ == 0) ? horizontalPadding : 0;
			unsigned char tileRightPadding = (maxTQ >= (outputWidth - iterQ)) ? horizontalPadding : 0;

			//Input activation width index that corresponds to iterQ
			unsigned short startN = (iterQ*(unsigned short) stride) < (unsigned short) horizontalPadding ?
				0 : iterQ*(unsigned short) stride - (unsigned short) horizontalPadding;


			unsigned char strideAcrossCol = maxTQ_A * stride;


			//Range of input height that are loaded
			unsigned short maxTM = (maxTP - 1) * (unsigned short) stride + (unsigned short) kernelSize - (unsigned short)(tileBottomPadding + tileTopPadding);
				//Range of input width to be loaded
			unsigned short maxTN = (maxTQ - 1) * (unsigned short) stride + (unsigned short) kernelSize - (unsigned short)(tileLeftPadding + tileRightPadding); 


			/*
			Load the streaming block addresses for all the INPUT activation strips in the X-Y region
			Assume the addresses are stored in Group-H-W-C layout.
			*/
				
			//SB addresses will be stored in NWC format in the cache
			int cacheAddress = 0;

			unsigned char iGroup = 0;
			unsigned short iMGlobal = startM;
			unsigned short iNGlobal = startN;
			unsigned int dramAddress = (unsigned int) startM * (unsigned int) inputWidth + (unsigned int) startN; //Start from depth 0
			while (iGroup < numGroups)
			{

				while (iNGlobal < (startN + maxTN))
				{
					cacheIAStreamBlockAddress[cacheAddress] = pIAStreamBlockAddress[dramAddress];
					cacheAddress++;
					iNGlobal++;
					dramAddress++;
				}
				//Update variables
				iNGlobal = startN;
				if (iMGlobal == (startM + maxTM - 1))
				{
					iMGlobal = startM;
					iGroup++;
				}
				else
				{
					iMGlobal++;
				} //iM
				dramAddress = ((unsigned int) iGroup* (unsigned int) inputHeight + (unsigned int) iMGlobal)*inputWidth + (unsigned int) startN;
			} //iGroup

			/*
			Filter parameters
			*/
			unsigned short iFilterGlobal = 0; //iL
			unsigned int iTransferBlockFilterBaseDDR = 0;


			for (unsigned short iGroup=0; iGroup<numGroups; iGroup++) //gl
			{
				/*
					Streaming input activations to the IA buffers
				*/
				unsigned char iterNInTile = 0;
				for (unsigned char iterA=0; iterA < maxPeCols; iterA++)
				{
					//Prepare control signals for one input buffer
					unsigned char localLeftPadding = (iterA==0) ? tileLeftPadding : 0;
					unsigned char localRightPadding = (iterA==maxPeCols) ? tileRightPadding : 0;
					unsigned char inputTileWidthCol = (maxTQ_A-1) * stride + kernelSize - localLeftPadding - localRightPadding;
					unsigned char inputTileHeightCol = (maxTP-1) * stride + kernelSize - tileTopPadding - tileBottomPadding;
					t_input_buffer_control control;
					control.topPadding = tileTopPadding;
					control.bottomPadding = tileBottomPadding;
					control.leftPadding = localLeftPadding;
					control.rightPadding = localRightPadding;
					control.inputTileWidth = inputTileWidthCol;
					control.inputTileHeight = inputTileHeightCol;
					control.stride = stride;
					control.kernelSize = kernelSize;
					control.numOutputChannelsInGroup = numFiltersInGroup;
					control.numInputChannelCompressionWindows = numInputChannelCompressionWindows;

					//Iterators that keep track of the input domain mapped to the pe column
					unsigned short iterMGlobal = startM;
					unsigned short iterNGlobal = startN + iterNInTile;

					unsigned char maxTransferCountAlongHeight = inputTileHeightCol+1; //Need one extra iteration to transmit the control block

					/*
					Send the transfer blocks in the inputTileHeight * inputTileWidth region to the select buffer
					Before sending each strip of transfer blocks, we need to send the number of transfer blocks in the strip.
					Before sending the transfer block counts and the strips, we need to send the control information
					*/
					for (unsigned char iterMInCol=0; iterMInCol <= maxTransferCountAlongHeight; iterMInCol++) //Need one extra iteration to transmit the control block
					{
						unsigned short indexIAAddressCache = (iterMInCol > 0) ?
							(iGroup * maxTM + (unsigned short) (iterMInCol-1))*(unsigned short) maxTN + (unsigned short) iterNInTile
							: 0;

						unsigned int iterTransferBlockIADramBase = (iterMInCol > 0) ?
							((unsigned int)(iGroup * inputHeight + iterMGlobal+iterMInCol-1) * inputWidth + iterNGlobal) * strideExternalMemoryIA
							: 0;

						unsigned char maxTransferCountAlongWidth = (iterMInCol==0) ? 1 : inputTileWidthCol;

						for (unsigned char iterNInCol=0; iterNInCol < maxTransferCountAlongWidth; iterNInCol++)
						{
							t_streamblock_address transferBlockCount = cacheIAStreamBlockAddress[indexIAAddressCache];

							unsigned short dramBlockCount = ((transferBlockCount & WIDE_SIZE_REMAINDER_MASK) > 0) ?
								(transferBlockCount >> WIDE_SIZE_OFFSET) + 1 : (transferBlockCount >> WIDE_SIZE_OFFSET);

							/*
							If iterMInCol and iterNInCol are both 0, then we transfer the control blog
							Otherwise, we first transfer the control count, then transfer the 
							*/
							unsigned short maxTranferCountAlongChannel = ((iterMInCol==0) && (iterNInCol==0)) ? 1 : (dramBlockCount + 1);
							unsigned int iterTransferBlockIADram = iterTransferBlockIADramBase;

							for (unsigned short iterC=0; iterC<maxTranferCountAlongChannel; iterC++)
							{
								t_dram_block dramBlock;

								if (iterMInCol==0)
								{
									dramBlock = inputBufferControl2DramBlock(control);
								}
								else 
								{
									if (iterC==0)
									{
										dramBlock = transferBlockCount2DramBlock(transferBlockCount);
									}
									else 
									{
										dramBlock = pInputActivation[iterTransferBlockIADram >> WIDE_SIZE_OFFSET];
										iterTransferBlockIADram += WIDE_SIZE_OFFSET;
									}
								}

								t_dram_block_ia_tagged iaBlock;
								iaBlock.dramBlock = dramBlock;
								iaBlock.destinationCol = iterA;

								write_channel_intel(channel_ia_wide[0], iaBlock);
							} //for-loop iterC

							iterTransferBlockIADramBase += strideExternalMemoryIA;	
							indexIAAddressCache++;
						} //for-loop iterNIncol
					} //for-loop iterMInCol
					iterNInTile += strideAcrossCol;
				}//iterA


				unsigned short iFilterInGroup = 0; //gf * F
				while (iFilterInGroup < numFiltersInGroup) //gf
				{
					unsigned char maxRowUsed = PE_ROWS < (numFiltersInGroup - iFilterInGroup) ?
						PE_ROWS : (numFiltersInGroup - iFilterInGroup); //maxF

					for (unsigned char iPeRow=0; iPeRow<maxRowUsed; iPeRow++)
					{
						unsigned short maxTransferBlockInFilter = cacheFilterStreamBlockAddress[iFilterGlobal];
						t_accumulator bias = cacheBias[iFilterGlobal];

						unsigned short maxDramBlockInFilter = ((maxTransferBlockInFilter & WIDE_SIZE_REMAINDER_MASK) > 0x0) ?
							(maxTransferBlockInFilter >> WIDE_SIZE_OFFSET) + 1 : maxTransferBlockInFilter >> WIDE_SIZE_OFFSET;
						unsigned short maxTransmitCount = maxDramBlockInFilter+1; //one extra for filter stream control;
						
						t_filter_streamer_control control;
						control.numOutputs = (unsigned short) maxTP * (unsigned short) maxTQ_A;
						control.bias = bias;
						control.numTransferBlocks = maxTransferBlockInFilter;
						control.maxPeCols = maxPeCols;

						t_dram_block dramControl = filterStreamerControl2dramBlock(control);

						unsigned int iTransferBlockDDR = iTransferBlockFilterBaseDDR;

						EMULATOR_PRINT(("[kernelFilterWriter] Sending filter %d to row %d. (iHeightGlobal, iterQ): (%d, %d). Number of transfer blocks: %d\n",
							iFilterGlobal, iPeRow, iHeightGlobal, iterQ, maxTransferBlockInFilter));
						for (unsigned short iTransmitCount=0; iTransmitCount<maxTransmitCount; iTransmitCount++)
						{
							t_dram_block block;
							if (iTransmitCount == 0) 
							{
								block = dramControl;
							}
							else
							{
								block = pDramWeights[iTransferBlockDDR >> WIDE_SIZE_OFFSET];
								iTransferBlockDDR += WIDE_SIZE;
							}

							t_dram_block_w_tagged taggedBlock;
							taggedBlock.dramBlock = block;
							taggedBlock.destinationRow = iPeRow;

							write_channel_intel(channel_weight_wide[0], taggedBlock);
						} // iTransmitCount

						iTransferBlockFilterBaseDDR += strideExternalMemoryWeights;
						iFilterGlobal++;

					} // iPeRow

					iFilterInGroup += maxRowUsed;	
				} // iFilterInGroup
			} //iGroup
			iterQ += maxTQ;
		} // iterQ
	iterP += maxTP;
	}	// iterP
}

#endif //MEMORY_READER

#ifdef IA_MEMORY
#define STATE_IA_BUFFER_WRITE_CACHE_DONE 0x0
#define STATE_IA_BUFFER_WRITE_CACHE_LOAD_TRANSFER_COUNT 0x1
#define STATE_IA_BUFFER_WRITE_CACHE_WRITE 0x2
#define IA_LOOP_LATENCY 7
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIABuffer ()
{
	typedef uint2_t t_state;
	int colID = get_compute_id(0);

	t_dram_block cacheIABlocks [IA_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE)));
	t_streamblock_address cacheIAStreamBlockAddress [256];


	while (true)
	{
		t_dram_block tileDramBlock = read_channel_intel(channel_ia_wide_local[colID]);
		t_input_buffer_control tileControl = dramBlock2InputBufferControl(tileDramBlock);

		unsigned short wNumInputWidthxHeight = (unsigned short) tileControl.inputTileWidth * (unsigned short) tileControl.inputTileHeight;
		//Starting position multipliles of each x-y strip in terms od DRAM block
		unsigned short stripStrideInCache = (tileControl.numInputChannelCompressionWindows*(COMPRESSION_WINDOW_SIZE+1) & WIDE_SIZE_REMAINDER_MASK) > 0 ?
			tileControl.numInputChannelCompressionWindows*(COMPRESSION_WINDOW_SIZE+1) >> WIDE_SIZE_OFFSET + 1 
			: tileControl.numInputChannelCompressionWindows*(COMPRESSION_WINDOW_SIZE+1) >> WIDE_SIZE_OFFSET;
		//Address of the iaCache, in terms of dram blocks
		unsigned short iaCacheWideAddressBase = 0; 
		
		//Import the NZ block count and the IA for one tile.
		for (unsigned short wCountInputWidthxHeight = 0; wCountInputWidthxHeight < wNumInputWidthxHeight; wCountInputWidthxHeight++)
		{
			//t_state state = STATE_IA_BUFFER_WRITE_CACHE_LOAD_TRANSFER_COUNT;
			t_dram_block dramBlockCount = read_channel_intel(channel_ia_wide_local[colID]);

			t_streamblock_address wNumInputTransferBlock = dramBlock2TransferBlockCount(dramBlockCount);
			cacheIAStreamBlockAddress[wCountInputWidthxHeight] = wNumInputTransferBlock;

			unsigned short iaCacheWideAddress = iaCacheWideAddressBase;

			for (t_streamblock_address wCountInputTransferBlock=0; 
				wCountInputTransferBlock < wNumInputTransferBlock;
				wCountInputTransferBlock++)
			{
				t_dram_block dramBlock = read_channel_intel(channel_ia_wide_local[colID]);
				cacheIABlocks[iaCacheWideAddress++] = dramBlock;
				wCountInputTransferBlock += WIDE_SIZE_OFFSET;
			}

			//t_state delayState[IA_LOOP_LATENCY];

			//#pragma unroll
			//for (unsigned char i=0; i<IA_LOOP_LATENCY; i++)
			//{
			//	delayState[i] = STATE_IA_BUFFER_WRITE_CACHE_LOAD_TRANSFER_COUNT;
			//}

			//Load one strip
			//TODO: This loop has large II
			/*
			while ( 
				delayState[0] != STATE_IA_BUFFER_WRITE_CACHE_DONE
			)
			{
				if (state != STATE_IA_BUFFER_WRITE_CACHE_DONE)
				{
					t_dram_block dramBlock = read_channel_intel(channel_ia_wide_local[colID]);

					if (state == STATE_IA_BUFFER_WRITE_CACHE_LOAD_TRANSFER_COUNT)
					{
						state = STATE_IA_BUFFER_WRITE_CACHE_WRITE;

						wNumInputTransferBlock = dramBlock2TransferBlockCount(dramBlock);
						cacheIAStreamBlockAddress[wCountInputWidthxHeight] = wNumInputTransferBlock;
						wCountInputTransferBlock = 0;
					}
					else
					{
						cacheIABlocks[iaCacheWideAddress++] = dramBlock;
						wCountInputTransferBlock += WIDE_SIZE_OFFSET;
						if (wCountInputTransferBlock >= wNumInputTransferBlock)
						{
							state = STATE_IA_BUFFER_WRITE_CACHE_DONE;
						}
					}
				}

				#pragma unroll
				for (unsigned char i=0; i<(IA_LOOP_LATENCY-1); i++)
				{
					delayState[i] = delayState[i+1];
				}
				delayState[IA_LOOP_LATENCY-1] = state;
			} //Load one strip
			*/
			iaCacheWideAddressBase += stripStrideInCache;
		} //Import the NZ block count and the IA for one tile.

		// Stream the input activation blocks
		unsigned short iFilterGlobal = 0;
		while (iFilterGlobal < tileControl.numOutputChannelsInGroup)
		{
			unsigned char numPeRows = (PE_ROWS < (tileControl.numOutputChannelsInGroup - iFilterGlobal)) ?
				PE_ROWS : (unsigned char) (tileControl.numOutputChannelsInGroup - iFilterGlobal);

			//Stream the activations
			unsigned char iPaddedTileHeight = 0;
			unsigned char iPaddedTileWidth = 0;
			while ( (iPaddedTileHeight+tileControl.kernelSize) <= (tileControl.inputTileHeight + tileControl.bottomPadding + tileControl.topPadding) )
			{
				unsigned char iPaddedHeightLocal = iPaddedTileHeight;
				unsigned char iPaddedWidthLocal = iPaddedTileWidth;
				unsigned char iNeuWidthLocal = 0;
				unsigned char iNeuHeightLocal = 0;

				while (iNeuHeightLocal < tileControl.kernelSize)
				{
					bool sendPadding = 
						(iPaddedHeightLocal < tileControl.topPadding)
						|| (iPaddedHeightLocal >= tileControl.inputTileHeight + tileControl.topPadding)
						|| (iPaddedWidthLocal < tileControl.leftPadding)
						|| (iPaddedWidthLocal >= tileControl.inputTileWidth + tileControl.leftPadding);

					unsigned char iTileHeightLocalTemp = (iPaddedHeightLocal > tileControl.topPadding) ? iPaddedHeightLocal - tileControl.topPadding : 0;
					unsigned char iTileHeightLocal = iTileHeightLocalTemp > tileControl.inputTileHeight ? 0 : iTileHeightLocalTemp;
					unsigned char iTileWidthLocalTemp = (iPaddedWidthLocal > tileControl.leftPadding) ? iPaddedWidthLocal - tileControl.leftPadding : 0;
					unsigned char iTileWidthLocal = iTileWidthLocalTemp > tileControl.inputTileWidth ? 0 : iTileWidthLocalTemp;
					
					unsigned short flatIndex = (unsigned short) iTileHeightLocal*(unsigned char) tileControl.inputTileWidth + (unsigned short) iTileWidthLocal;
					unsigned short iTransferBlockAddress = flatIndex;
					unsigned short iActivationBlockAddress = flatIndex * stripStrideInCache;

					t_streamblock_address numTransferBlocks = sendPadding ? tileControl.numInputChannelCompressionWindows : cacheIAStreamBlockAddress[iTransferBlockAddress];
					unsigned short iTransferBlock = 0;

					//Setting up the bitmasks in the padded transfer blocks
					t_transfer_block dummyBlock;
					dummyBlock.values[0].cluster_values[0] = 0x0;

					bool isLastNeuron = (iNeuWidthLocal == (tileControl.kernelSize - 1)) && (iNeuHeightLocal == (tileControl.kernelSize - 1));

					while (iTransferBlock < numTransferBlocks)
					{
						//bool success = false;
						t_transferblock_tagged taggedBlock;
						taggedBlock.isLast = (((iTransferBlock + 1) == numTransferBlocks) && isLastNeuron)
							? TRUE : FALSE;
						taggedBlock.maxTransportID = (numPeRows - 1);

						if (sendPadding)
						{
							taggedBlock.values = dummyBlock;
						}
						else
						{
							 t_dram_block dramBlock = cacheIABlocks[iActivationBlockAddress+(iTransferBlock >> WIDE_SIZE_OFFSET)];
							 taggedBlock.values = dramBlock.transferBlocks[iTransferBlock & WIDE_SIZE_REMAINDER_MASK];
						}

						write_channel_intel(channel_activation[0][colID], taggedBlock);

						iTransferBlock++;

						//if (success)
						//{
						//	iTransferBlock++;
						//}
					} //iTransferBlock

					if (iNeuWidthLocal >= (tileControl.kernelSize - 1) )
					{
						iNeuWidthLocal = 0;
						iPaddedWidthLocal = iPaddedTileWidth;
						iNeuHeightLocal++;
						iPaddedHeightLocal++;
					}
					else
					{
						iPaddedWidthLocal++;
						iNeuWidthLocal++;
					}
				} //iNeuHeightLocal


				if ( (iPaddedTileWidth + tileControl.stride + tileControl.kernelSize) > (tileControl.inputTileWidth + tileControl.leftPadding + tileControl.rightPadding) )
				{
					iPaddedTileWidth = 0;
					iPaddedTileHeight += tileControl.stride;
				}
				else
				{
					iPaddedTileWidth += tileControl.stride;
				}
			
			} //Loop over input tile

			//Update iFilterGloba
			iFilterGlobal += numPeRows;
		} //while. iFilterGlobal
	
	} //while
} //Input buffer kernel

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))

__kernel void kernelIATee ()
{
	int colID = get_compute_id(0);

	while (true)
	{
		t_dram_block_ia_tagged taggedBlock = read_channel_intel(channel_ia_wide[colID]);

		int destinationCol = (int) taggedBlock.destinationCol;

		if (destinationCol == colID)
		{
			write_channel_intel(channel_ia_wide_local[colID], taggedBlock.dramBlock);
		}
		else
		{
			if (colID < (PE_COLS - 1) )
			{
				write_channel_intel(channel_ia_wide[colID+1], taggedBlock);
			}
		}
	}
}
#endif //IA_MEMORY

#ifdef MEMORY_WRITER
__kernel void kenrelOutputWriter (
	//Pointer to the output activation
	volatile __global t_output_dram_block* restrict pOutputActivation,
	//Pointer to the output activation transfer block count
	volatile __global t_streamblock_address* restrict pOAStreamBlockAddress,

	unsigned int strideExterrnalMemoryOA, //In terms of output dram block

	/*
	Output width tiling parameters
	*/
	unsigned short outputWidth, //Q
	unsigned char sizeOutputTilePerColumnWidth, //TQ_A
	unsigned char sizeOutputTileWidthPerColumnPartial, //Output tile width per column for the final few columns
	//Ad-hoc parameters
	//They are employed to deal with the irregularities in output width tiling caused by 
	//imperfect division between number of PE columns and the overall activation width
	//if 0 <= iterQ < maxRegularQ, then all the PE columns receive tile of width TQ_A
	//else, PE columns 0 to APartial-1 each receives a tile of width TQ_APartial, the other columns are idle.
	unsigned short maxRegularQ, 
	unsigned char sizePartialColumns, //APartial 

	/*
	Output height tiling parameters
	*/
	unsigned short outputHeight, //P
	unsigned char sizeOutputHeightTile, //TP

	/*
	Auxillary
	*/
	unsigned short outputWidthxOutputHeight,

	//Number of groups in the output activations
	unsigned short numOutputChannels,
	unsigned short numChannelsInOutputGroupCurrentLayer,
	unsigned short numChannelsInInputGroupNextLayer,

	/*
	Output modification
	*/
	unsigned char numAccumulatorBitsToRightShift,
	unsigned char enableOutputRelu, //argument cannot be bool
	unsigned char enableSparsification //argument cannot be bool
	)
{
	//Cache of output activation stream block address
	t_streamblock_address cacheOAStreamBlockAddress [4096] __attribute__((numbanks(1)));

	unsigned short numGroups = 1 + (numOutputChannels-1) / numChannelsInInputGroupNextLayer;

	unsigned short iterP = 0;

	//Loops
	while (iterP < outputHeight)
	{
		//Calculate the effectual output tile height
		unsigned char maxTP = (((unsigned short) sizeOutputHeightTile) < (outputHeight - iterP) ) ?
			sizeOutputHeightTile :  (unsigned char) (outputHeight - iterP);


		unsigned short iterQ = 0; //countQCovertedByTQ

		while (iterQ < outputWidth) //tq
		{
			/*
			Input activation tile parameters
			*/
			unsigned char maxTQ_A = (iterQ < maxRegularQ) 
				? sizeOutputTilePerColumnWidth : sizeOutputTileWidthPerColumnPartial;

			unsigned char maxPeCols =  (iterQ < maxRegularQ)
				? PE_COLS : sizePartialColumns;

			unsigned short maxTQ = (unsigned short) maxTQ_A * (unsigned short) maxPeCols;

			//Send the output control
			{
				t_output_buffer_control outputControl;
				outputControl.numOutputTileHeightxWidth = maxTP*maxTQ_A;
				outputControl.outputModifierBits = generateOutputModifier(numAccumulatorBitsToRightShift, enableOutputRelu, enableSparsification);
				outputControl.numOutputChannels = numOutputChannels;
				outputControl.numChannelsInGroupCurrentLayer = numChannelsInOutputGroupCurrentLayer;
				outputControl.numChannelsInGroupNextLayer = numChannelsInInputGroupNextLayer;

				t_output_buffer_control_tagged controlTagged;
				controlTagged.control = outputControl;
				controlTagged.maxColID = (maxPeCols - 1);

				write_channel_intel(channel_output_buffer_control[0], controlTagged);
			}	

			for (unsigned short iGroup=0; iGroup < numGroups; iGroup++)
			{
				for (unsigned char iHeightInTile=0; iHeightInTile < maxTP; iHeightInTile++)
				{
					unsigned short iHeightGlobal = iterP + iHeightInTile;
					for (unsigned char iWidthInTile=0; iWidthInTile < maxTQ_A; iWidthInTile++)
					{
						unsigned short iWidthGlobal = iterQ + iWidthInTile;
						//Index of the x-y strip
						unsigned short indexStrip = 
								iGroup*outputWidthxOutputHeight
								+ iHeightGlobal*outputWidth + iWidthGlobal; //iCol*maxTQ_A is zero

						for (unsigned char iCol=0; iCol<maxPeCols; iCol++)
						{
							unsigned int addressOADramBlockInDram = (unsigned int) indexStrip * (unsigned int) strideExterrnalMemoryOA; //In terms of output dram block
							unsigned short addressTBCountInCache = indexStrip;
							unsigned short clusterCount;

							bool proceed = true;
							while (proceed)
							{
								t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[0]);
								if (receivedBlock.isLast)
								{
									proceed = false;
									clusterCount = outputDramBlock2ClusterCount(receivedBlock.block);
								}
								else 
								{
									//Store the dram count
									pOutputActivation[addressOADramBlockInDram++] = receivedBlock.block;
								}
							} //while

							t_streamblock_address tbBlockCount = clusterCount >> CLUSTER_TO_TRANSFER_BLOCK_SHIFT;
							//Store the cluster count
							cacheOAStreamBlockAddress[addressTBCountInCache] = tbBlockCount;

							indexStrip += maxTQ_A;
						} // for. iCol
					} //for. iWidthInTile
				} //for. iHeightInTile
			} //for. iGroup

			//Transfer the TB block counts from the cache to the DRAM
			{
				unsigned int dramAddressRowStride = (unsigned int) outputWidth;
				unsigned int dramAddressGroupStride = (unsigned int) outputHeight * (unsigned int) outputWidth;
				unsigned int dramAddressGroupBase = (unsigned int) iterP * dramAddressRowStride + iterQ;

				unsigned short cacheAddressRowStride = (unsigned short) maxTQ;
				unsigned short cacheAddressGroupStride = (unsigned short) maxTQ * (unsigned short) maxTP;
				unsigned short cacheAddressGroupBase = 0;

				for (unsigned char iGroup=0; iGroup<numGroups; iGroup++)
				{
					unsigned int dramAddressRowBase = dramAddressGroupBase;
					unsigned short cacheAddressRowBase = cacheAddressGroupBase;
					for (unsigned char iHeightInTile=0; iHeightInTile<maxTP; iHeightInTile++)
					{
						unsigned int dramAddress = dramAddressRowBase;
						unsigned int cacheAddress = cacheAddressRowBase;
						for (unsigned char iWidthInTile=0; iWidthInTile<maxTQ; iWidthInTile++)
						{
							pOAStreamBlockAddress[dramAddress++] = cacheOAStreamBlockAddress[cacheAddress++];
						} //iWidthInTile
						dramAddressRowBase += dramAddressRowStride;
						cacheAddressRowBase += cacheAddressRowStride;
					} //iHeightInTile
					dramAddressGroupBase += dramAddressGroupStride;
					cacheAddressGroupBase += cacheAddressGroupStride;
				} //iGroup
			} //Transfer the TB block counts from the cache to the DRAM

			iterQ += maxTQ;
		} //iterQ

		iterP += maxTP;
	} //iterP

}
#endif //MEMORY_WRITER 

#ifdef OA_MEMORY
#define STATE_OA_BUFFER_FETCH_CLUSTER 0x0
#define STATE_OA_BUFFER_FETCH_WAIT 0x1
#define STATE_OA_BUFFER_SEND_CLUSTER 0X0
#define STATE_OA_BUFFER_SEND_MASK 0X1
#define STATE_OA_BUFFER_SEND_PADDING 0X2
#define STATE_OA_BUFFER_SEND_WAIT 0x3
#define STATE_OA_BUFFER_SEND_END 0x4
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOABuffer ()
{
	int colID = get_compute_id(0);

	private char cacheOutputActivations[OA_CACHE_SIZE];
	t_cluster bufferCompression[16][2]; 

	typedef uint3_t t_state; 

	while (true)
	{
		//Read the output tile control. Stall until read.
		t_output_buffer_control outputControl = read_channel_intel(channel_output_buffer_local[colID]);
		unsigned char outputModifier = outputControl.outputModifierBits;
		
		//Unpack the output modifiers.
		//GOTTCHA: Somehow using function doesn't work !?

		//unsigned char numAccumulatorBitsToRightShift = outputModifier2RightShiftAmount(outputControl.outputModifierBits);
		//unsigned char enableRelu =  outputModifier2EnableSparsification(outputControl.outputModifierBits);
		//unsigned char enableSparsification =  outputModifier2EnableSparsification(outputControl.outputModifierBits);

		unsigned char numAccumulatorBitsToRightShift = outputModifier & 0xF;
		uint1_t enableRelu = (outputModifier >> 4) & 0x1;
		uint1_t enableSparsification = (outputModifier >> 5) & 0x1;
		
		unsigned short numChannelsInGroupCurrentLayer = outputControl.numChannelsInGroupCurrentLayer;
		unsigned short numChannelsInGroupNextLayer = outputControl.numChannelsInGroupNextLayer;

		unsigned char numOutputGroupsCurrentLayer = 1 + (outputControl.numOutputChannels-1) / numChannelsInGroupCurrentLayer;
		//Draining the ouput values from the PE columns, perform modification, and cache them
		{
			unsigned short iterOutChannelGlobal = 0;

			for (unsigned char iterGroup=0; iterGroup < numOutputGroupsCurrentLayer; iterGroup++)
			{
				unsigned short iterOutChannelInGroup = 0;
				while (iterOutChannelInGroup < numChannelsInGroupCurrentLayer)
				{
					unsigned char maxPeRows = ((numChannelsInGroupCurrentLayer - iterOutChannelInGroup) > PE_ROWS)
					? (unsigned char) PE_ROWS : (unsigned char) (numChannelsInGroupCurrentLayer - iterOutChannelInGroup);

					unsigned short outputIndexBase = iterOutChannelGlobal;

					for (unsigned char iterOutHxW=0; iterOutHxW<outputControl.numOutputTileHeightxWidth; iterOutHxW++)
					{
						unsigned short outputIndex = outputIndexBase;

						for (unsigned char iRow=0; iRow<maxPeRows; iRow++)
						{
							t_accumulator wideOutput = read_channel_intel(channel_drain[0][colID]);
							//t_accumulator wideOutput = 0;
							t_operand shortOutput = modifyOutput(wideOutput, numAccumulatorBitsToRightShift, enableRelu);
							cacheOutputActivations[outputIndex++] = shortOutput;
						}

						outputIndexBase += outputControl.numOutputChannels;
					} //iterOutHxW

					iterOutChannelInGroup += (unsigned short) maxPeRows;
					iterOutChannelGlobal += (unsigned short) maxPeRows;

				} //while iterOutChannelInGroup
				
			} //for-loop iterGroup 
		} //Draining the PEs	

		//Group the output activations into clusters, and send them to compression engine
		{
			unsigned short iterOutChannelGlobalBase = 0;;
			unsigned short numClustersInNextInputGroup = 1 + ((numChannelsInGroupNextLayer - 1) >> VALUE_TO_CLUSTER_SHIFT);
			unsigned short numWindowsInNextInputGroup = 1 + ((numClustersInNextInputGroup - 1) / COMPRESSION_WINDOW_SIZE);

			while (iterOutChannelGlobalBase < outputControl.numOutputChannels)
			{
				unsigned short outputIndexBase = iterOutChannelGlobalBase;
				for (unsigned char iterOutHxW=0; iterOutHxW<outputControl.numOutputTileHeightxWidth; iterOutHxW++)
				{
					unsigned short outputIndex = outputIndexBase;
					unsigned short iOCInGroup=0;

					//send clusters in a strip
					char regMask[2];
					unsigned char regSurvivingClusters[2];
					uint1_t regFetchSide = 0x0;
					#pragma unroll
					for (unsigned char i=0; i<2; i++)
					{
						regMask[i] = 0;
						regSurvivingClusters[i] = 0;
					}


					t_state fetchState = STATE_OA_BUFFER_FETCH_CLUSTER;
					unsigned char iClusterInWindowFetch = 0;
					unsigned char iClusterFetch = 0;

					t_state sendState = STATE_OA_BUFFER_SEND_WAIT;
					unsigned char iClusterInWindowSend = 0;
					unsigned char iWindowSend = 0;
					unsigned char iClusterInTransferBlockSend = 0;


					//Send and compress the clusters in a strip
					#pragma ivdep array(bufferCompression)
					//while (iWindowSend < numWindowsInNextInputGroup)
					while (sendState != STATE_OA_BUFFER_SEND_END)
					{
						//Fetch
						t_state nextFetchState = fetchState;
						if (fetchState == STATE_OA_BUFFER_FETCH_CLUSTER)
						{
							bool keep = (enableSparsification == FALSE);
							t_cluster cluster;
							#pragma unroll
							for (unsigned char i=0; i<CLUSTER_SIZE; i++)
							{
								unsigned short tempOC = iOCInGroup + i;
								char tempValue = (tempOC >= numChannelsInGroupNextLayer) ?
									0x0 : cacheOutputActivations[outputIndex+i];
								cluster.cluster_values[i] = tempValue;
								keep = keep || (tempValue != 0x0);
							}

							if (keep)
							{
								regMask[regFetchSide] |= ((char) 1) << iClusterInWindowFetch;
								regSurvivingClusters[regFetchSide]++;
							}

							bufferCompression[iClusterInWindowFetch][regFetchSide] = cluster;
							iClusterFetch++;
							iClusterInWindowFetch++;

							//Gotcha
							iOCInGroup += CLUSTER_SIZE;
							outputIndex += CLUSTER_SIZE;


							if ( (iClusterInWindowFetch == COMPRESSION_WINDOW_SIZE) || (iClusterFetch == numClustersInNextInputGroup) )
							{
								nextFetchState = STATE_OA_BUFFER_FETCH_WAIT; 
							}

						} //STATE_OA_BUFFER_FETCH_CLUSTER)

						//Send
						t_state nextSendState = sendState;
						t_cluster clusterSend;
						if (sendState == STATE_OA_BUFFER_SEND_MASK)
						{
							clusterSend.cluster_values[0] = regMask[(~regFetchSide) & 0x1];

							iClusterInTransferBlockSend++;

							if (0 == regSurvivingClusters[(~regFetchSide) & 0x1])
							{
								if (iClusterInTransferBlockSend < TRANSFER_SIZE)
								{
									nextSendState = STATE_OA_BUFFER_SEND_PADDING;
								}
								else
								{
									nextSendState = STATE_OA_BUFFER_SEND_WAIT;
									iWindowSend++;
								}
							}
							else
							{
								nextSendState = STATE_OA_BUFFER_SEND_CLUSTER;
							}
						} //STATE_OA_BUFFER_SEND_MASK
						else if (sendState == STATE_OA_BUFFER_SEND_CLUSTER)
						{
							clusterSend = bufferCompression[iClusterInWindowSend++][(~regFetchSide) & 0x1];

							iClusterInTransferBlockSend = (iClusterInTransferBlockSend == TRANSFER_SIZE) ? 1 : (iClusterInTransferBlockSend + 1);

							if (iClusterInWindowSend == regSurvivingClusters[(~regFetchSide) & 0x1])
							{
								if (iClusterInTransferBlockSend < TRANSFER_SIZE)
								{
									nextSendState = STATE_OA_BUFFER_SEND_PADDING;
								}
								else
								{
									nextSendState = STATE_OA_BUFFER_SEND_END;
									iWindowSend++;
								}
							}
						} //STATE_OA_BUFFER_SEND_CLUSTER
						else if (sendState == STATE_OA_BUFFER_SEND_PADDING)
						{
							//Make sure the extra padding we send are 0
							#pragma unroll
							for (unsigned int i=0; i<CLUSTER_SIZE; i++)
							{
								clusterSend.cluster_values[i] = 0;
							}

							iClusterInTransferBlockSend++;

							if (iClusterInTransferBlockSend == TRANSFER_SIZE)
							{
								nextSendState = STATE_OA_BUFFER_SEND_END;
								iWindowSend++;
							}
						} //STATE_OA_BUFFER_SEND_PADDING

						//Actual channel action
						if ( (sendState == STATE_OA_BUFFER_SEND_MASK) 
							|| (sendState == STATE_OA_BUFFER_SEND_CLUSTER) 
							|| (sendState == STATE_OA_BUFFER_SEND_PADDING) )
						{
							t_output_cluster_tagged taggedCluster;
							taggedCluster.isLastInStrip = ( ( iWindowSend == numWindowsInNextInputGroup) && (iClusterInTransferBlockSend == TRANSFER_SIZE) );
							taggedCluster.cluster = clusterSend;

							write_channel_intel(channel_output_buffer_to_tee[colID], taggedCluster);
						} //Channel write action

						if ((sendState == STATE_OA_BUFFER_SEND_WAIT) && (fetchState == STATE_OA_BUFFER_FETCH_WAIT))
						{
							regFetchSide = ~regFetchSide;

							iClusterInWindowFetch = 0;
							
							iClusterInTransferBlockSend = 0;
							iClusterInWindowSend = 0;

							if (iClusterFetch < numClustersInNextInputGroup)
							{
								//GOTCHA
								//If we were to assign to fetchState, the compiler will CRASH!!!!
								nextFetchState = STATE_OA_BUFFER_FETCH_CLUSTER;
							}

							if (enableSparsification == TRUE)
							{
								nextSendState = STATE_OA_BUFFER_SEND_MASK;
							}
							else
							{
								nextSendState = STATE_OA_BUFFER_SEND_CLUSTER;
							}
						} //SWAP

						sendState = nextSendState;
						fetchState = nextFetchState;
					} //Send ping-pong
					outputIndexBase += outputControl.numOutputChannels;
				} //iterOutHxW

				//Shift to a different group
				iterOutChannelGlobalBase += numChannelsInGroupNextLayer;
			} //iterOutChannelGlobalBase
		} //Streaming the output to the compressor
	} // while
}



#define STATE_OA_TEE_DRAIN_SELF 0x0
#define STATE_OA_TEE_DRAIN_SELF_SEND_COUNT 0X1
#define STATE_OA_TEE_DRAIN_OTHERS 0x2
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOATee ()
{
	typedef uint2_t t_state;
	int colID = get_compute_id(0);

	while (true)
	{
		/*
		Read the control
		*/
		t_output_buffer_control_tagged taggedControl =
			read_channel_intel(channel_output_buffer_control[colID]);


		/*
		Decode the control
		*/
		unsigned char numOutputTileHeightxWidth = taggedControl.control.numOutputTileHeightxWidth;
		unsigned char numGroups = 1 + (taggedControl.control.numOutputChannels - 1) / (taggedControl.control.numChannelsInGroupNextLayer);
		unsigned char maxColID = taggedControl.maxColID;
		unsigned char numOtherColToCollect = maxColID - colID;

		/*
		Pass on the control to the local
		*/
		write_channel_intel(channel_output_buffer_local[colID], taggedControl.control);

		/*
		Pass on the control to the right if needed
		*/
		if (colID < (PE_COLS - 1))
		{
			if (colID < maxColID)
			{
				write_channel_intel(channel_output_buffer_control[colID+1], taggedControl);
			}
		}

		/*
		Drain the outputs
		*/
		for (unsigned char iterOutput=0; iterOutput<numOutputTileHeightxWidth; iterOutput++)
		{
			t_state state = STATE_OA_TEE_DRAIN_SELF;
			unsigned short iClusters = 0;
			unsigned short iClusterInDram = 0;
			unsigned char iColDrained = 0;
			unsigned char numColToDrain = numOtherColToCollect + 1;
			t_output_dram_block dramBlock;

			while (iColDrained < numColToDrain)
			{
				t_output_dram_block_tagged dramBlockTagged;
				bool writeChannel = false;
				t_state nextState = state;

				if (state == STATE_OA_TEE_DRAIN_SELF)
				{
					t_output_cluster_tagged clusterTagged = read_channel_intel(channel_output_buffer_to_tee[colID]);
					dramBlock.clusters[iClusterInDram] = clusterTagged.cluster;
					iClusters++;

					if ( ( (iClusterInDram + 1) == NUM_CLUSTER_IN_DRAM_SIZE) || (clusterTagged.isLastInStrip) )
					{
						writeChannel = true;

						dramBlockTagged.block = dramBlock;
						dramBlockTagged.isLast = false;

						iClusterInDram = 0;
					}
					else
					{
						iClusterInDram++;
					}
					
					if (clusterTagged.isLastInStrip)
					{
						nextState = STATE_OA_TEE_DRAIN_SELF_SEND_COUNT;
					}
				} //STATE_OA_TEE_DRAIN_SELF
				else if (state == STATE_OA_TEE_DRAIN_SELF_SEND_COUNT)
				{
					writeChannel = true;

					t_output_dram_block countDramBlock = clusterCount2OutputDramBlock(iClusters);

					dramBlockTagged.block = countDramBlock;
					dramBlockTagged.isLast = true;

					nextState = STATE_OA_TEE_DRAIN_OTHERS;
					iColDrained++;
				} //STATE_OA_TEE_DRAIN_SELF_SEND_COUNT
				else if (state == STATE_OA_TEE_DRAIN_OTHERS)
				{

					if (colID < (PE_COLS - 1))
					{	
						writeChannel = true;
						t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[colID+1]);
						dramBlockTagged = receivedBlock;
						if (receivedBlock.isLast)
						{
							iColDrained++;
						}
					}

				} //STATE_OA_TEE_DRAIN_OTHERS

				if (writeChannel)
				{
					write_channel_intel(channel_output_wide[colID], dramBlockTagged);
				}

				state = nextState;
			} //while

		} //for

	} //while

}
#endif  //OA_MEMORY


#ifdef WEIGHT_MEMORY

#define STATE_FILTER_TEE_HEADER 0X0
#define STATE_FILTER_TEE_PAYLOAD 0X1

#define SWITCH_FILTER_TEE_STREAMER 0X0
#define SWITCH_FILTER_TEE_PASS 0x1
/*! kernelFilterTee
	\brief Pass a dram block to the assigned filter streamer or to the next filter tee
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterTee ()
{
	int rowID = get_compute_id(0);
	//uint1_t state = STATE_FILTER_TEE_HEADER;
	//uint1_t direction = SWITCH_FILTER_TEE_PASS;
	//unsigned short transferBlockCount;
	//unsigned short maxTransferBlockSize;
	while (true)
	{
		t_dram_block_w_tagged taggedBlock = read_channel_intel(channel_weight_wide[rowID]);

		int destinationRow = (int) taggedBlock.destinationRow;

		if (destinationRow == rowID)
		{
			write_channel_intel(channel_weight_wide_local[rowID], taggedBlock.dramBlock);
		}
		else
		{
			if (rowID < (PE_ROWS - 1) )
			{
				write_channel_intel(channel_weight_wide[rowID+1], taggedBlock);
			}
		}
	}
}

#define STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL 0X0
#define STATE_FILTER_STREAMER_WRITE_CACHE_WRITE 0X1
#define STATE_FILTER_STREAMER_WRITE_CACHE_WAIT 0X2

#define STATE_FILTER_STREAMER_READ_CACHE_SETUP 0X0
#define STATE_FILTER_STREAMER_READ_CACHE_READ 0X1
#define STATE_FILTER_STREAMER_READ_CACHE_WAIT 0X2

/*! kernelFilterStreamer
	\brief Stream filter values to the PE array
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterBuffer ()
{
	int rowID = get_compute_id(0);

	typedef uint2_t t_state;
	//important to size the bankwidth, otherwise the default 32 bit will be used, resulting in complex store logic
	t_dram_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE))); 
	uint1_t regWriteSide = 0x0;
	unsigned short maxOutputCount[2];
	//unsigned char maxOutputHeightTileSize[2]; //maxTP
	//unsigned char maxOutputWidthTileSize[2]; //maxTQ 
	unsigned short maxTransferBlockInFilter[2]; //maxCg
	unsigned char maxPeCols[2];
	t_accumulator cacheBias[2];

	//=================Write into cache variables=================
	t_state stateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
	unsigned short iTransferBlockInFilterWrite; //iCg

	//=================Read from cache variables=================
	t_state stateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
	unsigned short iTransferBlockInFilterRead = 0; //iCg
	//unsigned char iWidthInOutputTileRead; //pq*A
	//unsigned char iHeightInOutputTileRead; //p
	unsigned short iOutputRead = 0;

	#pragma ivdep array(cacheNzBlocks)
	while (true)
	{
		//===============Write side====================
		t_state nextStateWriteCache = stateWriteCache;
		{
			bool success = false;
			t_dram_block writeBlock;
			if ( (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL)
				|| (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WRITE) )
			{
				writeBlock = read_channel_nb_intel(channel_weight_wide_local[rowID], &success);
			}
			
			if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL)
			{
				if (success)
				{
					t_filter_streamer_control control = 
						dramBlock2FilterStreamerControl(writeBlock);

					//maxOutputHeightTileSize[regWriteSide] = control.maxOutputHeightTileSize;
					//maxOutputWidthTileSize[regWriteSide] = control.maxOutputWidthTileSize;
					maxOutputCount[regWriteSide] = control.numOutputs;
					maxPeCols[regWriteSide] = control.maxPeCols;
					maxTransferBlockInFilter[regWriteSide] = control.numTransferBlocks;
					cacheBias[regWriteSide] = control.bias;

					EMULATOR_PRINT(("[kernelFilterStreamer %d] Received setup packet for a new filter. Number of transfer blocks to follow: %d\n", rowID, control.numTransferBlocks));

					nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_WRITE;
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL
			else if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WRITE)
			{
				if (success)
				{
					unsigned short dramBlockIndex = (iTransferBlockInFilterWrite >> WIDE_SIZE_OFFSET);
					cacheNzBlocks[regWriteSide][dramBlockIndex] = writeBlock;
					iTransferBlockInFilterWrite += WIDE_SIZE;
					if (iTransferBlockInFilterWrite >= maxTransferBlockInFilter[regWriteSide])
					{
						nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_WAIT;
					}
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_WRITE
		} // WRITE

		t_state nextStateReadCache = stateReadCache;
		t_transferblock_tagged weightBlockTagged;
		
		if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_SETUP)
		{
			iTransferBlockInFilterRead = 0;
			if (iOutputRead == maxOutputCount[(~regWriteSide) & 0x1])
			{
				nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
				iOutputRead = 0;
			}
			else
			{
				nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_READ;
				//iOutputRead++;
			}
		} // STATE_FILTER_STREAMER_READ_CACHE_SETUP
		// Send bias, then followed by the clusters
		else if ( stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_READ)
		{
			t_transferblock_tagged weightBlockTagged;

			if (iTransferBlockInFilterRead > 0)
			{
				unsigned short dramIndex = (iTransferBlockInFilterRead - 1) >> WIDE_SIZE_OFFSET;
				unsigned short indexInDramBlock = (iTransferBlockInFilterRead - 1) & WIDE_SIZE_REMAINDER_MASK;
				t_dram_block dramBlock = cacheNzBlocks[(~regWriteSide) & 0x1][dramIndex];
				t_transfer_block tblock = dramBlock.transferBlocks[indexInDramBlock];
				t_transferblock_tagged taggedBlock;
				weightBlockTagged.values = tblock;
				weightBlockTagged.isLast = ((iTransferBlockInFilterRead) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1]) ?
					TRUE : FALSE;
			}
			else
			{
				t_accumulator bias = cacheBias[(~regWriteSide) & 0x1];
				t_transfer_block tblock = bias2TransferBlock(bias);
				weightBlockTagged.values = tblock;
				weightBlockTagged.isLast = false;
			}
			
			weightBlockTagged.maxTransportID = maxPeCols[(~regWriteSide) & 0x1];

			bool success = false;
			success = write_channel_nb_intel(channel_weight[rowID][0], weightBlockTagged);
			if (success)
			{
				/*
				EMULATOR_PRINT(("[kernelFilterStreamer %d] Sent tb %d: %d % d %d %d\n", 
					rowID, 
					iTransferBlockInFilterRead,
					tblock.values[0].cluster_values[0],d
					tblock.values[0].cluster_values[1],
					tblock.values[1].cluster_values[0],
					tblock.values[1].cluster_values[1]));
				*/
				//Omit plus 1 to send the bias
				if ((iTransferBlockInFilterRead) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1])
				{
					nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_SETUP;
					iOutputRead++;
				}
				else
				{
					iTransferBlockInFilterRead++;
				}
			}
		} // STATE_FILTER_STREAMER_READ_CACHE_READ

		if ( (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WAIT) 
			&& (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_WAIT) )
		{
			nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_READ;
			nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
			regWriteSide = (~regWriteSide) & 0x1; 
			EMULATOR_PRINT(("[kernelFilterStreamer %d] Swap\n", rowID));

		}

		stateReadCache = nextStateReadCache;
		stateWriteCache = nextStateWriteCache;

	} // while
}
#endif //WEIGHT_MEMORY

#ifdef PE_SYSTEM

//MAC Operands
typedef struct __attribute__((packed)) {
	char values [SIMD_SIZE*CLUSTER_SIZE];
} t_simd_operand;


__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__ ((autorun))
__kernel void kernelWeightTransport (
	)
{
	
#ifdef FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#else
	int idx = IDX;
	int idy = IDY;
#endif

	//t_simdblock_di_tagged block = read_channel_intel(channel_weightInput);
	//t_simdblock_di peBlock;
	#ifdef DIRECT_COMPRESSION_SIMD
	t_simdblock_bitmask_tagged block;
	t_simdblock_bitmask peBlock;
	#endif

	#ifdef FLEXIBLE_BITMASK_COMPRESSION
	t_transferblock_tagged block;
	t_transferblock_local peBlock;
	#endif
#ifdef FULL_SYSTEM
	block = read_channel_intel(channel_weight[idy][idx]);
#else
	block = read_channel_intel(channel_weight[0][0]);
#endif

	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		#ifdef DIRECT_COMPRESSION_SIMD
			peBlock.values.values[i] = block.values[i];
		#endif
		#ifdef FLEXIBLE_BITMASK_COMPRESSION
			peBlock.values.values[i] = block.values.values[i];
		#endif
	}
	//peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idx < (PE_COLS - 1)){
		if ( idx < block.maxTransportID ) {
			//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
#ifdef FULL_SYSTEM
			write_channel_intel(channel_weight[idy][idx+1], block);
#else
			write_channel_intel(channel_weight[0][1], block);
#endif
		}
	}
#ifdef FULL_SYSTEM
	write_channel_intel(channel_dpWeightInput[idy][idx], peBlock); 
#else
	write_channel_intel(channel_dpWeightInput[0][0], peBlock); 
#endif
}

#define STATE_ACTIVATION_TRANSPORT_READ 0X0
#define STATE_ACTIVATION_TRANSPORT_DRAIN_SELF 0x1
#define STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS 0x2

__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__ ((autorun))
__kernel void kernelActivationTransport (
	)
{
	typedef uint2_t t_state;

#ifdef FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#else
	int idx = IDX;
	int idy = IDY;
#endif

	t_state state = STATE_ACTIVATION_TRANSPORT_READ;
	unsigned char numOtherPSumToDrain;
	unsigned char countOtherPSum;

	while (true)
	{
		t_state nextState = state;
		t_accumulator pSum;
		if (state == STATE_ACTIVATION_TRANSPORT_READ)
		{
			#ifdef DIRECT_COMPRESSION_SIMD
			t_simdblock_bitmask_tagged block;
			t_simdblock_bitmask peBlock;
			#endif

			#ifdef FLEXIBLE_BITMASK_COMPRESSION
			t_transferblock_tagged block;
			t_transferblock_local peBlock;
			#endif
#ifdef FULL_SYSTEM
			block = read_channel_intel(channel_activation[idy][idx]);
#else
			block = read_channel_intel(channel_activation[0][0]);
#endif
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				#ifdef DIRECT_COMPRESSION_SIMD
					peBlock.values.values[i] = block.values[i];
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					peBlock.values.values[i] = block.values.values[i];
				#endif
			}
			//peBlock.streamingBlockIndex = block.streamingBlockIndex;
			peBlock.isLast = block.isLast;

			if (idy < (PE_ROWS - 1)){
				if ( idy < block.maxTransportID ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass an activation block to the output\n") );
#ifdef FULL_SYSTEM
			write_channel_intel(channel_activation[idy+1][idx], block);
#else
			write_channel_intel(channel_activation[0][1], block);
#endif
				}
			}

			if (block.isLast == TRUE)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_DRAIN_SELF;	
				numOtherPSumToDrain = block.maxTransportID - idy;
				countOtherPSum = 0;
			}

#ifdef FULL_SYSTEM
			write_channel_intel(channel_dpActivationInput[idx][idy], peBlock);
#else
			write_channel_intel(channel_dpActivationInput[0][0], peBlock);
#endif
			 

		} //STATE_ACTIVATION_TRANSPORT_READ
		else if (state == STATE_ACTIVATION_TRANSPORT_DRAIN_SELF)
		{
#ifdef FULL_SYSTEM
			pSum = read_channel_intel(channel_peDrainOutput[idy][idx]);
#else
			pSum = read_channel_intel(channel_peDrainOutput[0][0]);
#endif
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT] Drain from PE\n"));
			if (countOtherPSum == numOtherPSumToDrain)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_READ;
			}
			else
			{
				nextState = STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS;
			}
		} //STATE_ACTIVATION_TRANSPORT_DRAIN_SELF
		else if (state == STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS)
		{
			//TODO: change the following in deply
#ifdef FULL_SYSTEM
			if (idy < PE_ROWS - 1)
			{
				pSum = read_channel_intel(channel_drain[idy+1][idx]);
			}
#else
				pSum = read_channel_intel(channel_drain[1][0]);
#endif
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT] Drain from Others\n"));
			countOtherPSum++;
			if (countOtherPSum == numOtherPSumToDrain)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_READ;
			} 
		} //STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS

		if ((state == STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS) 
			|| 
			(state == STATE_ACTIVATION_TRANSPORT_DRAIN_SELF))
		{
#ifdef FULL_SYSTEM
			write_channel_intel(channel_drain[idy][idx], pSum);
#else
			write_channel_intel(channel_drain[0][0], pSum);
#endif
		}

		state = nextState;
	}
}

t_accumulator madd (t_simd_operand activations, t_simd_operand weights) {
	t_accumulator output = 0x0;

	//#ifdef DIRECT_COMPRESSION_SIMD
		#pragma unroll
		for(int i=0; i<SIMD_SIZE*CLUSTER_SIZE/4; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += (t_accumulator) a10_mac_8bitx4(
					activations.values[i*4],
					weights.values[i*4],
					activations.values[i*4+1],
					weights.values[i*4+1],
					activations.values[i*4+2],
					weights.values[i*4+2],
					activations.values[i*4+3],
					weights.values[i*4+3]
					);
			#elif defined (C5SOC)
				output += (t_accumulator) c5_mac_8bitx4(
						activations.values[i*4],
						weights.values[i*4],
						activations.values[i*4+1],
						weights.values[i*4+1],
						activations.values[i*4+2],
						weights.values[i*4+2],
						activations.values[i*4+3],
						weights.values[i*4+3]
						);
			#else
			#error Unsupported FPGA type!
			#endif
		}
	//#endif
	//#ifdef FLEXIBLE_BITMASK_COMPRESSION
	/*
		#pragma unroll
		for(int i=0; i<SIMD_SIZE/2; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += a10_mac_8bitx2(
					activations.values[i*2],
					weights.values[i*2],
					activations.values[i*2+1],
					weights.values[i*2+1]
					);
			#elif defined (C5SOC)
				output += c5_mac_8bitx2(
						activations.values[i*2],
						weights.values[i*2],
						activations.values[i*2+1],
						weights.values[i*2+1]
					);
			#else
			#error Unsupported FPGA type!
			#endif
		}
		*/
	//#endif

	return output;
}

#define ASSEMBLER_STATE_LOAD_BITMASK 0X0
#define ASSEMBLER_STATE_LOAD_VALUE 0X1
//#define ASSEMBLER_STATE_ALIGN 0x2
#define ASSEMBLER_STATE_WAIT 0x2
#define ASSEMBLER_STATE_LOAD_BIAS 0x3

#define BITWIDTH_COMPRESSION_WINDOW_INDEX 3
#define MASK_COMPRESSION_WINDOW_INDEX 0x7

#define MAC_STATE_WAIT 0x0
#define MAC_STATE_ALIGN 0x1
#define MAC_STATE_PROCESS_WINDOW 0x2
#define MAC_STATE_WRITE_PSUM 0x3
#define MAC_STATE_LOAD_BIAS 0x4

__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__((autorun))
__kernel void kernelPE ()
{
	
#if FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#endif

	//================Ping-ponged registers========================
	//BRAM for storing the compression windows
	t_cluster activationWindow[COMPRESSION_WINDOW_SIZE+1][2]; 
	t_cluster weightWindow[COMPRESSION_WINDOW_SIZE+1][2]; 

	//Flags that indicates whether we are at the last window
	uint1_t isLast[2] = {FALSE, TRUE};
	unsigned char bitmaskA[2];
	unsigned char bitmaskW[2];
	t_accumulator bias[2];

	uint1_t regLoadSide = 0x0;

	//========Assembler side registers====================
	unsigned char countActivation;
	unsigned char countWeight;
	unsigned char numActivation;
	unsigned char numWeight;
	uint2_t stateActivation = ASSEMBLER_STATE_LOAD_BIAS;
	uint2_t stateWeight = ASSEMBLER_STATE_LOAD_BIAS;
	//unsigned long alignmentData;


	//=========MAC side logic========================
	uint3_t stateMac = MAC_STATE_WAIT;
	t_accumulator pSum = 0;
	unsigned char countOperands;
	unsigned char numOperands;
	unsigned int indicesW;
	unsigned int indicesA;

	//================Debug====================
	//unsigned short debugCount = 0;

	#pragma ivdep array(activationWindow)
	#pragma ivdep array(weightWindow)
	//#pragma ivdep safelen(7)
	//#pragma ivdep
	while (true)
	{

		//================ACTIVATION========================
		
		uint2_t nextStateActivation = stateActivation;
		{ 
			if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateActivation == ASSEMBLER_STATE_LOAD_VALUE)
			{
				t_transferblock_local activationTransferBlock;
				bool activationReadSuccess;

#ifdef FULL_SYSTEM
				activationTransferBlock = read_channel_nb_intel (
							channel_dpActivationInput[idy][idx],
							&activationReadSuccess
						);
#else
				activationTransferBlock = read_channel_nb_intel (
							channel_dpActivationInput[0][0],
							&activationReadSuccess
						);
#endif
				if (activationReadSuccess)
				{
					//isLastActivation = activationTransferBlock.isLast;
					//DEBUG_PRINT(("[Assembler] Activation read!\n"));

					if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK)
					{
						unsigned char bitmask = activationTransferBlock.values.values[0].cluster_values[0];
						bitmaskA[regLoadSide & 0x01] = bitmask;
						numActivation = popCounter(bitmask);
						countActivation = 0;
						EMULATOR_PRINT(("[assembler] bitmaskA: %#04x \n", bitmask));
					}
					//else
					//{

						//uint3_t offset = (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK) ?
						//	0X1 : 0X0; 

						#pragma unroll
						for (uint3_t i=0; i<TRANSFER_SIZE; i++)
						{
							//if (i >= offset)
							//{
								activationWindow[countActivation+i][regLoadSide & 0x01]
									= activationTransferBlock.values.values[i];
								//EMULATOR_PRINT(("[assembler] activation value: %#04x %#04x \n"
								//	, activationTransferBlock.values.values[i].cluster_values[0] & 0xFF
								//	, activationTransferBlock.values.values[i].cluster_values[1] & 0xFF));
								//EMULATOR_PRINT(("[assembler] activation offset, countActivation: %#04x %#04x\n"
								//	, offset, countActivation));
							//}
						} // for. Transfer the values in the transfer block to the compression window

						//if (debugCount < maxDebugCount)
						//{
							DEBUG_PRINT(("[PE] ActivationTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
								activationTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
								activationTransferBlock.values.values[0].cluster_values[1] & 0xFF,
								activationTransferBlock.values.values[1].cluster_values[0] & 0xFF,
								activationTransferBlock.values.values[1].cluster_values[1] & 0xFF));
						//}

						countActivation += (unsigned char)(TRANSFER_SIZE);
					//}

					//State update
					if (countActivation > numActivation) //countActivation needs to be strictly larger than numActivation
					{
						nextStateActivation = ASSEMBLER_STATE_WAIT;
					}
					else {
						nextStateActivation = ASSEMBLER_STATE_LOAD_VALUE;
					}

				} // if activationReadSuccess
			} // ASSEMBLER_STATE_LOAD_BITMASK || ASSEMBLER_STATE_LOAD_VALUE 
			else if (stateActivation == ASSEMBLER_STATE_LOAD_BIAS)
			{
				EMULATOR_PRINT(("[ACTIVATION ASSEMBLER] Wait for bias\n"));
				nextStateActivation = ASSEMBLER_STATE_WAIT;
			}
		}
		//===================================================

		//================WEIGHT========================
		
		uint2_t nextStateWeight = stateWeight;
		{
			bool weightReadSuccess;
			t_transferblock_local weightTransferBlock;

			if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE
				|| stateWeight == ASSEMBLER_STATE_LOAD_BIAS) 
			{
#ifdef FULL_SYSTEM
				weightTransferBlock = read_channel_nb_intel (
							channel_dpWeightInput[idy][idx],
							&weightReadSuccess
						);
#else
				weightTransferBlock = read_channel_nb_intel (
							channel_dpWeightInput[0][0],
							&weightReadSuccess
						);
#endif
			}

			if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE)
			{
				if (weightReadSuccess)
				{
					isLast[regLoadSide & 0x01] = weightTransferBlock.isLast;
					//DEBUG_PRINT(("[Assembler] Weight read!\n"));

					if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK)
					{
						unsigned char bitmask =  weightTransferBlock.values.values[0].cluster_values[0];
						bitmaskW[regLoadSide & 0x01] = bitmask; 
						numWeight = popCounter(bitmask);
						countWeight = 0;
						EMULATOR_PRINT(("[assembler] bitmaskW: %#04x \n", bitmask));
					}
					//else
					//{

						//uint3_t offset = (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK) ?
						//	0X1 : 0X0; 

						#pragma unroll
						for (uint3_t i=0; i<TRANSFER_SIZE; i++)
						{
							//if (i >= offset)
							//{
								weightWindow[countWeight+i][regLoadSide & 0x01]
									= weightTransferBlock.values.values[i];
								//EMULATOR_PRINT(("[assembler] weight value: %#04x %#04x \n"
								//	, weightTransferBlock.values.values[i].cluster_values[0] & 0xFF
								//	, weightTransferBlock.values.values[i].cluster_values[1] & 0xFF));
							//}
						} // for. Transfer the values in the transfer block to the compression window

						//if (debugCount < maxDebugCount)
						//{
							DEBUG_PRINT(("[PE] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
								weightTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
								weightTransferBlock.values.values[0].cluster_values[1] & 0xFF,
								weightTransferBlock.values.values[1].cluster_values[0] & 0xFF,
								weightTransferBlock.values.values[1].cluster_values[1] & 0xFF));
						//}

						countWeight += (unsigned char)(TRANSFER_SIZE);
					//}

					//State update
					if (countWeight > numWeight) //countWeight needs to be strictly larger than numWeight
					{
						nextStateWeight = ASSEMBLER_STATE_WAIT;
					}
					else 
					{
						nextStateWeight = ASSEMBLER_STATE_LOAD_VALUE;
					}

				} // if weightReadSuccess
			} //ASSEMBLER_STATE_LOAD_BITMASK || ASSEMBLER_STATE_LOAD_VALUE
			else if (stateWeight == ASSEMBLER_STATE_LOAD_BIAS)
			{
				if (weightReadSuccess)
				{
					EMULATOR_PRINT(("[WEIGHT ASSEMBLER] Wait for bias\n"));
					bias[regLoadSide & 0x01] = transferBlock2Bias(weightTransferBlock.values);
					nextStateWeight = ASSEMBLER_STATE_WAIT;
				}
			}
		}
		//===================================================

		//==================MAC states===================
		uint3_t nextStateMac = stateMac;

		if (stateMac == MAC_STATE_ALIGN)
		{
			
			unsigned long alignmentData = operandMatcher8(
				bitmaskW [(~regLoadSide) & 0x1],
				bitmaskA [(~regLoadSide) & 0x1]
			);
			
			
			//unsigned long alignmentData = 0;
			numOperands = (alignmentData >> 48) & 0xFF;
			indicesW = (alignmentData >> 24) & 0xFFFFFF;
			indicesA = (alignmentData) & 0xFFFFFF;
			countOperands = 0; 
			//EMULATOR_PRINT ( ("[aligner]: indicesW: %#06x indicesA: %#06x numOperands: %#04x \n"
			//		, indicesW, indicesA,  numOperands) );

			/*
			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1])
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
			else
			{
				nextStateMac = MAC_STATE_PROCESS_WINDOW;
			}
			*/
			nextStateMac = MAC_STATE_PROCESS_WINDOW;
		}
		else if (stateMac == MAC_STATE_PROCESS_WINDOW)
		{

			t_simd_operand simdActivations;
			t_simd_operand simdWeights;
			t_cluster zeros;
			#pragma unroll
			for (int i=0; i<CLUSTER_SIZE; i++)
			{
				zeros.cluster_values[i] = 0x0;
			}


			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++)
			{
				unsigned char indexW = 
					(indicesW >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster w = ((countOperands + i) < numOperands) ?
					weightWindow[indexW+1][(~regLoadSide) & 0x01] : zeros;
				//char w = weightWindow[i][(~regLoadSide) & 0x1];
				//simdWeights.values[i] = w;

				unsigned char indexA = 
					(indicesA >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster a = ((countOperands + i) < numOperands) ?
					activationWindow[indexA+1][(~regLoadSide) & 0x01] : zeros;
				//char a = activationWindow[i][(~regLoadSide) & 0x1];

				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = a.cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = w.cluster_values[j];
				}

				//EMULATOR_PRINT ( ("[dispatcher]: w0: %#04x w1: %#04x a0: %#04x a1: %#04x \n"
				//	, w.cluster_values[0] & 0xFF, w.cluster_values[1] & 0xFF,  a.cluster_values[0] & 0xFF, a.cluster_values[1] & 0xFF) );
				//EMULATOR_PRINT ( ("[dispatcher]: wIndex: %u aIndex :%u \n", (indexW) & 0xFF, (indexA) & 0xFF));
			}


			t_accumulator tempPSum = madd(simdActivations, simdWeights);
			pSum += tempPSum;
			//if (debugCount < maxDebugCount)
			//	{
			//		DEBUG_PRINT(("[PE Dispatcher] a0, a1, a1, a2: %#04x %#04x %#04x %#04x\n",
			//			simdActivations.values[0] & 0xFF, 
			//			simdActivations.values[1] & 0xFF,
			//			simdActivations.values[2] & 0xFF,
			//			simdActivations.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Dispatcher] w0, w1, w2, w3: %#04x %#04x %#04x %#04x\n",
			//			simdWeights.values[0] & 0xFF, 
			//			simdWeights.values[1] & 0xFF,
			//			simdWeights.values[2] & 0xFF,
			//			simdWeights.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Madd] Psum %#04x\n", pSum));

			//	}		
			countOperands += SIMD_SIZE;
			indicesW = indicesW >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);
			indicesA = indicesA >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);

			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1] == TRUE)
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
		} // if state == MAC_STATE_PROCESS_WINDOW
		else if (stateMac == MAC_STATE_WRITE_PSUM)
		{
			bool writeSuccess;
#ifdef FULL_SYSTEM
			writeSuccess = write_channel_nb_intel(channel_peDrainOutput[idy][idx], pSum);
#else
			writeSuccess = write_channel_nb_intel(channel_peDrainOutput[0][0], pSum);
#endif
			

			//write_channel_intel(channel_peDrainOutput, pSum);
			if (writeSuccess)
			{
				//DEBUG_PRINT(("[MAC] Sending!\n"));
				EMULATOR_PRINT(("[MAC] Commit. pSum value: %#04x \n", pSum));
				//DEBUG_PRINT(("[PE Psum] Commit. %#04x\n", pSum));
				//pSum = 0;
				nextStateMac = MAC_STATE_WAIT;
				//pSum = 0;
			}
		}
		else if (stateMac == MAC_STATE_LOAD_BIAS)
		{
			EMULATOR_PRINT(("[MAC] Load Bias\n"));
			pSum = bias[(~regLoadSide) & 0x1];
			nextStateMac = MAC_STATE_WAIT;
		}


	//===================SWAP===========================
	//Take an extra iteration for swapping, otherwise Fmax is low
		if ( (stateActivation == ASSEMBLER_STATE_WAIT)
			&& (stateWeight == ASSEMBLER_STATE_WAIT)
			&& (stateMac == MAC_STATE_WAIT) )
		{
			nextStateWeight = (isLast[(regLoadSide) & 0x1] == TRUE) ? 
				ASSEMBLER_STATE_LOAD_BIAS : ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateActivation = (isLast[(regLoadSide) & 0x1] == TRUE) ? 
				ASSEMBLER_STATE_LOAD_BIAS : ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateMac = (isLast[(~regLoadSide) & 0x1] == TRUE) ?
				MAC_STATE_LOAD_BIAS: MAC_STATE_ALIGN;

			regLoadSide = ~regLoadSide;
			//countActivation = 0;
			//countWeight = 0;

		}

		//================DEBUG==============================
		//if (debugCount < maxDebugCount)
		//{
		//	DEBUG_PRINT(("[PE] countWeight, %#03x\n", countWeight));
		//	DEBUG_PRINT(("[PE] countActivation: %#03x\n", countActivation));
		//	DEBUG_PRINT(("[PE] countOperands: %#03x\n", countOperands));
		//	DEBUG_PRINT(("[PE] indicesW: %#03x\n", indicesW));
		//	DEBUG_PRINT(("[PE] indicesA: %#03x\n", indicesA));
		//	debugCount++;
		//}
		
		//===================================================

		//================Next state update==================
		stateWeight = nextStateWeight;
		stateActivation = nextStateActivation;
		stateMac = nextStateMac;
		//===================================================
	} // while true
} // end of kernel
#endif //PE_SYSTEM