`timescale 1 ns / 1 ps

`define CONST_TRANSFER_SIZE 2
`define CONST_LOG2_TRANSFER_SIZE 1
`define CONST_COMPRESSION_WINDOW_SIZE 8
`define CONST_LOG2_COMPRESSION_WINDOW_SIZE 3
`define CONST_CLUSTER_BITWIDTH 16

module smallBufferAccumulator 
	# 	(
			parameter MAX_NUM_OUTPUT = `CONST_TRANSFER_SIZE,
			parameter LOG2_MAX_NUM_OUTPUT = `CONST_LOG2_TRANSFER_SIZE,

			//DO NOT CHANGLE BELOW
			//parameter COUNT_BITWIDTH = $rtoi($clog2(MAX_NUM_OUTPUT) + 1.0)
			parameter COUNT_BITWIDTH = LOG2_MAX_NUM_OUTPUT + 1
		)
	(
		input wire inputBit,
		input wire [COUNT_BITWIDTH-1:0] previousAccum,
		output wire [COUNT_BITWIDTH-1:0] accum
	);

	reg [COUNT_BITWIDTH-1:0] operandA;
	reg [COUNT_BITWIDTH-1:0] operandB;

	always @ (*) begin
		operandA = previousAccum;
		if (previousAccum == MAX_NUM_OUTPUT) begin
			operandA = 0;
		end
		operandB = {{(COUNT_BITWIDTH-1){1'b0}}, {inputBit}};
	end
	assign accum = operandA + operandB;

endmodule

/**
 * \brief Count the number of 1s preceding and up to each bit in the bitmask, starting from the startIndex argument
 * The number of ones is capped by the parameter MAX_NUM_OUTPUT 
 * bitmask is counted from LSB to MSB
 * i.e. bitmask[N] is considered AFTER bitmask [N-1]
 * This piece of logic is combinational
 * \input bitmask The bitmask to generate the accumlated indices from
 * \output outAccumulation The accumulated index
 */
module smallBufferMaskAccumulator
	#	(
			parameter BITMASK_LENGTH = `CONST_COMPRESSION_WINDOW_SIZE, //Number of input bitmask length
			parameter MAX_NUM_OUTPUT = `CONST_TRANSFER_SIZE,
			parameter LOG2_MAX_NUM_OUTPUT = `CONST_LOG2_TRANSFER_SIZE,

			//DO NOT CHANGLE BELOW
			//parameter COUNT_BITWIDTH = $rtoi($clog2(MAX_NUM_OUTPUT) + 1.0),
			parameter COUNT_BITWIDTH = LOG2_MAX_NUM_OUTPUT + 1,
			parameter ACCUMULATION_MASK_WIDTH = COUNT_BITWIDTH*BITMASK_LENGTH
	  	) 
	(
		input wire [BITMASK_LENGTH-1 : 0] bitmask,
		output wire [ACCUMULATION_MASK_WIDTH-1 : 0] outAccumulation

    );

    //===============================================
	//Count the number of 1s preceding and up to each bit in the bit mask. 
	//bitmask is counted from LSB to MSB
	//i.e. bitmask[N] is considered AFTER bitmask [N-1]
	//===============================================
	wire [ACCUMULATION_MASK_WIDTH-1 : 0] accumulation;
	assign outAccumulation = accumulation;
	genvar i;
	generate
		for (i=0; i<BITMASK_LENGTH; i=i+1) begin: FOR_SELECT_GEN
			wire [COUNT_BITWIDTH-1:0] wireAccum = (i==0) ? 0 : accumulation[i*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH];
			smallBufferAccumulator #(.MAX_NUM_OUTPUT(MAX_NUM_OUTPUT), .LOG2_MAX_NUM_OUTPUT(LOG2_MAX_NUM_OUTPUT))
				accum_inst(.inputBit(bitmask[i]), .previousAccum(wireAccum), .accum(accumulation[(i+1)*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH]));
		end
	endgenerate
endmodule

//TODO; Need to create the C model for this module
module clMaskAccumulatorWrapper # (
		parameter BITMASK_LENGTH = `CONST_COMPRESSION_WINDOW_SIZE,
		parameter MAX_NUM_OUTPUT = `CONST_TRANSFER_SIZE,
		parameter LOG2_MAX_NUM_OUTPUT = `CONST_LOG2_TRANSFER_SIZE
	)
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		//Break up a long bitmask into bytes
		input wire [7:0] bitmask0,
		input wire [7:0] bitmask1,
		input wire [7:0] bitmask2,
		input wire [7:0] bitmask3,
		input wire [7:0] bitmask4,
		input wire [7:0] bitmask5,
		input wire [7:0] bitmask6,
		input wire [7:0] bitmask7,

		//Over provisioned to handle the most general case
		output wire [255:0] result
	);

	//TODO: Change these parameters if TRANSFER_SIZE changes
	localparam COUNT_BITWIDTH = LOG2_MAX_NUM_OUTPUT + 1;

	generate 
		if ((BITMASK_LENGTH*COUNT_BITWIDTH) < 256) begin
			assign result[255 -: (256 - BITMASK_LENGTH*COUNT_BITWIDTH)] = {(256 - BITMASK_LENGTH*COUNT_BITWIDTH){1'b0}};
		end
	endgenerate

	assign ovalid = ivalid;
	assign oready = 1'b1;

	wire [63:0] bitmask;
	assign bitmask = {bitmask7, bitmask6, bitmask5, bitmask4, bitmask3, bitmask2, bitmask1, bitmask0};

	smallBufferMaskAccumulator #(.BITMASK_LENGTH(BITMASK_LENGTH), .MAX_NUM_OUTPUT(MAX_NUM_OUTPUT), .LOG2_MAX_NUM_OUTPUT(LOG2_MAX_NUM_OUTPUT))
		mask_accumulator (.bitmask(bitmask[BITMASK_LENGTH - 1 : 0]), .outAccumulation(result[BITMASK_LENGTH*COUNT_BITWIDTH-1 : 0]));
endmodule


/*
 * \brief Filter and coalesce a sparse input bus with BITMASK_LENGTH elements to NUM_OUTPUT elements using the input mask
 * \input sparseInput: input with gaps. Each element has INPUT_ELEMENT_WIDTH bits.
 * \input bitmask: Little ENDIAN bitmask indicating which bit are dense
 * \output denseOutput: Little endian bus of the dense output. Contains up to NUM_OUTPUT elements.
 * \output numDenseInput: Number of 1s in the bitmask
 * e.g. 
 * sparseInput = {3'b000, 3'b101, 3'b000, 3'b111}
 * bitmask = 4'b0100
 * denseOutput = {3'b000, 3'b000, 3'b000, 3'b101}
 * numDenseInput = 3'b001
*/
module inputFilter
	#	(
			parameter BITMASK_LENGTH = `CONST_COMPRESSION_WINDOW_SIZE,
			parameter INPUT_ELEMENT_WIDTH = 1,
			parameter MAX_NUM_OUTPUT = `CONST_TRANSFER_SIZE,
			parameter LOG2_BITMASK_LENGTH = `CONST_LOG2_COMPRESSION_WINDOW_SIZE,
			parameter LOG2_MAX_NUM_OUTPUT = `CONST_LOG2_TRANSFER_SIZE,

			//DO NOT CHANGLE BELOW
			//parameter COUNT_BITWIDTH = $rtoi($clog2(MAX_NUM_OUTPUT) + 1.0), //Number of bits per element in the accumulated bitmask
			parameter COUNT_BITWIDTH = LOG2_MAX_NUM_OUTPUT + 1, //Number of bits per element in the accumulated bitmask
			parameter INDEX_BITWIDTH = LOG2_BITMASK_LENGTH , //Number of bits to encode the bitmask index position
			parameter INPUT_WIDTH = INPUT_ELEMENT_WIDTH*BITMASK_LENGTH,
			parameter OUTPUT_WIDTH = INPUT_ELEMENT_WIDTH*MAX_NUM_OUTPUT,
			parameter ACCUMULATION_MASK_WIDTH = BITMASK_LENGTH * COUNT_BITWIDTH
		)
	(
		input wire [INPUT_WIDTH - 1 : 0] sparseInput,
		input wire [ACCUMULATION_MASK_WIDTH-1 : 0] accumulatedBitmask,
		input wire [INDEX_BITWIDTH-1 : 0] startIndex,
		output reg [OUTPUT_WIDTH-1 : 0 ] denseOutput,
		output reg [COUNT_BITWIDTH-1 : 0] numDenseOutput,
		output reg [INDEX_BITWIDTH-1 : 0] nextStartIndex
	);

	//wire [ACCUMULATION_MASK_WIDTH-1 : 0] accumulatedIndex;
	//assign numDenseOutput = accumulatedIndex[COUNT_BITWIDTH*BITMASK_LENGTH-1 -: COUNT_BITWIDTH];

	//Select the appropriate filter
	genvar iGenOutput;
	generate
		for (iGenOutput = 0; iGenOutput < MAX_NUM_OUTPUT; iGenOutput=iGenOutput+1) begin: GENFOR_OUTPUT
			/**
			 * Priority encoder
			 */

			 //Index of accumulated index
			integer iAccumMask;
			always @ (*) begin
				denseOutput[(iGenOutput+1)*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH] = {INPUT_ELEMENT_WIDTH{1'b0}};
				for (iAccumMask = BITMASK_LENGTH; iAccumMask>0; iAccumMask=iAccumMask-1) begin:FOR_ACCUM
					//Generate the priority bitmask
					if (
							(accumulatedBitmask[iAccumMask*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] == (iGenOutput+1)) 
							&& ((iAccumMask-1) >= startIndex)
						)
					begin
						denseOutput[(iGenOutput+1)*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH] 
							= sparseInput[iAccumMask*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH];
					end
				end
			end
		end
	endgenerate

	//Search for the next start index.
	//It is the first index (counting from the LSB) at which the accumulated mask value is MAX_NUM_OUTPUT
	integer iNextIndex;
	always @ (*) begin
		nextStartIndex = BITMASK_LENGTH;
		numDenseOutput = accumulatedBitmask[ACCUMULATION_MASK_WIDTH-1 -: COUNT_BITWIDTH];
		for (iNextIndex = BITMASK_LENGTH; iNextIndex > 0; iNextIndex = iNextIndex-1) begin
			if 	(
					(accumulatedBitmask[iNextIndex*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] == MAX_NUM_OUTPUT)
					&&  ((iNextIndex-1) > startIndex)
				)
			begin
				nextStartIndex = iNextIndex; //Do not substract 1
				numDenseOutput = accumulatedBitmask[iNextIndex*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH];
			end
		end
	end

endmodule

//TODO: Change this module if OpenCL code changes
module clMaskFilter # (
		parameter COMPRESSION_WINDOW_SIZE = `CONST_COMPRESSION_WINDOW_SIZE, //Bitmask length 
		parameter TRANSFER_SIZE = `CONST_TRANSFER_SIZE, //MAX_NUM_OUTPUT,
		parameter LOG2_COMPRESSION_WINDOW_SIZE = `CONST_LOG2_COMPRESSION_WINDOW_SIZE,
		parameter LOG2_TRANSFER_SIZE = `CONST_LOG2_TRANSFER_SIZE
    )
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		//Bytes of the mutual mask
		input 	wire [7:0] mutualBitmask0,
		input 	wire [7:0] mutualBitmask1,
		input 	wire [7:0] mutualBitmask2,
		input   wire [7:0] mutualBitmask3,
		input 	wire [7:0] mutualBitmask4,
		input 	wire [7:0] mutualBitmask5,
		input 	wire [7:0] mutualBitmask6,
		input   wire [7:0] mutualBitmask7,


		//Bytes of the accumulated bitmask
		//Might not need all of them
		input 	wire [7:0]	accumulatedBitmask0,
		input 	wire [7:0]	accumulatedBitmask1,
		input 	wire [7:0]	accumulatedBitmask2,
		input 	wire [7:0]	accumulatedBitmask3,
		input 	wire [7:0]	accumulatedBitmask4,
		input 	wire [7:0]	accumulatedBitmask5,
		input 	wire [7:0]	accumulatedBitmask6,
		input 	wire [7:0]	accumulatedBitmask7,
		input 	wire [7:0]	accumulatedBitmask8,
		input 	wire [7:0]	accumulatedBitmask9,
		input 	wire [7:0]	accumulatedBitmask10,
		input 	wire [7:0]	accumulatedBitmask11,
		input 	wire [7:0]	accumulatedBitmask12,
		input 	wire [7:0]	accumulatedBitmask13,
		input 	wire [7:0]	accumulatedBitmask14,
		input 	wire [7:0]	accumulatedBitmask15,
		input 	wire [7:0]	accumulatedBitmask16,
		input 	wire [7:0]	accumulatedBitmask17,
		input 	wire [7:0]	accumulatedBitmask18,
		input 	wire [7:0]	accumulatedBitmask19,
		input 	wire [7:0]	accumulatedBitmask20,
		input 	wire [7:0]	accumulatedBitmask21,
		input 	wire [7:0]	accumulatedBitmask22,
		input 	wire [7:0]	accumulatedBitmask23,
		input 	wire [7:0]	accumulatedBitmask24,
		input 	wire [7:0]	accumulatedBitmask25,
		input 	wire [7:0]	accumulatedBitmask26,
		input 	wire [7:0]	accumulatedBitmask27,
		input 	wire [7:0]	accumulatedBitmask28,
		input 	wire [7:0]	accumulatedBitmask29,
		input 	wire [7:0]	accumulatedBitmask30,
		input 	wire [7:0]	accumulatedBitmask31,

		input 	wire [7:0] startIndex,


		//[7:0] Packed mutual bitmask. Only [TRANSFER_SIZE-1 : 0] are meaningful
		//[15:8] Next start index. Only [8 + INDEX_BITWIDTH - 1 -: INDEX_BITWIDTH] are meanintful
		output wire [15:0] result
	);

	localparam INDEX_BITWIDTH = LOG2_COMPRESSION_WINDOW_SIZE;

	localparam COUNT_BITWIDTH = LOG2_TRANSFER_SIZE + 1; //Number of bits per element in the accumulated bitmask
	localparam ACCUMULATION_MASK_WIDTH = COMPRESSION_WINDOW_SIZE * COUNT_BITWIDTH;

	assign ovalid = ivalid;
	assign oready = 1'b1;

	//Make sure to zero out the unused values
	generate 
		if (TRANSFER_SIZE < 8) begin
			assign result[7 -: (8 - TRANSFER_SIZE)] = 0;
		end
		if (INDEX_BITWIDTH < 8) begin
			assign result[15 -: (8 - INDEX_BITWIDTH)] = 0;
		end
	endgenerate

	wire [255 : 0] accumulatedBitmask
		= {
			accumulatedBitmask31,
			accumulatedBitmask30,
			accumulatedBitmask29,
			accumulatedBitmask28,
			accumulatedBitmask27,
			accumulatedBitmask26,
			accumulatedBitmask25,
			accumulatedBitmask24,
			accumulatedBitmask23,
			accumulatedBitmask22,
			accumulatedBitmask21,
			accumulatedBitmask20,
			accumulatedBitmask19,
			accumulatedBitmask18,
			accumulatedBitmask17,
			accumulatedBitmask16,
			accumulatedBitmask15,
			accumulatedBitmask14,
			accumulatedBitmask13,
			accumulatedBitmask12,
			accumulatedBitmask11,
			accumulatedBitmask10,
			accumulatedBitmask9,
			accumulatedBitmask8,
			accumulatedBitmask7,
			accumulatedBitmask6,
			accumulatedBitmask5,
			accumulatedBitmask4,
			accumulatedBitmask3,
			accumulatedBitmask2,
			accumulatedBitmask1,
			accumulatedBitmask0
		   };

	wire [63 : 0] mutualBitmask
		= {
			mutualBitmask7,
			mutualBitmask6,
			mutualBitmask5,
			mutualBitmask4,
			mutualBitmask3,
			mutualBitmask2,
			mutualBitmask1,
			mutualBitmask0
		};
	
	inputFilter #(
		.BITMASK_LENGTH     (COMPRESSION_WINDOW_SIZE),
		.INPUT_ELEMENT_WIDTH    (1),
		.MAX_NUM_OUTPUT     (TRANSFER_SIZE),
		.LOG2_BITMASK_LENGTH 	(LOG2_COMPRESSION_WINDOW_SIZE),
		.LOG2_MAX_NUM_OUTPUT 	(LOG2_TRANSFER_SIZE)
		)
	maskFilter (
		.sparseInput  (mutualBitmask[COMPRESSION_WINDOW_SIZE - 1 : 0]),
		.accumulatedBitmask      (accumulatedBitmask[ACCUMULATION_MASK_WIDTH - 1 : 0]),
		.denseOutput  (result [TRANSFER_SIZE - 1 : 0]),
		.startIndex    (startIndex [INDEX_BITWIDTH - 1 : 0]),
		.numDenseOutput	(),
		.nextStartIndex(result[8 + INDEX_BITWIDTH - 1 -: INDEX_BITWIDTH])
		);
endmodule

module clSparseMacBufferUpdate # (
		parameter TRANSFER_SIZE = `CONST_TRANSFER_SIZE,
		parameter CLUSTER_BITWIDTH = `CONST_CLUSTER_BITWIDTH,
		parameter LOG2_TRANSFER_SIZE = `CONST_LOG2_TRANSFER_SIZE
	)
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input 	wire [7:0] inputSelectBitmask,

		//Bytes of the input buffer
		input 	wire [7:0] inputTransferBlock0,
		input 	wire [7:0] inputTransferBlock1,
		input 	wire [7:0] inputTransferBlock2,
		input 	wire [7:0] inputTransferBlock3,
		input 	wire [7:0] inputTransferBlock4,
		input 	wire [7:0] inputTransferBlock5,
		input 	wire [7:0] inputTransferBlock6,
		input 	wire [7:0] inputTransferBlock7,

		//Bytes of the buffer
		input 	wire [7:0] currentBuffer0,
		input 	wire [7:0] currentBuffer1,
		input 	wire [7:0] currentBuffer2,
		input 	wire [7:0] currentBuffer3,
		input 	wire [7:0] currentBuffer4,
		input 	wire [7:0] currentBuffer5,
		input 	wire [7:0] currentBuffer6,
		input 	wire [7:0] currentBuffer7,

		input 	wire [7:0]	currentBufferSize,



		//[63:0] macOutput
		//[127:64] nextBuffer
		//result[128 + BUFFER_COUNT_WIDTH -2	-:	(BUFFER_COUNT_WIDTH-1)]  Next buffer size
		//[136] macOutputIsValid
		//Others: not used
		output wire [255:0] result
	);

	localparam INDEX_BITWIDTH = LOG2_TRANSFER_SIZE;

	localparam COUNT_BITWIDTH = LOG2_TRANSFER_SIZE + 1; //Number of bits per element in the accumulated bitmask
	localparam ACCUMULATION_MASK_WIDTH = TRANSFER_SIZE * COUNT_BITWIDTH;
	localparam BUFFER_BITWIDTH = CLUSTER_BITWIDTH * TRANSFER_SIZE;
	localparam BUFFER_COUNT_WIDTH = COUNT_BITWIDTH;
	localparam CONCATENTATED_BUFFER_COUNT_WIDTH = BUFFER_COUNT_WIDTH;

	assign ovalid = ivalid;
	assign oready = 1'b1;

	wire [ACCUMULATION_MASK_WIDTH - 1 : 0] accumulationMask;

	wire [BUFFER_BITWIDTH-1:0] currentBuffer;
	wire [BUFFER_BITWIDTH-1:0] inputTransferBlock;

	assign currentBuffer = {currentBuffer7, currentBuffer6, currentBuffer5, currentBuffer4, currentBuffer3, currentBuffer2, currentBuffer1, currentBuffer0};
	assign inputTransferBlock = {inputTransferBlock7, inputTransferBlock6, inputTransferBlock5, inputTransferBlock4, inputTransferBlock3, inputTransferBlock2, inputTransferBlock1, inputTransferBlock0};

	wire [BUFFER_COUNT_WIDTH-1	:	0] numClusterValid;
	wire [BUFFER_BITWIDTH-1	:	0] denseClusters;
	wire [CONCATENTATED_BUFFER_COUNT_WIDTH - 1	:	0] totalSize;
	wire macClustersValid;
	wire [BUFFER_COUNT_WIDTH - 2 	:	0] newSize;
	wire [BUFFER_BITWIDTH-1	:	0] newBuffer;
	wire [BUFFER_BITWIDTH-1	:	0] macClusters;
	reg  [2*BUFFER_BITWIDTH-1	:	0] concatenatedBuffer;
	wire [2*BUFFER_BITWIDTH-1	:	0] paddedCurrentBuffer;
	wire [2*BUFFER_BITWIDTH-1	:	0] paddedDenseClusters;

	assign result[63:0] 	= 	macClusters;
	assign result[127:64] 	= 	newBuffer;
	assign result[128 + BUFFER_COUNT_WIDTH -2	-:	(BUFFER_COUNT_WIDTH-1)] 	= 	newSize;
	assign result[136] 		=	macClustersValid;
	assign result[255:137] = {(255-137+1){1'b0}};

	smallBufferMaskAccumulator #(
			.BITMASK_LENGTH         (TRANSFER_SIZE),
			.MAX_NUM_OUTPUT         (TRANSFER_SIZE),
			.LOG2_MAX_NUM_OUTPUT    (LOG2_TRANSFER_SIZE)
		)
	inst_maskAccum (
			.bitmask (inputSelectBitmask[TRANSFER_SIZE  - 1 : 0]),
			.outAccumulation(accumulationMask)
		);

	inputFilter #(
			.BITMASK_LENGTH     (TRANSFER_SIZE),
			.INPUT_ELEMENT_WIDTH(CLUSTER_BITWIDTH),
			.MAX_NUM_OUTPUT     (TRANSFER_SIZE),
			.LOG2_BITMASK_LENGTH (LOG2_TRANSFER_SIZE),
			.LOG2_MAX_NUM_OUTPUT (LOG2_TRANSFER_SIZE)
		)
	operandFilter (
			.sparseInput   (inputTransferBlock [TRANSFER_SIZE*CLUSTER_BITWIDTH-1 : 0]),
			.accumulatedBitmask       (accumulationMask),
			.startIndex    (0),
			.denseOutput   (denseClusters),
			.nextStartIndex(),
			.numDenseOutput(numClusterValid)
		);

	assign totalSize = {1'b0, numClusterValid[BUFFER_COUNT_WIDTH - 1 : 0]} + {2'b00 ,currentBufferSize[BUFFER_COUNT_WIDTH - 2 : 0]};
	assign macClustersValid = totalSize[CONCATENTATED_BUFFER_COUNT_WIDTH - 1];
	assign newSize = totalSize[BUFFER_COUNT_WIDTH -2 : 0]; //New size has one less bit!!!!
	assign paddedCurrentBuffer = {{BUFFER_BITWIDTH{1'b0}}, currentBuffer};
	assign paddedDenseClusters = {{BUFFER_BITWIDTH{1'b0}}, denseClusters};

	//Select content for the concatenated buffer
	always @ (*) begin: FOR_CONCATENTATED_BUFFER_SELECT
		integer i;
		for (i=0; i<(2*TRANSFER_SIZE); i=i+1) begin
			if (i < {2'b00, currentBufferSize[BUFFER_COUNT_WIDTH-2 : 0]}) begin
				concatenatedBuffer[(i+1)*CLUSTER_BITWIDTH-1 -: CLUSTER_BITWIDTH] = paddedCurrentBuffer[(i+1)*CLUSTER_BITWIDTH-1 -: CLUSTER_BITWIDTH];
			end
			else begin
				if (i < ({2'b00, currentBufferSize[BUFFER_COUNT_WIDTH-2 : 0]} + {1'b0, numClusterValid})) begin
					concatenatedBuffer[(i+1)*CLUSTER_BITWIDTH-1 -: CLUSTER_BITWIDTH] 
						= paddedDenseClusters[(i-{2'b00, currentBufferSize[BUFFER_COUNT_WIDTH - 2:0]}+1)*CLUSTER_BITWIDTH-1 -: CLUSTER_BITWIDTH];
				end
				else begin
					concatenatedBuffer[(i+1)*CLUSTER_BITWIDTH-1 -: CLUSTER_BITWIDTH] = {CLUSTER_BITWIDTH{1'b0}};
				end
			end
		end
	end

	assign newBuffer = (macClustersValid == 1'b1) ? 
		concatenatedBuffer[2*TRANSFER_SIZE*CLUSTER_BITWIDTH-1 -: TRANSFER_SIZE*CLUSTER_BITWIDTH] 
		: concatenatedBuffer[TRANSFER_SIZE*CLUSTER_BITWIDTH-1 -: TRANSFER_SIZE*CLUSTER_BITWIDTH];

	assign macClusters = concatenatedBuffer[TRANSFER_SIZE*CLUSTER_BITWIDTH-1 -: TRANSFER_SIZE*CLUSTER_BITWIDTH];
endmodule


module smallBufferPopCount # (
		parameter BITMASK_LENGTH = `CONST_COMPRESSION_WINDOW_SIZE,
		parameter LOG2_BITMASK_LENGTH = `CONST_LOG2_COMPRESSION_WINDOW_SIZE,
		//DO NOT CHANGE BELOW
		parameter COUNT_BITWIDTH = LOG2_BITMASK_LENGTH + 1
	)
	(
		input wire [BITMASK_LENGTH - 1:0] bitmask,
		output wire [COUNT_BITWIDTH - 1:0] sum
	);

	reg [BITMASK_LENGTH*COUNT_BITWIDTH-1 : 0] counterBank;
	assign sum = counterBank[BITMASK_LENGTH*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH];
	integer i;
	always @ (*) begin
		counterBank[COUNT_BITWIDTH - 1 : 0] = {{(COUNT_BITWIDTH-1){1'b0}}, bitmask[0]};
		for (i=1; i<BITMASK_LENGTH; i=i+1) begin
			counterBank[(i+1)*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] = 
				counterBank[i*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] + {{(COUNT_BITWIDTH-1){1'b0}}, bitmask[i]};
		end
	end
endmodule

module clSmallBufferPopCount #(
		parameter BITMASK_LENGTH = `CONST_COMPRESSION_WINDOW_SIZE,
		parameter LOG2_BITMASK_LENGTH 	= 	`CONST_LOG2_COMPRESSION_WINDOW_SIZE
	)
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input wire [7:0] bitmask0,
		input wire [7:0] bitmask1,
		input wire [7:0] bitmask2,
		input wire [7:0] bitmask3,
		input wire [7:0] bitmask4,
		input wire [7:0] bitmask5,
		input wire [7:0] bitmask6,
		input wire [7:0] bitmask7,

		output wire [7:0] result

	);
	localparam COUNT_BITWIDTH = LOG2_BITMASK_LENGTH + 1;

	wire [COUNT_BITWIDTH*BITMASK_LENGTH-1 : 0] bitmask = 
		{bitmask7, bitmask6, bitmask5, bitmask4, bitmask3, bitmask2, bitmask1, bitmask0};

	generate 
		if (COUNT_BITWIDTH < 8) begin
			assign result[7 -: (8-COUNT_BITWIDTH)] = 0;
		end
	endgenerate

	smallBufferPopCount #(
			.BITMASK_LENGTH(BITMASK_LENGTH),
			.LOG2_BITMASK_LENGTH (LOG2_BITMASK_LENGTH)
		)
	inst_pop_counter (.bitmask(bitmask), .sum(result[COUNT_BITWIDTH-1 : 0]));
endmodule