`timescale 1 ns / 1 ps

module smallBufferAccumulator 
	# 	(
			parameter MAX_NUM_OUTPUT = 2,
			localparam COUNT_BITWIDTH = $rtoi($clog2(MAX_NUM_OUTPUT) + 1)
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
		operandB = {(COUNT_BITWIDTH-1){inputBit}};
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
			parameter BITMASK_LENGTH = 8, //Number of input bitmask length
			parameter MAX_NUM_OUTPUT = 2,
			localparam COUNT_BITWIDTH = $rtoi($clog2(MAX_NUM_OUTPUT) + 1),
			localparam ACCUMULATION_MASK_WIDTH = COUNT_BITWIDTH*BITMASK_LENGTH
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
			smallBufferAccumulator #(.MAX_NUM_OUTPUT(MAX_NUM_OUTPUT))
				accum_inst(.inputBit(bitmask[i]), .previousAccum(wireAccum), .accum(accumulation[(i+1)*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH]));
		end
	endgenerate
endmodule

//TODO; Need to create the C model for this module
module clMaskAccumulatorWrapper
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		//Break up a long bitmask into bytes
		input wire bitmask0[7:0],
		input wire bitmask1[7:0],
		input wire bitmask2[7:0],
		input wire bitmask3[7:0],
		input wire bitmask4[7:0],
		input wire bitmask5[7:0],
		input wire bitmask6[7:0],
		input wire bitmask7[7:0],

		//Over provisioned to handle the most general case
		output wire [254:0] result
	);

	localparam BITMASK_LENGTH = 8;
	localparam MAX_NUM_OUTPUT = 2;
	localparam COUNT_BITWIDTH = $rtoi($clog2(MAX_NUM_OUTPUT) + 1);

	assign ovalid = ivalid;
	assign oready = 1'b1;

	wire [63:0] bitmask;
	assign bitmask = {bitmask7, bitmask6, bitmask5, bitmask4, bitmask3, bitmask2, bitmask1, bitmask0};

	smallBufferMaskAccumulator #(.BITMASK_LENGTH(BITMASK_LENGTH), .MAX_NUM_OUTPUT(MAX_NUM_OUTPUT))
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
			parameter ENABLE_NEXT_START_INDEX = 1, //1 for adding logic to generate the start index. 0 else
			parameter BITMASK_LENGTH = 16,
			parameter INDEX_BITWIDTH = 5,
			parameter INPUT_ELEMENT_WIDTH = 1,
			parameter MAX_NUM_OUTPUT = 4,
			parameter COUNT_BITWIDTH = 4
		)
	(
		input wire [INPUT_ELEMENT_WIDTH*BITMASK_LENGTH-1 : 0] sparseInput,
		input wire [BITMASK_LENGTH-1 : 0] bitmask,
		input wire [INDEX_BITWIDTH-1 : 0] startIndex,
		output reg [INPUT_ELEMENT_WIDTH*MAX_NUM_OUTPUT-1 : 0 ] denseOutput,
		output wire [COUNT_BITWIDTH-1 : 0] numDenseOutput,
		output wire [INDEX_BITWIDTH-1 : 0] nextStartIndex
	);

	wire [COUNT_BITWIDTH*BITMASK_LENGTH-1 : 0] accumulatedIndex;
	assign numDenseOutput = accumulatedIndex[COUNT_BITWIDTH*BITMASK_LENGTH-1 -: COUNT_BITWIDTH];

	selectGenerator # (
		.ENABLE_NEXT_START_INDEX(ENABLE_NEXT_START_INDEX),
		.BITMASK_LENGTH(BITMASK_LENGTH),
		.MAX_NUM_OUTPUT(MAX_NUM_OUTPUT),
		.COUNT_BITWIDTH(COUNT_BITWIDTH),
		.INDEX_BITWIDTH(INDEX_BITWIDTH)
		)
	inst_select_generator (
		.bitmask(bitmask),
		.startIndex     (startIndex),
		.outAccumulation(accumulatedIndex),
		.nextStartIndex (nextStartIndex)
		);

	genvar iGenOutput;
	generate
		for (iGenOutput = 0; iGenOutput < MAX_NUM_OUTPUT; iGenOutput=iGenOutput+1) begin: GENFOR_OUTPUT
			integer iAccumMask;
			always @ (*) begin
				denseOutput[(iGenOutput+1)*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH] = {INPUT_ELEMENT_WIDTH{1'b0}};
				for (iAccumMask = BITMASK_LENGTH; iAccumMask>0; iAccumMask=iAccumMask-1) begin:FOR_ACCUM
					if (accumulatedIndex[iAccumMask*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] == (iGenOutput+1)) begin
						denseOutput[(iGenOutput+1)*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH] 
							= sparseInput[iAccumMask*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH];
					end
				end
			end
		end
	endgenerate
endmodule

//TODO: Change this module if OpenCL code changes
module clMaskFilter
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input 	wire [15:0] bitmask,
		input 	wire [15:0] sparseInput, //Mutual bitmask
		input 	wire [7:0] startIndex,


		//[7:0] Next start index
		//[15:8] Packed output, only bit 9 and 8 are meaningful
		output wire [15:0] result
	);

	assign ovalid = ivalid;
	assign oready = 1'b1;

	//Make sure to zero out the unused values
	assign {result[7:4], result[15:10]} = 0;

	inputFilter #(
		.ENABLE_NEXT_START_INDEX(1),
		.BITMASK_LENGTH     (8),
		.INDEX_BITWIDTH     (4),
		.INPUT_ELEMENT_WIDTH(1),
		.MAX_NUM_OUTPUT     (2),
		.COUNT_BITWIDTH     (2)
	)
	maskFilter (
		.sparseInput  (sparseInput[7:0]),
		.bitmask      (bitmask[7:0]),
		.denseOutput  (result[9:8]),
		.startIndex    (startIndex[3:0]),
		.numDenseOutput	(),
		.nextStartIndex(result[3:0])
		);
endmodule

module clSparseMacBufferUpdate
	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input 	wire [7:0] inputSelectBitmask,
		//input 	wire [31:0] inputTransferBlock,
		//input 	wire [31:0] currentBuffer,
		input 	wire [7:0] inputTransferBlockA0,
		input 	wire [7:0] inputTransferBlockA1,
		input 	wire [7:0] inputTransferBlockB0,
		input 	wire [7:0] inputTransferBlockB1,

		input 	wire [7:0] currentBufferA0,
		input 	wire [7:0] currentBufferA1,
		input 	wire [7:0] currentBufferB0,
		input 	wire [7:0] currentBufferB1,

		input 	wire [7:0]	currentBufferSize,



		//[31:0] macOutput
		//[63:32] nextBuffer
		//[65:64] Next buffer size
		//[72] macOutputIsValid
		output wire [127:0] result
	);
	assign ovalid = ivalid;
	assign oready = 1'b1;

	wire [31:0] currentBuffer;
	wire [31:0] inputTransferBlock;

	assign currentBuffer = {currentBufferB1, currentBufferB0, currentBufferA1, currentBufferA0};
	assign inputTransferBlock = {inputTransferBlockB1, inputTransferBlockB0, inputTransferBlockA1, inputTransferBlockA0};

	wire [1:0] numClusterValid;
	wire [31:0] denseClusters;
	wire [1:0] totalSize;
	wire macClustersValid;
	wire [1:0] newSize;
	wire [31:0] newBuffer;
	wire [31:0] macClusters;
	reg [63:0] concatenatedBuffer;
	wire [63:0] paddedCurrentBuffer;
	wire [63:0] paddedDenseClusters;

	assign result[63:0] = {newBuffer, macClusters};
	assign result[65:64] = newSize;
	assign result[72] = macClustersValid;

	inputFilter #(
			.ENABLE_NEXT_START_INDEX(0),
			.BITMASK_LENGTH     (2),
			.INDEX_BITWIDTH     (2),
			.INPUT_ELEMENT_WIDTH(16),
			.MAX_NUM_OUTPUT     (2),
			.COUNT_BITWIDTH     (2)
		)
	operandFilter (
			.sparseInput   (inputTransferBlock),
			.bitmask       (inputSelectBitmask[1:0]),
			.startIndex    (2'd0),
			.denseOutput   (denseClusters),
			.nextStartIndex(),
			.numDenseOutput(numClusterValid)
		);

	assign totalSize = numClusterValid + currentBufferSize[1:0];
	assign macClustersValid = totalSize[1];
	assign newSize = {1'b0, totalSize[0]};
	assign paddedCurrentBuffer = {32'd0, currentBuffer};
	assign paddedDenseClusters = {32'd0, denseClusters};

	//Select content for the concatenated buffer
	always @ (*) begin: FOR_CONCATENTATED_BUFFER_SELECT
		integer i;
		for (i=0; i<4; i=i+1) begin
			if (i < {1'b0, currentBufferSize[1:0]}) begin
				concatenatedBuffer[(i+1)*16-1 -: 16] = paddedCurrentBuffer[(i+1)*16-1 -: 16];
			end
			else begin
				if (i < ({1'b0, currentBufferSize[1:0]} + {1'b0, numClusterValid})) begin
					concatenatedBuffer[(i+1)*16-1 -: 16] = paddedDenseClusters[(i-{1'b0, currentBufferSize[1:0]}+1)*16-1 -: 16];
				end
				else begin
					concatenatedBuffer[(i+1)*16-1 -: 16] = {16{1'b0}};
				end
			end
		end
	end

	assign newBuffer = (macClustersValid == 1'b1) ? concatenatedBuffer[4*16-1 -: 32] : concatenatedBuffer[2*16-1 -: 32];
	assign macClusters = concatenatedBuffer[2*16-1 -: 32];
endmodule

module clMaskMatcher
	#	(
			parameter BITMASK_LENGTH = 16,
			parameter INDEX_BITWIDTH = 5,
			parameter INPUT_ELEMENT_WIDTH = 1,
			parameter COUNT_BITWIDTH = 2,
			parameter MAX_NUM_OUTPUT = 2
		) 

	(
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input 	wire [BITMASK_LENGTH-1:0] bitmaskW,
		input 	wire [BITMASK_LENGTH-1:0] bitmaskA,
		input 	wire [INDEX_BITWIDTH-1:0] startIndexA,
		input 	wire [INDEX_BITWIDTH-1:0] startIndexW,

		//[15:0] packed bitmask W
		//[31:16] packed bitmask A
		//[36:32] Next start index of W
		//[38:37] Number of dense output for W
		//[44:40] Next start index of A
		//[46:45] Number of dense output for A
		output wire [63:0] result
	);


	assign ovalid = 1'b1;
	assign oready = 1'b1;

	wire [BITMASK_LENGTH-1:0] bitmaskMutual = bitmaskA & bitmaskW;

	inputFilter #(
		.ENABLE_NEXT_START_INDEX(1),
		.BITMASK_LENGTH     (BITMASK_LENGTH),
		.INDEX_BITWIDTH     (INDEX_BITWIDTH),
		.INPUT_ELEMENT_WIDTH(INPUT_ELEMENT_WIDTH),
		.MAX_NUM_OUTPUT     (MAX_NUM_OUTPUT),
		.COUNT_BITWIDTH     (COUNT_BITWIDTH)
	)
	maskWFilter (
		.sparseInput  (bitmaskMutual),
		.bitmask      (bitmaskW),
		.denseOutput  (result[0+BITMASK_LENGTH*INPUT_ELEMENT_WIDTH-1:0]),
		.startIndex    (startIndexW),
		.numDenseOutput (result[38:37]),
		.nextStartIndex(result[32+INDEX_BITWIDTH-1:32])
		);

	inputFilter #(
		.BITMASK_LENGTH     (BITMASK_LENGTH),
		.INDEX_BITWIDTH     (INDEX_BITWIDTH),
		.INPUT_ELEMENT_WIDTH(INPUT_ELEMENT_WIDTH),
		.MAX_NUM_OUTPUT     (MAX_NUM_OUTPUT),
		.COUNT_BITWIDTH     (COUNT_BITWIDTH)
	)
	maskAFilter (
		.sparseInput  (bitmaskMutual),
		.bitmask      (bitmaskA),
		.denseOutput  (result[16+BITMASK_LENGTH*INPUT_ELEMENT_WIDTH-1:16]),
		.startIndex    (startIndexA),
		.numDenseOutput(result[46:45]),
		.nextStartIndex(result[40+INDEX_BITWIDTH-1:40])
		);

endmodule



