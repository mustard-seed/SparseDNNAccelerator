`timescale 1 ns / 1 ps

module smallBufferAccumulator 
	# 	(
			parameter POSITION = 4,
			parameter COUNT_BITWIDTH = 5,
			parameter INDEX_BITWIDTH = 5,
			parameter MAX_NUM_OUTPUT = 2
		)
	(
		input wire [INDEX_BITWIDTH-1:0] startIndex,
		input wire b,
		input wire [COUNT_BITWIDTH-1:0] previousAccum,
		output reg [COUNT_BITWIDTH-1:0] accum
	);
	always @ (*) begin
		//Priority mux
		if (previousAccum <  MAX_NUM_OUTPUT) begin
			if (startIndex > POSITION) begin
				accum = {COUNT_BITWIDTH{1'b0}};
			end
			else begin
				accum = previousAccum + {{(COUNT_BITWIDTH-1){1'b0}}, b};
			end
		end
		else begin
			accum = previousAccum;
		end
	end
endmodule

/**
 * \brief Count the number of 1s preceding and up to each bit in the bit mask, starting from the startIndex argument
 * The number of ones is capped by the parameter MAX_NUM_OUTPUT 
 * bitmask is counted from LSB to MSB
 * i.e. bitmask[N] is considered AFTER bitmask [N-1]
 * This piece of logic is combinational
 * \input bitmask The bitmask to generate the accumlated indices from
 * \input startIndex The first position from which the accumulation starts
 * \output outAccumulation The accumulated index
 * \output nextStartIndex The index from which the next accumulation cycle should start
 */
module selectGenerator
	#	(
			parameter ENABLE_NEXT_START_INDEX = 1, //1 for adding logic to generate the start index. 0 else
			parameter BITMASK_LENGTH = 16, //Number of input bitmask length
			parameter MAX_NUM_OUTPUT = 16, //Max number of the number of 1s that we care, must be at least 1
			parameter COUNT_BITWIDTH = 5, //Number of bits per accumulated index. LOG2(MAX_NUM_OUTPUT)
			parameter INDEX_BITWIDTH = 5 //Number of bits per position index. floor(LOG2(BITMASK_LENGTH)) + 1
	  	) 
	(
		input wire [BITMASK_LENGTH-1 : 0] bitmask,
		input wire [INDEX_BITWIDTH-1 : 0] startIndex,
		output wire [COUNT_BITWIDTH*BITMASK_LENGTH-1 : 0] outAccumulation,
		output reg [INDEX_BITWIDTH-1 : 0] nextStartIndex

    );

    //===============================================
	//Count the number of 1s preceding and up to each bit in the bit mask. 
	//bitmask is counted from LSB to MSB
	//i.e. bitmask[N] is considered AFTER bitmask [N-1]
	//===============================================
	wire [COUNT_BITWIDTH*BITMASK_LENGTH-1 : 0] accumulation;
	assign outAccumulation = accumulation;
	genvar i;
	generate
		for (i=0; i<BITMASK_LENGTH; i=i+1) begin: FOR_SELECT_GEN
			wire [COUNT_BITWIDTH-1:0] wireAccum = (i==0) ? 0 : accumulation[i*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH];
			smallBufferAccumulator #(.POSITION(i), .COUNT_BITWIDTH(COUNT_BITWIDTH), .INDEX_BITWIDTH(INDEX_BITWIDTH), .MAX_NUM_OUTPUT(MAX_NUM_OUTPUT))
				accum_inst(.startIndex(startIndex), .b(bitmask[i]), .previousAccum(wireAccum), .accum(accumulation[(i+1)*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH]));
		end
	endgenerate

	if (ENABLE_NEXT_START_INDEX == 1) begin
		integer idxI;

		always @(*) begin
			if (accumulation[BITMASK_LENGTH*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] == {COUNT_BITWIDTH{1'b0}}) begin
				nextStartIndex = BITMASK_LENGTH;
			end
			else begin
				for (idxI=BITMASK_LENGTH; idxI>0; idxI=idxI-1) begin
					if (accumulation[idxI*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH] == accumulation[BITMASK_LENGTH*COUNT_BITWIDTH-1 -: COUNT_BITWIDTH]) begin
						nextStartIndex = idxI;
					end
				end
			end
		end
	end
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
module clMaskFilter16c2_1bit
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

	assign ovalid = 1'b1;
	assign oready = 1'b1;

	assign {result[7:5], result[15:10]} = 0;

	inputFilter #(
		.ENABLE_NEXT_START_INDEX(1),
		.BITMASK_LENGTH     (16),
		.INDEX_BITWIDTH     (5),
		.INPUT_ELEMENT_WIDTH(1),
		.MAX_NUM_OUTPUT     (2),
		.COUNT_BITWIDTH     (2)
	)
	maskFilter (
		.sparseInput  (sparseInput),
		.bitmask      (bitmask),
		.denseOutput  (result[9:8]),
		.startIndex    (startIndex[4:0]),
		.numDenseOutput	(),
		.nextStartIndex(result[4:0])
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
	assign ovalid = 1'b1;
	assign oready = 1'b1;

	wire [31:0] currentBuffer;
	wire [31:0] inputTransferBlock;

	assign currentBuffer = {currentbufferB1, currentBufferB0, currentBufferA1, currentBufferA0};
	assign inputTransferBlock = {inputTransferBlockB1, inputTransferBlockB0, inputTransferBlockA1, inputTransferBlockA0};

	wire [1:0] numClusterValid;
	wire [31:0] denseClusters;
	wire [1:0] totalSize;
	wire macClustersValid;
	wire [1:0] newSize;
	reg [31:0] newBuffer;
	reg [31:0] macClusters;
	reg [63:0] concatenatedBuffer;

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
			.bitmask       (inputSelectBitmask),
			.startIndex    (2'd0),
			.denseOutput   (denseClusters),
			.nextStartIndex(),
			.numDenseOutput(numClusterValid)
		);

	assign totalSize = numClusterValid + currentBufferSize[1:0];
	assign macClustersValid = totalSize[1];
	assign newSize = {1'b0, totalSize[0]};

	//Select content for the concatenated buffer
	always @ (*) begin
		integer i;
		for (i=0; i<4; i=i+1) begin
			if (i < {1'b0, currentBufferSize[1:0]}) begin
				concatenatedBuffer[(i+1)*16-1 -: 16] = currentBuffer[(i+1)*16-1 -: 16];
			end
			else begin
				if (i < ({1'b0, currentBufferSize[1:0]} + {1'b0, numClusterValid})) begin
					concatenatedBuffer[(i+1)*16-1 -: 16] = denseClusters[(i-{1'b0, currentBufferSize[1:0]}+1)*16-1 -: 16];
				end
				else begin
					concatenatedBuffer[(i+1)*16-1 -: 16] = {16{1'b0}};
				end
			end
		end
	end

	assign newBuffer = (macClustersValid == 1'b0) ? concatenatedBuffer[4*16-1 -: 32] : concatenatedBuffer[2*16-1 -: 32];
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



