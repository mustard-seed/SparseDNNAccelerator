`timescale 1 ns / 1ps

module small_buffer_lib_tb ();
	parameter HALF_PERIOD = 5;
	parameter TRANSFER_SIZE = 4;
	parameter CLUSTER_BITWIDTH  = 8;
	parameter COMPRESSION_WINDOW_SIZE = 32;

	localparam COUNT_BITWIDTH = $rtoi($clog2(TRANSFER_SIZE) + 1.0); //Number of bits per element in the accumulated bitmask
	localparam BUFFER_COUNT_WIDTH = COUNT_BITWIDTH;
	localparam BUFFER_BITWIDTH = CLUSTER_BITWIDTH * TRANSFER_SIZE;

	localparam ACCUMULATION_MASK_WIDTH = COUNT_BITWIDTH*COMPRESSION_WINDOW_SIZE;
	localparam COMPRESSION_WINDOW_INDEX_BITWIDTH = $rtoi($ceil($clog2(COMPRESSION_WINDOW_SIZE)));


	reg [COMPRESSION_WINDOW_SIZE-1	:	0] mutualBitmask;
	reg [COMPRESSION_WINDOW_SIZE-1	:	0] bitmask;

	reg[3:0] transerBlockIndex;

	reg [CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] compressedInputs [7:0];
	wire [CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] busInputs;

	assign busInputs = compressedInputs[transerBlockIndex];


	reg [BUFFER_BITWIDTH-1	:	0] regBuffer;
	reg [BUFFER_COUNT_WIDTH-2	:	0] regBufferSize;
	wire [BUFFER_BITWIDTH-1	:	0] macOperands;
	wire [BUFFER_BITWIDTH-1	:	0] newBuffer;
	wire macOutputValid;
	wire [BUFFER_COUNT_WIDTH-2	:	0] newBufferSize;

	wire [ACCUMULATION_MASK_WIDTH-1	:	0] accumulatedIndex;

	wire [COMPRESSION_WINDOW_INDEX_BITWIDTH - 1	:	0]	newCompressionWindowIndex;
	reg [COMPRESSION_WINDOW_INDEX_BITWIDTH - 1	:	0]	regCompressionWindowIndex;

	wire [15:0] maskFilterOutput;
	wire [191:0]	updaterOutput;

	wire [TRANSFER_SIZE-1	:	0] denseBitmask = maskFilterOutput[TRANSFER_SIZE - 1 : 0];
	assign newCompressionWindowIndex = maskFilterOutput[8 + COMPRESSION_WINDOW_INDEX_BITWIDTH - 1	-:	COMPRESSION_WINDOW_INDEX_BITWIDTH];

	assign newBuffer = updaterOutput[127:64];
	assign macOperands = updaterOutput[63:0];
	assign newBufferSize = updaterOutput[128 + BUFFER_COUNT_WIDTH -2	-:	(BUFFER_COUNT_WIDTH-1)];
	assign macOutputValid = updaterOutput[136];

	clMaskAccumulatorWrapper #(
		.BITMASK_LENGTH(COMPRESSION_WINDOW_SIZE),
		.MAX_NUM_OUTPUT(TRANSFER_SIZE)
	)
	inst_mask_accum (
			.bitmask0(bitmask[7:0]),
			.bitmask1(bitmask[15:8]),
			.bitmask2(bitmask[23:16]),
			.bitmask3(bitmask[31:24]),

			.result(accumulatedIndex)
		);

	clMaskFilter #(
		.COMPRESSION_WINDOW_SIZE(COMPRESSION_WINDOW_SIZE),
		.TRANSFER_SIZE          (TRANSFER_SIZE)
	)
	inst_mask_filter (
			.mutualBitmask0      (mutualBitmask[7:0]),
			.mutualBitmask1      (mutualBitmask[15:8]),
			.mutualBitmask2      (mutualBitmask[23:16]),
			.mutualBitmask3      (mutualBitmask[31:24]),

			.accumulatedBitmask0 (accumulatedIndex[7:0]),
			.accumulatedBitmask1 (accumulatedIndex[15:8]),
			.accumulatedBitmask2 (accumulatedIndex[23:16]),
			.accumulatedBitmask3 (accumulatedIndex[31:24]),
			.accumulatedBitmask4 (accumulatedIndex[39:32]),
			.accumulatedBitmask5 (accumulatedIndex[47:40]),
			.accumulatedBitmask6 (accumulatedIndex[55:48]),
			.accumulatedBitmask7 (accumulatedIndex[63:56]),
			.accumulatedBitmask8 (accumulatedIndex[71:64]),
			.accumulatedBitmask9 (accumulatedIndex[79:72]),
			.accumulatedBitmask10(accumulatedIndex[87:80]),
			.accumulatedBitmask11(accumulatedIndex[95:88]),

			.startIndex          (regCompressionWindowIndex),
			.result              (maskFilterOutput)
		);

	clSparseMacBufferUpdate #(
			.TRANSFER_SIZE   (TRANSFER_SIZE),
			.CLUSTER_BITWIDTH(CLUSTER_BITWIDTH)
		)
	update_inst (
			.inputSelectBitmask (denseBitmask),

			.inputTransferBlock0(busInputs[7:0]),
			.inputTransferBlock1(busInputs[15:8]),
			.inputTransferBlock2(busInputs[23:16]),
			.inputTransferBlock3(busInputs[31:24]),

			.currentBuffer0     (regBuffer[7:0]),
			.currentBuffer1     (regBuffer[15:8]),
			.currentBuffer2     (regBuffer[23:16]),
			.currentBuffer3     (regBuffer[31:24]),

			.currentBufferSize  (regBufferSize),

			.result             (updaterOutput)
		);

	wire [7:0] popCount;

	clSmallBufferPopCount # (
			.BITMASK_LENGTH (COMPRESSION_WINDOW_SIZE)
		)
	pop_counter_inst (
			.bitmask0(bitmask[7:0]),
			.bitmask1(bitmask[15:8]),
			.bitmask2(bitmask[23:16]),
			.bitmask3(bitmask[31:24]),

			.result  (popCount)
		);

	reg clock;
	/**
	 * Prepare the inputs
	 */
	 initial begin
	 	clock = 0;
	 	compressedInputs[0][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h04_03_02_01;
	 	compressedInputs[1][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h08_07_06_05;
	 	compressedInputs[2][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h0C_0B_0A_09;
	 	compressedInputs[3][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h10_0F_0E_0D;
	 	compressedInputs[4][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h14_13_12_11;
	 	compressedInputs[5][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h18_17_16_15;
	 	compressedInputs[6][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h1C_1B_1A_19;
	 	compressedInputs[7][CLUSTER_BITWIDTH*TRANSFER_SIZE-1 : 0] = 32'h20_1F_1E_1D;

	 	bitmask = 32'hFF_FF_FF_FF;
	 	mutualBitmask = 32'hF0_0F_F0_0F;
	 	transerBlockIndex = 0;

	 	regBufferSize = 0;
	 	regBuffer = 0;
	 	regCompressionWindowIndex = 0;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;

	 	#(2*HALF_PERIOD)
	 	transerBlockIndex = transerBlockIndex+1;
	 end

	 always begin
	 	#HALF_PERIOD	clock = ~clock;
	 end

	always @ (posedge clock) begin
		regBufferSize <= newBufferSize;
		regBuffer <= newBuffer;
		regCompressionWindowIndex <= newCompressionWindowIndex;
	end
endmodule