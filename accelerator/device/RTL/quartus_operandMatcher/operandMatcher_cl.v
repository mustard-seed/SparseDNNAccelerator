`timescale 1 ps / 1 ps


module accumulator 
	# (
		parameter BITMASK_LENGTH = 8,
		parameter INDEX_BITWIDTH = 3
		)
	(
	input wire [BITMASK_LENGTH-1 : 0] bitmask,
	output wire [INDEX_BITWIDTH*BITMASK_LENGTH-1 : 0] index
	);

	//===============================================
	//Count the number of 1s preceding each bit in the bit mask. 
	//bitmask is counted from MSB to LSB
	//i.e. bitmask[N] is considered BEFORE bitmask [N-1]
	//===============================================

	reg [INDEX_BITWIDTH*BITMASK_LENGTH-1 : 0] wireIndex;
	integer i;
	assign index = wireIndex;
	//assign wireIndex[63:60] = 4'b0000;

	always @ (*) begin
		wireIndex[INDEX_BITWIDTH-1 : 0] = {INDEX_BITWIDTH{1'b0}};
		for (i=1; i<BITMASK_LENGTH; i=i+1) begin: accum
			//assign wireIndex [i*4-1 -: 4] = {3'b000, bitmask[i]} + wireIndex[(i+1)*4-1-:4];
			wireIndex[(i+1)*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH] = 
			{ {(INDEX_BITWIDTH - 1){1'b0}}, bitmask[i-1] } + wireIndex[i*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH];
		end
	end
	
endmodule

module extendAndMask 
	# (
		parameter BITMASK_LENGTH = 8,
		parameter INDEX_BITWIDTH = 3
	)
	(
	input wire [INDEX_BITWIDTH*BITMASK_LENGTH-1:0] unmaskedIndices,
	input wire [BITMASK_LENGTH-1:0] bitmask,
	output reg [INDEX_BITWIDTH*BITMASK_LENGTH-1:0] maskedIndices
	);
	//===================================================
	//Extend and applies a N-bit binary mask to an array of N M-bit indicies.
	//===================================================

	integer i;
	always @ (*) begin
		for (i=BITMASK_LENGTH; i>0; i=i-1) begin: genloop
			maskedIndices[i*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH] = 
				{INDEX_BITWIDTH{bitmask[i-1]}} & unmaskedIndices[i*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH];
		end
	end
endmodule


module indexExtraction 
	# (
		parameter NUM_INDEX = 8,
		parameter BITWIDTH_INDEX = 3,
		parameter BITWIDTH_NUMBER = 1 
	)
	(
		input wire [NUM_INDEX*BITWIDTH_INDEX-1:0] shiftList,
		input wire [NUM_INDEX*BITWIDTH_NUMBER-1:0] inputWithGap,
		output reg [BITWIDTH_NUMBER-1:0] outputNumber
	);
	//===================================================
	//Finds the last N-bit integer inside a list of integers (a.k.a shiftList)
	//that has a position index k equal to its value
	//The integers are counted from the left to the right
	//Then assign the output Q-bit integer with the Q-bit integer at k in the input list
	//===================================================
	integer i;
	// combinational block
	always @ (*) begin
		//Priority encoder
		//Default value
		outputNumber = {BITWIDTH_NUMBER{1'b0}};

		for (i=0; i<NUM_INDEX; i=i+1) begin
			if (shiftList[ (i+1)*BITWIDTH_INDEX-1 -: BITWIDTH_INDEX] == i) begin
				outputNumber = inputWithGap[(i+1)*BITWIDTH_NUMBER-1 -: BITWIDTH_NUMBER];
			end
		end
	end

endmodule

module collapseBubble 
	# (
		parameter BITMASK_LENGTH = 8,
		parameter BITWIDTH_INDEX = 3,
		parameter BITWIDTH_NUMBER = 1
	)
	(
	input wire [BITMASK_LENGTH*BITWIDTH_NUMBER-1:0] inputWithGap,
	input wire [BITMASK_LENGTH*BITWIDTH_INDEX-1:0] positions,
	output wire [BITMASK_LENGTH*BITWIDTH_NUMBER-1:0] outputWithoutGap
	);

	//Take the a list of INDEX_BITWIDTH-bit indices separated by 0s and collaspse them
	//positions indicate the number of 0s in front of each index in the index with gap list
	//indexWithoutGap is the output
	genvar i;
	generate
		for (i=0; i<BITMASK_LENGTH; i=i+1) begin: genloop
			indexExtraction #(.NUM_INDEX(BITMASK_LENGTH - i), .BITWIDTH_INDEX(BITWIDTH_INDEX), .BITWIDTH_NUMBER(BITWIDTH_NUMBER)) 
			inst_indexExtraction (
					.shiftList(positions[BITMASK_LENGTH*BITWIDTH_INDEX-1 : i*BITWIDTH_INDEX]),
					.inputWithGap(inputWithGap[BITMASK_LENGTH*BITWIDTH_NUMBER-1 : i*BITWIDTH_NUMBER]),
					.outputNumber    (outputWithoutGap[(i+1)*BITWIDTH_NUMBER-1 -: BITWIDTH_NUMBER])
				);
		end
	endgenerate

endmodule

module popCounter
	# (
		parameter BITMASK_LENGTH = 8,
		parameter BITWIDTH_OUTPUT = 4
	)
	(
		input wire [BITMASK_LENGTH-1 : 0] bitmask,
		output wire [BITWIDTH_OUTPUT-1 : 0] count
	);

	reg [BITMASK_LENGTH*BITWIDTH_OUTPUT-1 : 0] counterBank;

	integer i;
	always @ (*) begin
		counterBank [BITWIDTH_OUTPUT-1 : 0] = {{(BITWIDTH_OUTPUT-1){1'b0}}, bitmask[0]};
		for (i=1; i<BITMASK_LENGTH; i=i+1) begin
				counterBank[(i+1)*BITWIDTH_OUTPUT-1 -: BITWIDTH_OUTPUT]
					= counterBank[i*BITWIDTH_OUTPUT-1 -: BITWIDTH_OUTPUT] + {{(BITWIDTH_OUTPUT-1){1'b0}}, bitmask[i]};
		end
	end

	assign count = counterBank[BITMASK_LENGTH*BITWIDTH_OUTPUT-1 -: BITWIDTH_OUTPUT];

endmodule

module operandMatcher8 (
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,
		
		input  wire [7:0]  bitmaskW, //weight bitmask
		input  wire [7:0]  bitmaskA, //Activation bitmask
		
		output wire [63:0] result  //  [23:0] Packed indices of A; [47:24] Packed indices of W; [51:48] Number of pairs; [63:52]: Padding
	);
	
	//Ignoring resetn, ivalid, and iready

	//tie the following signals to high
	assign ovalid = 1'b1;
	assign oready = 1'b1;

	//Localdef
	localparam BITMASK_LENGTH = 8;
	localparam INDEX_BITWIDTH = 3;
	localparam ACCUM_LENGTH = BITMASK_LENGTH * INDEX_BITWIDTH;
	localparam BITWIDTH_COUNT = 4;


	//Declare the registers
	//reg [BITMASK_LENGTH-1:0] regBitmaskW;
	//reg [BITMASK_LENGTH-1:0] regBitmaskA;
	//reg [BITMASK_LENGTH-1:0] regMutualBitmask;
	//reg [ACCUM_LENGTH-1:0] regShiftAccum;
	//reg [ACCUM_LENGTH-1:0]  regActivationMaskedAccum;
	//reg [ACCUM_LENGTH-1:0]  regWeightMaskedAccum;
	reg [ACCUM_LENGTH-1:0]  regActivationDenseAccum;
	reg [ACCUM_LENGTH-1:0]  regWeightDenseAccum;
	reg [BITWIDTH_COUNT-1:0] regBitWidthCount;

	//Structural wires
	wire [BITMASK_LENGTH-1:0] wireMutualBitmask;
	wire [BITMASK_LENGTH-1:0] wireNegatedMutualBitmask;
	wire [ACCUM_LENGTH-1:0] wireShiftAccum;
	wire [ACCUM_LENGTH-1:0]  wireActivationAccum;
	wire [ACCUM_LENGTH-1:0]  wireWeightAccum;
	wire [ACCUM_LENGTH-1:0]  wireActivationMaskedAccum;
	wire [ACCUM_LENGTH-1:0]  wireWeightMaskedAccum;
	wire [ACCUM_LENGTH-1:0]  wireActivationDenseAccum;
	wire [ACCUM_LENGTH-1:0]  wireWeightDenseAccum;
	wire [BITWIDTH_COUNT-1:0] wireBitWidthCount;

	//Structural coding
	assign wireMutualBitmask = bitmaskA & bitmaskW;
	assign wireNegatedMutualBitmask = ~wireMutualBitmask;

	accumulator # (.BITMASK_LENGTH(BITMASK_LENGTH),
			.INDEX_BITWIDTH(INDEX_BITWIDTH))
	inst_accumulator_shiftAccumulator 
		(.bitmask(wireNegatedMutualBitmask),
		 .index(wireShiftAccum));


	accumulator # (.BITMASK_LENGTH(BITMASK_LENGTH),
			.INDEX_BITWIDTH(INDEX_BITWIDTH))
	inst_accumulator_activationAccumulator 
		(.bitmask(bitmaskA),
		 .index(wireActivationAccum));

	accumulator # (.BITMASK_LENGTH(BITMASK_LENGTH),
			.INDEX_BITWIDTH(INDEX_BITWIDTH))
	inst_accumulator_weightAccumulator 
		(.bitmask(bitmaskW),
		 .index(wireWeightAccum));


	extendAndMask 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.INDEX_BITWIDTH (INDEX_BITWIDTH)
		)
		inst_extendAndMask_activationMask
		(
			.unmaskedIndices(wireActivationAccum),
			.bitmask(wireMutualBitmask),
			.maskedIndices(wireActivationMaskedAccum)
		);

	extendAndMask 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.INDEX_BITWIDTH (INDEX_BITWIDTH)
		)
		inst_extendAndMask_weightMask
		(
			.unmaskedIndices(wireWeightAccum),
			.bitmask(wireMutualBitmask),
			.maskedIndices(wireWeightMaskedAccum)
		);


	//Logic for generating the condensed bitmask
	popCounter
		# (
			.BITMASK_LENGTH (8),
			.BITWIDTH_OUTPUT (4)
		)
		inst_popCounter_pairCount
		(
			.bitmask(wireMutualBitmask),
			.count(wireBitWidthCount)
		);

	//Logic for generating the condensed activation indices
	collapseBubble 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.BITWIDTH_INDEX (INDEX_BITWIDTH),
			.BITWIDTH_NUMBER (3)
		)
		inst_collapseBubble_activation
		(
			.inputWithGap(wireActivationMaskedAccum),
			.positions(wireShiftAccum),
			.outputWithoutGap(wireActivationDenseAccum)
		);

	//Logic for generating the condensed activation indices
	collapseBubble 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.BITWIDTH_INDEX (INDEX_BITWIDTH),
			.BITWIDTH_NUMBER (3)
		)
		inst_collapseBubble_weight
		(
			.inputWithGap(wireWeightMaskedAccum),
			.positions(wireShiftAccum),
			.outputWithoutGap(wireWeightDenseAccum)
		);

	//Registers update
	always @ (posedge clock) begin
		if (resetn == 1'b0) begin
			regActivationDenseAccum <= {ACCUM_LENGTH{1'b0}};
			regWeightDenseAccum <= {ACCUM_LENGTH{1'b0}};
			regBitWidthCount <= {BITWIDTH_COUNT{1'b0}};
		end
		else begin
			regActivationDenseAccum <= wireActivationDenseAccum;
			regWeightDenseAccum <= wireWeightDenseAccum;
			regBitWidthCount <= wireBitWidthCount;
		end
	end

	//Assign the final output
	assign result = 
		{{12{1'b0}}, regBitWidthCount, regWeightDenseAccum, regActivationDenseAccum};
	//assign result = 
	//	{{12{1'b0}}, wireBitWidthCount, wireWeightDenseAccum, wireActivationDenseAccum};
	
endmodule

module operandMatcher16 (
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,
		
		input  wire [15:0]  bitmaskW, //weight bitmask
		input  wire [15:0]  bitmaskA, //Activation bitmask
		
		output wire [255:0] result  //  [63:0] Packed indices of A; [127:64] Packed indices of W; [132:128] Number of pairs; [255:133]: Padding
	);
	
	//Ignoring resetn, ivalid, and iready

	//tie the following signals to high
	assign ovalid = 1'b1;
	assign oready = 1'b1;

	//Localdef
	localparam BITMASK_LENGTH = 16;
	localparam INDEX_BITWIDTH = 4;
	localparam ACCUM_LENGTH = BITMASK_LENGTH * INDEX_BITWIDTH;
	localparam BITWIDTH_COUNT = 5;


	//Declare the registers
	reg [BITMASK_LENGTH-1:0] regBitmaskW;
	reg [BITMASK_LENGTH-1:0] regBitmaskA;
	//reg [BITMASK_LENGTH-1:0] regMutualBitmask;
	//reg [ACCUM_LENGTH-1:0] regShiftAccum;
	//reg [ACCUM_LENGTH-1:0]  regActivationMaskedAccum;
	//reg [ACCUM_LENGTH-1:0]  regWeightMaskedAccum;
	reg [ACCUM_LENGTH-1:0]  regActivationDenseAccum;
	reg [ACCUM_LENGTH-1:0]  regWeightDenseAccum;
	reg [BITWIDTH_COUNT-1:0] regBitWidthCount;

	//Structural wires
	wire [BITMASK_LENGTH-1:0] wireMutualBitmask;
	wire [BITMASK_LENGTH-1:0] wireNegatedMutualBitmask;
	wire [ACCUM_LENGTH-1:0] wireShiftAccum;
	wire [ACCUM_LENGTH-1:0]  wireActivationAccum;
	wire [ACCUM_LENGTH-1:0]  wireWeightAccum;
	wire [ACCUM_LENGTH-1:0]  wireActivationMaskedAccum;
	wire [ACCUM_LENGTH-1:0]  wireWeightMaskedAccum;
	wire [ACCUM_LENGTH-1:0]  wireActivationDenseAccum;
	wire [ACCUM_LENGTH-1:0]  wireWeightDenseAccum;
	wire [BITWIDTH_COUNT-1:0] wireBitWidthCount;

	//Structural coding
	assign wireMutualBitmask = regBitmaskA & regBitmaskW;
	assign wireNegatedMutualBitmask = ~wireMutualBitmask;

	accumulator # (.BITMASK_LENGTH(BITMASK_LENGTH),
			.INDEX_BITWIDTH(INDEX_BITWIDTH))
	inst_accumulator_shiftAccumulator 
		(.bitmask(wireNegatedMutualBitmask),
		 .index(wireShiftAccum));


	accumulator # (.BITMASK_LENGTH(BITMASK_LENGTH),
			.INDEX_BITWIDTH(INDEX_BITWIDTH))
	inst_accumulator_activationAccumulator 
		(.bitmask(regBitmaskA),
		 .index(wireActivationAccum));

	accumulator # (.BITMASK_LENGTH(BITMASK_LENGTH),
			.INDEX_BITWIDTH(INDEX_BITWIDTH))
	inst_accumulator_weightAccumulator 
		(.bitmask(regBitmaskW),
		 .index(wireWeightAccum));


	extendAndMask 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.INDEX_BITWIDTH (INDEX_BITWIDTH)
		)
		inst_extendAndMask_activationMask
		(
			.unmaskedIndices(wireActivationAccum),
			.bitmask(wireMutualBitmask),
			.maskedIndices(wireActivationMaskedAccum)
		);

	extendAndMask 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.INDEX_BITWIDTH (INDEX_BITWIDTH)
		)
		inst_extendAndMask_weightMask
		(
			.unmaskedIndices(wireWeightAccum),
			.bitmask(wireMutualBitmask),
			.maskedIndices(wireWeightMaskedAccum)
		);


	//Logic for generating the condensed bitmask
	popCounter
		# (
			.BITMASK_LENGTH (8),
			.BITWIDTH_OUTPUT (4)
		)
		inst_popCounter_pairCount
		(
			.bitmask(wireMutualBitmask),
			.count(wireBitWidthCount)
		);

	//Logic for generating the condensed activation indices
	collapseBubble 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.BITWIDTH_INDEX (INDEX_BITWIDTH),
			.BITWIDTH_NUMBER (INDEX_BITWIDTH)
		)
		inst_collapseBubble_activation
		(
			.inputWithGap(wireActivationMaskedAccum),
			.positions(wireShiftAccum),
			.outputWithoutGap(wireActivationDenseAccum)
		);

	//Logic for generating the condensed activation indices
	collapseBubble 
		# (
			.BITMASK_LENGTH (BITMASK_LENGTH),
			.BITWIDTH_INDEX (INDEX_BITWIDTH),
			.BITWIDTH_NUMBER (INDEX_BITWIDTH)
		)
		inst_collapseBubble_weight
		(
			.inputWithGap(wireWeightMaskedAccum),
			.positions(wireShiftAccum),
			.outputWithoutGap(wireWeightDenseAccum)
		);

	//Registers update
	always @ (posedge clock) begin
		if (resetn == 1'b0) begin
			regBitmaskW <= {BITMASK_LENGTH{1'b0}};
			regBitmaskA <= {BITMASK_LENGTH{1'b0}};
			regActivationDenseAccum <= {ACCUM_LENGTH{1'b0}};
			regWeightDenseAccum <= {ACCUM_LENGTH{1'b0}};
			regBitWidthCount <= {BITWIDTH_COUNT{1'b0}};
		end
		else begin
			regBitmaskW	<= bitmaskW;
			regBitmaskA <= bitmaskA;
			regActivationDenseAccum <= wireActivationDenseAccum;
			regWeightDenseAccum <= wireWeightDenseAccum;
			regBitWidthCount <= wireBitWidthCount;
		end
	end

	//Assign the final output
	assign result = 
		{{123{1'b0}}, regBitWidthCount, regWeightDenseAccum, regActivationDenseAccum};
	//assign result = 
	//	{{12{1'b0}}, wireBitWidthCount, wireWeightDenseAccum, wireActivationDenseAccum};
	



endmodule
