`timescale 1 ns / 1 ps

/**
 * \brief Count the number of 1s preceding and up to each bit in the bit mask. 
 * bitmask is counted from LSB to MSB
 * i.e. bitmask[N] is considered AFTER bitmask [N-1]
 */
module selectGenerator
	#	(
			parameter BITMASK_LENGTH = 16, //Number of input bitmask length
			parameter INDEX_BITWIDTH = 5 //Number of bits per index in the accumulated output
	  	) 
	(
		input wire [BITMASK_LENGTH-1 : 0] bitmask,
		output wire [INDEX_BITWIDTH*BITMASK_LENGTH-1 : 0] index
    );
    //===============================================
	//Count the number of 1s preceding and up to each bit in the bit mask. 
	//bitmask is counted from LSB to MSB
	//i.e. bitmask[N] is considered AFTER bitmask [N-1]
	//===============================================

	reg [INDEX_BITWIDTH*BITMASK_LENGTH-1 : 0] wireIndex;
	integer i;
	assign index = wireIndex;
	//assign wireIndex[63:60] = 4'b0000;

	always @ (*) begin
		wireIndex[INDEX_BITWIDTH-1 : 0] = { {(INDEX_BITWIDTH - 1){1'b0}}, bitmask[0] };
		for (i=1; i<BITMASK_LENGTH; i=i+1) begin: accum
			//assign wireIndex [i*4-1 -: 4] = {3'b000, bitmask[i]} + wireIndex[(i+1)*4-1-:4];
			wireIndex[(i+1)*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH] = 
			{ {(INDEX_BITWIDTH - 1){1'b0}}, bitmask[i] } + wireIndex[i*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH];
		end
	end
endmodule

/*
 * \brief Filter and coalesce a sparse input bus using the input mask
 * \input sparseInput: input with gaps. Each element has INPUT_ELEMENT_WIDTH bits.
 * \input bitmask: Little ENDIAN bitmask indicating which bit are dense
 * \output denseOutput: Little endian bus of the dense output
 * \output numDenseInput: Number of 1s in the bitmask
 * e.g. 
 * sparseInput = {3'b000, 3'b101, 3'b000, 3'b111}
 * bitmask = 4'b0100
 * denseOutput = {3'b000, 3'b000, 3'b000, 3'b101}
 * numDenseInput = 3'b001
*/
module inputFilter
	#	(
			parameter BITMASK_LENGTH = 16,
			parameter INDEX_BITWIDTH = 5,
			parameter INPUT_ELEMENT_WIDTH = 1
		)
	(
		input wire [INPUT_ELEMENT_WIDTH*BITMASK_LENGTH-1 : 0] sparseInput,
		input wire [BITMASK_LENGTH-1 : 0] bitmask,
		output reg [INPUT_ELEMENT_WIDTH*BITMASK_LENGTH-1 : 0 ] denseOutput,
		output wire [INDEX_BITWIDTH-1 : 0] numDenseInput
	);

	wire [INDEX_BITWIDTH*BITMASK_LENGTH-1 : 0] accumulatedIndex;
	assign numDenseInput = accumulatedIndex[INDEX_BITWIDTH*BITMASK_LENGTH-1 -: INDEX_BITWIDTH];

	selectGenerator # (
		.BITMASK_LENGTH(BITMASK_LENGTH),
		.INDEX_BITWIDTH(INDEX_BITWIDTH)
		)
	inst_select_generator (
		.bitmask(bitmask),
		.index(accumulatedIndex)
		);

	genvar iGenOutput;
	generate
			for (iGenOutput = 0; iGenOutput < BITMASK_LENGTH; iGenOutput=iGenOutput+1) begin: GENFOR_OUTPUT
				integer iAccumMask;
				always @ (*) begin
					denseOutput[(iGenOutput+1)*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH] = {INPUT_ELEMENT_WIDTH{1'b0}};
					for (iAccumMask = BITMASK_LENGTH; iAccumMask>0; iAccumMask=iAccumMask-1) begin:FOR_ACCUM
						if (accumulatedIndex[iAccumMask*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH] == (iGenOutput+1)) begin
							denseOutput[(iGenOutput+1)*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH] 
								= sparseInput[iAccumMask*INPUT_ELEMENT_WIDTH-1 -: INPUT_ELEMENT_WIDTH];
						end
					end
				end
			end
	endgenerate
endmodule

module clMaskMatcher16 (
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input 	wire [15:0] bitmaskW,
		input 	wire [15:0] bitmaskA,

		//[15:0] packed bitmask W
		//[31:16] packed bitmask A
		//[36:32] number of W operand blocks
		//[44:40] number of A operand blocks
		output wire [63:0] result
	);

	assign ovalid = 1'b1;
	assign oready = 1'b1;

	wire [15:0] bitmaskMutual = bitmaskA & bitmaskW;

	inputFilter #(
		.BITMASK_LENGTH     (16),
		.INDEX_BITWIDTH     (5),
		.INPUT_ELEMENT_WIDTH(1)
	)
	maskWFilter (
		.sparseInput  (bitmaskMutual),
		.bitmask      (bitmaskW),
		.denseOutput  (result[15:0]),
		.numDenseInput(result[36:32])
		);

	inputFilter #(
		.BITMASK_LENGTH     (16),
		.INDEX_BITWIDTH     (5),
		.INPUT_ELEMENT_WIDTH(1)
	)
	maskAFilter (
		.sparseInput  (bitmaskMutual),
		.bitmask      (bitmaskA),
		.denseOutput  (result[31:16]),
		.numDenseInput(result[44:40])
		);

endmodule



