`timescale 1ps/1ns

module selectGenerator
	#	(
			parameter BITMASK_LENGTH = 16,
			parameter INDEX_BITWIDTH = 5
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

	reg [INDEX_BITWIDTH*(BITMASK_LENGTH+1)-1 : 0] wireIndex;
	integer i;
	assign index = wireIndex[INDEX_BITWIDTH*(BITMASK_LENGTH+1)-1 -: INDEX_BITWIDTH*BITMASK_LENGTH];
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