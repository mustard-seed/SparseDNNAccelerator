module popCounterWrapper (
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input wire [7:0] bitmask,

		output wire [7:0] count
	);

	assign {ovalid, oready} = 2'b11;

	assign count [7:4] = 4'b0000;
	popCounter #(.BITMASK_LENGTH(8), .BITWIDTH_OUTPUT(4))
	inst_popCounter (.bitmask(bitmask), .count(count[3:0]));

endmodule

module popCounterWrapper16 (
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,

		input wire [15:0] bitmask,

		output wire [7:0] count
	);

	assign {ovalid, oready} = 2'b11;

	assign count [7:5] = 3'b000;
	popCounter #(.BITMASK_LENGTH(16), .BITWIDTH_OUTPUT(5))
	inst_popCounter (.bitmask(bitmask), .count(count[4:0]));

endmodule