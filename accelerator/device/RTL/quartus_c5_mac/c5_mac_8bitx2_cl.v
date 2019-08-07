`timescale 1 ps / 1 ps
module c5_mac_8bitx2 (
		input   clock,
		input   resetn,
		input   ivalid, 
		input   iready,
		output  ovalid, 
		output  oready,
		
		input  wire [7:0]  dataa_0, // dataa_0.dataa_0
		input  wire [7:0]  datab_0, // datab_0.datab_0
		
		input  wire [7:0]  dataa_1, // dataa_1.dataa_1
		input  wire [7:0]  datab_1, // datab_1.datab_1
		
		output wire [31:0] result  //  result.result
	);

	assign ovalid = 1'b1;
	assign oready = 1'b1;
	// ivalid, iready, resetn are ignored

	c5_mac_8bitx2_0002 inst (
		.result  (result),  //  result.result
		.dataa_0 (dataa_0), // dataa_0.dataa_0
		.dataa_1 (dataa_1), // dataa_1.dataa_1
		.datab_0 (datab_0), // datab_0.datab_0
		.datab_1 (datab_1), // datab_1.datab_1
		.clock0  (clock)   //  clock0.clock0
	);

endmodule
