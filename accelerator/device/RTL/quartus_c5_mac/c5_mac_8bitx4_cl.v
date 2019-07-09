`timescale 1 ps / 1 ps
module c5_mac_8bitx4 (
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
		
		input  wire [7:0]  dataa_2, // dataa_2.dataa_2
		input  wire [7:0]  datab_2, // datab_2.datab_2
		
		input  wire [7:0]  dataa_3, // dataa_3.dataa_3
		input  wire [7:0]  datab_3, // datab_3.datab_3
		
		output wire [31:0] result  //  result.result
	);

	wire [17:0] result_18b;

	assign ovalid = 1'b1;
	assign oready = 1'b1;
	// ivalid, iready, resetn are ignored

    //Sign extend the output
	assign result = {{14{result_18b[17]}}, result_18b};

	c5_mac_8bitx4_0002 inst (
		.result  (result_18b),  //  result.result
		.dataa_0 (dataa_0), // dataa_0.dataa_0
		.dataa_1 (dataa_1), // dataa_1.dataa_1
		.dataa_2 (dataa_2), // dataa_2.dataa_2
		.dataa_3 (dataa_3), // dataa_3.dataa_3
		.datab_0 (datab_0), // datab_0.datab_0
		.datab_1 (datab_1), // datab_1.datab_1
		.datab_2 (datab_2), // datab_2.datab_2
		.datab_3 (datab_3), // datab_3.datab_3
		.clock0  (clock)   //  clock0.clock0
	);

endmodule
