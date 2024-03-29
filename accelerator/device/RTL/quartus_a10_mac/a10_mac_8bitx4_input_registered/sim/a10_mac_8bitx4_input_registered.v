// a10_mac_8bitx4_input_registered.v

// Generated using ACDS version 17.1.1 273

`timescale 1 ps / 1 ps
module a10_mac_8bitx4_input_registered (
		input  wire        clock0,  //  clock0.clock0
		input  wire [7:0]  dataa_0, // dataa_0.dataa_0
		input  wire [7:0]  dataa_1, // dataa_1.dataa_1
		input  wire [7:0]  dataa_2, // dataa_2.dataa_2
		input  wire [7:0]  dataa_3, // dataa_3.dataa_3
		input  wire [7:0]  datab_0, // datab_0.datab_0
		input  wire [7:0]  datab_1, // datab_1.datab_1
		input  wire [7:0]  datab_2, // datab_2.datab_2
		input  wire [7:0]  datab_3, // datab_3.datab_3
		output wire [17:0] result   //  result.result
	);

	a10_mac_8bitx4_input_registered_altera_mult_add_171_i4fixgy mult_add_0 (
		.result  (result),  //  output,  width = 16,  result.result
		.dataa_0 (dataa_0), //   input,   width = 8, dataa_0.dataa_0
		.dataa_1 (dataa_1), //   input,   width = 8, dataa_1.dataa_1
		.dataa_2 (dataa_2), //   input,   width = 8, dataa_2.dataa_2
		.dataa_3 (dataa_3), //   input,   width = 8, dataa_3.dataa_3
		.datab_0 (datab_0), //   input,   width = 8, datab_0.datab_0
		.datab_1 (datab_1), //   input,   width = 8, datab_1.datab_1
		.datab_2 (datab_2), //   input,   width = 8, datab_2.datab_2
		.datab_3 (datab_3), //   input,   width = 8, datab_3.datab_3
		.clock0  (clock0)   //   input,   width = 1,  clock0.clock0
	);

endmodule
