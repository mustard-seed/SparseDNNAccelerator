`timescale 1ns/1ps

module tb_a10_mac_8bitx4();
    reg [7:0] dataa0;
    reg [7:0] dataa1;
    reg [7:0] dataa2;
    reg [7:0] dataa3;

    reg [7:0] datab0;
    reg [7:0] datab1;
    reg [7:0] datab2;
    reg [7:0] datab3;

    reg clk;
    wire [17:0] result;

    a10_mac_8bitx4 dut (
			.result  (result),  //  output,  width = 18,  result.result
		.dataa_0 (dataa0), //   input,   width = 8, dataa_0.dataa_0
		.dataa_1 (dataa1), //   input,   width = 8, dataa_1.dataa_1
		.dataa_2 (dataa2), //   input,   width = 8, dataa_2.dataa_2
		.dataa_3 (dataa3), //   input,   width = 8, dataa_3.dataa_3
		.datab_0 (datab0), //   input,   width = 8, datab_0.datab_0
		.datab_1 (datab1), //   input,   width = 8, datab_1.datab_1
		.datab_2 (datab2), //   input,   width = 8, datab_2.datab_2
		.datab_3 (datab3), //   input,   width = 8, datab_3.datab_3
		.clock0  (clk)   //   input,   width = 1,  clock0.clock0
	);
    
    initial begin
        clk = 1'b0;
        {dataa3, dataa2, dataa1, dataa0, datab3, datab2, datab1, datab0} = 64'h0000000000000000;
        #10 ;
        {dataa3, dataa2, dataa1, dataa0, datab3, datab2, datab1, datab0} = 64'h0101010101010101;
    end

    always #5 clk = ~clk;
       
endmodule
