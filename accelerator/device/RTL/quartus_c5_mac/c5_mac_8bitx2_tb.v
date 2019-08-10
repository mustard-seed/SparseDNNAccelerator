`timescale 1ns/1ps

module tb_c5_mac_8bitx2();
    reg [7:0] dataa0;
    reg [7:0] dataa1;
    //reg [7:0] dataa2;
    //reg [7:0] dataa3;

    reg [7:0] datab0;
    reg [7:0] datab1;
    //reg [7:0] datab2;
    //reg [7:0] datab3;

    reg clk;
    wire [31:0] result;

    c5_mac_8bitx2 dut (
			.result  (result),  //  output,  width = 18,  result.result
		.dataa_0 (dataa0), //   input,   width = 8, dataa_0.dataa_0
		.dataa_1 (dataa1), //   input,   width = 8, dataa_1.dataa_1
		.datab_0 (datab0), //   input,   width = 8, datab_0.datab_0
		.datab_1 (datab1), //   input,   width = 8, datab_1.datab_1
		.clock0  (clk)   //   input,   width = 1,  clock0.clock0
	);
    
    initial begin
        clk = 1'b0;
        {dataa1, dataa0, datab1, datab0} = 32'h00000000;
        #10 ;
        {dataa1, dataa0, datab1, datab0} = 32'h01010101;
        #10
        {dataa1, dataa0, datab1, datab0} = 32'h0101FFFF;
    end

    always #5 clk = ~clk;
       
endmodule
