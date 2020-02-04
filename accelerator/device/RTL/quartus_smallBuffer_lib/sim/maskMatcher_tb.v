`timescale 1 ns / 1ps

module maskMatcher_tb ();
	reg [15:0] bitmaskW;
	reg [15:0] bitmaskA;
	wire [63:0] result;
	wire [15:0] packedBitmaskW;
	wire [15:0] packedBitmaskA;
	wire [4:0] numAOperands;
	wire [4:0] numWOperands;

	clMaskMatcher16 #()
	dut (.bitmaskW(bitmaskW), .bitmaskA(bitmaskA), .result  (result));

	assign packedBitmaskW = result[15:0];
	assign packedBitmaskA = result[31:16];
	assign numWOperands = result[36:32];
	assign numAOperands = result[44:40];

	initial begin
		bitmaskW = 16'hFFFF; bitmaskA = 16'h0000; 
		#10;
		bitmaskW = 16'hFFFF; bitmaskA = 16'hFFFF;
		#10
		bitmaskW = 16'hF00F; bitmaskA = 16'hFFFF;
		#10
		bitmaskW = 16'h0000; bitmaskA = 16'h0000;

	end
endmodule