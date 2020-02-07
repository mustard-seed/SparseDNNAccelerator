`timescale 1 ns / 1ps

module maskMatcher_tb ();
	reg [15:0] bitmaskW;
	reg [15:0] bitmaskA;
	reg [4:0] startIndexA;
	reg [4:0] startIndexW;
	wire [63:0] result;
	wire [15:0] packedBitmaskW;
	wire [15:0] packedBitmaskA;
	wire [4:0] nextAStartIndex;
	wire [4:0] nextWStartIndex;
	wire [1:0] numDenseW, numDenseA;

	clMaskMatcher #()
	dut (.bitmaskW(bitmaskW), .bitmaskA(bitmaskA), .startIndexA(startIndexA), .startIndexW(startIndexW), .result(result));

	assign packedBitmaskW = result[15:0];
	assign packedBitmaskA = result[31:16];
	assign nextWStartIndex = result[36:32];
	assign nextAStartIndex = result[44:40];
	assign numDenseW = result[38:37];
	assign numDenseA = result[46:45];

	initial begin
		bitmaskW = 16'hFFFF; bitmaskA = 16'h0000; startIndexA = 5'd0; startIndexW = 5'd0; 
		#10;
		bitmaskW = 16'hFFFF; bitmaskA = 16'hFFFF; startIndexA = 5'd4; startIndexW = 5'd0;
		#10
		bitmaskW = 16'hF00F; bitmaskA = 16'hFFFF; startIndexA = 5'd0; startIndexW = 5'd0;
		#10
		bitmaskW = 16'h0000; bitmaskA = 16'h0000; startIndexA = 5'd4; startIndexW = 5'd4;

	end
endmodule