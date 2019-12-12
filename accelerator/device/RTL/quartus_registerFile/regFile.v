`timescale 1ns/1ps

module regFile 
	# (
		parameter PORT_WIDTH = 16,
		parameter ADDR_WIDTH = 4
	)
	(
	//Write side 
	input writeBank,
	input writeEnable,
	input [ADDR_WIDTH-2:0] writeAddrTransferBlock,
	//input [ADDR_WIDTH-1:0] writeAddr1,
	input [PORT_WIDTH-1:0] writeData0,
	input [PORT_WIDTH-1:0] writeData1,

	//Read side,
	input readBank,
	input [ADDR_WIDTH-1:0] readAddr0,
	input [ADDR_WIDTH-1:0] readAddr1,
	output reg [PORT_WIDTH-1:0] readData0,
	output reg [PORT_WIDTH-1:0] readData1,

	//Interface
	input clock,
	input resetn
	);

	reg [2**(ADDR_WIDTH+1)*PORT_WIDTH-1:0] registerFile /* synthesis preserve */;
	//wire [2**(ADDR_WIDTH+1)-1:0] writeEnableRegisterFile0;
	//wire [2**(ADDR_WIDTH+1)-1:0] writeEnableRegisterFile1;
	wire [2**(ADDR_WIDTH)-1:0] writeEnableRegisterFileTransferBlock;

	reg regReadBank;
	reg [ADDR_WIDTH-1:0] regReadAddress0 /* synthesis preserve */;
	reg [ADDR_WIDTH-1:0] regReadAddress1 /* synthesis preserve */;

	//See intel.com/content/dam/www/programmable/us/en/pdfs/literature/catalogs/lpm.pdf
	// lpm_decode	#(.lpm_decodes(2**(ADDR_WIDTH+1)), .lpm_type("LPM_DECODE"), .lpm_width(ADDR_WIDTH+1))
	// write_decoder_bank0 (
	// 	.data ({writeBank, writeAddr0}),
	// 	.eq (writeEnableRegisterFile0)
	// 	// synopsys translate_on
	// 	);

	// lpm_decode	#(.lpm_decodes(2**(ADDR_WIDTH+1)), .lpm_type("LPM_DECODE"), .lpm_width(ADDR_WIDTH+1))
	// write_decoder_bank1 (
	// 	.data ({writeBank, writeAddr1}),
	// 	.eq (writeEnableRegisterFile1)
	// 	// synopsys translate_on
	// 	);
	lpm_decode	#(.lpm_decodes(2**(ADDR_WIDTH)), .lpm_type("LPM_DECODE"), .lpm_width(ADDR_WIDTH))
	write_decoder_wide (
		.data ({writeBank, writeAddrTransferBlock}),
		.eq (writeEnableRegisterFileTransferBlock)
		// synopsys translate_on
		);


	//Generate the write side
	integer regI;
	always @ (posedge clock) begin
		if (resetn == 1'b0) begin
			registerFile <= {(2**(ADDR_WIDTH+1)*PORT_WIDTH){1'b0}};
		end
		else begin
			for (regI=0; regI<2**(ADDR_WIDTH); regI=regI+1) begin
				if (writeEnable == 1'b1) begin
					if (writeEnableRegisterFileTransferBlock[regI] == 1'b1) begin
						registerFile[(2*regI+2)*PORT_WIDTH-1 -: 2*PORT_WIDTH] <= {writeData1, writeData0};
					end
				end
			end
		end
	end

	//Generate the delay registers
	always @ (posedge clock) begin
		if (resetn == 1'b0) begin
			regReadBank <= 1'b0;
			{regReadAddress1, regReadAddress0} <= {(2*ADDR_WIDTH){1'b0}};
		end
		else begin
			{regReadBank, regReadAddress1, regReadAddress0} <= {readBank, readAddr1, readAddr0};
		end
	end

	//Generate the output select
	always @ (*) begin
		readData0 = registerFile[({regReadBank, regReadAddress0}+1)*PORT_WIDTH-1 -: PORT_WIDTH];
		readData1 = registerFile[({regReadBank, regReadAddress1}+1)*PORT_WIDTH-1 -: PORT_WIDTH];
	end


endmodule
