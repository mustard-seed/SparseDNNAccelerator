module top (
		///////// CLOCK /////////
      input              CLOCK2_50,
      input              CLOCK3_50,
      input              CLOCK4_50,
      input              CLOCK_50,

      ///////// KEY /////////
      input    [ 3: 0]   KEY,

      ///////// SW /////////
      input    [ 9: 0]   SW,

      ///////// LED /////////
      output   [ 9: 0]   LEDR,

      ///////// Seg7 /////////
      output   [ 6: 0]   HEX0,
      output   [ 6: 0]   HEX1,
      output   [ 6: 0]   HEX2,
      output   [ 6: 0]   HEX3,
      output   [ 6: 0]   HEX4,
      output   [ 6: 0]   HEX5,

      ///////// GPIO /////////
      inout     [35:0]         GPIO_0,
      inout     [35:0]         GPIO_1
	);

  localparam BITMASK_LENGTH = 4;
  localparam INDEX_BITWIDTH = 3;
  localparam COUNT_BITWIDTH = 3;
  localparam MAX_NUM_OUTPUT = 4;

	wire [63:0] result;
	reg [15:0] regW, regA;
	reg [63:0] regResult;
	reg [15:0] muxMaskOutput;
	reg [4:0] muxNextStartIndexOutput;

	clMaskMatcher #(.BITMASK_LENGTH(BITMASK_LENGTH), .INDEX_BITWIDTH(INDEX_BITWIDTH), .INPUT_ELEMENT_WIDTH(1), .COUNT_BITWIDTH(COUNT_BITWIDTH), .MAX_NUM_OUTPUT(MAX_NUM_OUTPUT) )
  maskMatcher (
			.bitmaskW(regW[BITMASK_LENGTH-1:0]),
			.bitmaskA(regA[BITMASK_LENGTH-1:0]),
			.startIndexA({INDEX_BITWIDTH{1'b0}}),
			.startIndexW({INDEX_BITWIDTH{1'b0}}),
			.result(result)
		);

	hexDriver H0 (muxMaskOutput[3:0], HEX0);
	hexDriver H1 (muxMaskOutput[7:4], HEX1);
	hexDriver H2 (muxMaskOutput[11:8], HEX2);
	hexDriver H3 (muxMaskOutput[15:12], HEX3);
	hexDriver H4 (muxNextStartIndexOutput[3:0], HEX4);
	hexDriver H5 ({3'b000, muxNextStartIndexOutput[4]}, HEX5);

	assign LEDR[9:5] = SW[9:5];
	assign LEDR[4:0] = SW[4:0];

	always @ (posedge CLOCK_50) begin
		if (KEY[0] == 1'b1) begin
			regW <= 16'h0000;
			regA <= 16'h0000;
			regResult <= 64'h00_00_00_00_00_00_00_00;
		end
		else begin
			regW <= {GPIO_0[10:0], SW[4:0]};
			regA <= {GPIO_0[21:11], SW[9:5]};
			regResult <= result;
		end
	end

	always @ (*) begin
		if (KEY[1] == 1'b1) begin
			muxMaskOutput = regResult[BITMASK_LENGTH-1:0];
			muxNextStartIndexOutput = regResult[32+INDEX_BITWIDTH-1:32];
		end
		else begin
			muxMaskOutput = regResult[16+BITMASK_LENGTH-1:16];
			muxNextStartIndexOutput = regResult[40+INDEX_BITWIDTH-1:40];
		end
	end
	
endmodule

module hexDriver ( 
    input [3:0] in,
    output [6:0] hex
);

    assign hex[0] = |{ &{~in[3], ~in[2], ~in[1], in[0]},
                         &{~in[3], in[2], ~in[1], ~in[0] },
                        &{in[3], ~in[2], in[1], in[0]},
                        &{in[3], in[2], ~in[1], in[0]}};

    assign hex[1] = |{ &{in[2], in[1], ~in[0]},
                       &{in[3], in[1], in[0]},
                       &{in[3], in[2], ~in[0]},
                       &{~in[3], in[2], ~in[1], in[0]}
                        };

    assign hex[2] = |{ &{~in[3], ~in[2], in[1], ~in[0]},
                       &{in[3], in[2], ~in[0]},
                       &{in[3], in[2], in[1]}
                        };

    assign hex[3] = |{ &{~in[3], in[2], ~in[1], ~in[0]},
                       &{in[2], in[1], in[0]},
                       &{in[3], ~in[2], in[1], ~in[0]},
                       &{~in[2], ~in[1], in[0]}
                     };

    assign hex[4] = | { &{~in[3], in[2], ~in[1], ~in[0]},
                         &{~in[2], ~in[1], in[0]},
                         &{~in[3], in[1], in[0]},
                         &{~in[3], ~in[1], in[0]}
                      };

    assign hex[5] = | { &{~in[3], ~in[2], in[0]},
                        &{~in[3], in[1], in[0]},
                        &{in[3], in[2], ~in[1], in[0]},
                        &{~in[3], ~in[2], in[1], ~in[0]}};

    assign hex[6] = | { &{~in[3], ~in[2], ~in[1]},
                        &{~in[3], in[2], in[1], in[0]},
                        &{in[3], in[2], ~in[1], ~in[0]}};
endmodule