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

	wire [63:0] result;
	reg [15:0] regW, regA;
	reg [63:0] regResult;
	reg [15:0] muxMaskOutput;
	reg [4:0] muxCountOutput;

	clMaskMatcher16 maskMatcher (
			.bitmaskW(regW),
			.bitmaskA(regA),
			.result(result)
		);

	hexDriver H0 (muxMaskOutput[3:0], HEX0);
	hexDriver H1 (muxMaskOutput[7:4], HEX1);
	hexDriver H2 (muxMaskOutput[11:8], HEX2);
	hexDriver H3 (muxMaskOutput[15:12], HEX3);
	hexDriver H4 (muxCountOutput[3:0], HEX4);
	hexDriver H5 ({3'b000, muxCountOutput[4]}, HEX5);

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
			muxMaskOutput = regResult[15:0];
			muxCountOutput = regResult[36:32];
		end
		else begin
			muxMaskOutput = regResult[31:16];
			muxCountOutput = regResult[44:40];
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