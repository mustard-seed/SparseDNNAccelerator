`timescale 1 ps / 1 ps

module dsp_no_input_pipeline_reg(
		input wire [7:0] ax,  //multiplier-a input X
		input wire [7:0] ay,  //multiplier-a input Y
		input wire [7:0] bx,  //multiplier-b input X
		input wire [7:0] by,  //multiplier-b input Y
		input wire [63:0] chainin, //chain-adder input
		input wire [2:0] ena, //clock anbles, {ena2, ena1, ena0}
		output wire [63:0] chainout, //chain out
		output wire [31:0] resulta,
        input wire  aclr0,
        input wire  aclr1,
        input  clk0,
        input  clk1,
        input  clk2
	);

	twentynm_mac        twentynm_mac_component (
                                        .ax (ax),
                                        .ay (ay),
                                        .bx (bx),
                                        .by (by),
                                        .chainin (chainin),
                                        .ena (ena),
                                        .chainout (chainout),
                                        .resulta (resulta),
                                        .aclr ({aclr1,aclr0}),
                                        .clk ({clk2,clk1,clk0}),
                                        .accumulate (),
                                        .az (),
                                        .bz (),
                                        .coefsela (),
                                        .coefselb (),
                                        .dftout (),
                                        .loadconst (),
                                        .negate (),
                                        .resultb (),
                                        .scanin (),
                                        .scanout (),
                                        .sub ());
            defparam
                    twentynm_mac_component.ax_width = 8,
                    twentynm_mac_component.ay_scan_in_width = 8,
                    twentynm_mac_component.bx_width = 8,
                    twentynm_mac_component.by_width = 8,
                    twentynm_mac_component.operation_mode = "m18x18_sumof2",
                    twentynm_mac_component.mode_sub_location = 0,
                    twentynm_mac_component.operand_source_max = "input",
                    twentynm_mac_component.operand_source_may = "input",
                    twentynm_mac_component.operand_source_mbx = "input",
                    twentynm_mac_component.operand_source_mby = "input",
                    twentynm_mac_component.signed_max = "true",
                    twentynm_mac_component.signed_may = "true",
                    twentynm_mac_component.signed_mbx = "true",
                    twentynm_mac_component.signed_mby = "true",
                    twentynm_mac_component.preadder_subtract_a = "false",
                    twentynm_mac_component.preadder_subtract_b = "false",
                    twentynm_mac_component.ay_use_scan_in = "false",
                    twentynm_mac_component.by_use_scan_in = "false",
                    twentynm_mac_component.delay_scan_out_ay = "false",
                    twentynm_mac_component.delay_scan_out_by = "false",
                    twentynm_mac_component.use_chainadder = "true",
                    twentynm_mac_component.enable_double_accum = "false",
                    twentynm_mac_component.load_const_value = 0,
                    twentynm_mac_component.coef_a_0 = 0,
                    twentynm_mac_component.coef_a_1 = 0,
                    twentynm_mac_component.coef_a_2 = 0,
                    twentynm_mac_component.coef_a_3 = 0,
                    twentynm_mac_component.coef_a_4 = 0,
                    twentynm_mac_component.coef_a_5 = 0,
                    twentynm_mac_component.coef_a_6 = 0,
                    twentynm_mac_component.coef_a_7 = 0,
                    twentynm_mac_component.coef_b_0 = 0,
                    twentynm_mac_component.coef_b_1 = 0,
                    twentynm_mac_component.coef_b_2 = 0,
                    twentynm_mac_component.coef_b_3 = 0,
                    twentynm_mac_component.coef_b_4 = 0,
                    twentynm_mac_component.coef_b_5 = 0,
                    twentynm_mac_component.coef_b_6 = 0,
                    twentynm_mac_component.coef_b_7 = 0,
                    twentynm_mac_component.ax_clock = "0",
                    twentynm_mac_component.ay_scan_in_clock = "0",
                    twentynm_mac_component.az_clock = "none",
                    twentynm_mac_component.bx_clock = "0",
                    twentynm_mac_component.by_clock = "0",
                    twentynm_mac_component.bz_clock = "none",
                    twentynm_mac_component.coef_sel_a_clock = "none",
                    twentynm_mac_component.coef_sel_b_clock = "none",
                    twentynm_mac_component.sub_clock = "none",
                    twentynm_mac_component.sub_pipeline_clock = "none",
                    twentynm_mac_component.negate_clock = "none",
                    twentynm_mac_component.negate_pipeline_clock = "none",
                    twentynm_mac_component.accumulate_clock = "none",
                    twentynm_mac_component.accum_pipeline_clock = "none",
                    twentynm_mac_component.load_const_clock = "none",
                    twentynm_mac_component.load_const_pipeline_clock = "none",
                    twentynm_mac_component.input_pipeline_clock = "none",
                    twentynm_mac_component.output_clock = "0",
                    twentynm_mac_component.scan_out_width = 8,
                    twentynm_mac_component.result_a_width = 32;
endmodule //dsp_no_input_pipeline_reg

module dsp_input_pipeline_reg(
		input wire [7:0] ax,  //multiplier-a input X
		input wire [7:0] ay,  //multiplier-a input Y
		input wire [7:0] bx,  //multiplier-b input X
		input wire [7:0] by,  //multiplier-b input Y
		input wire [63:0] chainin, //chain-adder input
		input wire [2:0] ena, //clock anbles, {ena2, ena1, ena0}
		output wire [63:0] chainout, //chain out
		output wire [31:0] resulta,
        input wire  aclr0,
        input wire  aclr1,
        input  clk0,
        input  clk1,
        input  clk2
	);

	twentynm_mac        twentynm_mac_component (
                                        .ax (ax),
                                        .ay (ay),
                                        .bx (bx),
                                        .by (by),
                                        .chainin (chainin),
                                        .ena (ena),
                                        .chainout (chainout),
                                        .resulta (resulta),
                                        .aclr ({aclr1,aclr0}),
                                        .clk ({clk2,clk1,clk0}),
                                        .accumulate (),
                                        .az (),
                                        .bz (),
                                        .coefsela (),
                                        .coefselb (),
                                        .dftout (),
                                        .loadconst (),
                                        .negate (),
                                        .resultb (),
                                        .scanin (),
                                        .scanout (),
                                        .sub ());
            defparam
                    twentynm_mac_component.ax_width = 8,
                    twentynm_mac_component.ay_scan_in_width = 8,
                    twentynm_mac_component.bx_width = 8,
                    twentynm_mac_component.by_width = 8,
                    twentynm_mac_component.operation_mode = "m18x18_sumof2",
                    twentynm_mac_component.mode_sub_location = 0,
                    twentynm_mac_component.operand_source_max = "input",
                    twentynm_mac_component.operand_source_may = "input",
                    twentynm_mac_component.operand_source_mbx = "input",
                    twentynm_mac_component.operand_source_mby = "input",
                    twentynm_mac_component.signed_max = "true",
                    twentynm_mac_component.signed_may = "true",
                    twentynm_mac_component.signed_mbx = "true",
                    twentynm_mac_component.signed_mby = "true",
                    twentynm_mac_component.preadder_subtract_a = "false",
                    twentynm_mac_component.preadder_subtract_b = "false",
                    twentynm_mac_component.ay_use_scan_in = "false",
                    twentynm_mac_component.by_use_scan_in = "false",
                    twentynm_mac_component.delay_scan_out_ay = "false",
                    twentynm_mac_component.delay_scan_out_by = "false",
                    twentynm_mac_component.use_chainadder = "true",
                    twentynm_mac_component.enable_double_accum = "false",
                    twentynm_mac_component.load_const_value = 0,
                    twentynm_mac_component.coef_a_0 = 0,
                    twentynm_mac_component.coef_a_1 = 0,
                    twentynm_mac_component.coef_a_2 = 0,
                    twentynm_mac_component.coef_a_3 = 0,
                    twentynm_mac_component.coef_a_4 = 0,
                    twentynm_mac_component.coef_a_5 = 0,
                    twentynm_mac_component.coef_a_6 = 0,
                    twentynm_mac_component.coef_a_7 = 0,
                    twentynm_mac_component.coef_b_0 = 0,
                    twentynm_mac_component.coef_b_1 = 0,
                    twentynm_mac_component.coef_b_2 = 0,
                    twentynm_mac_component.coef_b_3 = 0,
                    twentynm_mac_component.coef_b_4 = 0,
                    twentynm_mac_component.coef_b_5 = 0,
                    twentynm_mac_component.coef_b_6 = 0,
                    twentynm_mac_component.coef_b_7 = 0,
                    twentynm_mac_component.ax_clock = "0",
                    twentynm_mac_component.ay_scan_in_clock = "0",
                    twentynm_mac_component.az_clock = "none",
                    twentynm_mac_component.bx_clock = "0",
                    twentynm_mac_component.by_clock = "0",
                    twentynm_mac_component.bz_clock = "none",
                    twentynm_mac_component.coef_sel_a_clock = "none",
                    twentynm_mac_component.coef_sel_b_clock = "none",
                    twentynm_mac_component.sub_clock = "none",
                    twentynm_mac_component.sub_pipeline_clock = "none",
                    twentynm_mac_component.negate_clock = "none",
                    twentynm_mac_component.negate_pipeline_clock = "none",
                    twentynm_mac_component.accumulate_clock = "none",
                    twentynm_mac_component.accum_pipeline_clock = "none",
                    twentynm_mac_component.load_const_clock = "none",
                    twentynm_mac_component.load_const_pipeline_clock = "none",
                    twentynm_mac_component.input_pipeline_clock = "0",
                    twentynm_mac_component.output_clock = "0",
                    twentynm_mac_component.scan_out_width = 8,
                    twentynm_mac_component.result_a_width = 32;
endmodule

module a10_chain_madd_8bitx8 (
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

		input  wire [7:0]  dataa_4, // dataa_0.dataa_0
		input  wire [7:0]  datab_4, // datab_0.datab_0
		
		input  wire [7:0]  dataa_5, // dataa_1.dataa_1
		input  wire [7:0]  datab_5, // datab_1.datab_1
		
		input  wire [7:0]  dataa_6, // dataa_2.dataa_2
		input  wire [7:0]  datab_6, // datab_2.datab_2
		
		input  wire [7:0]  dataa_7, // dataa_3.dataa_3
		input  wire [7:0]  datab_7, // datab_3.datab_3

		// input  wire [31:0] dataBias,
		
		output wire [31:0] result  //  result.result
	);
	
	//Pipeline balancing registers
	reg [7:0] regA4_0, regB4_0, regA5_0, regB5_0;
	reg [7:0] regA6_0, regB6_0, regA7_0, regB7_0;
	reg [7:0] regA6_1, regB6_1, regA7_1, regB7_1;

	//Chain buses
	wire [63:0] chain_d0d1, chain_d1d2, chain_d2d3;

	//Output valid propagation stage
	// reg [4:0] regOutputValid;

	//DSP0
	dsp_no_input_pipeline_reg dsp0(
			.ax(dataa_0),
			.ay(datab_0),
			.bx(dataa_1),
			.by(datab_1),
			.chainin ({64{1'b0}}),
			.ena({3'b001}),
			.chainout(chain_d0d1),
			.clk0    (clock)
		);

	dsp_input_pipeline_reg dsp1(
			.ax(dataa_2),
			.ay(datab_2),
			.bx(dataa_3),
			.by(datab_3),
			.chainin (chain_d0d1),
			.ena({3'b001}),
			.chainout(chain_d1d2),
			.clk0    (clock)
		);

	dsp_input_pipeline_reg dsp2(
			.ax(regA4_0),
			.ay(regB4_0),
			.bx(regA5_0),
			.by(regB5_0),
			.chainin (chain_d1d2),
			.ena({3'b001}),
			.chainout(chain_d2d3),
			.clk0    (clock)
		);

	dsp_input_pipeline_reg dsp3(
			.ax(regA6_1),
			.ay(regB6_1),
			.bx(regA7_1),
			.by(regB7_1),
			.chainin (chain_d2d3),
			.ena({3'b001}),
			.resulta (result),
			.clk0    (clock)
		);

	assign ovalid = 1'b1;
	assign oready = 1'b1;
	// ivalid, iready, resetn are ignored

	//Register updates
	always @(posedge clock) begin
		{regA6_1, regB6_1, regA7_1, regB7_1} <= {regA6_0, regB6_0, regA7_0, regB7_0};
		{regA7_0, regB7_0, regA6_0, regB6_0} <= {dataa_7, datab_7, dataa_6, datab_6};
		{regA5_0, regB5_0, regA4_0, regB4_0} <= {dataa_5, datab_5, dataa_4, datab_4};
	end

	// always @(posedge clk0) begin
	// 	if (resetn == 1'b0) begin
	// 		regOutputValid <= 5'b000;
	// 	end
	// 	else begin:
	// 		regOutputValid <= {regOutputValid[3:0], ivalid};
	// 	end
	// end


endmodule
