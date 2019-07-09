// Copyright (C) 2017  Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions 
// and other software and tools, and its AMPP partner logic 
// functions, and any output files from any of the foregoing 
// (including device programming or simulation files), and any 
// associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License 
// Subscription Agreement, the Intel Quartus Prime License Agreement,
// the Intel FPGA IP License Agreement, or other applicable license
// agreement, including, without limitation, that your use is for
// the sole purpose of programming logic devices manufactured by
// Intel and sold by Intel or its authorized distributors.  Please
// refer to the applicable agreement for further details.

// VENDOR "Altera"
// PROGRAM "Quartus Prime"
// VERSION "Version 17.1.1 Build 273 12/19/2017 SJ Pro Edition"

// DATE "07/08/2019 14:51:22"

// 
// Device: Altera 10AX115S2F45I2SGES Package FBGA1932
// 

// 
// This greybox netlist file is for third party Synthesis Tools
// for timing and resource estimation only.
// 


module a10_mac_8bitx4 (
	result,
	clock0,
	datab_3,
	dataa_3,
	datab_2,
	dataa_2,
	datab_1,
	dataa_1,
	datab_0,
	dataa_0)/* synthesis synthesis_greybox=0 */;
output 	[17:0] result;
input 	clock0;
input 	[7:0] datab_3;
input 	[7:0] dataa_3;
input 	[7:0] datab_2;
input 	[7:0] dataa_2;
input 	[7:0] datab_1;
input 	[7:0] dataa_1;
input 	[7:0] datab_0;
input 	[7:0] dataa_0;

wire gnd;
wire vcc;
wire unknown;

assign gnd = 1'b0;
assign vcc = 1'b1;
// unknown value (1'bx) is not needed for this tool. Default to 1'b0
assign unknown = 1'b0;

wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[0] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[1] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[2] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[3] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[4] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[5] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[6] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[7] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[8] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[9] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[10] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[11] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[12] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[13] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[14] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[15] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[16] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[17] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~12 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~13 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~14 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~15 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~16 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~17 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~18 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~19 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~20 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~21 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~22 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~23 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~24 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~25 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~26 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~27 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~28 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~29 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~30 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~31 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~32 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~33 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~34 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~35 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~36 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~37 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~38 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~39 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~40 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~41 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~42 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~43 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~44 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~45 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~46 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~47 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~48 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~49 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~50 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~51 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~52 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~53 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~54 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~55 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~56 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~57 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[0] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[1] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[2] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[3] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[4] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[5] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[6] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[7] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[8] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[9] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[10] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[11] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[12] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[13] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[14] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[15] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[16] ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~12 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~13 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~14 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~15 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~16 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~17 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~18 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~19 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~20 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~21 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~22 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~23 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~24 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~25 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~26 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~27 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~28 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~29 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~30 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~31 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~32 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~33 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~34 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~35 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~36 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~37 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~38 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~39 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~40 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~41 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~42 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~43 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~44 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~45 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~46 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~47 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~48 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~49 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~50 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~51 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~52 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~53 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~54 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~55 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~56 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~57 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~58 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~123 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~124 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~125 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~126 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~127 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~128 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~129 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~130 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~131 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~132 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~133 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~134 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~135 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~136 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~137 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~138 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~139 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~140 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~141 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~142 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~143 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~144 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~145 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~146 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~147 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~148 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~149 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~150 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~151 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~152 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~153 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~154 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~155 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~156 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~157 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~158 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~159 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~160 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~161 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~162 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~163 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~164 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~165 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~166 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~167 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~168 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~169 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~170 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~171 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~172 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~173 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~174 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~175 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~176 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~177 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~178 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~179 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~180 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~181 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~182 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~183 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~184 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~185 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~186 ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[0]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[1]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[2]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[3]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[4]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[5]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[6]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[7]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[8]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[9]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[10]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[11]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[12]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[13]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[14]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[15]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[16]~q ;
wire \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[17]~q ;

wire [63:0] \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus ;
wire [63:0] \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus ;
wire [63:0] \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus ;

assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[0]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [0];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[1]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [1];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[2]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [2];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[3]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [3];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[4]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [4];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[5]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [5];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[6]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [6];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[7]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [7];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[8]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [8];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[9]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [9];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[10]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [10];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[11]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [11];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[12]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [12];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[13]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [13];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[14]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [14];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[15]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [15];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[16]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [16];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[17]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [17];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~12  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [18];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~13  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [19];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~14  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [20];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~15  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [21];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~16  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [22];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~17  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [23];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~18  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [24];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~19  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [25];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~20  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [26];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~21  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [27];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~22  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [28];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~23  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [29];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~24  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [30];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~25  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [31];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~26  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [32];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~27  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [33];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~28  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [34];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~29  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [35];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~30  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [36];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~31  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [37];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~32  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [38];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~33  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [39];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~34  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [40];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~35  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [41];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~36  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [42];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~37  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [43];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~38  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [44];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~39  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [45];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~40  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [46];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~41  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [47];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~42  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [48];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~43  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [49];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~44  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [50];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~45  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [51];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~46  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [52];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~47  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [53];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~48  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [54];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~49  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [55];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~50  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [56];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~51  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [57];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~52  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [58];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~53  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [59];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~54  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [60];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~55  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [61];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~56  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [62];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~57  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus [63];

assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[0]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [0];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[1]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [1];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[2]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [2];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[3]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [3];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[4]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [4];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[5]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [5];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[6]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [6];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[7]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [7];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[8]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [8];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[9]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [9];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[10]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [10];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[11]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [11];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[12]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [12];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[13]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [13];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[14]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [14];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[15]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [15];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|adder_result_1[16]  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [16];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~12  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [17];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~13  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [18];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~14  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [19];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~15  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [20];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~16  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [21];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~17  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [22];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~18  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [23];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~19  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [24];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~20  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [25];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~21  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [26];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~22  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [27];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~23  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [28];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~24  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [29];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~25  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [30];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~26  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [31];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~27  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [32];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~28  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [33];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~29  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [34];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~30  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [35];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~31  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [36];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~32  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [37];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~33  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [38];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~34  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [39];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~35  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [40];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~36  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [41];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~37  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [42];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~38  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [43];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~39  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [44];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~40  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [45];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~41  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [46];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~42  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [47];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~43  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [48];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~44  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [49];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~45  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [50];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~46  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [51];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~47  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [52];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~48  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [53];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~49  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [54];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~50  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [55];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~51  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [56];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~52  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [57];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~53  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [58];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~54  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [59];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~55  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [60];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~56  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [61];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~57  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [62];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~58  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus [63];

assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~123  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [0];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~124  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [1];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~125  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [2];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~126  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [3];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~127  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [4];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~128  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [5];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~129  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [6];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~130  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [7];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~131  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [8];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~132  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [9];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~133  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [10];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~134  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [11];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~135  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [12];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~136  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [13];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~137  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [14];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~138  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [15];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~139  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [16];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~140  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [17];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~141  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [18];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~142  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [19];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~143  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [20];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~144  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [21];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~145  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [22];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~146  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [23];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~147  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [24];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~148  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [25];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~149  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [26];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~150  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [27];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~151  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [28];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~152  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [29];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~153  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [30];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~154  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [31];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~155  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [32];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~156  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [33];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~157  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [34];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~158  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [35];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~159  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [36];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~160  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [37];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~161  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [38];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~162  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [39];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~163  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [40];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~164  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [41];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~165  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [42];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~166  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [43];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~167  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [44];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~168  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [45];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~169  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [46];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~170  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [47];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~171  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [48];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~172  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [49];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~173  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [50];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~174  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [51];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~175  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [52];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~176  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [53];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~177  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [54];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~178  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [55];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~179  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [56];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~180  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [57];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~181  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [58];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~182  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [59];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~183  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [60];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~184  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [61];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~185  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [62];
assign \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~186  = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus [63];

twentynm_mac \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac (
	.sub(gnd),
	.negate(gnd),
	.accumulate(gnd),
	.loadconst(gnd),
	.ax({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,datab_3[7],datab_3[6],datab_3[5],datab_3[4],datab_3[3],datab_3[2],datab_3[1],datab_3[0]}),
	.ay({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,dataa_3[7],dataa_3[6],dataa_3[5],dataa_3[4],dataa_3[3],dataa_3[2],dataa_3[1],dataa_3[0]}),
	.az(26'b00000000000000000000000000),
	.bx({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,datab_2[7],datab_2[6],datab_2[5],datab_2[4],datab_2[3],datab_2[2],datab_2[1],datab_2[0]}),
	.by({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,dataa_2[7],dataa_2[6],dataa_2[5],dataa_2[4],dataa_2[3],dataa_2[2],dataa_2[1],dataa_2[0]}),
	.bz(18'b000000000000000000),
	.coefsela(3'b000),
	.coefselb(3'b000),
	.clk(3'b000),
	.aclr(2'b00),
	.ena(3'b111),
	.scanin(27'b000000000000000000000000000),
	.chainin({\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~186 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~185 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~184 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~183 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~182 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~181 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~180 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~179 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~178 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~177 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~176 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~175 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~174 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~173 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~172 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~171 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~170 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~169 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~168 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~167 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~166 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~165 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~164 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~163 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~162 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~161 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~160 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~159 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~158 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~157 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~156 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~155 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~154 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~153 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~152 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~151 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~150 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~149 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~148 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~147 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~146 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~145 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~144 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~143 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~142 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~141 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~140 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~139 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~138 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~137 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~136 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~135 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~134 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~133 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~132 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~131 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~130 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~129 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~128 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~127 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~126 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~125 ,
\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~124 ,\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~123 }),
	.dftout(),
	.resulta(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac_RESULTA_bus ),
	.resultb(),
	.scanout(),
	.chainout());
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .accum_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .accumulate_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .ax_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .ax_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .ay_scan_in_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .ay_scan_in_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .ay_use_scan_in = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .az_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .bx_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .bx_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .by_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .by_use_scan_in = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .by_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .bz_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_0 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_1 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_2 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_3 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_4 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_5 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_6 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_a_7 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_0 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_1 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_2 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_3 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_4 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_5 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_6 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_b_7 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_sel_a_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .coef_sel_b_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .delay_scan_out_ay = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .delay_scan_out_by = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .enable_double_accum = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .input_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .load_const_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .load_const_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .load_const_value = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .mode_sub_location = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .negate_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .negate_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .operand_source_max = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .operand_source_may = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .operand_source_mbx = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .operand_source_mby = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .operation_mode = "m18x18_sumof2";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .output_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .preadder_subtract_a = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .preadder_subtract_b = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .result_a_width = 64;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .signed_max = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .signed_may = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .signed_mbx = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .signed_mby = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .sub_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .sub_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_1~mac .use_chainadder = "true";

twentynm_mac \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac (
	.sub(gnd),
	.negate(gnd),
	.accumulate(gnd),
	.loadconst(gnd),
	.ax({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,datab_1[7],datab_1[6],datab_1[5],datab_1[4],datab_1[3],datab_1[2],datab_1[1],datab_1[0]}),
	.ay({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,dataa_1[7],dataa_1[6],dataa_1[5],dataa_1[4],dataa_1[3],dataa_1[2],dataa_1[1],dataa_1[0]}),
	.az(26'b00000000000000000000000000),
	.bx({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,datab_0[7],datab_0[6],datab_0[5],datab_0[4],datab_0[3],datab_0[2],datab_0[1],datab_0[0]}),
	.by({gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,gnd,dataa_0[7],dataa_0[6],dataa_0[5],dataa_0[4],dataa_0[3],dataa_0[2],dataa_0[1],dataa_0[0]}),
	.bz(18'b000000000000000000),
	.coefsela(3'b000),
	.coefselb(3'b000),
	.clk(3'b000),
	.aclr(2'b00),
	.ena(3'b111),
	.scanin(27'b000000000000000000000000000),
	.chainin(1'b0),
	.dftout(),
	.resulta(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_RESULTA_bus ),
	.resultb(),
	.scanout(),
	.chainout(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac_CHAINOUT_bus ));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .accum_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .accumulate_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .ax_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .ax_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .ay_scan_in_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .ay_scan_in_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .ay_use_scan_in = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .az_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .bx_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .bx_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .by_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .by_use_scan_in = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .by_width = 8;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .bz_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_0 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_1 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_2 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_3 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_4 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_5 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_6 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_a_7 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_0 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_1 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_2 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_3 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_4 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_5 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_6 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_b_7 = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_sel_a_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .coef_sel_b_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .delay_scan_out_ay = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .delay_scan_out_by = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .enable_double_accum = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .input_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .load_const_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .load_const_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .load_const_value = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .mode_sub_location = 0;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .negate_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .negate_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .operand_source_max = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .operand_source_may = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .operand_source_mbx = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .operand_source_mby = "input";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .operation_mode = "m18x18_sumof2";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .output_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .preadder_subtract_a = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .preadder_subtract_b = "false";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .result_a_width = 64;
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .signed_max = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .signed_may = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .signed_mbx = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .signed_mby = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .sub_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .sub_pipeline_clock = "none";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|add_0~mac .use_chainadder = "false";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[0] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[0] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[0]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[0] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[0] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[1] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[1] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[1]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[1] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[1] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[2] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[2] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[2]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[2] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[2] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[3] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[3] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[3]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[3] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[3] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[4] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[4] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[4]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[4] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[4] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[5] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[5] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[5]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[5] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[5] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[6] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[6] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[6]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[6] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[6] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[7] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[7] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[7]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[7] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[7] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[8] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[8] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[8]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[8] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[8] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[9] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[9] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[9]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[9] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[9] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[10] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[10] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[10]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[10] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[10] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[11] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[11] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[11]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[11] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[11] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[12] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[12] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[12]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[12] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[12] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[13] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[13] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[13]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[13] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[13] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[14] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[14] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[14]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[14] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[14] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[15] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[15] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[15]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[15] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[15] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[16] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[16] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[16]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[16] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[16] .power_up = "low";

dffeas \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[17] (
	.clk(clock0),
	.d(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|final_adder_block|data_out_wire[17] ),
	.asdata(vcc),
	.clrn(vcc),
	.aload(gnd),
	.sclr(gnd),
	.sload(gnd),
	.ena(vcc),
	.q(\mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[17]~q ),
	.prn(vcc));
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[17] .is_wysiwyg = "true";
defparam \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[17] .power_up = "low";

assign result[0] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[0]~q ;

assign result[1] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[1]~q ;

assign result[2] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[2]~q ;

assign result[3] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[3]~q ;

assign result[4] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[4]~q ;

assign result[5] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[5]~q ;

assign result[6] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[6]~q ;

assign result[7] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[7]~q ;

assign result[8] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[8]~q ;

assign result[9] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[9]~q ;

assign result[10] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[10]~q ;

assign result[11] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[11]~q ;

assign result[12] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[12]~q ;

assign result[13] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[13]~q ;

assign result[14] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[14]~q ;

assign result[15] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[15]~q ;

assign result[16] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[16]~q ;

assign result[17] = \mult_add_0|altera_mult_add_component|auto_generated|altera_mult_add_rtl1|output_reg_block|data_out_wire[17]~q ;

endmodule
