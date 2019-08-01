// ============================================================================
// Copyright (c) 2016 by Terasic Technologies Inc.
// ============================================================================
//
// Permission:
//
//   Terasic grants permission to use and modify this code for use
//   in synthesis for all Terasic Development Boards and Altera Development 
//   Kits made by Terasic.  Other use of this code, including the selling 
//   ,duplication, or modification of any portion is strictly prohibited.
//
// Disclaimer:
//
//   This VHDL/Verilog or C/C++ source code is intended as a design reference
//   which illustrates how these types of functions can be implemented.
//   It is the user's responsibility to verify their design for
//   consistency and functionality through the use of formal
//   verification methods.  Terasic provides no warranty regarding the use 
//   or functionality of this code.
//
// ============================================================================
//           
//  Terasic Technologies Inc
//  9F., No.176, Sec.2, Gongdao 5th Rd, East Dist, Hsinchu City, 30070. Taiwan
//  
//  
//                     web: http://www.terasic.com/  
//                     email: support@terasic.com
//
// ============================================================================
//Date:  Thu Nov  3 15:01:20 2016
// ============================================================================

//`define ENABLE_HSMC
//`define ENABLE_HPS

module top(

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

      ///////// SDRAM /////////
      output             DRAM_CLK,
      output             DRAM_CKE,
      output   [12: 0]   DRAM_ADDR,
      output   [ 1: 0]   DRAM_BA,
      inout    [15: 0]   DRAM_DQ,
      output             DRAM_LDQM,
      output             DRAM_UDQM,
      output             DRAM_CS_N,
      output             DRAM_WE_N,
      output             DRAM_CAS_N,
      output             DRAM_RAS_N,

      ///////// Video-In /////////
      input              TD_CLK27,
      input              TD_HS,
      input              TD_VS,
      input    [ 7: 0]   TD_DATA,
      output             TD_RESET_N,

      ///////// VGA /////////
      output             VGA_CLK,
      output             VGA_HS,
      output             VGA_VS,
      output   [ 7: 0]   VGA_R,
      output   [ 7: 0]   VGA_G,
      output   [ 7: 0]   VGA_B,
      output             VGA_BLANK_N,
      output             VGA_SYNC_N,

      ///////// Audio /////////
      inout              AUD_BCLK,
      output             AUD_XCK,
      inout              AUD_ADCLRCK,
      input              AUD_ADCDAT,
      inout              AUD_DACLRCK,
      output             AUD_DACDAT,

      ///////// PS2 /////////
      inout              PS2_CLK,
      inout              PS2_CLK2,
      inout              PS2_DAT,
      inout              PS2_DAT2,

      ///////// ADC /////////
      output             ADC_SCLK,
      input              ADC_DOUT,
      output             ADC_DIN,
      output             ADC_CONVST,

      ///////// I2C for Audio and Video-In /////////
      output             FPGA_I2C_SCLK,
      inout              FPGA_I2C_SDAT,

      ///////// GPIO /////////
      inout    [35: 0]   GPIO,

`ifdef ENABLE_HSMC
      ///////// HSMC /////////
      input              HSMC_CLKIN_P1,
      input              HSMC_CLKIN_N1,
      input              HSMC_CLKIN_P2,
      input              HSMC_CLKIN_N2,
      output             HSMC_CLKOUT_P1,
      output             HSMC_CLKOUT_N1,
      output             HSMC_CLKOUT_P2,
      output             HSMC_CLKOUT_N2,
      inout    [16: 0]   HSMC_TX_D_P,
      inout    [16: 0]   HSMC_TX_D_N,
      inout    [16: 0]   HSMC_RX_D_P,
      inout    [16: 0]   HSMC_RX_D_N,
      input              HSMC_CLKIN0,
      output             HSMC_CLKOUT0,
      inout    [ 3: 0]   HSMC_D,
      output             HSMC_SCL,
      inout              HSMC_SDA,
`endif /*ENABLE_HSMC*/

`ifdef ENABLE_HPS
      ///////// HPS /////////
      inout              HPS_CONV_USB_N,
      output   [14: 0]   HPS_DDR3_ADDR,
      output   [ 2: 0]   HPS_DDR3_BA,
      output             HPS_DDR3_CAS_N,
      output             HPS_DDR3_CKE,
      output             HPS_DDR3_CK_N,
      output             HPS_DDR3_CK_P,
      output             HPS_DDR3_CS_N,
      output   [ 3: 0]   HPS_DDR3_DM,
      inout    [31: 0]   HPS_DDR3_DQ,
      inout    [ 3: 0]   HPS_DDR3_DQS_N,
      inout    [ 3: 0]   HPS_DDR3_DQS_P,
      output             HPS_DDR3_ODT,
      output             HPS_DDR3_RAS_N,
      output             HPS_DDR3_RESET_N,
      input              HPS_DDR3_RZQ,
      output             HPS_DDR3_WE_N,
      output             HPS_ENET_GTX_CLK,
      inout              HPS_ENET_INT_N,
      output             HPS_ENET_MDC,
      inout              HPS_ENET_MDIO,
      input              HPS_ENET_RX_CLK,
      input    [ 3: 0]   HPS_ENET_RX_DATA,
      input              HPS_ENET_RX_DV,
      output   [ 3: 0]   HPS_ENET_TX_DATA,
      output             HPS_ENET_TX_EN,
      inout    [ 3: 0]   HPS_FLASH_DATA,
      output             HPS_FLASH_DCLK,
      output             HPS_FLASH_NCSO,
      inout              HPS_GSENSOR_INT,
      inout              HPS_I2C1_SCLK,
      inout              HPS_I2C1_SDAT,
      inout              HPS_I2C2_SCLK,
      inout              HPS_I2C2_SDAT,
      inout              HPS_I2C_CONTROL,
      inout              HPS_KEY,
      inout              HPS_LCM_BK,
      inout              HPS_LCM_D_C,
      inout              HPS_LCM_RST_N,
      output             HPS_LCM_SPIM_CLK,
      output             HPS_LCM_SPIM_MOSI,
	  input 			 HPS_LCM_SPIM_MISO,
      output             HPS_LCM_SPIM_SS,
      inout              HPS_LED,
      inout              HPS_LTC_GPIO,
      output             HPS_SD_CLK,
      inout              HPS_SD_CMD,
      inout    [ 3: 0]   HPS_SD_DATA,
      output             HPS_SPIM_CLK,
      input              HPS_SPIM_MISO,
      output             HPS_SPIM_MOSI,
      output             HPS_SPIM_SS,
      input              HPS_UART_RX,
      output             HPS_UART_TX,
      input              HPS_USB_CLKOUT,
      inout    [ 7: 0]   HPS_USB_DATA,
      input              HPS_USB_DIR,
      input              HPS_USB_NXT,
      output             HPS_USB_STP,
`endif /*ENABLE_HPS*/


      ///////// IR /////////
      output             IRDA_TXD,
      input              IRDA_RXD
);


//=======================================================
//  REG/WIRE declarations
//=======================================================
wire [63:0] wireResult;
reg [7:0] regSwitchInput;
reg [7:0] regSwitchA;
//reg [63:0] regOutput;

//assign LEDR [9:0] = {2'b00, regLedOutput};
//=======================================================
//  Structural coding
//=======================================================
operandMatcher8 dut (
         .clock (CLOCK_50),
         .bitmaskW(regSwitchInput),
         .resetn  (KEY[0]),
        .bitmaskA (regSwitchA),
        .result (wireResult)
    );

always @ (posedge CLOCK_50) begin
    if (KEY[0] == 1'b0) begin
        //regOutput <= 64'h0;
        regSwitchInput<= 8'h00;
        regSwitchA<=8'h00;
    end
    else begin
        regSwitchInput <= {SW[9:5], GPIO[2:0]};
        regSwitchA <= {SW[4:0], GPIO[5:3]};
        //regSwitchInput <= SW[7:0];
        //regSwitchA <= SW[7:0];
        //regOutput <= wireResult;
    end
end

hexDriver hexD0 (
  .in(wireResult[3:0]),
  .hex(HEX0)
  );

hexDriver hexD1 (
  .in(wireResult[7:4]),
  .hex(HEX1)
  );

hexDriver hexD2 (
  .in(wireResult[11:8]),
  .hex(HEX2)
  );

hexDriver hexD3 (
  .in(wireResult[15:12]),
  .hex(HEX3)
  );

hexDriver hexD4 (
  .in(wireResult[19:16]),
  .hex(HEX4)
  );

hexDriver hexD5 (
  .in(wireResult[23:20]),
  .hex(HEX5)
  );

//assign LEDR[0] = &(wireResult[63:24]);
assign GPIO[29:6] = wireResult[47:24];
assign LEDR[3:0] = wireResult[51:48];
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
