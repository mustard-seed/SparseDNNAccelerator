//--------------------------------------------------------------------------//
// Title:       a10_fpga_golden_top.v                                        //
// Rev:         Rev 1                                                       //
//--------------------------------------------------------------------------//
// Description: All Arria 10 FPGA Dev Kit I/O      //
//              FPGA signals and settings such as termination, drive       //
//              strength, etc...  Some toggle_rate=0 where needed for       // 
//					 fitter rules.(TR=0)														 //
//--------------------------------------------------------------------------//
// Revision History:                                                        //
// Rev 1:       Board Revision A FPGA pinout.		 								 //
//----------------------------------------------------------------------------
//------ 1 ------- 2 ------- 3 ------- 4 ------- 5 ------- 6 ------- 7 ------7
//------ 0 ------- 0 ------- 0 ------- 0 ------- 0 ------- 0 ------- 0 ------8
//----------------------------------------------------------------------------
//Copyright 2013 Altera Corporation. All rights reserved.  Altera products  
//are protected under numerous U.S. and foreign patents, maskwork rights,     
//copyrights and other intellectual property laws.                            
//                                                                            
//This reference design file, and your use thereof, is subject to and         
//governed by the terms and conditions of the applicable Altera Reference     
//Design License Agreement.  By using this reference design file, you         
//indicate your acceptance of such terms and conditions between you and       
//Altera Corporation.  In the event that you do not agree with such terms and 
//conditions, you may not use the reference design file. Please promptly      
//destroy any copies you have made.                                           
//                                                                            
//This reference design file being provided on an "as-is" basis and as an     
//accommodation and therefore all warranties, representations or guarantees   
//of any kind (whether express, implied or statutory) including, without      
//limitation, warranties of merchantability, non-infringement, or fitness for 
//a particular purpose, are specifically disclaimed.  By making this          
//reference design file available, Altera expressly does not recommend,       
//suggest or require that this reference design file be used in combination   
//with any other product not provided by Altera.           

module a10_fpga_golden_top (
//Clocks Inputs
	input		clk_emi_p,	//LVDS - 100MHz (Programmable Si5338)
	input		clk_fpga_b2_p,	//LVDS - 100MHz (Programmable Si570) or SMA input
	input		clk_fpga_b3_p,	//LVDS - 100MHz (Programmable Si570) or SMA input
	input		clk_125_p,	//LVDS - 125MHz
	input		clk_50,		//1.8V - 50MHz
//	input		refclk1_p,	//LVDS XCVR reference clock - 100MHz (Programmable Si570) or SMA input
//	input 	refclk4_p,	//LVDS XCVR reference clock - 100MHz (Programmable Si570) or SMA input
	
	output	sdi_clk148_up,	 //Voltage Control for SDI VCXO
	output	sdi_clk148_down,//Voltage Control for SDI VCXO
	
	output	sma_clk_out,	//1.8V - SMA Clock Output
	
	output	clock_scl,	//1.8V, Level Translator? 2.5V Si5338 I2C scl -> Tri-state when not in use.
	inout		clock_sda,	//1.8V, Level Translator? 2.5V Si5338 I2C sda -> Tri-state when not in use.

//LCD ------------------------------------------------------------------
	output	disp_spiss,	//SPI Slave Select (NC in I2C mode)
	output	disp_i2c_scl,	//LCD Serial Clock
	inout		disp_i2c_sda,	//LCD Serial Data In (SPI)/Serial Data (I2C)
	
//External Memory Connector   ///Variable voltage to support DDR3, DDR4, RLD3, QDR4
	output [31:0]	mem_addr_cmd,
	output			mem_clk_p,

	inout  [8:0]	mem_dq_addr_cmd,	//DDR3/DDR4 DQS8 group and QDR4 ADDR/CMD group
	inout				mem_dqs_addr_cmd_p,
	inout				mem_dqs_addr_cmd_n,
	
	inout	 [3:0]	mem_dma, 
	inout	 [33:0]	mem_dqa,		//4 x8DQS groups & 2 x16DQS groups
	inout	 [3:0]	mem_dqsa_p,
	inout	 [3:0]	mem_dqsa_n,
	inout	 [1:0]	mem_qka_n,		//qka_p0 = mem_dqa7, qka_p1 = mem_dqa23
	
	inout	 [3:0]	mem_dmb, 
	inout	 [33:0]	mem_dqb,		//4 x8DQS groups & 2 x16DQS groups
	inout	 [3:0]	mem_dqsb_p,	
	inout	 [3:0]	mem_dqsb_n,
	inout	 [1:0]	mem_qkb_n,		//qkb_p0 = mem_dqb7, qkb_p1 = mem_dqb23
	
	
//OCT Termination
	input				rzq_b2k,              	//EMI oct.rzqin for DDR3, DDR4, RLD3 and QDR4

//Dual Purpose Config Pins ------------//33 pins//----------------------------
	inout				fpga_cvp_confdone,	//1.8V
	inout	[31:0]	fpga_config_data,	//1.8V

//Max V System Controller -------------// 8 pins//----------------------------
	inout	[3:0]	max5_ben,	//1.8V
	inout			max5_clk,	//1.8V
	inout			max5_csn,	//1.8v
	inout			max5_oen,	//1.8V
	inout			max5_wen,	//1.8V

//User-IO------------------------------//28 pins //--------------------------
   input				cpu_resetn,	//1.8V    //CPU Reset Pushbutton (TR=0)

   input	 [7:0] 	user_dipsw,	//1.8V    //User DIP Switches (TR=0)
   output [7:0]	user_led_g,	//1.8V    //User LEDs
	output [7:0]	user_led_r,	//1.8V    //User LEDs
   input  [2:0]	user_pb,	//1.8V    //User Pushbuttons (TR=0)

//Display Port-------------------------//17 pins//----------------------------
//	output	[3:0]	dp_ml_lane_p,	//XCVR data out.
//	input				refclk_dp_p,	//LVDS 270MHz default, programmable
	input				dp_aux_p,	//Input LVDS, Output Diff SSTL-1.8V (BLVDS)
//	inout				dp_aux_p,	//LVDS
	input				dp_hot_plug,	//1.8V, need level translator
	output			dp_return,	//1.8V, need level translator
	inout				dp_config1, //1.8V, need level translator
	inout				dp_config2, //1.8V, need level translator

//Ethernet-----------------------------// 8 pins//----------------------------
	input				enet_rx_p,	//LVDS SGMII RX Data
	output			enet_tx_p,	//LVDS SGMII TX Data
	output			enet_intn,	//1.8V, need level translator
	output			enet_resetn,	//1.8V, need level translator
	output			enet_mdc,	//1.8V, need level translator
	inout				enet_mdio,	//1.8V, need level translator

//Flash ------------------------------//66 pins//----------------------------
	output [26:1]	fm_a,	//1.8V
	inout	 [31:0]	fm_d,	//1.8V
	output			flash_advn,	//1.8V
	output [1:0]	flash_cen,	//1.8V
	output			flash_clk,	//1.8V
	output			flash_oen,	//1.8V
	input				flash_rdybsyn0,	//1.8V
	input				flash_rdybsyn1,	//1.8V
	output			flash_resetn,	//1.8V
	output			flash_wen,	//1.8V
	
//FPGA Mezzanine Card (FMCA)----------//149 Pins//----------------------------
	//XCVR Interface///
//	input	[1:0]	fmca_gbtclk_m2c_p,	//Transciever Ref Clocks
//	input 	[15:0]	fmca_dp_m2c_p,		//Transceiver Data FPGA RX
//	output	[15:0]	fmca_dp_c2m_p,		//Transceiver Data FPGA TX
//	input		refclk_fmca_p,		//On-board ref clock for FMCA - Default 625MHz
	
	input	[1:0]		fmca_clk_m2c_p,		//LVDS - Dedicated Clock Input
	input	[1:0]		fmca_la_rx_clk_p,	//LVDS - Clock Input

//Arria 10 LDVS can be Input or Output, not Bidirectional	
	output[14:0]	fmca_la_rx_p,		//LVDS
	output[16:0]	fmca_la_tx_p,		//LVDS
//	input	[14:0]	fmca_la_rx_p,		//LVDS
//	input	[16:0]	fmca_la_tx_p,		//LVDS
	
	inout	[1:0]		fmca_ga,		//1.8V
	input				fmca_prsntn,		//1.8V
	output			fmca_scl,		//1.8V
	inout				fmca_sda,		//1.8V
	
	inout				fmca_rx_led,		//1.8V
	inout				fmca_tx_led,		//1.8V
	
//FPGA Mezzanine Card (FMCB)----------//149 pins//----------------------------
	//XCVR Interface///
//	input	[1:0]	fmcb_gbtclk_m2c_p,	//Transciever Ref Clocks
//	input 	[15:0]	fmcb_dp_m2c_p,		//Transceiver Data FPGA RX
//	output	[15:0]	fmcb_dp_c2m_p,		//Transceiver Data FPGA TX
//	input		refclk_fmcb_p,		//On-board ref clock for FMCB - Default 625MHz
	
	input	[1:0]		fmcb_clk_m2c_p,		//LVDS - Dedicated Clock Input
	input	[1:0]		fmcb_la_rx_clk_p,	//LVDS - Clock Input
	
//Arria 10 LDVS can be Input or Output, not Bidirectional	
	output[14:0]	fmcb_la_rx_p,		//LVDS
	output[16:0]	fmcb_la_tx_p,		//LVDS
//	input	[14:0]	fmcb_la_rx_p,		//LVDS
//	input	[16:0]	fmcb_la_tx_p,		//LVDS
	
	inout	[1:0]		fmcb_ga,		//1.8V
	input				fmcb_prsntn,		//1.8V
	output			fmcb_scl,		//1.8V
	inout				fmcb_sda,		//1.8V
	
	inout				fmcb_rx_led,		//1.8V
	inout				fmcb_tx_led,		//1.8V
	
//PCI-Express--------------------------//49 pins //--------------------------
//	input	[7:0]	pcie_rx_p,           	//PCML14  //PCIe Receive Data-req's OCT
//	output	[7:0]	pcie_tx_p,           	//PCML14  //PCIe Transmit Data
//	input		pcie_edge_refclk_p,     //HCSL    //PCIe Clock- Terminate on MB
//	input		pcie_ob_refclk_p,	//LVDS On-board programmable Ref Clock, Default 100MHz
	
   input			pcie_perstn,		//1.8V    //PCIe Reset 
	input			pcie_smbclk,		//2.5V    //SMBus Clock (TR=0)
	inout			pcie_smbdat,		//2.5V    //SMBus Data (TR=0)
	output		pcie_waken,		//2.5V    //PCIe Wake-Up (TR=0) 
                                                 //must install 0-ohm resistor
   output		pcie_led_g3,		//1.8V    //User LED - Labeled Gen3
   output		pcie_led_g2,		//1.8V    //User LED - Labeled Gen2
   output		pcie_led_x1,		//1.8V    //User LED - Labeled x1
   output		pcie_led_x4, 		//1.8V    //User LED - Labeled x4
  	output		pcie_led_x8,		//1.8V    //User LED - Labeled x8

//Partial Reconfig
	inout			fpga_pr_done,		//1.8V
	inout			fpga_pr_error,		//1.8V
	inout			fpga_pr_ready,		//1.8V
	inout			fpga_pr_request,	//1.8V
	
//QSFP--------------------------------//25 pins//----------------------------
//	input	[3:0]	qsfp_rx_p,		//1.4V-PCML QSFP XCVR RX Data
//	output	[3:0]	qsfp_tx_p,		//1.4V-PCML QSFP XCVR TX Data
//	input		refclk_qsfp_p,		//LVDS - 644.53125MHz default, programmable
	
	input			qsfp_interruptn,	//1.8V, need level translator from 3.3V
	output		qsfp_lp_mode,		//1.8V, need level translator from 3.3V
	input			qsfp_mod_prsn,		//1.8V, need level translator from 3.3V
	output		qsfp_mod_seln,		//1.8V, need level translator from 3.3V
	output		qsfp_rstn,		//1.8V, need level translator from 3.3V
	output		qsfp_scl,		//1.8V, need level translator from 3.3V
	inout			qsfp_sda,		//1.8V, need level translator from 3.3V

//SDI --------------------------------//10 pins//----------------------------
//	input			sdi_rx_p,		//1.4V-PCML SDI XCVR RX Data
//	output		sdi_tx_p,		//1.4V-PCML SDI XCVR TX Data
//	input			refclk_sdi_p,		//LVDS Si516 148.5MHz/148.35MHz
	
	output		sdi_mf0_bypass,		//1.8V, need level translator from 3.3V
	output		sdi_mf1_auto_sleep,		//1.8V, need level translator from 3.3V
	output		sdi_mf2_mute,		//1.8V, need level translator from 3.3V
	output		sdi_tx_sd_hdn,		//1.8V, need level translator from 3.3V
	
//SFP+--------------------------------//15 pins//----------------------------
//	input			sfp_rx_p,		//1.4V-PCML SFP XCVR RX Data
//	output		sfp_tx_p,		//1.4V-PCML SFP XCVR TX Data
//	input			refclk_sfp_p,		//LVDS - 644.53125MHz default, programmable
//	input			refclk_sfp_clean_p,	//LVDS recoverd & cleaned refernece clock

	output		sfp_rx_los,		//1.8V, need level translator from 3.3V
	input			sfp_mod0_prsntn,	//1.8V, need level translator from 3.3V
	output		sfp_mod1_scl,		//1.8V, need level translator from 3.3V
	inout			sfp_mod2_sda,		//1.8V, need level translator from 3.3V
	output		sfp_rs0,		//1.8V, need level translator from 3.3V
	output		sfp_rs1,		//1.8V, need level translator from 3.3V
	output		sfp_tx_disable,		//1.8V, need level translator from 3.3V
	output		sfp_tx_fault,		//1.8V, need level translator from 3.3V

	//LMK Clock Cleaner
	output	spi_lmk_clk, 	//1.8V SPI Clock, Level Translator from 3.3V
	output	spi_lmk_csn, 	//1.8V LMK CSn, Level Translator from 3.3V
	inout		spi_lmk_sdio, 	//1.8V SPI SDIO, Level Translator from 3.3V
	inout		lmk_reset,		//1.8V LMK Reset, Level Translator from 3.3V	
	output	rclock_out_p,	//LVDS  recovered clock to LMK Clock Cleaner
	input		lmk_clean_clk_p,//LVDS Device Clock
	input		lmk_sysref_p,	//LVDS SYSREF/Device Clock
	
//SMA---------------------------------// 2 pins//----------------------------
//	output		sma_tx_p,		//1.4V-PCML SMA Transmit XCVR
//	input			refclk_sma_p,		//LVDS - 348MHz default, programmable
	
//Arria 10 VID
	///Use caution when adjusting these signals.  Setting the incorrect values
	///can cause the regulator voltage to be too high damaging the FPGA.
	///See the reference manual for VID settings, as they are different (offset)  
	///from the table in the ISL6306 datasheet.
	output [6:0] vid,		//1.8V VID signals to the controller.
	input			 vid_en, //1.8V from dipswitch to enable VID control from FPGA.
	
//USB Blaster II - System Console ---//19 pins  //--------------------------
	inout	[7:0] usb_data,		//1.8V from Max II
	inout	[1:0]	usb_addr,		//1.8V from Max II
	inout			usb_fpga_clk,	//1.8V - Need level translator from 3.3V from Cypress USB
	output		usb_empty,		//1.8V from Max II
	output		usb_full,		//1.8V from Max II
	input			usb_oen,		//1.8V from Max II
	input			usb_rdn,		//1.8V from Max II
	input			usb_resetn,		//1.8V from Max II
	inout			usb_scl,		//1.8V from Max II
	inout			usb_sda,		//1.8V from Max II
	input			usb_wrn			//1.8V from Max II

);

	assign clock_scl = 1'bZ;
	assign clock_sda = 1'bZ;
	assign lcd_spiss = 1'bZ;
	assign lcd_scl = 1'bZ;
	assign lcd_sda = 1'bZ;
	assign vid = 7'bZ;
	
endmodule