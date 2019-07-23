`timescale 1 ps / 1 ps
//`default netttype none
module leadingZeroCounter (
		input   wire clock,
		input   wire resetn,
		input   wire ivalid, 
		input   wire iready,
		output  wire ovalid, 
		output  wire oready,
		
		input  wire [7:0]  bitmask, // dataa_0.dataa_0
		
		output reg [7:0] result  //  7:4 are the number of leading zero plus 1, 3:0 encode the nubmer of leading zeros
	);

	assign ovalid = 1'b1;
	assign oready = 1'b1;
	// ivalid, iready, resetn are ignored

    //Use a priority encoder to derive value from bitmask
	always @ (*) begin
        //default value
        result[7:0] = {4'h9, 4'h8};
        if (bitmask[0] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd1;
            //leading zero count
            result[3:0] = 4'd0;
        end
        else if (bitmask[1] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd2;
            //leading zero count
            result[3:0] = 4'd1;
        end
        else if (bitmask[2] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd3;
            //leading zero count
            result[3:0] = 4'd2;
        end
        else if (bitmask[3] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd4;
            //leading zero count
            result[3:0] = 4'd3;
        end
        else if (bitmask[4] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd5;
            //leading zero count
            result[3:0] = 4'd4;
        end
        else if (bitmask[5] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd6;
            //leading zero count
            result[3:0] = 4'd5;
        end
        else if (bitmask[6] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd7;
            //leading zero count
            result[3:0] = 4'd6;
        end
        else if (bitmask[7] == 1'b1) begin
            //leading zero count plus 1
            result[7:4] = 4'd8;
            //leading zero count
            result[3:0] = 4'd7;
        end
    end
endmodule
