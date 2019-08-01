`timescale 1 ps / 1 ps
module operandMatcher_tb;
	reg clock, resetn;
	reg [7:0] bitmaskW, bitmaskA;
	reg [63:0] goldenResult;
	wire [63:0] result;

	integer inputFile;

	//File I/O
	initial begin
		inputFile = $fopen("testVector.txt", "r");
	end

	//Output waveform
	initial begin
		$dumpfile("output.vcd");
		$dumpvars;
	end

	initial begin
		clock = 0;
		resetn = 0;
		#5 resetn = 1;
	end

	//Clock logic
	always begin
		#5 clock = !clock;
	end

	//Connect the device under test
	operandMatcher8 dut (
		.clock(clock),
		.resetn(resetn),
		.ivalid(),
		.iready(),
		.ovalid  (),
		.oready (),
		.bitmaskW(bitmaskW),
		.bitmaskA(bitmaskA),
		.result(result)
		);

	//The actual test driving block
	integer testCaseCount;
	initial begin
		testCaseCount = 0;
		//Wait for the reset cycle to pass
		#10;
		while (! $feof(inputFile) ) begin
			//Read the test vector
			$fscanf (inputFile, "%b %b %h\n",
				bitmaskA, bitmaskW, goldenResult);
			//Wait for 2 clock cycles to pass
			#20;
			if (result != goldenResult) begin
				$display("DUT Error at time %d, test case %d\n",
				 $time, testCaseCount);
				$display("Expected valud %h, actual value %h\n", 
					goldenResult, result);
			end
			testCaseCount = testCaseCount+1;
		end
		$fclose(inputFile);
		$stop;

	end






endmodule