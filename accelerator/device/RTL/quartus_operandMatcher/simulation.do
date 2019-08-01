# set the working dir, where all compiled verilog goes
vlib work

# compile all verilog modules in mux.v to working dir
# could also have multiple verilog files
vlog tb_operandMatcher.v operandMatcher_cl.v

#load simulation using mux as the top level simulation module
vsim operandMatcher_tb

#log all signals and add some signals to waveform window
log {/*}
# add wave {/*} would add all items in top level simulation module
add wave {/*}

# Run the testbench
run -all
