# stop any simulation that is currently running
quit -sim

# create the default "work" library
vlib work;

vlog ../small_buffer_lib.v
vlog small_buffer_lib_tb.v

# start the Simulator
vsim small_buffer_lib_tb -Lf 220model -Lf altera_mf_ver -Lf verilog
# show waveforms specified in wave.do
do wave.do
# advance the simulation the desired amount of time
run 80 ns
