set_global_assignment -entity "c5_mac_8bitx4_input_registered" -library "lib_c5_mac_8bitx4_input_registered" -name IP_TOOL_NAME "altera_mult_add"
set_global_assignment -entity "c5_mac_8bitx4_input_registered" -library "lib_c5_mac_8bitx4_input_registered" -name IP_TOOL_VERSION "18.0"
set_global_assignment -entity "c5_mac_8bitx4_input_registered" -library "lib_c5_mac_8bitx4_input_registered" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_c5_mac_8bitx4_input_registered" -name SPD_FILE [file join $::quartus(sip_path) "c5_mac_8bitx4_input_registered.spd"]

set_global_assignment -library "lib_c5_mac_8bitx4_input_registered" -name MISC_FILE [file join $::quartus(sip_path) "c5_mac_8bitx4_input_registered_sim/c5_mac_8bitx4_input_registered.v"]
