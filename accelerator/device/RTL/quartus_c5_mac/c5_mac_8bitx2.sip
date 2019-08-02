set_global_assignment -entity "c5_mac_8bitx2" -library "lib_c5_mac_8bitx2" -name IP_TOOL_NAME "altera_mult_add"
set_global_assignment -entity "c5_mac_8bitx2" -library "lib_c5_mac_8bitx2" -name IP_TOOL_VERSION "18.0"
set_global_assignment -entity "c5_mac_8bitx2" -library "lib_c5_mac_8bitx2" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_c5_mac_8bitx2" -name SPD_FILE [file join $::quartus(sip_path) "c5_mac_8bitx2.spd"]

set_global_assignment -library "lib_c5_mac_8bitx2" -name MISC_FILE [file join $::quartus(sip_path) "c5_mac_8bitx2_sim/c5_mac_8bitx2.v"]
