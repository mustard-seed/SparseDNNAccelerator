#ifndef _SMALL_BUFFER_HPP_
#define _SMALL_BUFFER_HPP_

typedef struct __attribute__((packed)) {
#ifdef INTELFPGA_CL
		cl_char values[4];
#else
		unsigned char values[4];
#endif
} t_smb_tb;

#endif