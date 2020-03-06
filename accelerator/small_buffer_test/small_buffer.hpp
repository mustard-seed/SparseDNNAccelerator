#ifndef _SMALL_BUFFER_HPP_
#define _SMALL_BUFFER_HPP_

#define SMB_TRANSFER_SIZE 2
#define SMB_CLUSTER_SIZE 2
#define SMB_BUFFER_SIZE (SMB_TRANSFER_SIZE*SMB_CLUSTER_SIZE)
#define SMB_COMPRESSION_WINDOW_SIZE 8

typedef struct __attribute__((packed)) {
#ifdef INTELFPGA_CL
		unsigned char values[SMB_BUFFER_SIZE];
#else
		cl_char values[SMB_BUFFER_SIZE];
#endif
} t_smb_tb;

#endif