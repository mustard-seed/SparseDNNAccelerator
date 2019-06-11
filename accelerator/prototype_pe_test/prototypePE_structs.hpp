#ifndef PROTOTYPE_STRUCTS_HPP
#endif PROTOTYPE_STRUCTS_HPP

#include "params.hpp"

#ifdef INTELFPGA_CL
typedef struct __attribute__((aligned(64))) __attribute__((packed)) {
  ushort lastIndex;
  unsigned char maxIDX;
  unsigned char maxIDY;
  unsigned char mode;
  char fracW;
  char fracDin;
  char fracDout;
} t_pe_prototype_instruction;
#else
typedef struct __attribute__((aligned(64))) __attribute__((packed)) {
  cl_ushort lastIndex;
  cl_uchar maxIDX;
  cl_uchar maxIDY;
  cl_uchar mode;
  cl_char fracW;
  cl_char fracDin;
  cl_char fracDout;
} t_pe_prototype_instruction;
#endif

#endif
