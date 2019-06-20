#ifndef PROTOTYPE_STRUCTS_HPP
#define PROTOTYPE_STRUCTS_HPP
#include "params.hpp"

#ifdef INTELFPGA_CL
typedef struct __attribute__((aligned(64))) __attribute__((packed)) {
  //The index of the start of the streaming block. 
  //If the first element in the block is effectual, this is its index
  unsigned short transmissionStartIndex;

  //The index of the last element in the uncompressed streaming block
  //The compression algorithms guarantees that this element is preserved, regardless of its value
  unsigned short transmissionEndIndex;

  //The index of the first element of interest 
  unsigned short selectStartIndex;

  // PEs with id that satisfies 0<=idx<=maxIDX and -<=idy<=maxIDY will participate 
  unsigned char maxIDX;
  unsigned char maxIDY;

  // Mode of the instruction
  unsigned char mode;

  // Number of bits assigned to the fraction width
  char fracW;
  char fracDin;
  char fracDout;
} t_pe_prototype_instruction;
#else
typedef struct __attribute__((aligned(64))) __attribute__((packed)) {
  cl_ushort transmissionStartIndex;
  cl_ushort transmissionEndIndex;
  cl_ushort selectStartIndex;
  cl_uchar mode;
  cl_char fracW;
  cl_char fracDin;
  cl_char fracDout;
} t_pe_prototype_instruction;
#endif

#endif
