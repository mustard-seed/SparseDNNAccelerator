#ifndef PROTOTYPE_STRUCTS_HPP
#define PROTOTYPE_STRUCTS_HPP
#include "params.hpp"

#ifdef INTELFPGA_CL
typedef struct __attribute__((packed)) {

  // PEs with id that satisfies 0<=idx<=maxIDX and -<=idy<=maxIDY will participate 
  unsigned char maxIDX;
  unsigned char maxIDY;


  // Number of bits assigned to the fraction width
  unsigned char fracW;
  unsigned char fracDin;
  unsigned char fracDout;
} t_pe_prototype_instruction;

typedef struct __attribute__((aligned(8))) __attribute__((packed)) {

  // PEs with id that satisfies 0<=idx<=maxIDX and -<=idy<=maxIDY will participate 
  unsigned char maxIDX;
  unsigned char maxIDY;


  // Number of bits assigned to the fraction width
  unsigned char fracW;
  unsigned char fracDin;
  unsigned char fracDout;
} t_pe_prototype_instruction_host;

typedef struct __attribute__((aligned(8))) __attribute__((packed)) {
  unsigned char numPSumToProcess;
  unsigned char numBitsToRightShift;
  bool enableRelu;
} t_output_instruction_host;
#else
typedef struct __attribute__((aligned(8))) __attribute__((packed)) {

  // PEs with id that satisfies 0<=idx<=maxIDX and -<=idy<=maxIDY will participate 
  cl_uchar maxIDX;
  cl_uchar maxIDY;


  // Number of bits assigned to the fraction width
  cl_uchar fracW;
  cl_uchar fracDin;
  cl_uchar fracDout;
} t_pe_prototype_instruction_host;

typedef struct __attribute__((aligned(8))) __attribute__((packed)) {
  cl_uchar numPSumToProcess;
  cl_uchar numBitsToRightShift;
  cl_bool enableRelu;
} t_output_instruction_host;
#endif

#endif
