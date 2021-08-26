# Bitstreams and Pre-trained Sparse CNN Models
## Bitstreams
Please download the bitstreams from [this Google Drive folder](https://drive.google.com/drive/folders/1q8oZ6War5Nk5p4HQUqdUmNJliHKt2Inr?usp=sharing)

| Bitstream ID      | ROW   | COL   | Sparse | Pruning Range Size (R)| Cluster Size (C)| Processing Window Size (P) | PE Coalescing Size (G)|
|-------------------|-------|-------|--------|-----------------------|-----------------|----------------------------|-----------------------|
|24x7M1_c1r4p16g8_28b_ab64_wb64 | 24 	| 7 | Y | 4 | 1 | 16 | 8 |
|24x7M1_c2r4p8g8_28b_ab64_wb64  | 24	| 7	| Y | 4 | 2 | 8  | 8 |
|24x7M1_c2p8g8_16b_ab64_wb64    | 24    | 7 | N | --| 2 | 8  | 8 |

All of these accelerator instances target Intel Arria 10 FPGA development kit, and are generated using AOCL 19.3.0.222. Moreoever, all of these instances move neurons and weights between the off-chip DRAM and the systolic-array in blocks of size 64 weights/neurons.


## Pre-trained CNN Models
[Download link](https://drive.google.com/drive/folders/1Wohoq8upaZUw-7jKgh04gP6tRKq8Xzol?usp=sharing)

Each pre-trained model folder contains two files that are parsed by the accelerator:

1. \*_trace.yaml: The topology description.
2. \*_parameters.npz: The weights and biases.

In addition, the reference output file, \*\_inout\_\*.npz, contains one set of golden input and output values that are derived by passing an image through the model in PyTorch. It is used by some tests to validate the correctness of the inference.


