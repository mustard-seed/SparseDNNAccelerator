# MCBBS: A Sparse CNN Inference Accelerator
## About the project
Weight sparsity in convoluional neural networks (CNNs) can potentially benefit inference in two ways: 1) reducing the amount of necessary storage and memory access inference, and 2) reducing the number of required multiply-accumulate (MAC) operations. In this project, we exploit weight sparsity on a 2D systolic-array accelerator architecture, implement the accelerator on an Intel Arria 10 FPGA, and achieve 2.86x and 1.75x speed-up over a dense baseline on the MAC operations of VGG-16 and ResNet-50 v1.5. 

This repository contains the source code of the CNN inference accelerator. Below are some technical highlights:

1. A fine-grained structured sparsity pattern, Micro-range Clustered Bank-Balanced (MCBBS) Sparsity, is proposed to simplify the required hardware support.

2. Most of the accelerator is implemented using Intel FPGA SDK for OpenCL. This choice is inspired by *PipeCNN* [1], another open-sourced CNN inference accelerator that targets FPGAs.

## File Organization
```
accelerator/
├── accelerator_wrapper 	# Host-side encapsulation of the accelerator
├── cnpy 					        # 3rd party library for loading NumPy data into C++
├── common 					      # Host-side utilities. Include: value quantization, sparse tensor compression, and accelerator instruction generation.
├── device 					      # Description of the accelerator in OpenCL (mostly) and Verilog.
├── full_system 			    # Tests that cover the correctness of the accelerator, instruciton generation, and the accelerator
├── graph_factory 			  # Encapsulation of CNN model
├── imagenet_demo 			  # Demo that runs inference on ImageNet
├── latency_model_validation # Tests that cover the correctness of the latency model, which is part of the instruction generation process.
├── model_container 		  # Data structures useful for loadiing CNN layer descriptions from YAML files.
├── spw_pe_test 			    # Tests that focus on the PE correctness.
├── spw_tensor_test 		  # Tests that validate the correcctness of the tensor compression and decompression.
└── yaml-cpp 				      # 3rd party library for loading YAML file into C++.
```

## How to Run the Tests
### Prerequisites
### ImageNet Demo
### Graph Factory Test
### Validation Tests

## Design of the Sparse CNN Inference Accelerator
### Overview of the Accelerator

![System architecture](docs/system_architecture.png)

The accelerator is made of OpenCL kernels connected by *channels*, which are FIFO-like features supported by Intel FPGA SDK for OpenCL. The modules are grouped into blocks according to their functionalities, and we briefly describe the blocks below:

- **Convolution Engine (CONV Engine)** processes convolutional (CONV) and fully-connected layers (FC). 
- **Miscellaneous Engine (MISC Engine)** processes max-pooling, average-pooling, concatenation, and element-wse addition layers.
- **Input Tile Controller (ITC)** and **Output Tile Controller (OTC)** control data access of buffers inside the CONV engine.
- **MISC Tile Controller (MTC)** guides the MISC engine to perform operations in a tile-by-tile fashion.
- **Input Reader, Weight Reader, and Output Writer** move data (weights, neurons) between the accelerator and the off-chip memory (DRAM).

All OpenCL kernels are implemented in [accelerator/device/sparse_pe_system.cl](accelerator/device/sparse_pe_system.cl)

A CPU-host is responsible for converting a given CNN into instructions for the accelerator blocks, and transfer the instructions as well as weights/biases to the accelerator before making any inference call. All instructions and weights are stored on the off-chip DRAM. At the start of each inference call, the host transfers the input neurons to the off-chip DRAM. For the execution of each layer, the host directs the accelerator to access instructions from the off-chip DRAM. Although the accelerator leverages data-reuse opportunity within each layer by buffering data on the on-chip memory, the intermediate results that proceed from each layer are buffered in the off-chip DRAM. At the end of each inference call, the host transfers the results from the off-chip DRAM to the host-side.

### The Convolution Engine
![Convolution Engine](docs/sys_conv_engine.png)

The convolution engine consists of a systolic-array of processing elements (PEs), weight buffers, input neuron buffers, and output neuron buffers. Weights and input neurons are propagated along the rows and columns, respectively. The topology is inspired by the work of Wei et al. [2]. Each PE performs MAC operations between weights and input neurons in a SIMD fashion. MAC operation results are shifted out of the systolic-array along the columns into the output neuron buffers, where ReLU activation function is applied.  

The buffers are implemented as ping-pong buffers to overlap data access from off-chip DRAM with transfers to and from the systolic array.

Since the size of input and output neuron tensors are large in typicial convolutional and fully-connected layers, tiling is used to breakdown the tensors. Please refer to the author's thesis (will be made available by November 2021) for details. 
### Micro-range Clustered Bank-Balanced Sparsity
There are two challenges that we need to address in order to reap gains from weight sparsity.

1. Gathering the right input neurons to the sparse weights at the right cycles. This is trivial if both weights and neurons are dense. However, in the presence of weight sparsity, crossbars are required for selecting the right input neurons to be multiplied with the weights. The fewer the constraints on the pattern of weight sparsity, the more complex the crossbars are.

2. Balancing the workload across several rows of PE. Without any constraint, some PE rows might see more sparse weights than the rest, and the overall latency of the systolic array is limited by the slowest-moving rows.

We propose a fine-grained constraint on the sparsity pattern, Micro-range Clustered Bank-Balanced Sparsity (MCBBS), to address these issues. MCBBS is inspired by Bank-Balanced Sparsity [3], which is proposed for a 1D systolic array accelerator for long short-term memories (LSTMs).

Below is a toy example on how MCBBS is enforced on 4 1x1 filters. Note that other filter sizes are actually more common, but we choose 1x1 for simplicity.

1. The filters before pruning:

![1. The filters before pruning](docs/mcbbs_dense.png) 

2. Grouping adjacent weights into clusters, and grouping consecutive clusters into pruning ranges:

![2. Forming clusters and pruning ranges](docs/mcbbs_pruning_ranges.png)

3. Rank the clusters in each pruning range according to their L1 norms, and prune the least significant clusters. All the prunings ranges of filtes from the same layer are pruned with the same sparsity level.

![3. Prune each pruning range](docs/mcbbs_prune.png)

4. After pruning, all filters retain the same number of sparse weights. Morever, the matching input neuron of each sparse weight is guaranteed to reside in a relative small window that has the same size as a pruning range.

![4. After MCBBS pruning](docs/mcbbs_result.png)

Generally, MCBBS is parametrized by the following values:

- C: The size of each cluster in terms of the number of weights.
- R: The size of each pruning range in terms of the number of clusters.


The purpose of clustering is to allow consecutive sparse weights to share indices, which are used by the PEs to select the right input neurons for multiplication with the weights. On the other hand, pruning ranges narrows the selection window of input neurons for each sparse weight.

### Supporting MCBBS in PEs
To support MCBBS in each systolic array PE, we introduce neuron registers to buffer input neurons and multiplexer to select neurons that should be multiplied with incoming sparse weights. The SIMD MAC operations in each PE is spread over several consecutive pruning ranges, which form a *processing window*. The number of multipliers inside a PE is the product between the cluster size and the processing window size: C \* P. Sparse weights arrive at each PE as weight fetch blocks. Each weight fetch blocks is made of interleaving weight clusters from pruning ranges inside the same processing window. We also assign an index to each weight cluster. The value of the index is the position of the cluster inside its pruning range. The PE uses weight indices as selection signals for the neuron multiplexer. The neuron registers buffer all the input neurons that span over one processing window, an they are updated as weights from a new processing window arrives. The following figure shows a PE performing MAC operations between a sparse filter and dense neuron from the convolution window:

![pe_processing.png](docs/pe_processing.png)

### Optimization: Coalescing Adjacent PEs in the same Systolic Array Column
Since PEs in the same systolic array column see the same input neurons, and all take the same number of cycles to processing filters that pass through them, we can coalescing adjacent PEs in the same systolic array column. This enables the PEs to share neuron registers, neuron FIFO connections, and control logic across the coalesced PEs, leading to FPGA resource saving. The number of PEs that are packed to the same group is denoted as *G*. 

![pe_groups.png](docs/pe_groups.png)

## Results
|    Configurations    |  ALM       |  DSP      |  M20K     |  fMax (MHz)  | VGG-16 Latency (ms)        | ResNet-50 v1.5 Latency (ms)    |
|----------------------|----------- |---------  |-----------|--------------|----------------------------|--------------------------------|
|    24x7M1_C1R4P16G8  | 336K (79%) | 1352 (89%)| 1785 (66%)| 242          | 18.70 (FC+CONV only: 13.69)| 16.53 (FC+CONV only: 6.65)     |
|    24x7M1_C2R4P8G8   | 322K (75%) | 1352 (89%)| 1716 (63%)| 226          | 19.25 (FC+CONV only: 14.23)| 17.13 (FC+CONV only: 7.10)     |

Details of the configuration labels:

- "24x7": Both configurations contain systolic-array of size 24 rows by 7 columns.
- "M1": Both configurations contain one MISC engine. In fact, this is fixed.
- "C1", "C2": The cluster sizes are 1 and 2, respectively.
- "R4": Each pruning range contains 4 clusters.
- "P16", "P8": The processing windows in the accelerators contain 16 and 8 pruning ranges, respectively.  
- "G8": Both configurations coalesce PEs in the same systolic-array column into groups of 8 (G=8).

Details of the CNN models:

- Weight sparsity levels: If a layer has the same number of input feature maps as the number of output feature maps, then prune at 75% weight sparsity. Otherwise, prune at 50% weight sparsity. The first convolution layers are not pruned.
- Weight and neuron precisions: INT8
- VGG-16: Retains 30% of the MAC operations after pruning. 
- ResNet-50 v1.5: Retains 40% of the MAC operations after pruning. The average-pooling layer is modified to divide the sum of each feature map by 64 instead of the size of the feature map. 

## Citation
If your research benefits from this work, please kindly cite it as below:

Lin Qiao Liu, Stephen Dean Brown: **Leveraging Fine-grained Structured Sparsity for CNN Inference on Systolic Array Architectures**. FPL 2021

## Acknowledgments
This project makes use of the following third-party open-sourced work:

- [Google Test](https://github.com/google/googletest)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [cnpy](https://github.com/rogersce/cnpy)

## References
[1] D. Wang, K. Xu, and D. Jiang, “Pipecnn: An opencl-based open-source FPGA accelerator for convolution neural networks,” in International Conference on Field Programmable Technology, FPT 2017, Melbourne, Australia, December 11-13, 2017. Available: https://doi.org/10.1109/FPT.2017.8280160 IEEE, 2017, pp. 279–282. [Online]. [Source Link](https://github.com/doonny/PipeCNN)

[2] X. Wei, C. H. Yu et al., “Automated systolic array architecture synthesis for high throughput CNN inference on fpgas,” in Proceedings of the 54th Annual Design Automation Conference, DAC 2017, Austin, TX, USA, June 18-22, 2017. ACM, 2017, pp. 29:1–29:6. [Online]. Available: https://doi.org/10.1145/3061639.3062207

[3] S. Cao, C. Zhang et al., “Efficient and effective sparse LSTM on FPGA with bank-balanced sparsity,” in Proceedings of the 2019 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, FPGA 2019, Seaside, CA, USA, February 24-26, 2019. [Online]. Available:
https://doi.org/10.1145/3289602.3293898



