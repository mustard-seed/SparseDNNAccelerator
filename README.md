# MCBBS: A Sparse CNN Inference Accelerator
## About the project
Weight sparsity in convoluional neural networks (CNNs) can potentially benefit inference in two ways: 1) reducing the amount of necessary storage and memory access inference, and 2) reducing the number of required multiply-accumulate (MAC) operations. In this project, we exploit weight sparsity on a 2D systolic-array accelerator architecture, implement the accelerator on an Intel Arria 10 FPGA, and achieve 2.86x and 1.75x speed-up over a dense baseline on the MAC operations of VGG-16 and ResNet-50 v1.5. 

This repository contains the source code of the CNN inference accelerator. Below are some technical highlights:

1. A fine-grained structured sparsity pattern, Micro-range Clustered Bank-Balanced (MCBBS) Sparsity, is proposed to simplify the required hardware support.

2. Most of the accelerator is implemented using Intel FPGA SDK for OpenCL. This choice is inspired by *PipeCNN* [1], another open-sourced CNN inference accelerator that targets FPGAs.

## File Organization
```
accelerator/
├── accelerator_wrapper 		  # Host-side encapsulation of the accelerator
├── cnpy 					      # 3rd party library for loading NumPy data into C++
├── common 					      # Host-side utilities. Include: value quantization, sparse tensor compression, and accelerator instruction generation.
├── device 					      # Description of the accelerator in OpenCL (mostly) and Verilog.
├── full_system 			      # Tests that cover the correctness of the accelerator, instruciton generation, and the accelerator
├── graph_factory 			  	  # Encapsulation of CNN model
├── imagenet_demo 			  	  # Demo that runs inference on ImageNet
├── latency_model_validation 	  # Tests that cover the correctness of the latency model, which is part of the instruction generation process.
├── model_container 		  	  # Data structures useful for loadiing CNN layer descriptions from YAML files.
├── spw_pe_test 			      # Tests that focus on the PE correctness.
├── spw_tensor_test 		      # Tests that validate the correcctness of the tensor compression and decompression.
└── yaml-cpp 				      # 3rd party library for loading YAML file into C++.
```

## How to build and run the tests?

### Prerequisites

	- Boost 1.56
	- zlib
  
   In addition, OpenCV 4.0.0 or above is required if you need to run the ImageNet demo. Moreover, if you intend to cross-compile the project's software (e.g. for the ARM SoC on DE10-Standard), these libraries should be cross-compiled too.

   To compile FPGA bitstreams, please make sure you have setted-up Intel FPGA SDK for OpenCL and BSP for the target boards.
 
### Clone this repository and the submodules
```
git clone git@github.com:mustard-seed/SparseDNNAccelerator.git
cd SparseDNNAccelerator
git submodule update --init --recursive
```

### Initialize a build directory and invoke CMake

In the examples below, let us assume the initial path is SparseDNNAccelerator.

Example 1: Host software will run on an x86 platform. Target board is Intel Arria 10 FPGA Dev. Kit.

```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../accelerator/a10ref.cmake \
	-DBOARD_NAME=A10REF \
	-DOpenCV_DIR=<path to OpenCV> \
	../accelerator
```

The flag **CMAKE_TOOLCHAIN_FILE** is mandatory. It must point to a CMake configuration file that specifies the C++ compiler. See [accelerator/a10ref.cmake](../accelerator/a10ref.cmake), [accelerator/a10pac.cmake](../accelerator/a10pac.cmake), and [accelerator/de10standard.cmake](../accelerator/de10standard.cmake)

The flag **BOARD_NAME** is mandatory. Currently, the supported values are DE10Standard, A10PAC, and A10REF. These correspond to Terasic DE10-Standard, Intel FPGA Arria 10 PAC, and Intel FPGA Arria 10 Dev. Kit, respectively. Under the hood, these flags are used by some targets' CMakeLists to link the correct Intel FPGA libraries to the host-side software. For instance, see [accelerator/full_system/CMakeLists.txt](accelerator/full_system/CMakeLists.txt).

Example 2: Host software will run on a DE10-Standard, which has ARM processors. Cross-compilation is required.
Make sure to modify the CMAKE_FIND_ROOT_PATH variable in accelerator/de10standard.cmake first.

```
mkdir build
cd build
cmake -DOpenCV_DIR=<path to OpenCV> \
	-DBOOST_ROOT=<path to Boost> \
	-DBOOST_INCLUDEDIR=<path to Boost headers> \
	-DBOOST_LIBRARYDIR=<path to Boost libraries> \
	-DBoost_NO_SYSTEM_PATHS=ON -DZLIB_ROOT=<path to zlib>  \
	-DZLIB_LIBRARIES=<path to the zlib static library (.a)> \
	-DZLIB_INCLUDE_DIRS=<path to zlib headers> \
	-DCMAKE_TOOLCHAIN_FILE=../accelerator/de10standard.cmake \
	-DBOARD_NAME=DE10Standard \
	./accelerator
```

### ImageNet Demo
We assume that test is run on an x86 host that is equipped with an Intel Arria 10 FPGA Dev. Kit. First, change path to the build directory and build the ImageNet Demo.

```
make imagenet_demo -j8
```

This will create a binary, imagenet_demo, under 
```
build/imagenet_demo
```

Then, download the accelerator bitstreams and CNN models according to the [instructions](docs/bitstreams_and_models.md).

Next, copy the bitstream that you are interested in testing into the work directory, where the image\_demo binary is, and rename the bitstream file to "sparse_pe_system.aocx". Alternatively, one can modify the variable "std::string aocxBinaryFile" inside image_demo.cpp, but doing so requires re-compiling the code.

Also copy [accelerator/imagenet_demo/caffe_words.yaml](accelerator/imagenet_demo/caffe_words.yaml), [accelerator/imagenet_demo/demo_ground_truth.yaml](accelerator/imagenet_demo/demo_ground_truth.yaml), and [accelerator/imagenet_demo/preprocess.yaml](accelerator/imagenet_demo/preprocess.yaml) into the work directory.

Finally, to test the accelerator on one image (not necessarily from ImageNet dataset):
```
./imagenet_demo --model=<path to the model topology file, *_trace.yaml> \
  --param=<path to the model parameter file, *_parameters.npz> \
  --image=<path to the image>
```

Alternatively, to validate the inference accelerator on the entire ImageNet validation set:
```
./imagenet_demo --model=<path to the model topology file, *_trace.yaml> \
  --param=<path to the model parameter file, *_parameters.npz> \
  --val=true \
  --folder=<path to the ImageNet validation folder> \
  --numSamples=<number of samples to validate on. If not specified, then the entire validation set is used>
```


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

## How to modify the accelerator?
First, play around with the accelerator configurations in [accelerator/device/params.hpp](accelerator/device/params.hpp). Please see the [detailed explanation](docs/accelerator_configuration.md) of the user-configurable flags.

For a more detailed understanding of the design, please consult our FPL2021 paper and the M.A.Sc. thesis on this project (will be released by November 2021). We have also included a [short description](docs/accelerator_design.md) in this repository.

## Known Issues
To be added.

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



