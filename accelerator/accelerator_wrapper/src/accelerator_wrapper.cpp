#include "accelerator_wrapper.hpp"
#include "params.hpp"
#include "floatFixedPointConversion.hpp"

/**
CL INTEL FPGA MEM BANKS
Only available on certain Arria 10 and Stratix 10 boards
 */
#if defined(A10PAC)
    #define MEM_BANK_ACTIVATION CL_CHANNEL_1_INTELFPGA
    #define MEM_BANK_WEIGHT     CL_CHANNEL_2_INTELFPGA
    #define MEM_BANK_BIAS       CL_CHANNEL_2_INTELFPGA
    #define MEM_BANK_INSTRUCTIONS   CL_CHANNEL_2_INTELFPGA
#else
    #define MEM_BANK_ACTIVATION 0x0
    #define MEM_BANK_WEIGHT     0x0
    #define MEM_BANK_BIAS       0x0
    #define MEM_BANK_INSTRUCTIONS   0x0
#endif

#if defined(C5SOC) //Hack for ARMv7, otherwise chrono won't work
__asm__(".symver _ZNSt6chrono3_V212system_clock3nowEv,_ZNSt6chrono12system_clock3nowEv@GLIBCXX_3.4.11");
#endif

namespace GraphRuntime {
    std::vector<float> convert2Float(std::vector<fixedPointNumber> fpVector)
    {
        //TODO: Maybe parallelize this?
        std::vector<float> result;
        result.resize(fpVector.size());
        int index = 0;
        for (auto& fpVal: fpVector)
        {
            result.at(index++) = fpVal.convert2Float();
        }
        return result;
    }

    std::vector<fixedPointNumber> quantize(std::vector<float> floatVector, char _fracWidth)
    {
        //TODO: Maybe parallelize this?
        std::vector<fixedPointNumber> result;
        result.resize(floatVector.size());
        int index = 0;
        char intWidth = 7 - _fracWidth;
        for (auto& fpVal: floatVector)
        {
            result.at(index++) = fixedPointNumber(fpVal, _fracWidth, intWidth);
        }
        return result;
    }

    AcceleratorWrapper::AcceleratorWrapper(std::string _bitstreamFileName,
                                           t_execution_graph& _executionGraph,
                                           t_accelerator_info& _acceleratorInfo,
                                           int _fpgaID)
    {
        cl_int status = CL_SUCCESS;

        binaryFile = _bitstreamFileName;
        #if defined(EMULATE)
            clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
        #else
            clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
        #endif
        //Setup and platform and the context
        std::vector<cl::Device> devices;
        status = clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        aocl_utils_cpp::checkError(status, "Failed to query the devices");
        clDevice = devices[_fpgaID];

        #if defined(C5SOC)
            clContext = cl::Context({clDevice}
                                    ,NULL
                                    ,&aocl_utils_cpp::oclContextCallback
                                    ,NULL
                                    ,&status);
        #else
            clContext = cl::Context(clDevice
                                    ,NULL
                                    ,&aocl_utils_cpp::oclContextCallback
                                    ,NULL
                                    ,&status);
            aocl_utils_cpp::checkError(status, "Failed to create context");
        #endif

        //Parse the binary and invoke the kernel
        program = aocl_utils_cpp::createProgramFromBinary(
                    clContext,
                    binaryFile.c_str(),
                    {clDevice}
                    );
        status = program.build({clDevice});
        aocl_utils_cpp::checkError(status, "Failed to build program");

        //Instantiate the host-side kernel objects
        kernelIAMover = cl::Kernel(program, "kernelIAMover", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelIAMover!");

        kernelWMover = cl::Kernel(program, "kernelWMover", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelWMover!");

        kernelOAMover = cl::Kernel(program, "kernelOAMover", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelOutputWriter!");

        kernelIATileController = cl::Kernel(program, "kernelIATileController", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelIATileController!");

        KernelOATileController = cl::Kernel(program, "kernelOATileController", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelOATileController!");

        kernelMKInstructionMover = cl::Kernel(program, "kernelMiscControlMover", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelMiscControlMover!");

    #if defined(NOOP)
        kernelNoop = cl::Kernel(program, "kernelNoop", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelNoop!");
    #endif
        //Instantiate the command queues
        clCQIAMover = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQIAMover!");

        clCQOAMover = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQOAMover!");

        clCQWMover = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQWMover!");

        clCQIATileController = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQIATileController!");

        clCQOATileController = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQOATileController!");

        clCQMKController = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clMKController!");

    #if defined(NOOP)
        clCQNoop = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQNoop!");
    #endif
        //Instantiate the buffers
        cl_ulong maxBufferSizeByte = clDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE> (&status);
        aocl_utils_cpp::checkError(status, "Failed to query the maximum buffer size in bytes!");

        typedef struct {
            cl_ulong bufferSizeByte;
            cl::Buffer& bufferObject;
            cl_mem_flags memFlag;
            std::string bufferName;
        } t_buffer_setup_info;

        std::vector<t_buffer_setup_info> vecBufferInfo;

        cl_ulong weightMoverInstructionBufferSize = maxBufferSizeByte < MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION;
        vecBufferInfo.push_back({weightMoverInstructionBufferSize, bufferWMoverInstructions, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferWMoverInstructions"});

        cl_ulong inputWeightBufferSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT;
        vecBufferInfo.push_back({inputWeightBufferSize, bufferWMoverWDramBlocks, CL_MEM_READ_ONLY | MEM_BANK_WEIGHT, "bufferWMoverWDramBlocks"});

        cl_ulong inputBiasSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_BIAS ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_BIAS;
        vecBufferInfo.push_back({inputBiasSize, bufferWMoverBias, CL_MEM_READ_ONLY | MEM_BANK_BIAS, "bufferWMoverBias"});

        cl_ulong activationSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION;
        vecBufferInfo.push_back({activationSize, bufferActivationDramBlocks, CL_MEM_READ_WRITE | MEM_BANK_ACTIVATION, "bufferActivationDramBlocks"});

        cl_ulong inputIAMoverInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION;
        vecBufferInfo.push_back({inputIAMoverInstructionSize, bufferIAMoverInstructions, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferIAMoverInstructions"});

        cl_ulong inputIATileControllerInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION;
        vecBufferInfo.push_back({inputIATileControllerInstructionSize, bufferIATileControllerInstructions, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferIATileControllerInstructions"});

        cl_ulong outputOAMoverInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION;
        vecBufferInfo.push_back({outputOAMoverInstructionSize, bufferOAMoverInstructions, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferOAMoverInstructions"});

        cl_ulong outoutOATileControllerInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION;
        vecBufferInfo.push_back({outoutOATileControllerInstructionSize, bufferOATileControllerInstructions, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferOATileControllerInstructions"});

        cl_ulong mkInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION;
        vecBufferInfo.push_back({mkInstructionSize, bufferMKInstructions, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferMKInstructions"});

    #if defined(SPARSE_SYSTEM)
        //If the device is PAC, place the TB count on the same memory bank as the instructions
        cl_ulong inputWeightSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT;
        vecBufferInfo.push_back({inputWeightSBSize, bufferWMoverWTBCounts, CL_MEM_READ_ONLY | MEM_BANK_INSTRUCTIONS, "bufferWMoverWTBCounts"});

        cl_ulong activationTBCountSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT;
        vecBufferInfo.push_back({activationTBCountSize, bufferActivationTBCounts, CL_MEM_READ_WRITE | MEM_BANK_INSTRUCTIONS, "bufferActivationTBCounts"});
    #endif

        for (auto& info : vecBufferInfo)
        {
            cl_int localStatus = CL_SUCCESS;
            std::cout <<"Setting the buffer "<<info.bufferName<<". Size (bytes): "<<info.bufferSizeByte<<std::endl;
            info.bufferObject = cl::Buffer (
                        clContext,
                        info.memFlag,
                        info.bufferSizeByte,
                        NULL,
                        &localStatus
                    );
            aocl_utils_cpp::checkError(status, "Failed to setup the buffer!");
        }
        std::cout <<"AOCL setup compelete"<<std::endl;

        acceleratorInfo = _acceleratorInfo;

        //Load the network, allocate buffers, and transfer the instructions
        loadGraph(_executionGraph);

        std::cout <<"Accelerator setup complete."<<std::endl;
    } // AcceleratorWrapper parametrized constructor

    void AcceleratorWrapper::loadGraph (t_execution_graph& _executionGraph)
    {
        std::cout <<"Allocate memory for the input and output blobs."<<std::endl;
        for (auto& inputInfo: _executionGraph.vecInputInfo)
        {
            bool flagCanBeSparse = inputInfo.flagCanBeSparse;
            if (flagCanBeSparse == true)
            {
                vecInputBlobsInternal.emplace_back(
                            std::shared_ptr<AlignedTensor>(new FlexibleDirectCompressedTensor(
                                                1, //_num3DTensors
                                                inputInfo.group * inputInfo.channelPerGroup, //_channel
                                                inputInfo.width, //_width
                                                inputInfo.height, //_height
                                                inputInfo.channelPerGroup-1, //_maxScalarIndexInChannelGroup
                                                COMPRESSION_WINDOW_SIZE-1, //_maxClusterIndexInCompressionBlock
                                                TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                                CLUSTER_SIZE-1, //_maxScalarIndexInCluster
                                                false //isKernel
                                                )
                                            )
                            );
            }
            else
            {
                vecInputBlobsInternal.emplace_back(
                            std::shared_ptr<AlignedTensor>(new AlignedTensor(
                                                1, //_num3DTensors
                                                inputInfo.group * inputInfo.channelPerGroup, //_channel
                                                inputInfo.width, //_width
                                                inputInfo.height, //_height
                                                inputInfo.channelPerGroup-1, //_maxScalarIndexInChannelGroup
                                                TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                                CLUSTER_SIZE-1, //_maxScalarIndexInCluster
                                                false //isKernel
                                                )
                                            )
                            );
            }
        }
        vecInputBlobsInfo = _executionGraph.vecInputInfo;

        for (auto& outputInfo: _executionGraph.vecOutputInfo)
        {
            bool flagCanBeSparse = outputInfo.flagCanBeSparse;
            if (flagCanBeSparse == true)
            {
                vecOutputBlobsInternal.emplace_back(
                            std::shared_ptr<AlignedTensor>(new FlexibleDirectCompressedTensor(
                                                1, //_num3DTensors
                                                outputInfo.group * outputInfo.channelPerGroup, //_channel
                                                outputInfo.width, //_width
                                                outputInfo.height, //_height
                                                outputInfo.channelPerGroup-1, //_maxScalarIndexInChannelGroup
                                                COMPRESSION_WINDOW_SIZE-1, //_maxClusterIndexInCompressionBlock
                                                TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                                CLUSTER_SIZE-1, //_maxScalarIndexInCluster
                                                false //isKernel
                                                )
                                            )
                            );
            }
            else
            {
                vecOutputBlobsInternal.emplace_back(
                            std::shared_ptr<AlignedTensor>(new AlignedTensor(
                                                1, //_num3DTensors
                                                outputInfo.group * outputInfo.channelPerGroup, //_channel
                                                outputInfo.width, //_width
                                                outputInfo.height, //_height
                                                outputInfo.channelPerGroup-1, //_maxScalarIndexInChannelGroup
                                                TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                                                CLUSTER_SIZE-1, //_maxScalarIndexInCluster
                                                false //isKernel
                                                )
                                            )
                            );
            }
        }
        vecOutputBlobsInfo = _executionGraph.vecOutputInfo;

        vecActivationInfo = _executionGraph.vecLayerInfo;

        std::cout <<"Transferring the graph to the FPGA."<<std::endl;
        int stepCount = 0;
        cl_int status = CL_SUCCESS;

        std::cout <<stepCount++<<". Setting kernel arguments for the IA Mover."<<std::endl;
        {
            cl_uint argIdx = 0;
            //volatile __global t_dram_block* restrict pIA
            kernelIAMover.setArg(argIdx, bufferActivationDramBlocks);
            argIdx++;
            #if defined(SPARSE_SYSTEM)
                //volatile __global t_streamblock_address* restrict pTBCount,
                kernelIAMover.setArg(argIdx, bufferActivationTBCounts);
                argIdx++;
            #endif
            //volatile __global t_ia_mover_instruction* restrict pInstruction,
            kernelIAMover.setArg(argIdx, bufferIAMoverInstructions);
            argIdx++;
            //unsigned int numInstruction
            kernelIAMover.setArg(argIdx, (cl_uint) (_executionGraph.vecIAMoverInstruction.size()) );
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the IA Tile controller."<<std::endl;
        {
            cl_uint argIdx = 0;
            //__global volatile t_ia_tile_controller_instruction* restrict pInstruction,
            kernelIATileController.setArg(argIdx, bufferIATileControllerInstructions);
            argIdx++;

            //unsigned int numInstruction
            kernelIATileController.setArg(argIdx, (cl_uint) (_executionGraph.vecIATileControllerInstruction.size()) );
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the Filter mover."<<std::endl;
        {
            cl_uint argIdx = 0;
            //volatile __global t_weight_mover_instruction* restrict pInst,
            kernelWMover.setArg(argIdx++, bufferWMoverInstructions);
            //volatile __global t_dram_block* restrict pW,
            kernelWMover.setArg(argIdx++, bufferWMoverWDramBlocks);
            //vola<tile __global t_accumulator* restrict pBias,
            kernelWMover.setArg(argIdx++, bufferWMoverBias);
            #if defined(SPARSE_SYSTEM)
                //volatile __global t_streamblock_address* restrict pFilterTBCount,
                kernelWMover.setArg(argIdx++, bufferWMoverWTBCounts);
            #endif //SPARSE_SYSTEM
            //unsigned int numInstruction
            kernelWMover.setArg(argIdx++, (cl_uint) (_executionGraph.vecWMoverInstruction.size()) );
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the Miscellaneous controller."<<std::endl;
        {
            cl_uint argIdx=0;
            //__global t_misc_instruction* restrict pInstruction,
            kernelMKInstructionMover.setArg(argIdx++, bufferMKInstructions);
            //unsigned int numInstruction
            kernelMKInstructionMover.setArg(argIdx++, (cl_uint) (_executionGraph.vecMiscInstruction.size()) );
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the OA mover."<<std::endl;
        {
            cl_uint argIdx=0;
            //volatile __global t_output_dram_block* restrict pOA,
            kernelOAMover.setArg(argIdx++, bufferActivationDramBlocks);
            #if defined(SPARSE_SYSTEM)
                //volatile __global t_streamblock_address* restrict pTBCount,
                kernelOAMover.setArg(argIdx++, bufferActivationTBCounts);
            #endif
            //volatile __global t_oa_mover_instruction* restrict pInstruction,
            kernelOAMover.setArg(argIdx++, bufferOAMoverInstructions);
            //unsigned int numInstruction
            kernelOAMover.setArg(argIdx++, (cl_uint) (_executionGraph.vecOAMoverInstruction.size()) );
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the OA tile controller."<<std::endl;
        {
            cl_uint argIdx=0;
            //volatile  __global t_oa_tile_controller_instruction* restrict pInst,
            KernelOATileController.setArg(argIdx++, bufferOATileControllerInstructions);
            //unsigned int numInstructions
            KernelOATileController.setArg(argIdx++, (cl_uint) (_executionGraph.vecOATileControllerInstruction.size()));
        }

        std::cout <<stepCount++<<". Transfer the IA Mover instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecIAMoverInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecIAMoverInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferIAMoverInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION && "Too many IA mover instructions to fit inside the global memory" );

            status = clCQIAMover.enqueueWriteBuffer(bufferIAMoverInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 _executionGraph.vecIAMoverInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the IA mover instructions");
            clCQIAMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the IA mover instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the IA Tile Controller instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecIATileControllerInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecIATileControllerInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferIATileControllerInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION && "Too many IA Tile instructions to fit inside the global memory" );

            status = clCQIATileController.enqueueWriteBuffer(bufferIATileControllerInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 _executionGraph.vecIATileControllerInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the IA tile controller instructions");
            clCQIATileController.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the IA tile controller instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the OA Mover instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecOAMoverInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecOAMoverInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferOAMoverInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION && "Too many OA mover instructions to fit inside the global memory" );

            status = clCQOAMover.enqueueWriteBuffer(bufferOAMoverInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 _executionGraph.vecOAMoverInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the OA mover instructions");
            clCQOAMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the OA mover instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the OA Tile Controller instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecOATileControllerInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecOATileControllerInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferOAMoverInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION && "Too many OA TILE instructions to fit inside the global memory" );

            status = clCQOATileController.enqueueWriteBuffer(bufferOATileControllerInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 _executionGraph.vecOATileControllerInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the IA tile controller instructions");
            clCQOATileController.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the OA tile controller instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        //If the operation is CONVOLUTION,
        //then transfer the WMover instructions, the weights, the weight TB count, and the biases
        std::cout <<stepCount++<<". Transfer the W Mover instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecWMoverInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecWMoverInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferWMoverInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION && "Too many Weight Mover instructions to fit inside the global memory" );


            status = clCQWMover.enqueueWriteBuffer(bufferWMoverInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 _executionGraph.vecWMoverInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the W Mover instructions");
            clCQWMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the W Mover instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the filter biases"<<std::endl;
        {
            cl_double elapsedTimeUs = 0.0;
            int transferBytes = 0;
            int idxTensor = 0;
            for (std::shared_ptr<t_aligned_short_vector>& ptr : _executionGraph.pBiasVector)
            {
                cl::Event event;
                auto pBiases = ptr.get();
                auto numElements = pBiases->size();
                auto sizeElement = sizeof(typeof(pBiases->at(0)));
                transferBytes += sizeElement * numElements;

                int offsetIndex = _executionGraph.vecBiasStart.at(idxTensor++) * sizeElement;
                assert(transferBytes <= MAX_DRAM_BYTE_INPUT_BIAS && "Too many biases to fit inside the global memory" );

                status = clCQWMover.enqueueWriteBuffer(bufferWMoverBias, //buffer
                                                     CL_TRUE, //blocking_write
                                                     offsetIndex, //offset
                                                     transferBytes, //size
                                                     pBiases->data(), //data pointer
                                                     NULL, //dependency list
                                                     &event //events generated
                                                    );
                aocl_utils_cpp::checkError(status, "Failed to write the filter biases");
                clCQWMover.finish();
                cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                elapsedTimeUs += (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            }

            std::cout <<"Transfer the filter biases took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the filter weights"<<std::endl;
        {
            cl_double weightTransferElapsedTimeUs = 0.0;
            int weightTransferBytes = 0;
            int idxTensor = 0;
            for (std::shared_ptr<AlignedTensor>& ptr : _executionGraph.pWeights)
            {
                cl::Event event;
                auto pWeights = ptr.get();
                auto numElements =  (pWeights->getTransferBlockVector()).size();
                auto sizeElement = sizeof(typeof((pWeights->getTransferBlockVector()).at(0)));

                int byteOffset = _executionGraph.vecWeightDramBlockStart.at(idxTensor++)
                        * (BURST_SIZE_BYTE);

                weightTransferBytes += numElements*sizeElement;
                assert(weightTransferBytes <= MAX_DRAM_BYTE_INPUT_WEIGHT && "Too many weights to fit inside the global memory" );
                status = clCQWMover.enqueueWriteBuffer(bufferWMoverWDramBlocks, //buffer
                                                     CL_TRUE, //blocking_write
                                                     byteOffset, //offset
                                                     weightTransferBytes, //size
                                                     (pWeights->getTransferBlockVector()).data(), //data pointer
                                                     NULL, //dependency list
                                                     &event //events generated
                                                    );
                aocl_utils_cpp::checkError(status, "Failed to write the filter weight tensors");
                clCQWMover.finish();
                cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                weightTransferElapsedTimeUs += (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            }

            std::cout <<"Transfer the filter weight tensors took "<<weightTransferElapsedTimeUs<<" us"<<std::endl;
        }

    #if defined(SPARSE_SYSTEM)
        std::cout <<stepCount++<<". Transfer the filter weight TB counts"<<std::endl;
        {
            cl_double transferElapsedTimeUs = 0.0;
            int transferBytes = 0;
            int idxTensor = 0;
            for (std::shared_ptr<AlignedTensor>& ptr : _executionGraph.pWeights)
            {
                cl::Event event;
                auto pWeights = ptr.get();
                auto numElements =  (pWeights->getTransferBlockCountVector()).size();
                auto sizeElement = sizeof(typeof((pWeights->getTransferBlockCountVector()).at(0)));
                transferBytes += sizeElement * numElements;

                int byteOffset = _executionGraph.vecWeightTBCountStart.at(idxTensor++) * 2;

                assert(transferBytes <= MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT && "Too many weight TB counts to fit inside the global memory" );

                status = clCQWMover.enqueueWriteBuffer(bufferWMoverWTBCounts, //buffer
                                                     CL_TRUE, //blocking_write
                                                     byteOffset, //offset
                                                     transferBytes, //size
                                                     (pWeights->getTransferBlockCountVector()).data(), //data pointer
                                                     NULL, //dependency list
                                                     &event //events generated
                                                    );
                aocl_utils_cpp::checkError(status, "Failed to write the filter weight TB counts");
                clCQWMover.finish();
                cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                transferElapsedTimeUs += (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            }
            std::cout <<"Transfer the filter weight TB counts took "<<transferElapsedTimeUs<<" us"<<std::endl;
        }
    #endif
        std::cout <<stepCount++<<". Transfer the MK controller instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecMiscInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecMiscInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferMKInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION && "Too many MK instructions to fit inside the global memory" );

            status = clCQMKController.enqueueWriteBuffer(bufferMKInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 _executionGraph.vecMiscInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the MK controller instructions");
            clCQMKController.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the MK controller instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }
    }

    void AcceleratorWrapper::prepareInputBlob (std::vector<float>& floatBlob, int inputBlobID)
    {
        //Sanity checks:
        //1. inputBlobID should be in range
        //2. Number of elements inside floatBlob should be consistent with the inputBlobID info.
        assert(inputBlobID < vecInputBlobsInfo.size() && "inputBlobID is out of range");
        auto blobInfo = vecInputBlobsInfo.at(inputBlobID);
        assert(blobInfo.group*blobInfo.channelPerGroup*blobInfo.height*blobInfo.width
               == floatBlob.size() && "Number of elements in the provided vector of float mismatch that of the requested input tensor size.");

        //Quantize the input
        std::vector<fixedPointNumber> inputBlobQuantized = quantize(floatBlob, blobInfo.numFracBits);
        if (vecInputBlobsInfo.at(inputBlobID).flagCanBeSparse == true)
        {
            vecInputBlobsInternal.at(inputBlobID).reset(
                        new FlexibleDirectCompressedTensor(
                            inputBlobQuantized,
                            1, //_num3DTensors
                            blobInfo.group * blobInfo.channelPerGroup, //_channel
                            blobInfo.width, //_width
                            blobInfo.height, //_height
                            blobInfo.channelPerGroup-1, //_maxScalarIndexInChannelGroup
                            COMPRESSION_WINDOW_SIZE-1, //_maxClusterIndexInCompressionBlock
                            TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                            CLUSTER_SIZE-1, //_maxScalarIndexInCluster
                            false //isKernel
                            )
                        );
        }
        else
        {
            vecInputBlobsInternal.at(inputBlobID).reset(
                        new AlignedTensor(
                            inputBlobQuantized,
                            1, //_num3DTensors
                            blobInfo.group * blobInfo.channelPerGroup, //_channel
                            blobInfo.width, //_width
                            blobInfo.height, //_height
                            blobInfo.channelPerGroup-1, //_maxScalarIndexInChannelGroup
                            TRANSFER_SIZE-1, //_maxClusterIndexInTransferBlock
                            CLUSTER_SIZE-1, //_maxScalarIndexInCluster
                            false //isKernel
                            )
                        );
        }
    } //prepareInputBlob

    std::vector<float> AcceleratorWrapper::extractOutputBlob (int outputBlobID)
    {
        assert(outputBlobID < vecOutputBlobsInfo.size() && "outputBlobID is out of range.");
        t_blob_info blobInfo = vecOutputBlobsInfo.at(outputBlobID);
        std::vector<fixedPointNumber> quantizedResult;
        int fracWidth = blobInfo.numFracBits;
        int intWidth = 7-fracWidth;
        vecOutputBlobsInternal.at(outputBlobID)->decodeTensor(quantizedResult, fracWidth, intWidth);
        std::vector<float> result = convert2Float(quantizedResult);
        return result;
    } //extractOutputBlob

    std::vector<t_blob_info> AcceleratorWrapper::getInputBlobsInfo()
    {
        return vecInputBlobsInfo;
    }
    std::vector<t_blob_info> AcceleratorWrapper::getOutputBlobsInfo()
    {
        return vecOutputBlobsInfo;
    }

    void AcceleratorWrapper::inference()
    {
        cl_int status = CL_SUCCESS;
        /*
          1. Launch all kernels, except for the IA mover and the OA mover
        */
        status = clCQIATileController.enqueueTask(kernelIATileController, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelIATileController!");

        status = clCQWMover.enqueueTask(kernelWMover, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelWMover!");

        status = clCQMKController.enqueueTask(kernelMKInstructionMover, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelMKInstructionMover!");

        status = clCQOATileController.enqueueTask(KernelOATileController, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch KernelOATileController!");

        /*
         *2. Transfer all input blobs to the FPGA
        */


        /*
         *3. Ochestrate the layer execution
        */

        /*
         *4. Transfer output blobs from the FPGA to the host
        */

    }


}
