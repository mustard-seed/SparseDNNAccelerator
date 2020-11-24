#include "accelerator_wrapper.hpp"
#include "params.hpp"
#include "floatFixedPointConversion.hpp"
#include "timer.hpp"

#if !defined(C5SOC)
#include "CL/cl_ext_intelfpga.h" //For CL_CHANNEL_<X>_INTELFPGA
#endif

#include <iomanip>
#include <limits>
#include <fstream> //For file I/O

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

//#if defined(C5SOC) //Hack for ARMv7, otherwise chrono won't work
//__asm__(".symver _ZNSt6chrono3_V212system_clock3nowEv,_ZNSt6chrono12system_clock3nowEv@GLIBCXX_3.4.11");
//#endif

//See https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224.html#yng1552497976376
typedef cl_int (*clGetProfileDataDevice_fn) (cl_device_id, cl_program,
                                                       cl_bool, cl_bool, cl_bool,
                                                       size_t, void *,
                                                       size_t *, cl_int *);

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

    std::vector<fixedPointNumber> quantize(std::vector<float> floatVector, signed char _fracWidth)
    {
        //TODO: Maybe parallelize this?
        std::vector<fixedPointNumber> result;
        result.resize(floatVector.size());
        int index = 0;
        signed char intWidth = 7 - _fracWidth;
        for (auto& fpVal: floatVector)
        {
            result.at(index++) = fixedPointNumber(fpVal, _fracWidth, intWidth);
        }
        return result;
    }

    AcceleratorWrapper::AcceleratorWrapper(std::string _bitstreamFileName,
                                           std::string _platformName,
                                           t_accelerator_info _acceleratorInfo,
                                           int _fpgaID) :
        minInferenceDuration(std::numeric_limits<double>::max()),
        maxInferenceDuration(0.00),
        averageInferenceDuration(0.00)
    {
        cl_int status = CL_SUCCESS;

        binaryFile = _bitstreamFileName;
        clPlatform = aocl_utils_cpp::findPlatform(_platformName);

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
    }

    AcceleratorWrapper::AcceleratorWrapper(std::string _bitstreamFileName,
                                           std::string _platformName,
                                           t_execution_graph& _executionGraph,
                                           t_accelerator_info& _acceleratorInfo,
                                           int _fpgaID) :
        AcceleratorWrapper(_bitstreamFileName,  _platformName, _acceleratorInfo, _fpgaID)
    {
        resetGraph();

        //Load the network, allocate buffers, and transfer the instructions
        loadGraph(_executionGraph);

    } // AcceleratorWrapper parametrized constructor

    void AcceleratorWrapper::resetGraph()
    {
        vecInputBlobsInfo.clear();
        vecInputBlobsInternal.clear();
        vecOutputBlobsInfo.clear();
        vecOutputBlobsInternal.clear();
        vecInputTransferTime.clear();
        vecOutputTransferTime.clear();
        vecLayerExecutionTime.clear();
        numRunExecuted = 0;
        minInferenceDuration = std::numeric_limits<double>::max();
        maxInferenceDuration = 0.0;
        averageInferenceDuration = 0.00;
        launchIATileController = false;
        launchMKController = false;
        launchWMover = false;
        numIAInstructions = 0;
    }

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
        vecInputTransferTime = std::vector<cl_double>(vecInputBlobsInfo.size(), 0.0);

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
        vecOutputTransferTime = std::vector<cl_double>(vecOutputBlobsInfo.size(), 0.0);

        vecLayerInfo = _executionGraph.vecLayerInfo;
        vecLayerExecutionTime = std::vector<cl_double>(vecLayerInfo.size(), 0.0);

        numRunExecuted = 0;

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

            auto numElements = _executionGraph.vecIAMoverInstruction.size();
            //unsigned int numInstruction,
            kernelIAMover.setArg(argIdx, (cl_uint) numElements);
            argIdx++;

            //unsigned int offsetInstruction
            kernelIAMover.setArg(argIdx, (cl_uint) 0);
            argIdx++;
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the IA Tile controller."<<std::endl;
        {
            cl_uint argIdx = 0;
            //__global volatile t_ia_tile_controller_instruction* restrict pInstruction,
            kernelIATileController.setArg(argIdx, bufferIATileControllerInstructions);
            argIdx++;

            //unsigned int numInstruction
            unsigned int numInustruction = _executionGraph.vecIATileControllerInstruction.size();
            kernelIATileController.setArg(argIdx, (cl_uint) (numInustruction) );

            launchIATileController = numInustruction > 0 ? true : false;
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the Filter mover."<<std::endl;
        {
            cl_uint argIdx = 0;
            //volatile __global t_weight_mover_instruction* restrict pInst,
            kernelWMover.setArg(argIdx++, bufferWMoverInstructions);
            //volatile __global t_dram_block* restrict pW,
            kernelWMover.setArg(argIdx++, bufferWMoverWDramBlocks);
            //vola<tile __global t_bias* restrict pBias,
            kernelWMover.setArg(argIdx++, bufferWMoverBias);
            #if defined(SPARSE_SYSTEM)
                //volatile __global t_streamblock_address* restrict pFilterTBCount,
                kernelWMover.setArg(argIdx++, bufferWMoverWTBCounts);
            #endif //SPARSE_SYSTEM
            //unsigned int numInstruction
            unsigned int numInustruction = _executionGraph.vecWMoverInstruction.size();
            kernelWMover.setArg(argIdx++, (cl_uint) (_executionGraph.vecWMoverInstruction.size()) );
            launchWMover = numInustruction > 0 ? true: false;
        }

        std::cout <<stepCount++<<". Setting kernel arguments for the Miscellaneous controller."<<std::endl;
        {
            cl_uint argIdx=0;
            //__global t_misc_instruction* restrict pInstruction,
            kernelMKInstructionMover.setArg(argIdx++, bufferMKInstructions);
            //unsigned int numInstruction
            unsigned int numInustruction = _executionGraph.vecMiscInstruction.size();
            kernelMKInstructionMover.setArg(argIdx++, (cl_uint) (_executionGraph.vecMiscInstruction.size()) );
            launchMKController = numInustruction > 0 ? true: false;
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
            if (transferBytes > MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION)
            {
                std::cout << "Too many IA mover instructions to fit inside the global memory."<<std::endl;
                throw;
            }

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

            if (transferBytes > 0)
            {
                std::cout <<"Transfering "<<transferBytes<<" bytes into bufferIATileControllerInstructions"<<std::endl;
                if (transferBytes > MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION)
                {
                    std::cout << "Too many IA Tile instructions to fit inside the global memory."<<std::endl;
                    throw;
                }

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
        }

        std::cout <<stepCount++<<". Transfer the OA Mover instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecOAMoverInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecOAMoverInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferOAMoverInstructions"<<std::endl;
            if (transferBytes > MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION)
            {
                std::cout << "Too many OA mover instructions to fit inside the global memory."<<std::endl;
                throw;
            }

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
            if (transferBytes > MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION)
            {
                std::cout << "Too many OA TILE instructions to fit inside the global memory."<<std::endl;
                throw;
            }

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

        std::cout <<stepCount++<<". Transfer the W Mover instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = _executionGraph.vecWMoverInstruction.size();
            auto sizeElement = sizeof(typeof(_executionGraph.vecWMoverInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            if (transferBytes > 0)
            {
                std::cout <<"Transfering "<<transferBytes<<" bytes into bufferWMoverInstructions"<<std::endl;
                if (transferBytes > MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION)
                {
                    std::cout << "Too many Weight Mover instructions to fit inside the global memory."<<std::endl;
                    throw;
                }


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
                if (transferBytes > MAX_DRAM_BYTE_INPUT_BIAS)
                {
                    std::cout << "Too many biases to fit inside the global memory."<<std::endl;
                    throw;
                }

                status = clCQWMover.enqueueWriteBuffer(bufferWMoverBias, //buffer
                                                     CL_TRUE, //blocking_write
                                                     offsetIndex, //offset
                                                     sizeElement * numElements, //size
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
                if (weightTransferBytes > MAX_DRAM_BYTE_INPUT_WEIGHT)
                {
                    std::cout << "Too many weights to fit inside the global memory."<<std::endl;
                    throw;
                }
                status = clCQWMover.enqueueWriteBuffer(bufferWMoverWDramBlocks, //buffer
                                                     CL_TRUE, //blocking_write
                                                     byteOffset, //offset
                                                     numElements*sizeElement, //size
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
                if (transferBytes > MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT)
                {
                    std::cout << "Too many weight TB counts to fit inside the global memory."<<std::endl;
                    throw;
                }
                status = clCQWMover.enqueueWriteBuffer(bufferWMoverWTBCounts, //buffer
                                                     CL_TRUE, //blocking_write
                                                     byteOffset, //offset
                                                     numElements*sizeElement, //size
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

            if (transferBytes > 0)
            {
                std::cout <<"Transfering "<<transferBytes<<" bytes into bufferMKInstructions"<<std::endl;
                if (transferBytes > MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION)
                {
                    std::cout << "Too many MK instructions to fit inside the global memory."<<std::endl;
                    throw;
                }
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
    }

    void AcceleratorWrapper::prepareInputBlob (std::vector<float>& floatBlob, int inputBlobID)
    {
        //Sanity checks:
        //1. inputBlobID should be in range
        //2. Number of elements inside floatBlob should be consistent with the inputBlobID info.
        auto blobInfo = vecInputBlobsInfo.at(inputBlobID);
        if(blobInfo.group*blobInfo.channelPerGroup*blobInfo.height*blobInfo.width
               != floatBlob.size())
        {
            std::cout << "Number of elements in the provided vector of float mismatch that"
                      <<" of the requested input tensor size."<<std::endl;
        }

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
        t_blob_info blobInfo = vecOutputBlobsInfo.at(outputBlobID);
        std::vector<fixedPointNumber> quantizedResult;
        signed char fracWidth = blobInfo.numFracBits;
        signed char intWidth = 7-fracWidth;
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

    void AcceleratorWrapper::inference(bool flagEnableProfile)
    {
        Timer t;
        t.start();
        cl_int status = CL_SUCCESS;
        /*
          1. Launch all kernels, except for the IA mover and the OA mover
        */
       //TODO: add conditional checks to see whether we need to launch some of the kernels
        if (launchIATileController)
        {
            status = clCQIATileController.enqueueTask(kernelIATileController, NULL);
            aocl_utils_cpp::checkError(status, "Failed to launch kernelIATileController!");
        }

        if (launchWMover)
        {
            status = clCQWMover.enqueueTask(kernelWMover, NULL);
            aocl_utils_cpp::checkError(status, "Failed to launch kernelWMover!");
        }

        if (launchMKController)
        {
            status = clCQMKController.enqueueTask(kernelMKInstructionMover, NULL);
            aocl_utils_cpp::checkError(status, "Failed to launch kernelMKInstructionMover!");
        }

        status = clCQOATileController.enqueueTask(KernelOATileController, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch KernelOATileController!");

        /*
         *2. Transfer all input blobs to the FPGA
        */
        {
            int index = 0;
            for (const auto& blobInfo : vecInputBlobsInfo)
            {
                cl::Event event;
                auto pInput = vecInputBlobsInternal.at(index).get();

                auto numTransferBlocks = (pInput->getTransferBlockVector()).size();
                auto sizeTransferBlockElement = sizeof(typeof((pInput->getTransferBlockVector()).at(0)));
                auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

                int activationOffsetByte = blobInfo.memoryRegionID * MEM_ACTIVATION_REGION_SIZE_PER_SLICE * BURST_SIZE_BYTE;
                if (valueVectorSizeBytes > (BURST_SIZE_BYTE * MEM_ACTIVATION_REGION_SIZE_PER_SLICE))
                {
                    std::cout << "Too many input activation bytes to fit inside the global memory."<<std::endl;
                    throw;
                }
                status = clCQIAMover.enqueueWriteBuffer(bufferActivationDramBlocks, //buffer
                                                         CL_TRUE, //blocking_write
                                                         activationOffsetByte, //offset
                                                         valueVectorSizeBytes, //size
                                                         (pInput->getTransferBlockVector()).data(), //data pointer
                                                         NULL, //dependency list
                                                         &event //events generated
                                                            );
                aocl_utils_cpp::checkError(status, "Failed to write an input activation vector");
                cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                vecInputTransferTime.at(index) += (cl_double)((endTime - startTime)*(cl_double)(1e-3));

                #if defined(SPARSE_SYSTEM)
                    if (blobInfo.flagCanBeSparse == true)
                    {
                        auto numElements = (pInput->getTransferBlockCountVector()).size();
                        auto sizeElement = sizeof(typeof((pInput->getTransferBlockCountVector()).at(0)));
                        auto transferBytes = sizeElement * numElements;

                        int tbCountOffsetByte = blobInfo.memoryRegionID * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE * sizeof(t_streamblock_address);

                        //std::cout <<"Transfering "<<transferBytes<<" bytes into bufferActivationTBCounts"<<std::endl;
                        if (transferBytes > (2 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE))
                        {
                            std::cout << "Too many input activation TB count bytes to fit inside the global memory."<<std::endl;
                            throw;
                        }
                        status = clCQIAMover.enqueueWriteBuffer(bufferActivationTBCounts, //buffer
                                                             CL_TRUE, //blocking_write
                                                             tbCountOffsetByte, //offset
                                                             transferBytes, //size
                                                             (pInput->getTransferBlockCountVector()).data(), //data pointer
                                                             NULL, //dependency list
                                                             NULL //events generated
                                                            );
                        aocl_utils_cpp::checkError(status, "Failed to write the input activation TB count");
                    }
                #endif
                index++;
            }
        }


        /*
         *3. Ochestrate the layer execution
        */
        std::vector<cl::Event> vecOAFinishes;
        int numLayers = vecLayerInfo.size();
        vecOAFinishes.resize(numLayers);

        if (flagEnableProfile == true)
        {
#if !defined(C5SOC)
            clGetProfileDataDevice_fn get_profile_data_ptr = (clGetProfileDataDevice_fn) clGetExtensionFunctionAddressForPlatform (clPlatform(), "clGetProfileDataDeviceIntelFPGA");
            (get_profile_data_ptr) (
                        //cl_device_id
                        clDevice(),
                        //cl_program
                        program(),
                        //bool read_enqueue_kernels, no effect
                        false,
                        //cl_bool read_auto_enqueued,
                        true,
//                        cl_bool clear_counters_after_readback,
//                        size_t param_value_size,
//                        void *param_value,
//                        size_t *param_value_size_ret,
//                        cl_int *errcode_ret
                        false,
                        0,
                        NULL,
                        0,
                        NULL
                        );
            (get_profile_data_ptr) (
                        //cl_device_id
                        clDevice(),
                        //cl_program
                        program(),
                        //bool read_enqueue_kernels, no effect
                        false,
                        //cl_bool read_auto_enqueued,
                        true,
//                        cl_bool clear_counters_after_readback,
//                        size_t param_value_size,
//                        void *param_value,
//                        size_t *param_value_size_ret,
//                        cl_int *errcode_ret
                        false,
                        0,
                        NULL,
                        0,
                        NULL);
#else
            clGetProfileDataDeviceIntelFPGA(
                        //cl_device_id
                        clDevice(),
                        //cl_program
                        program(),
                        //bool read_enqueue_kernels, no effect
                        false,
                        //cl_bool read_auto_enqueued,
                        true,
//                        cl_bool clear_counters_after_readback,
//                        size_t param_value_size,
//                        void *param_value,
//                        size_t *param_value_size_ret,
//                        cl_int *errcode_ret
                        false,
                        0,
                        NULL,
                        0,
                        NULL
                  );
#endif

        }


        for (int i=0; i<numLayers; i++)
        {
            #if defined(HOST_DEBUG)
                std::cout<<"Launching layer "<<vecLayerInfo.at(i).layerName<<std::endl;
            #endif
            #if defined(SPARSE_SYSTEM)
                cl_uint oaArgIdx = 3;
            #else
                cl_uint oaArgIdx = 2;
            #endif

            //unsigned int numInstruction,
            kernelOAMover.setArg(oaArgIdx++, (cl_uint) vecLayerInfo.at(i).numOAMoverInstructions);
            //unsigned int offsetInstruction
            kernelOAMover.setArg(oaArgIdx++, (cl_uint) vecLayerInfo.at(i).offsetOAMoverInstruction);

            status = clCQOAMover.enqueueTask(kernelOAMover, NULL, &(vecOAFinishes.at(i)));
            #if defined(HOST_DEBUG)
            aocl_utils_cpp::checkError(status, "Failed to launch kernelOAMover!");
            #endif
            if (i==0)
            {
                #if defined(HOST_DEBUG)
                    std::cout<<"Launching kernelIAMover."<<std::endl;
                #endif
                status = clCQIAMover.enqueueTask(kernelIAMover, NULL, NULL);
                #if defined(HOST_DEBUG)
                aocl_utils_cpp::checkError(status, "Failed to launch kernelIAMover!");
                #endif
            }
            #if defined(HOST_DEBUG)
                clCQOAMover.finish();
            #endif
        }

        #if defined(HOST_DEBUG)
            std::cout<<"Done all layers."<<std::endl;
        #else
            clCQOAMover.finish();
        #endif

            if (flagEnableProfile == true)
            {
#if !defined(C5SOC)
                clGetProfileDataDevice_fn get_profile_data_ptr = (clGetProfileDataDevice_fn) clGetExtensionFunctionAddressForPlatform (clPlatform(), "clGetProfileDataDeviceIntelFPGA");
                (get_profile_data_ptr) (
                            //cl_device_id
                            clDevice(),
                            //cl_program
                            program(),
                            //bool read_enqueue_kernels, no effect
                            false,
                            //cl_bool read_auto_enqueued,
                            true,
    //                        cl_bool clear_counters_after_readback,
    //                        size_t param_value_size,
    //                        void *param_value,
    //                        size_t *param_value_size_ret,
    //                        cl_int *errcode_ret
                            false,
                            0,
                            NULL,
                            0,
                            NULL);
#else
            clGetProfileDataDeviceIntelFPGA(
                        //cl_device_id
                        clDevice(),
                        //cl_program
                        program(),
                        //bool read_enqueue_kernels, no effect
                        false,
                        //cl_bool read_auto_enqueued,
                        true,
//                        cl_bool clear_counters_after_readback,
//                        size_t param_value_size,
//                        void *param_value,
//                        size_t *param_value_size_ret,
//                        cl_int *errcode_ret
                        false,
                        0,
                        NULL,
                        0,
                        NULL
                  );
#endif

            }

        /*
         *4. Transfer output blobs from the FPGA to the host
        */
        {
            int index = 0;
            for (const auto& blobInfo : vecOutputBlobsInfo)
            {
                cl::Event event;
                auto pOutput = vecOutputBlobsInternal.at(index).get();

                auto numTransferBlocks = (pOutput->getTransferBlockVector()).size();
                auto sizeTransferBlockElement = sizeof(typeof((pOutput->getTransferBlockVector()).at(0)));
                auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

                int activationOffsetByte = blobInfo.memoryRegionID * MEM_ACTIVATION_REGION_SIZE_PER_SLICE * BURST_SIZE_BYTE;
                if (valueVectorSizeBytes > (BURST_SIZE_BYTE * MEM_ACTIVATION_REGION_SIZE_PER_SLICE))
                {
                    std::cout << "Too many output activation bytes to read from global memory."<<std::endl;
                    throw;
                }
                status = clCQOAMover.enqueueReadBuffer(bufferActivationDramBlocks, //buffer
                                                         CL_TRUE, //blocking_write
                                                         activationOffsetByte, //offset
                                                         valueVectorSizeBytes, //size
                                                         (pOutput->getTransferBlockVector()).data(), //data pointer
                                                         NULL, //dependency list
                                                         &event //events generated
                                                            );
                #if defined(HOST_DEBUG)
                    aocl_utils_cpp::checkError(status, "Failed to read an output activation vector");
                #endif
                cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                vecOutputTransferTime.at(index) += (cl_double)((endTime - startTime)*(cl_double)(1e-3));

                #if defined(SPARSE_SYSTEM)
                    if (blobInfo.flagCanBeSparse == true)
                    {
                        auto numElements = (pOutput->getTransferBlockCountVector()).size();
                        auto sizeElement = sizeof(typeof((pOutput->getTransferBlockCountVector()).at(0)));
                        auto transferBytes = sizeElement * numElements;

                        int tbCountOffsetByte = blobInfo.memoryRegionID * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE * sizeof(t_streamblock_address);

                        if (transferBytes > (2 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE))
                        {
                            std::cout << "Too many output activation TB count bytes to be read from global memory."<<std::endl;
                            throw;
                        }
                        status = clCQOAMover.enqueueReadBuffer(bufferActivationTBCounts, //buffer
                                                             CL_TRUE, //blocking_write
                                                             tbCountOffsetByte, //offset
                                                             transferBytes, //size
                                                             (pOutput->getTransferBlockCountVector()).data(), //data pointer
                                                             NULL, //dependency list
                                                             NULL //events generated
                                                               );
                        #if defined(HOST_DEBUG)
                        aocl_utils_cpp::checkError(status, "Failed to read the output activation TB count");
                        #endif
                    }
                #endif
                index++;
            }
        }
        t.stop();

        //Update runtime of eacy layer
        {
            int index = 0;
            for (const auto& event : vecOAFinishes)
            {
                cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                vecLayerExecutionTime.at(index++) += (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            }
        }
        numRunExecuted++;

        //Updat clocks
        double timeElapsed = (double) t.get_time_s();
        averageInferenceDuration = (averageInferenceDuration*(numRunExecuted-1) + timeElapsed) / numRunExecuted;
        maxInferenceDuration = std::max(timeElapsed, maxInferenceDuration);
        minInferenceDuration = std::min(timeElapsed, minInferenceDuration);

    }

    float AcceleratorWrapper::getInvocationOverhead()
    {
        float usTime = 0;
        for (int i=0; i<100; i++)
        {
            cl::Event event;
            clCQNoop.enqueueTask(kernelNoop, NULL, &event);
            clCQNoop.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            usTime += (float)((endTime - startTime)*(cl_double)(1e-3));

        }
        usTime = usTime  / 100.0;
        return usTime;
    }

    std::string AcceleratorWrapper::reportRuntime()
    {
        //TODO: finish implementing this
        std::ostringstream buffer;
        buffer <<"===========Performance counts ========="<<std::endl;
        buffer <<std::setw(30)<<std::left<<"Number of inferences: ";
        buffer <<std::setw(30)<<std::left<<std::to_string(numRunExecuted)<<std::endl;
        buffer <<std::setw(30)<<std::left<<"Average Inference Latency (us): ";
        buffer <<std::setw(30)<<std::left<<std::to_string(averageInferenceDuration * 1000000.0)<<std::endl;
        buffer <<std::setw(30)<<std::left<<"Maximum Inference Latency (us): ";
        buffer <<std::setw(30)<<std::left<<std::to_string(maxInferenceDuration * 1000000.0)<<std::endl;
        buffer <<std::setw(30)<<std::left<<"Minimum Inference Latency (us): ";
        buffer <<std::setw(30)<<std::left<<std::to_string(minInferenceDuration * 1000000.0)<<std::endl;

        const int maxName = 29;
        if (numRunExecuted > 0)
        {
            //Print average input blob transfer time
            buffer <<"===========Input blob transfer time"<<std::endl;
            buffer <<std::setw(30)<<std::left<<"Input Blob Name";
            buffer <<std::setw(30)<<std::left<<"Average transfer latency (us)"<<std::endl;
            for (unsigned int i=0; i<vecInputBlobsInfo.size(); i++)
            {
                auto blobInfo = vecInputBlobsInfo.at(i);
                auto blobTime = vecInputTransferTime.at(i);
                std::string name = blobInfo.blobName;
                if (name.length() > maxName)
                {
                    name = name.substr(0, maxName-4);
                    name += "...";
                }
                double averageTimeUs = blobTime / ((double) numRunExecuted);
                buffer <<std::setw(30)<<std::left<<name;
                buffer <<std::setw(30)<<std::left<<std::to_string(averageTimeUs)<<std::endl;
            }
            //Print average layer transfer time
            buffer <<"===========Layer inference time"<<std::endl;
            buffer <<std::setw(30)<<std::left<<"Layer Name";
            buffer <<std::setw(30)<<std::left<<"Average latency (us)"<<std::endl;
            for (unsigned int i=0; i<vecLayerExecutionTime.size(); i++)
            {
                auto layerInfo = vecLayerInfo.at(i);
                auto layerTime = vecLayerExecutionTime.at(i);
                std::string name = layerInfo.layerName;
                if (name.length() > maxName)
                {
                    name = name.substr(0, maxName-4);
                    name += "...";
                }
                double averageTimeUs = layerTime / ((double) numRunExecuted);
                buffer <<std::setw(30)<<std::left<<name;
                buffer <<std::setw(30)<<std::left<<std::to_string(averageTimeUs)<<std::endl;
            }
            //Print average outout blob transfer time
            buffer <<"===========Output blob transfer time"<<std::endl;
            buffer <<std::setw(30)<<std::left<<"Output Blob Name";
            buffer <<std::setw(30)<<std::left<<"Average transfer latency (us)"<<std::endl;
            for (unsigned int i=0; i<vecOutputBlobsInfo.size(); i++)
            {
                auto blobInfo = vecOutputBlobsInfo.at(i);
                auto blobTime = vecOutputTransferTime.at(i);
                std::string name = blobInfo.blobName;
                if (name.length() > maxName)
                {
                    name = name.substr(0, maxName-4);
                    name += "...";
                }
                double averageTimeUs = blobTime / ((double) numRunExecuted);
                buffer <<std::setw(30)<<std::left<<name;
                buffer <<std::setw(30)<<std::left<<std::to_string(averageTimeUs)<<std::endl;
            }
        }
        return buffer.str();
    }

    void AcceleratorWrapper::dumpRuntimeToCSV(std::string csvFilePath)
    {
        std::ofstream dumpFile;
        dumpFile.open(csvFilePath);
        std::string sep = ",";

        const int maxName = 29;
        if (numRunExecuted > 0)
        {
            //Print average input blob transfer time
            for (unsigned int i=0; i<vecInputBlobsInfo.size(); i++)
            {
                auto blobInfo = vecInputBlobsInfo.at(i);
                auto blobTime = vecInputTransferTime.at(i);
                std::string name = blobInfo.blobName;
                if (name.length() > maxName)
                {
                    name = name.substr(0, maxName-4);
                    name += "...";
                }
                double averageTimeUs = blobTime / ((double) numRunExecuted);
                dumpFile <<name<<sep<<std::to_string(averageTimeUs)<<std::endl;
            }
            //Print average layer transfer time
            for (unsigned int i=0; i<vecLayerExecutionTime.size(); i++)
            {
                auto layerInfo = vecLayerInfo.at(i);
                int sizeOutputTileHeight = layerInfo.outputTileHeight;
                int sizeOutputTileWidthPerCol = layerInfo.outputTileWidthPerCol;
                int numActiveColsPartialCol = layerInfo.numActiveColsPartialOutputTile;
                unsigned int expectedLatency = layerInfo.expectedLatency;
                unsigned int isComputeBound = layerInfo.isComputeBound;
                unsigned int ops = layerInfo.ops;
                unsigned int inputLatency = layerInfo.inputTransferLatency;
                unsigned int outputLatency = layerInfo.outputTransferLatency;
                unsigned int weightLatency = layerInfo.weightTransferLatency;
                unsigned int computeLatency = layerInfo.computeLatency;
                auto layerTime = vecLayerExecutionTime.at(i);
                std::string name = layerInfo.layerName;
                if (name.length() > maxName)
                {
                    name = name.substr(0, maxName-4);
                    name += "...";
                }
                double averageTimeUs = layerTime / ((double) numRunExecuted);
                dumpFile <<name<<sep<<std::to_string(averageTimeUs)
                        <<sep<<sizeOutputTileHeight
                        <<sep<<sizeOutputTileWidthPerCol
                        <<sep<<numActiveColsPartialCol
                        <<sep<<expectedLatency
                        <<sep<<ops
                        <<sep<<isComputeBound
                        <<sep<<inputLatency
                        <<sep<<weightLatency
                        <<sep<<outputLatency
                        <<sep<<computeLatency
                        <<std::endl;
            }
            //Print average outout blob transfer time
            for (unsigned int i=0; i<vecOutputBlobsInfo.size(); i++)
            {
                auto blobInfo = vecOutputBlobsInfo.at(i);
                auto blobTime = vecOutputTransferTime.at(i);
                std::string name = blobInfo.blobName;
                if (name.length() > maxName)
                {
                    name = name.substr(0, maxName-4);
                    name += "...";
                }
                double averageTimeUs = blobTime / ((double) numRunExecuted);
                dumpFile <<name<<sep<<std::to_string(averageTimeUs)<<std::endl;
            }
        }

        dumpFile <<"Number of inferences"<<sep<<std::to_string(numRunExecuted)<<std::endl;
        dumpFile <<"Average Inference Latency (us)"<<sep<<std::to_string(averageInferenceDuration * 1000000.0)<<std::endl;
        dumpFile <<"Maximum Inference Latency (us)"<<sep<<std::to_string(maxInferenceDuration * 1000000.0)<<std::endl;
        dumpFile <<"Minimum Inference Latency (us)"<<sep<<std::to_string(minInferenceDuration * 1000000.0)<<std::endl;
        dumpFile.close();
    }
}
