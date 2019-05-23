#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/cl.hpp"
#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"
#include "params.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string> //for std::to_string
#include "device_structures.hpp"
#include "boost/align/aligned_allocator.hpp"
#include <unistd.h> //usleep
#include <random>

#define MATRIX_ROWS 64
#define MATRIX_COLS 10
#define SEED 10
#define BERN_P 0.5

typedef
std::vector<cl_ushort, boost::alignment::aligned_allocator<cl_ushort, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
aligned_ushort_vector;

typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
aligned_short_vector;


/*! \brief Initialize the matrix
*/
void matrix_initialization ( std::vector<char> & _matrix);

/*! \brief Compresses a given 2D matrix.
*/
void matrix_compression (std::vector<char> & _matrix,
    unsigned int numberOfWeightsPerRow,
    unsigned int numberOfRows,
    unsigned int cbPerRow,
    aligned_short_vector &outEffectualValues,
    aligned_ushort_vector &outCBOffets
  );

/*! \brief Comppare the output matrix with the original matrix
    \return True if the two matrices match. False otherwise.
*/
bool check_matrix (aligned_short_vector &originalMatrix,
  aligned_ushort_vector &originalIndex,
  aligned_short_vector &outputMatrix,
  unsigned int numberOfRow,
  unsigned int numberOfWeightsPerRow,
  unsigned int cbPerRow
  );

/*!
 * \brief clInit
 * \details Set up the OpenCL context, creates the command queue, and create the kernels
 * \param binaryFile
 * \return The status code
 */
cl_int clInit (const std::string binaryFile,
               cl::Platform &clPlatform,
               cl::Context & clContext,
               cl::Device & clDevice,
               cl::CommandQueue & clDMAQueue,
               cl::CommandQueue & clSequencerQueue,
               std::vector<cl::CommandQueue> & clCollectorQueue,
               cl::Kernel & krnSpWDMA,
               cl::Kernel & krnSequencer,
               std::vector<cl::Kernel> & krnWeightCollectorVec);

t_instruction cmdGenFillWeightBuffer(
        unsigned int ddrIndexOffset,
        unsigned int ddrWeightOffset,
        unsigned short filterStart,
        unsigned char numFilterToStream,
        unsigned short cbStart,
        unsigned short cbEnd,
        unsigned short numCbInFilter,
        unsigned int nWInFilter
        );


t_instruction cmdGenDrainWeightBuffer(
        unsigned char laneStart,
        unsigned char laneEnd,
        unsigned short cbStart,
        unsigned short cbEnd
        );

t_instruction cmdGenCollectWeight(
        unsigned int ddrWeightOffset,
        unsigned short filterStart,
        unsigned short numFiltersToCollect,
        unsigned int numWeightInFilter
        );

t_instruction cmdGentWeightBufferSwap();

void cleanup();

int main(int argc, char* argv[]) {

    aocl_utils_cpp::Options options(argc, argv);

    std::string binaryFile;

    if(options.has("help")) {
        std::cout<<"Usage: "<<argv[0]
        <<" -aocx=<abs path to .aocx>"<<std::endl;
        return 1;
    }

    if (options.has("aocx")) {
        binaryFile = options.get<std::string>("aocx");
      }
      else {
        std::cout<<"Error: aocx file path is not supplied"<<std::endl;
        return 1;
      }


    std::cout <<"Prepare the test matrix"<<std::endl;
    std::vector<char> matrix;

    aligned_short_vector effectualValues (MATRIX_ROWS * MATRIX_COLS, 3);
    //effectualValues.reserve(MATRIX_ROWS * MATRIX_COLS);


    aligned_short_vector outputEffectualValues (MATRIX_ROWS * MATRIX_COLS, 4);
    //outputEffectualValues.reserve(MATRIX_ROWS * MATRIX_COLS);

    unsigned int numberOfWeightsPerRow = MATRIX_COLS;

    unsigned int CBPerRow =
    (unsigned int) std::ceil( (double) numberOfWeightsPerRow / (double) ENCODING_LENGTH);

    aligned_ushort_vector cbOffsets ((CBPerRow+1) * MATRIX_ROWS, 0);
    //cbOffsets.reserve((CBPerRow+1) * MATRIX_COLS);




    matrix_initialization(matrix);
    matrix_compression(matrix, numberOfWeightsPerRow, MATRIX_ROWS, CBPerRow,
    effectualValues, cbOffsets);

    /*
    for (unsigned short j=cbOffsets[98*(CBPerRow+1)]; j<cbOffsets[99*(CBPerRow+1)-1]; j++) {
    unsigned short valueAndZero = effectualValues[98*numberOfWeightsPerRow + (unsigned int) j];
    std::cout <<"("<<( (valueAndZero & WEIGHT_ZCOUNT_MASK) >> WEIGHT_ZCOUNT_BITOFFSET )
              <<" "
              <<( (valueAndZero & WEIGHT_MASK) >> WEIGHT_BITOFFSET )
              <<") ";
    }
    std::cout <<std::endl;
    */

        std::cout <<"Check the self-consistency of the generated matrix"<<std::endl;

    bool checkResult =
    check_matrix(
      effectualValues,
      cbOffsets,
      effectualValues,
      MATRIX_ROWS,
      MATRIX_COLS,
      CBPerRow
      );
    assert (checkResult);

    //Generate the instructions
    unsigned int numInstructions;
//    std::vector<t_instruction,
//            boost::alignment::aligned_allocator<t_instruction, aocl_utils_cpp::AOCL_ALIGNMENT>
//            > instructionVector;
    std::vector<t_instruction> instructionVector;
    for (unsigned int r=0; r<MATRIX_ROWS; r+=KERNEL_CACHE_LANES) {
            t_instruction instructionCollect =
                cmdGenCollectWeight(
                    0, //ddrWeightOffset
                    (unsigned short) r, //filterStart
                    (unsigned short) (std::min((unsigned short) KERNEL_CACHE_LANES, (unsigned short) (MATRIX_ROWS-r))), //numFiltersToCollect
                    (unsigned int) MATRIX_COLS //number of uncompressed weights in the filter
                );
            instructionVector.push_back(instructionCollect);
        for (unsigned int c=0; c<CBPerRow; c+= KERNEL_INDEX_CACHE_DEPTH) {
            t_instruction instructionFill =
                    cmdGenFillWeightBuffer(
                            0, //ddrIndexOffset
                            0, //ddrWeightOffset
                            (unsigned short) r, //Filter start
                            (unsigned char) std::min((unsigned char) KERNEL_CACHE_LANES, (unsigned char) (MATRIX_ROWS-r)), //numFilterToStream
                            (unsigned short) c, //cbStart
                            (unsigned short) (c + (unsigned int) std::min((unsigned int)KERNEL_INDEX_CACHE_DEPTH - 1, (unsigned int) CBPerRow - c) - 1), //cbEnd
                            CBPerRow,
                            MATRIX_COLS
                        );
            instructionVector.push_back(instructionFill);

            t_instruction instructionSwap = cmdGentWeightBufferSwap();
            instructionVector.push_back(instructionSwap);

            t_instruction instructionDrain =
                   cmdGenDrainWeightBuffer(
                        0, //laneStart
                        (unsigned char) std::min((unsigned char) KERNEL_CACHE_LANES, (unsigned char) (MATRIX_ROWS-r)), //laneEnd
                        0, //cbStart
                        (unsigned short) (std::min((unsigned int)KERNEL_INDEX_CACHE_DEPTH - 1, (unsigned int) CBPerRow - c) - 1) //cbEnd
                    );
            instructionVector.push_back(instructionDrain);
        }
    }

    numInstructions = instructionVector.size();

    try{
        cl_int status = CL_SUCCESS;
        //Platform
        cl::Platform clPlatform;

        //Device ID. Assumes that there is only one device.
        cl::Device clDevice;

        //Context
        cl::Context clContext;

        //Command queue used for data transfer
        cl::CommandQueue clDMAQueue, clSequencerQueue;
        std::vector<cl::CommandQueue> vecClCollectorQueues;
        for (unsigned int i=0; i < KERNEL_CACHE_LANES; i++) {
            vecClCollectorQueues.emplace_back();
        }

        //The sparse weight feeder DMA kernel
        cl::Kernel krnSpWDMA;

        //The sequencer kernel
        cl::Kernel krnSequencer;

        //The weight collector kernel
        std::vector<cl::Kernel> krnWeightCollectorVec;

        //Set up the OpenCL environment and create the kernels
        clInit(binaryFile,
               clPlatform,
               clContext,
               clDevice,
               clDMAQueue,
               clSequencerQueue,
               vecClCollectorQueues,
               krnSpWDMA,
               krnSequencer,
               krnWeightCollectorVec);

        //Create the buffers
        cl::Buffer inputSpWBuffer(
                    clContext,
                    CL_MEM_READ_WRITE,
                    MATRIX_ROWS * MATRIX_COLS * sizeof(typeof(effectualValues.at(0))),
                    NULL,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to create the input sparse weight buffer");

        cl::Buffer inputPointerBuffer(
                    clContext,
                    CL_MEM_READ_WRITE,
                    (CBPerRow+1) * MATRIX_ROWS * sizeof(typeof(cbOffsets.at(0))),
                    NULL,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to create the input block pointer buffer");

        cl::Buffer outputSpWBuffer(
                    clContext,
                    CL_MEM_READ_WRITE,
                    MATRIX_ROWS * MATRIX_COLS * sizeof(typeof(outputEffectualValues.at(0))),
                    NULL,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to create the output sparse weight buffer");

        cl::Buffer inputInstructionBuffer (
                    clContext,
                    CL_MEM_READ_WRITE,
                    instructionVector.size() * sizeof(t_instruction),
                    NULL,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to create the input sparse weight buffer");

        std::cout <<"Set up the kernel arguments"<<std::endl;
        status = krnSequencer.setArg(0, inputInstructionBuffer);
        status = krnSequencer.setArg(1, (cl_ushort) numInstructions);
        aocl_utils_cpp::checkError(status, "Failed to set up arguments for the instruction sequencer");

        for (auto iter=krnWeightCollectorVec.begin();
             iter < krnWeightCollectorVec.end();
             iter++) {
            status = iter->setArg(0, outputSpWBuffer);
        }
        aocl_utils_cpp::checkError(status, "Failed to set up arguments for the weight collector");

        status = krnSpWDMA.setArg(0, inputSpWBuffer);
        status = krnSpWDMA.setArg(1, inputPointerBuffer);
        aocl_utils_cpp::checkError(status, "Failed to set up arguments for the sparse weight DMA");

        std::cout <<"Transer data to the accelerator"<<std::endl;
        cl::Event inputTransferEvent, inputPointerTransferEvent, instructionTransferEvent;
        status = clSequencerQueue.enqueueWriteBuffer(
                    inputSpWBuffer,
                    CL_TRUE,
                    0,
                    sizeof(typeof(effectualValues.at(0))) * effectualValues.size(),
                    effectualValues.data(),
                    NULL,
                    &inputTransferEvent
                    );

        //CAUTION: Specail trick. Pollut the output region with all 4s, and see whether the FPGA pollute it.
        status = clSequencerQueue.enqueueWriteBuffer(outputSpWBuffer,
                                  CL_TRUE
                                  ,0
                                  ,sizeof(typeof(outputEffectualValues.at(0))) * outputEffectualValues.size()
                                  , outputEffectualValues.data()
                                  );

        aocl_utils_cpp::checkError(status, "Failed to transfer sparse weights to the accelerator");
        status = clSequencerQueue.enqueueWriteBuffer(
                    inputPointerBuffer,
                    CL_TRUE,
                    0,
                    sizeof(typeof(cbOffsets.at(0))) * cbOffsets.size(),
                    cbOffsets.data(),
                    NULL,
                    &inputPointerTransferEvent
                    );
        aocl_utils_cpp::checkError(status, "Failed to transfer weight pointers to the accelerator");
        status = clSequencerQueue.enqueueWriteBuffer(
                    inputInstructionBuffer,
                    CL_TRUE,
                    0,
                    sizeof(typeof(instructionVector.at(0))) * instructionVector.size(),
                    instructionVector.data(),
                    NULL,
                    &instructionTransferEvent
                    );
        clSequencerQueue.finish();
        std::cout <<"Number of instruction is "<<instructionVector.size()<<std::endl;
        aocl_utils_cpp::checkError(status, "Failed to transfer instructions to the accelerator");

        std::vector<cl::Event> transferEvents{
            inputTransferEvent, inputPointerTransferEvent, inputPointerTransferEvent};

        std::cout <<"Launching the kernels"<<std::endl;
        for (unsigned int i = 0;
             i < krnWeightCollectorVec.size();
             i ++) {
            status = vecClCollectorQueues.at(i).enqueueTask(
                        krnWeightCollectorVec.at(i)
                        );
        }
        status = clDMAQueue.enqueueTask(krnSpWDMA);
        status = clSequencerQueue.enqueueTask(krnSequencer);
        aocl_utils_cpp::checkError(status, "Failed to launch at least one kernel");
        //usleep (10000000);
        std::cout <<"Wait for result to be transferred back"<<std::endl;
        clSequencerQueue.finish();
        status = clSequencerQueue.enqueueReadBuffer(outputSpWBuffer,
                                  CL_TRUE
                                  ,0
                                  ,sizeof(typeof(outputEffectualValues.at(0))) * outputEffectualValues.size()
                                  , outputEffectualValues.data()
                                  );
       clSequencerQueue.finish();
        aocl_utils_cpp::checkError(status, "Failed to read the results back");
      }
    catch (const std::runtime_error & e) {
        std::cout <<e.what()<<std::endl;
        return 1;
    }
    catch (...) {
        std::cout <<"Unspecified error occured!"<<std::endl;
        return 1;
    }

    std::cout <<"Checking the output matrix"<<std::endl;
    checkResult =
        check_matrix(
          effectualValues,
          cbOffsets,
          outputEffectualValues,
          MATRIX_ROWS,
          MATRIX_COLS,
          CBPerRow
          );

    if (!checkResult) {
        std::cout <<"FAILED: Values do not match"<<std::endl;
        std::cout <<"Content of the output buffer"<<std::endl;
        unsigned int rowIter=0, colIter=0;
        for (auto value : outputEffectualValues) {
            if (colIter == 0) {
                std::cout <<"[Row "<<rowIter<<"]: ";
            }
            std::cout <<value<<" ";
            colIter++;
            if (colIter == MATRIX_COLS) {
                colIter = 0;
                rowIter++;
                std::cout<<std::endl;
            }
        }
    }
    else {
        std::cout <<"SUCCESS!"<<std::endl;
    }


    return 0;
}

void matrix_initialization ( std::vector<char> & _matrix)
{
  std::mt19937 generator (SEED);
  std::bernoulli_distribution distribution (BERN_P);
  for (unsigned int i=0; i<MATRIX_ROWS; i++){
    for (unsigned int j=0; j<MATRIX_COLS; j++) {
//      if (j==i) {
//        _matrix.push_back((char) 1);
//      }
//      else {
//        _matrix.push_back(0);
//      }
        bool result = distribution(generator);
        unsigned char number = result ? 1 : 0;
        _matrix.push_back(number);
    }
  }
}

void matrix_compression (std::vector<char> & _matrix,
    unsigned int numberOfWeightsPerRow,
    unsigned int numberOfRows,
    unsigned int cbPerRow,
    aligned_short_vector &outEffectualValues,
    aligned_ushort_vector &outCBOffets
  )
{
  
  for (unsigned int iRow=0; iRow < numberOfRows; iRow++){
    unsigned short jCol=0;
    unsigned int jCBIter = 0;
    unsigned int effectualValueIdx = iRow * numberOfWeightsPerRow;
    unsigned short effectualValueColIdx = 0;
    unsigned char zeroCount=0;
    unsigned int CBOffsetIdx = iRow * (cbPerRow + 1);
    bool isFirst = true;

    while (jCol < numberOfWeightsPerRow) {
      char value = _matrix[iRow * numberOfWeightsPerRow + jCol];
      if (value != 0 
          || zeroCount == WEIGHT_ZCOUNT_MAX 
          || jCol == (numberOfWeightsPerRow-1) 
          || jCBIter == ENCODING_LENGTH - 1) {
        //Four cases for preserving the value
        //1) it is not 0
        //2) the consecutive number of preceeding zero has reached the max
        //3) it is the last value in a filter strip
        //4) it is the last value in an encoding block
        unsigned short effectualWeightAndZCount
          = ((unsigned short) (zeroCount & 0X0F) << WEIGHT_ZCOUNT_BITOFFSET)
          | (( (unsigned short) (value & WEIGHT_MASK) ) << WEIGHT_BITOFFSET);

        outEffectualValues[effectualValueIdx] = effectualWeightAndZCount;

        //Insert a new element to the offset array if this is the first
        //effectual element in the encoding block.
        if (isFirst) {
          outCBOffets[CBOffsetIdx] = effectualValueColIdx;
          //std::cout <<"effectualValueColIdx: "<<effectualValueColIdx<<std::endl;

          CBOffsetIdx++;
          isFirst=false;
        }

        effectualValueIdx++;
        effectualValueColIdx++;
        zeroCount=0;
      }
      else {
        zeroCount++;
      }

      jCol++;  
      jCBIter++;
      jCBIter =  (jCBIter == ENCODING_LENGTH) ? 0 : jCBIter;
      isFirst = (jCBIter == 0) ? true:isFirst;
    } //while

    //CAUTION: Add the extra encodiing index block at the end of the row
    outCBOffets[CBOffsetIdx] = effectualValueColIdx;
    CBOffsetIdx++;

    assert(CBOffsetIdx == (iRow+1) * (cbPerRow+1));

  } //for
}

bool check_matrix (aligned_short_vector & originalMatrix,
      aligned_ushort_vector & originalIndex,
      aligned_short_vector &outputMatrix,
      unsigned int numberOfRow,
      unsigned int numberOfWeightsPerRow,
      unsigned int cbPerRow
  )
{
   bool result = true;
   for (unsigned int iterRow = 0;
        iterRow < numberOfRow;
        iterRow++) {
    unsigned int beginOffset = originalIndex[iterRow*(cbPerRow+1)];
    unsigned int endOffset = originalIndex[iterRow*(cbPerRow+1) + cbPerRow];

    unsigned int iterCSRColIndex = 0;
    for (unsigned int weightAddress = numberOfWeightsPerRow * iterRow + beginOffset;
           weightAddress < numberOfWeightsPerRow * iterRow + endOffset;
           weightAddress++, iterCSRColIndex++
        ) {
      if ( (originalMatrix[weightAddress] & WEIGHT_MASK) >> WEIGHT_BITOFFSET
              != (outputMatrix[weightAddress] & WEIGHT_MASK) ) {
        std::cout <<"Mismatch detected! Row is "<<iterRow<<std::endl;
        std::cout <<"Index of the first non-matching element in the compressed row is "<<iterCSRColIndex<<std::endl;
        std::cout <<"Displaying the mismatched filter."<<std::endl;
        std::cout <<"The expected filter row: "<<std::endl;
        unsigned int numEffectualWeights = 0;
        std::cout <<"["<<numEffectualWeights<<"] ";
        for (unsigned int weightAddress = numberOfWeightsPerRow * iterRow + beginOffset;
               weightAddress < numberOfWeightsPerRow * iterRow + endOffset;
               weightAddress++
             ){
            std::cout << ((originalMatrix[weightAddress] & WEIGHT_MASK) >> WEIGHT_BITOFFSET )<<" ";
            numEffectualWeights++;

            if (numEffectualWeights % 50 == 0) {
                std::cout <<std::endl<<"["<<numEffectualWeights<<"] ";
            }
         }
        std::cout << std::endl;
        std::cout <<"Number of expected effectual weights is "<<numEffectualWeights<<std::endl;

        numEffectualWeights = 0;
        std::cout <<"["<<numEffectualWeights<<"] ";
        std::cout <<"The actual output"<<std::endl;
        for (unsigned int weightAddress = numberOfWeightsPerRow * iterRow + beginOffset;
               weightAddress < numberOfWeightsPerRow * iterRow + endOffset;
               weightAddress++
             ) {
            std::cout <<( (outputMatrix[weightAddress] & WEIGHT_MASK) )<<" ";
            numEffectualWeights++;

            if (numEffectualWeights % 50 == 0) {
                std::cout <<std::endl<<"["<<numEffectualWeights<<"] ";
            }
         }
        std::cout << std::endl;
        result = false;
        break;
      }
    }

   }

   return result;
}

cl_int clInit (const std::string binaryFile,
               cl::Platform &clPlatform,
               cl::Context & clContext,
               cl::Device & clDevice,
               cl::CommandQueue & clDMAQueue,
               cl::CommandQueue & clSequencerQueue,
               std::vector<cl::CommandQueue> & clCollectorQueue,
               cl::Kernel & krnSpWDMA,
               cl::Kernel & krnSequencer,
               std::vector<cl::Kernel> & krnWeightCollectorVec)
{
    cl_int status = CL_SUCCESS;
    clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    std::vector<cl::Device> devices;
    status = clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    aocl_utils_cpp::checkError(status, "Failed to query the devices");

    std::cout <<"Selecting the device[0]"<<std::endl;
    clDevice = devices[0];
    clContext = cl::Context({devices[0]}
                            ,NULL
                            ,&aocl_utils_cpp::oclContextCallback
                            ,NULL
                            ,&status);
    aocl_utils_cpp::checkError(status, "Failed to create context");

    clDMAQueue = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );

    clSequencerQueue = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );

    for (auto iter = clCollectorQueue.begin();
         iter != clCollectorQueue.end();
         iter++) {
        *iter = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
    }
    aocl_utils_cpp::checkError(status, "Failed to create at least one command queue");

    std::cout <<"Using AOCX: "<<binaryFile<<std::endl;
    cl::Program program = aocl_utils_cpp::createProgramFromBinary(
                clContext,
                binaryFile.c_str(),
                {clDevice}
                );
    status = program.build({clDevice});
    aocl_utils_cpp::checkError(status, "Failed to build program");

    //Instantiate the kernels
    krnSequencer = cl::Kernel(program, "kernelSequencer", &status);
    aocl_utils_cpp::checkError(status, "Failed to create the squencer kernel");

    krnSpWDMA = cl::Kernel(program, "kernelSparseWeightDMA", &status);
    aocl_utils_cpp::checkError(status, "Failed to create the sparse weight DMA kernel");

    for (unsigned int i=0; i<KERNEL_CACHE_LANES; i++){
        std::string kernelName = "kernelWeightCollector"+std::to_string(i);
        krnWeightCollectorVec.push_back(
                    cl::Kernel(program, kernelName.c_str(), &status)
                    );
    }
    aocl_utils_cpp::checkError(status, "Failed to create at least one of the weight collector kernel");

    return status;
}

t_instruction cmdGenFillWeightBuffer(
        unsigned int ddrIndexOffset,
        unsigned int ddrWeightOffset,
        unsigned short filterStart,
        unsigned char numFilterToStream,
        unsigned short cbStart,
        unsigned short cbEnd,
        unsigned short numCbInFilter,
        unsigned int nWInFilter
        ){
    t_instruction instruction;
    instruction.header = OPCODE_FILL_WEIGHT_BUFFER;
    instruction.instructionSizeBytes = 21;

    //Generate the dependeny list
    //Should wait for the previous command of the same type to finish
    //Should wait for all previous swap to finish
    unsigned short dependency =
            (1 << (OPCODE_FILL_WEIGHT_BUFFER)) | (1 << OPCODE_SWAP_WEIGHT_BUFFER);
    instruction.dependencyList[0] = (unsigned char) (dependency & 0x0FF);
    instruction.dependencyList[1] = (unsigned char) ((dependency & 0X0FF00) >> 8);

    //Generate the instruction themselves
    instruction.words[0] = (unsigned char) (ddrIndexOffset & 0xFF);
    instruction.words[1] = (unsigned char) ( (ddrIndexOffset & 0xFF00) >> 8);
    instruction.words[2] = (unsigned char) ( (ddrIndexOffset & 0xFF0000) >> 16);
    instruction.words[3] = (unsigned char) ( (ddrIndexOffset & 0xFF000000) >> 24);

    instruction.words[4] = (unsigned char) (ddrWeightOffset & 0xFF);
    instruction.words[5] = (unsigned char) ( (ddrWeightOffset & 0xFF00) >> 8);
    instruction.words[6] = (unsigned char) ( (ddrWeightOffset & 0xFF0000) >> 16);
    instruction.words[7] = (unsigned char) ( (ddrWeightOffset & 0xFF000000) >> 24);

    instruction.words[8] = (unsigned char) ( filterStart & 0xFF );
    instruction.words[9] = (unsigned char) ( (filterStart & 0xFF00) >> 8 );

    instruction.words[10] = (unsigned char) numFilterToStream;

    instruction.words[11] = (unsigned char) (cbStart & 0xFF);
    instruction.words[12] = (unsigned char) ( (cbStart & 0xFF00) >> 8);

    instruction.words[13] = (unsigned char) (cbEnd & 0xFF);
    instruction.words[14] = (unsigned char) ( (cbEnd & 0xFF00) >> 8);

    instruction.words[15] = (unsigned char) (numCbInFilter & 0xFF);
    instruction.words[16] = (unsigned char) ( (numCbInFilter & 0xFF00) >> 8);

    instruction.words[17] = (unsigned char) (nWInFilter & 0xFF);
    instruction.words[18] = (unsigned char) ( (nWInFilter & 0xFF00) >> 8);
    instruction.words[19] = (unsigned char) ( (nWInFilter & 0xFF0000) >> 16);
    instruction.words[20] = (unsigned char) ( (nWInFilter & 0xFF000000) >> 24);

    return instruction;
}

t_instruction cmdGenDrainWeightBuffer(
        unsigned char laneStart,
        unsigned char laneEnd,
        unsigned short cbStart,
        unsigned short cbEnd
        ){
    t_instruction instruction;
    instruction.header = OPCODE_DRAIN_WEIGHT_BUFFER;
    instruction.instructionSizeBytes = 6;

    //Generate the dependeny list
    //Should wait for the previous command of the same type to finish
    //Should wait for all previous swap to finish
    unsigned short dependency =
            (1 << (OPCODE_DRAIN_WEIGHT_BUFFER)) | (1 << OPCODE_SWAP_WEIGHT_BUFFER);
    instruction.dependencyList[0] = (unsigned char) (dependency & 0xFF);
    instruction.dependencyList[1] = (unsigned char) ((dependency & 0XFF00) >> 8);

    //Generate the instruction themselves
    instruction.words[0] = (unsigned char) (laneStart & 0x0FF);

    instruction.words[1] = (unsigned char) ( (laneEnd & 0x0FF) );


    instruction.words[2] = (unsigned char) (cbStart & 0x0FF);
    instruction.words[3] = (unsigned char) ((cbStart & 0x0FF00) >> 8);

    instruction.words[4] = (unsigned char) (cbEnd & 0x0FF);
    instruction.words[5] = (unsigned char) ((cbEnd & 0x0FF00) >> 8);

    return instruction;
}

t_instruction cmdGenCollectWeight(
        unsigned int ddrWeightOffset,
        unsigned short filterStart,
        unsigned short numFiltersToCollect,
        unsigned int numWeightInFilter
        ){
    t_instruction instruction;
    instruction.header = OPCODE_COLLECT_WEIGHT;
    instruction.instructionSizeBytes = 11;

    //Generate the dependeny list
    //Should wait for the previous command of the same type to finish
    unsigned short dependency =
            (1 << (OPCODE_COLLECT_WEIGHT));
    instruction.dependencyList[0] = (unsigned char) (dependency & 0x0FF);
    instruction.dependencyList[1] = (unsigned char) ((dependency & 0X0FF00) >> 8);

    //Generate the instruction themselves
    instruction.words[0] = (unsigned char) (ddrWeightOffset & 0x0FF);
    instruction.words[1] = (unsigned char) ( (ddrWeightOffset & 0x0FF00) >> 8);
    instruction.words[2] = (unsigned char) ( (ddrWeightOffset & 0x0FF0000) >> 16);
    instruction.words[3] = (unsigned char) ( (ddrWeightOffset & 0x0FF000000) >> 24);

    instruction.words[4] = (unsigned char) ( filterStart & 0x0FF );
    instruction.words[5] = (unsigned char) ( (filterStart & 0x0FF00) >> 8 );

    instruction.words[6] = (unsigned char) ( numFiltersToCollect & 0x0FF );
    instruction.words[7] = (unsigned char) ( (numFiltersToCollect & 0x0FF00) >> 8 );


    instruction.words[8] = (unsigned char) (numWeightInFilter & 0x0FF);
    instruction.words[9] = (unsigned char) ( (numWeightInFilter & 0x0FF00) >> 8);
    instruction.words[10] = (unsigned char) ( (numWeightInFilter & 0x0FF0000) >> 16);

    return instruction;
}

t_instruction cmdGentWeightBufferSwap(){
    t_instruction instruction;
    instruction.header = OPCODE_SWAP_WEIGHT_BUFFER;
    instruction.instructionSizeBytes = 1;

    //Generate the dependeny list
    //Should wait for the previous command of the same type to finish
    unsigned short dependency =
            (1 << (OPCODE_DRAIN_WEIGHT_BUFFER)) | (1 << OPCODE_FILL_WEIGHT_BUFFER);
    instruction.dependencyList[0] = (unsigned char) (dependency & 0x0FF);
    instruction.dependencyList[1] = (unsigned char) ((dependency & 0X0FF00) >> 8);

    return instruction;
}
