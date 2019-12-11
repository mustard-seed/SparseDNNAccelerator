// Copyright (C) 2013-2017 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"
#include <algorithm>
#include <stdarg.h>
#include <fstream>
#include <exception>
#include "boost/align/align.hpp"

#ifdef _WIN32 // Windows
#include <windows.h>
#else         // Linux
#include <stdio.h> 
#include <unistd.h> // readlink, chdir
#endif

namespace aocl_utils_cpp {

static const char *const VERSION_STR = "170";

//////////////////////////////////////////
// Host allocation functions for alignment
//////////////////////////////////////////

#ifdef _WIN32 // Windows
void *alignedMalloc(size_t size) {
  return _aligned_malloc (size, AOCL_ALIGNMENT);
}

void alignedFree(void * ptr) {
  _aligned_free(ptr);
}
#else          // Linux
void *alignedMalloc(size_t size) {
  void *result = NULL;
  int rc;
  rc = posix_memalign (&result, AOCL_ALIGNMENT, size);
  return result;
}

void alignedFree(void * ptr) {
  free (ptr);
}
#endif

///////////////////////////////
// Error functions
///////////////////////////////

// Print the error associciated with an error code
void printError(cl_int error) {
  // Print error message
  switch(error)
  {
    case -1:
      printf("CL_DEVICE_NOT_FOUND ");
      break;
    case -2:
      printf("CL_DEVICE_NOT_AVAILABLE ");
      break;
    case -3:
      printf("CL_COMPILER_NOT_AVAILABLE ");
      break;
    case -4:
      printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
      break;
    case -5:
      printf("CL_OUT_OF_RESOURCES ");
      break;
    case -6:
      printf("CL_OUT_OF_HOST_MEMORY ");
      break;
    case -7:
      printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
      break;
    case -8:
      printf("CL_MEM_COPY_OVERLAP ");
      break;
    case -9:
      printf("CL_IMAGE_FORMAT_MISMATCH ");
      break;
    case -10:
      printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
      break;
    case -11:
      printf("CL_BUILD_PROGRAM_FAILURE ");
      break;
    case -12:
      printf("CL_MAP_FAILURE ");
      break;
    case -13:
      printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
      break;
    case -14:
      printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
      break;

    case -30:
      printf("CL_INVALID_VALUE ");
      break;
    case -31:
      printf("CL_INVALID_DEVICE_TYPE ");
      break;
    case -32:
      printf("CL_INVALID_PLATFORM ");
      break;
    case -33:
      printf("CL_INVALID_DEVICE ");
      break;
    case -34:
      printf("CL_INVALID_CONTEXT ");
      break;
    case -35:
      printf("CL_INVALID_QUEUE_PROPERTIES ");
      break;
    case -36:
      printf("CL_INVALID_COMMAND_QUEUE ");
      break;
    case -37:
      printf("CL_INVALID_HOST_PTR ");
      break;
    case -38:
      printf("CL_INVALID_MEM_OBJECT ");
      break;
    case -39:
      printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
      break;
    case -40:
      printf("CL_INVALID_IMAGE_SIZE ");
      break;
    case -41:
      printf("CL_INVALID_SAMPLER ");
      break;
    case -42:
      printf("CL_INVALID_BINARY ");
      break;
    case -43:
      printf("CL_INVALID_BUILD_OPTIONS ");
      break;
    case -44:
      printf("CL_INVALID_PROGRAM ");
      break;
    case -45:
      printf("CL_INVALID_PROGRAM_EXECUTABLE ");
      break;
    case -46:
      printf("CL_INVALID_KERNEL_NAME ");
      break;
    case -47:
      printf("CL_INVALID_KERNEL_DEFINITION ");
      break;
    case -48:
      printf("CL_INVALID_KERNEL ");
      break;
    case -49:
      printf("CL_INVALID_ARG_INDEX ");
      break;
    case -50:
      printf("CL_INVALID_ARG_VALUE ");
      break;
    case -51:
      printf("CL_INVALID_ARG_SIZE ");
      break;
    case -52:
      printf("CL_INVALID_KERNEL_ARGS ");
      break;
    case -53:
      printf("CL_INVALID_WORK_DIMENSION ");
      break;
    case -54:
      printf("CL_INVALID_WORK_GROUP_SIZE ");
      break;
    case -55:
      printf("CL_INVALID_WORK_ITEM_SIZE ");
      break;
    case -56:
      printf("CL_INVALID_GLOBAL_OFFSET ");
      break;
    case -57:
      printf("CL_INVALID_EVENT_WAIT_LIST ");
      break;
    case -58:
      printf("CL_INVALID_EVENT ");
      break;
    case -59:
      printf("CL_INVALID_OPERATION ");
      break;
    case -60:
      printf("CL_INVALID_GL_OBJECT ");
      break;
    case -61:
      printf("CL_INVALID_BUFFER_SIZE ");
      break;
    case -62:
      printf("CL_INVALID_MIP_LEVEL ");
      break;
    case -63:
      printf("CL_INVALID_GLOBAL_WORK_SIZE ");
      break;
    default:
      printf("UNRECOGNIZED ERROR CODE (%d)", error);
  }
}

// Print line, file name, and error code if there is an error. Exits the
// application upon error.
void _checkError(int line,
                 const char *file,
                 cl_int error,
                 const char *msg,
                 ...) {
  // If not successful
  if(error != CL_SUCCESS) {
    // Print line and file
    printf("ERROR: ");
    printError(error);
    printf("\nLocation: %s:%d\n", file, line);

    // Print custom message.
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    throw std::runtime_error("OpenCL error occured");

    // Cleanup and bail.
    //cleanup();
    //exit(error);
  }
}

// Sets the current working directory to be the same as the directory
// containing the running executable.
bool setCwdToExeDir() {
#ifdef _WIN32 // Windows
  HMODULE hMod = GetModuleHandle(NULL);
  char path[MAX_PATH];
  GetModuleFileNameA(hMod, path, MAX_PATH);

#else         // Linux
  // Get path of executable.
  char path[300];
  ssize_t n = readlink("/proc/self/exe", path, sizeof(path)/sizeof(path[0]) - 1);
  if(n == -1) {
    return false;
  }
  path[n] = 0;
#endif

  // Find the last '\' or '/' and terminate the path there; it is now
  // the directory containing the executable.
  size_t i;
  for(i = strlen(path) - 1; i > 0 && path[i] != '/' && path[i] != '\\'; --i);
  path[i] = '\0';

  // Change the current directory.
#ifdef _WIN32 // Windows
  SetCurrentDirectoryA(path);
#else         // Linux
  int rc;
  rc = chdir(path);
#endif

  return true;
}

// Searches all platforms for the first platform whose name
// contains the search string (case-insensitive).
cl::Platform findPlatform(const std::string platform_name_search) {
  cl_int status;

  std::string search = platform_name_search;
  std::transform(platform_name_search.begin()
                 ,platform_name_search.end()
                 ,search.begin()
                 ,tolower);

  //Get a list of all platform ids
  std::vector<cl::Platform> platforms;
  status = cl::Platform::get(&platforms);

  checkError(status, "Query for the list of platforms failed");

  // For each platform, get name and compare against the search string.
  for(auto iterP : platforms) {
    std::string name;
    status = iterP.getInfo(CL_PLATFORM_NAME, & name);
    checkError(status, "Query for platform name failed");

    // Convert to lower case.
    std::transform(name.begin(), name.end(), name.begin(), tolower);

    if(name.find(search) != std::string::npos) {
      // Found!
      return iterP;
    }
  }

  // No platform found.
  cl::Platform empty;
  return empty;
}

// Create a program for all devices associated with the context.
cl::Program createProgramFromBinary(
        cl::Context context, const char *binary_file_name, std::vector<cl::Device> devices) {
  // Early exit for potentially the most common way to fail: AOCX does not exist.
  if(!fileExists(binary_file_name)) {
    printf("AOCX file '%s' does not exist.\n", binary_file_name);
    checkError(CL_INVALID_PROGRAM, "Failed to load binary file");
  }

  unsigned int num_devices = devices.size();

  // Load the binary.
  ::std::ifstream aocx_stream(binary_file_name, std::ios::in|std::ios::binary);
  if(!aocx_stream.is_open()) {
    checkError(CL_INVALID_PROGRAM, "Failed to load binary file");
  }

  ::std::string prog(std::istreambuf_iterator<char>(aocx_stream),
                   (std::istreambuf_iterator<char>()));

  cl::Program::Binaries binaries(num_devices, ::std::make_pair(prog.::std::string::c_str(), prog.length()+1));

  cl_int status;
  ::std::vector<cl_int> binary_status;

  cl::Program program = cl::Program(context, devices, binaries, &binary_status, &status);
  checkError(status, "Failed to create program with binary");
  for(unsigned i = 0; i < binary_status.size(); ++i) {
    checkError(binary_status[i], "Failed to load binary for device");
  }

  return program;
}

bool fileExists(const char *file_name) {
#ifdef _WIN32 // Windows
  DWORD attrib = GetFileAttributesA(file_name);
  return (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY));
#else         // Linux
  return access(file_name, R_OK) != -1;
#endif
}

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

cl_ulong getStartEndTime(cl_event event) {
  cl_int status;

  cl_ulong start, end;
  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
  checkError(status, "Failed to query event start time");
  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
  checkError(status, "Failed to query event end time");

  return end - start;
}

cl_ulong getStartEndTime(cl_event *events, unsigned num_events) {
  cl_int status;

  cl_ulong min_start = 0;
  cl_ulong max_end = 0;
  for(unsigned i = 0; i < num_events; ++i) {
    cl_ulong start, end;
    status = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    checkError(status, "Failed to query event start time");
    status = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    checkError(status, "Failed to query event end time");

    if(i == 0) {
      min_start = start;
      max_end = end;
    }
    else {
      if(start < min_start) {
        min_start = start;
      }
      if(end > max_end) {
        max_end = end;
      }
    }
  }

  return max_end - min_start;
}

void waitMilliseconds(unsigned ms) {
#ifdef _WIN32 // Windows
  Sleep(ms);
#else         // Linux
  timespec sleeptime = {0, 0};
  sleeptime.tv_sec = ms / 1000;
  sleeptime.tv_nsec = long(ms % 1000) * 1000000L;  // convert to nanoseconds
  nanosleep(&sleeptime, NULL);
#endif
}

void oclContextCallback(const char *errinfo, const void *, size_t, void *) {
  printf("Context callback: %s\n", errinfo);
}

} // ns aocl_utils_cpp

