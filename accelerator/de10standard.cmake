# this one is important
SET(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
#this one not so much
SET(CMAKE_SYSTEM_VERSION 1)

# specify the cross compiler
SET(tools /home/jamesliu/intelFPGA/18.0/embedded/ds-5/sw/gcc/bin)
SET(CMAKE_C_COMPILER   ${tools}/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER ${tools}/arm-linux-gnueabihf-g++)

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

