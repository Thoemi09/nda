# This file allows other CMake Projects to find us
# We provide general project information
# and reestablish the exported CMake Targets

# Multiple inclusion guard
if(NOT nda_FOUND)
set(nda_FOUND TRUE)
set_property(GLOBAL PROPERTY nda_FOUND TRUE)

# version
set(nda_VERSION 1.3.0 CACHE STRING "nda version")
set(nda_GIT_HASH 7a9c653a032c1b25ff911497ddac5765c7adeb27 CACHE STRING "nda git hash")

# Root of the installation
set(nda_ROOT /tmp/install CACHE STRING "nda root directory")

# Find OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Find CUDAToolkit
if(OFF)
  find_package(CUDAToolkit REQUIRED)
endif()

# Disable the by default enabled fast math functions for intel compilers
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS "-fno-fast-math ${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)
  message(STATUS "-------- Intel Compiler Warning --------")
  message(STATUS "By default, we disable -ffast-math for Intel compilers")
  message(STATUS "If you want to enable it, add -ffast-math to CMAKE_CXX_FLAGS or to the CXXFLAGS")
endif()

## Find the target dependencies
function(find_dep)
  get_property(${ARGV0}_FOUND GLOBAL PROPERTY ${ARGV0}_FOUND)
  if(NOT ${ARGV0}_FOUND)
    find_package(${ARGN} REQUIRED HINTS /tmp/install)
  endif()
endfunction()
find_dep(itertools 1.2)
find_dep(mpi 1.2)
find_dep(h5 1.2)
if(OFF)
  find_dep(Cpp2Py 2.0)
endif()

# Include the exported targets of this project
include(/tmp/install/lib/cmake/nda/nda-targets.cmake)

message(STATUS "Found nda-config.cmake with version 1.3.0, hash = 7a9c653a032c1b25ff911497ddac5765c7adeb27, root = /tmp/install")

# Was the Project built with Documentation?
set(nda_WITH_DOCUMENTATION OFF CACHE BOOL "Was nda build with documentation?")

# Was the Project built with PythonSupport?
set(nda_WITH_PYTHON_SUPPORT OFF CACHE BOOL "Was nda build with python support?")
if(OFF)
  set(nda_MODULE_DIR /tmp/install/ CACHE BOOL "The nda python module directory")
endif()

# Was the Project built with CudaSupport?
set(nda_WITH_CUDA_SUPPORT OFF CACHE BOOL "Was nda build with Cuda support?")

# Was the Project built with HDF5Support?
set(nda_WITH_HDF5_SUPPORT ON CACHE BOOL "Was nda build with HDF5 support?")

# Was the Project built with MPISupport?
set(nda_WITH_MPI_SUPPORT ON CACHE BOOL "Was nda build with MPI support?")

# Was the Project built with OpenMPSupport?
set(nda_WITH_OPENMP_SUPPORT ON CACHE BOOL "Was nda build with OpenMP support?")

endif()
