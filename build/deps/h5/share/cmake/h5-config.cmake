# This file allows other CMake Projects to find us
# We provide general project information
# and reestablish the exported CMake Targets

# Multiple inclusion guard
if(NOT h5_FOUND)
set(h5_FOUND TRUE)
set_property(GLOBAL PROPERTY h5_FOUND TRUE)

# version
set(h5_VERSION 1.3.0 CACHE STRING "h5 version")
set(h5_GIT_HASH c650e2a032fafbd6a04a2f3a282005287d565c64 CACHE STRING "h5 git hash")

# Root of the installation
set(h5_ROOT /tmp/install CACHE STRING "h5 root directory")

## Find the target dependencies
#function(find_dep)
#  get_property(${ARGV0}_FOUND GLOBAL PROPERTY ${ARGV0}_FOUND)
#  if(NOT ${ARGV0}_FOUND)
#    find_package(${ARGN} REQUIRED HINTS /tmp/install)
#  endif()
#endfunction()
#find_dep(depname 1.0)

# cmake versions prior to 3.26 require C language enabled
# c.f. https://gitlab.kitware.com/cmake/cmake/-/issues/24241
if(CMAKE_VERSION VERSION_LESS "3.26")
  enable_language(C)
endif()

# Find hdf5 dependency
find_package(HDF5 REQUIRED C HL)

# Include the exported targets of this project
include(/tmp/install/lib/cmake/h5/h5-targets.cmake)

message(STATUS "Found h5-config.cmake with version 1.3.0, hash = c650e2a032fafbd6a04a2f3a282005287d565c64, root = /tmp/install")

# Was the Project built with Documentation?
set(h5_WITH_DOCUMENTATION OFF CACHE BOOL "Was h5 build with documentation?")

# Was the Project built with PythonSupport?
set(h5_WITH_PYTHON_SUPPORT OFF CACHE BOOL "Was h5 build with python support?")
if(OFF)
  set(h5_MODULE_DIR /tmp/install/ CACHE BOOL "The h5 python module directory")
endif()

endif()
