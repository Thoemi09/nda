# This file allows other CMake Projects to find us
# We provide general project information
# and reestablish the exported CMake Targets

# Multiple inclusion guard
if(NOT itertools_FOUND)
set(itertools_FOUND TRUE)
set_property(GLOBAL PROPERTY itertools_FOUND TRUE)

# version
set(itertools_VERSION 1.3.0 CACHE STRING "itertools version")
set(itertools_GIT_HASH aafd6771b56718030571590fb9a1fde7c137625f CACHE STRING "itertools git hash")

# Root of the installation
set(itertools_ROOT /tmp/install CACHE STRING "itertools root directory")

## Find the target dependencies
#function(find_dep)
#  get_property(${ARGV0}_FOUND GLOBAL PROPERTY ${ARGV0}_FOUND)
#  if(NOT ${ARGV0}_FOUND)
#    find_package(${ARGN} REQUIRED HINTS /tmp/install)
#  endif()
#endfunction()
#find_dep(depname 1.0)

# Include the exported targets of this project
include(/tmp/install/lib/cmake/itertools/itertools-targets.cmake)

message(STATUS "Found itertools-config.cmake with version 1.3.0, hash = aafd6771b56718030571590fb9a1fde7c137625f, root = /tmp/install")

# Was the Project built with Documentation?
set(itertools_WITH_DOCUMENTATION OFF CACHE BOOL "Was itertools build with documentation?")

endif()
