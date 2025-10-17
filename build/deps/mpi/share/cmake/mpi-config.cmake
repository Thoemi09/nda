# This file allows other CMake Projects to find us
# We provide general project information
# and reestablish the exported CMake Targets

# Multiple inclusion guard
if(NOT mpi_FOUND)
set(mpi_FOUND TRUE)
set_property(GLOBAL PROPERTY mpi_FOUND TRUE)

# version
set(mpi_VERSION 1.3.0 CACHE STRING "mpi version")
set(mpi_GIT_HASH f9348d2ae723998a6cc81e4ab13b6eef47cc894b CACHE STRING "mpi git hash")

# Root of the installation
set(mpi_ROOT /tmp/install CACHE STRING "mpi root directory")

# Find the target dependencies
function(find_dep)
  get_property(${ARGV0}_FOUND GLOBAL PROPERTY ${ARGV0}_FOUND)
  if(NOT ${ARGV0}_FOUND)
    find_package(${ARGN} REQUIRED HINTS /tmp/install)
  endif()
endfunction()
find_dep(itertools 1.0)

# Include the exported targets of this project
include(/tmp/install/lib/cmake/mpi/mpi-targets.cmake)

message(STATUS "Found mpi-config.cmake with version 1.3.0, hash = f9348d2ae723998a6cc81e4ab13b6eef47cc894b, root = /tmp/install")

# Was the Project built with Documentation?
set(mpi_WITH_DOCUMENTATION OFF CACHE BOOL "Was mpi build with documentation?")

# MPIEXEC Variables
set(MPIEXEC_EXECUTABLE /usr/bin/mpiexec CACHE STRING "Executable for running MPI programs.")
set(MPIEXEC_NUMPROC_FLAG -n CACHE STRING "Flag to pass to mpiexec before giving it the number of processors to run on.")
set(MPIEXEC_MAX_NUMPROCS 2 CACHE STRING "Flag to pass to mpiexec to set the number of mpi ranks to run on.")
set(MPIEXEC_PREFLAGS --oversubscribe CACHE STRING "Flags to pass to mpiexec directly before the executable to run.")
set(MPIEXEC_POSTFLAGS  CACHE STRING "Flags to pass to mpiexec after other flags.")

endif()
