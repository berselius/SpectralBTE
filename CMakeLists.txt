cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

enable_testing()

# set Release build type as default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)"
    FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

project(SpectralBTE CXX C)

# don't allow building in the source directory
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed.")
endif()

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

# enable C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

# generate debugging symbols for release and debug
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

# explicitly set DEBUG flag in Debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")

# to make shared libraries we need position independent code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# in this variable we collect all the external libraries that we need to link
# against, add math library
set(SPECTRALBTE_EXTERNAL_LIBS "m")

# find MPI
find_package(MPI REQUIRED)
# the name of the variable containing the MPI include path has changed in the
# most recent version of CMake
if(NOT "${MPI_C_INCLUDE_DIRS}" STREQUAL "")
  include_directories(${MPI_C_INCLUDE_DIRS})
elseif(NOT "${MPI_C_INCLUDE_PATH}" STREQUAL "")
  include_directories(${MPI_C_INCLUDE_PATH})
else()
  message(FATAL_ERROR "Could not find MPI include path")
endif()
set(SPECTRALBTE_EXTERNAL_LIBS "${SPECTRALBTE_EXTERNAL_LIBS};${MPI_C_LIBRARIES}")

# find OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(SPECTRALBTE_EXTERNAL_LIBS "${SPECTRALBTE_EXTERNAL_LIBS};${OpenMP_C_LIBRARIES}")

# find FFTW
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDES})
set(SPECTRALBTE_EXTERNAL_LIBS "${SPECTRALBTE_EXTERNAL_LIBS};${FFTW_LIBRARIES}")

# find GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
set(SPECTRALBTE_EXTERNAL_LIBS "${SPECTRALBTE_EXTERNAL_LIBS};${GSL_LIBRARIES}")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(SPECTRALBTE_EXTERNAL_LIBS "${SPECTRALBTE_EXTERNAL_LIBS};${CUDA_LIBRARIES}")
set(SPECTRALBTE_EXTERNAL_LIBS "${SPECTRALBTE_EXTERNAL_LIBS};${CUDA_CUFFT_LIBRARIES}")
set(CUDA_SEPARABLE_COMPILATION On)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35) # Tesla K40
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_37,code=sm_37) # Tesla K80
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52) # GeForce GTX TITAN X
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60) # Tesla P100
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70) # Tesla V100, GeForce TITAN V
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--maxrregcount 32)

# add the src directory to the include list
include_directories(src)

macro(add_SpectralBTE_library library sources)
  add_library(${library} OBJECT ${sources})
  set(SPECTRALBTE_LIBS "${SPECTRALBTE_LIBS};$<TARGET_OBJECTS:${library}>" PARENT_SCOPE)
endmacro()

set(SPECTRALBTE_CUDA_SRCS "")
macro(add_SpectralBTE_CUDA_sources sources)
  foreach(src IN ITEMS ${sources})
    get_filename_component(abs_path "${src}" ABSOLUTE)
    list(APPEND SPECTRALBTE_CUDA_SRCS "${abs_path}")
  endforeach()
  set(SPECTRALBTE_CUDA_SRCS "${SPECTRALBTE_CUDA_SRCS}" PARENT_SCOPE)
endmacro()

add_subdirectory(src)

message(STATUS "SPECTRALBTE_CUDA_SRCS = ${SPECTRALBTE_CUDA_SRCS}")

# compile CUDA sources
cuda_add_library(SpectralBTE_CUDA ${SPECTRALBTE_CUDA_SRCS})
CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(SpectralBTE_CUDA_link SpectralBTE_CUDA "${SpectralBTE_CUDA_SEPARABLE_COMPILATION_OBJECTS}")
set(SPECTRALBTE_LIBS "${SPECTRALBTE_LIBS};${SpectralBTE_CUDA_SEPARABLE_COMPILATION_OBJECTS};${SpectralBTE_CUDA_link}")

# create static library that will be used to link tests and SWIG library
add_library(SpectralBTE_static
  ${SPECTRALBTE_LIBS}
)

# create shared library that can be used to link other C++ applications
add_library(SpectralBTE SHARED
  ${SPECTRALBTE_LIBS}
)

set_target_properties(SpectralBTE_static SpectralBTE
  PROPERTIES LINKER_LANGUAGE CXX
)

add_subdirectory(exec)
add_subdirectory(tests)

# the compiler flags
message(STATUS "")
message(STATUS "")
message(STATUS "Configuration options:")
message(STATUS "")
message(STATUS "            CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "       CMAKE_CXX_LINK_FLAGS: " ${CMAKE_CXX_LINK_FLAGS})
message(STATUS "      CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "    CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "              CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS})
message(STATUS "         CMAKE_C_LINK_FLAGS: " ${CMAKE_C_LINK_FLAGS})
message(STATUS "        CMAKE_C_FLAGS_DEBUG: " ${CMAKE_C_FLAGS_DEBUG})
message(STATUS "      CMAKE_C_FLAGS_RELEASE: " ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "")
message(STATUS "  SPECTRALBTE_EXTERNAL_LIBS: " ${SPECTRALBTE_EXTERNAL_LIBS})
message(STATUS "")
message(STATUS "")
