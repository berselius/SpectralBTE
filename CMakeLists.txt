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
include_directories(${MPI_C_INCLUDE_DIRS})
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

# add the src directory to the include list
include_directories(src)

macro(add_SpectralBTE_library library sources)
  add_library(${library} OBJECT ${sources})
  set(SPECTRALBTE_LIBS "${SPECTRALBTE_LIBS};$<TARGET_OBJECTS:${library}>" PARENT_SCOPE)
endmacro()

add_subdirectory(src)

# create static library that will be used to link tests and SWIG library
add_library(SpectralBTE_static
  ${SPECTRALBTE_LIBS}
)

# create shared library that can be used to link other C++ applications
add_library(SpectralBTE SHARED
  ${SPECTRALBTE_LIBS}
)

add_subdirectory(exec)
#add_subdirectory(tests)

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
