

cmake_minimum_required(VERSION 3.18)
project(bell LANGUAGES CXX CUDA)

#–– 1) Set up root dirs
set(CUTENSOR_ROOT    /usr/local/cutensor)
set(CUQUANTUM_ROOT   /usr/local/cuquantum)

#–– 2) Find CUDA runtime
find_package(CUDAToolkit REQUIRED)

# Automatically gather all .cu, .cpp, .c files in src/
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    src/*.cu
    src/*.cpp
    src/*.c
)

#–– 3) Create our executable
add_executable(${PROJECT_NAME}
  ${SRC_FILES}
)

#–– 4) Headers for cuTENSOR & cuQuantum
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CUTENSOR_ROOT}/include
  ${CUQUANTUM_ROOT}/include
)

#–– 5) Locate the libraries
find_library(CUTENSOR_LIB
  NAMES cutensor
  HINTS ${CUTENSOR_ROOT}/lib
)
if (NOT CUTENSOR_LIB)
  message(FATAL_ERROR "Could not locate libcutensor.so in ${CUTENSOR_ROOT}/lib")
endif()

find_library(CUDENSITYMAT_LIB
  NAMES cudensitymat
  HINTS ${CUQUANTUM_ROOT}/lib
)
if (NOT CUDENSITYMAT_LIB)
  message(FATAL_ERROR "Could not locate libcudensitymat.so in ${CUQUANTUM_ROOT}/lib")
endif()

find_library(CUSTATEVEC_LIB
  NAMES custatevec
  HINTS ${CUQUANTUM_ROOT}/lib
)
if (NOT CUSTATEVEC_LIB)
  message(FATAL_ERROR "Could not locate libcustatevec.so in ${CUQUANTUM_ROOT}/lib")
endif()

#–– 6) Link everything
target_link_libraries(${PROJECT_NAME} PRIVATE
  CUDA::cudart
  ${CUTENSOR_LIB}
  ${CUDENSITYMAT_LIB}
  ${CUSTATEVEC_LIB}
)

#–– 7) RPATH so loader finds .so files at runtime
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH
    "/usr/local/cuda-12.6/targets/x86_64-linux/lib"
  BUILD_RPATH
    "${CUTENSOR_ROOT}/lib;${CUQUANTUM_ROOT}/lib"
)
