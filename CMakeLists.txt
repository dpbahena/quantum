# cmake_minimum_required(VERSION 3.18)

# # Project name and languages
# project(cudensity VERSION 0.1.0 LANGUAGES C CXX CUDA)

# # Find the CUDA runtime
# find_package(CUDAToolkit REQUIRED)

# # Where you installed cuQuantum
# # (change this if your cuQuantum lives elsewhere)
# set(CUQUANTUM_ROOT /usr/local/cuquantum)

# # Make sure CMake looks here for cuQuantum headers
# include_directories(
#   ${CUQUANTUM_ROOT}/include
#   ${CUDAToolkit_INCLUDE_DIRS}
# )

# # If you want to use link_directories instead of full paths:
# link_directories(
#   ${CUQUANTUM_ROOT}/lib
#   /usr/local/cuda-12.6/targets/x86_64-linux/lib
# )

# # Your executable
# add_executable(${PROJECT_NAME} cuDensityMinimal.c)

# # Link against CUDA runtime, cuStateVec, and cuDensityMat
# target_link_libraries(${PROJECT_NAME}
#   PRIVATE
#     CUDA::cudart
#     # cuStateVec library
#     ${CUQUANTUM_ROOT}/lib/libcustatevec.so
#     # cuDensityMat library
#     ${CUQUANTUM_ROOT}/lib/libcudensitymat.so
# )

# # (Optional) define _GNU_SOURCE if you need it
# target_compile_definitions(${PROJECT_NAME}
#   PRIVATE
#     _GNU_SOURCE
# )


# cmake_minimum_required(VERSION 3.18)
# project(cudensity VERSION 0.1.0 LANGUAGES C CXX CUDA)

# find_package(CUDAToolkit REQUIRED)

# # cuQuantum install prefix
# set(CUQUANTUM_ROOT /usr/local/cuquantum)

# # Tell the compiler where to find headers
# include_directories(
#   ${CUQUANTUM_ROOT}/include
#   ${CUDAToolkit_INCLUDE_DIRS}
# )

# # Add your executable
# add_executable(${PROJECT_NAME} cuDensityMinimal.cpp)

# # Make sure the linker can find the CUDA toolkit & cuQuantum libs
# link_directories(
#   /usr/local/cuda-12.6/targets/x86_64-linux/lib
#   ${CUQUANTUM_ROOT}/lib
# )

# # Link libraries in the right order:
# # 1) CUDA runtime
# # 2) cuStateVec
# # 3) cuDensityMat
# # 4) cuTENSOR (for the symbols cuDensityMat needs)
# target_link_libraries(${PROJECT_NAME}
#   PRIVATE
#     CUDA::cudart
#     ${CUQUANTUM_ROOT}/lib/libcustatevec.so
#     ${CUQUANTUM_ROOT}/lib/libcudensitymat.so
#     cutensor                  # this pulls in libcutensor.so from the CUDA toolkit
# )

# # Optionally embed an rpath so the executable knows where to find these at runtime
# set_target_properties(${PROJECT_NAME} PROPERTIES
#   BUILD_RPATH
#     "/usr/local/cuda-12.6/targets/x86_64-linux/lib;${CUQUANTUM_ROOT}/lib"
# )

# cmake_minimum_required(VERSION 3.18)
# project(mycutensor LANGUAGES CXX CUDA)

# find_package(CUDAToolkit REQUIRED)

# set(CUQUANTUM_ROOT /usr/local/cuquantum)

# include_directories(
#   ${CUQUANTUM_ROOT}/include
#   ${CUDAToolkit_INCLUDE_DIRS}
# )

# add_executable(${PROJECT_NAME}  cutensor_minimal.cpp)

# target_link_libraries(${PROJECT_NAME} 
#   PRIVATE
#     CUDA::cudart
#     ${CUQUANTUM_ROOT}/lib/libcustatevec.so
#     ${CUQUANTUM_ROOT}/lib/libcudensitymat.so
#     # and make sure cutensor is available:
#     /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcutensor.so
# )

# set_target_properties(${PROJECT_NAME}  PROPERTIES
#   BUILD_RPATH
#     "/usr/local/cuda-12.6/targets/x86_64-linux/lib;${CUQUANTUM_ROOT}/lib"
# )


cmake_minimum_required(VERSION 3.18)
project(contraction LANGUAGES CXX CUDA)

# 1) Tell CMake where to find cuTENSOR headers & libs
set(CUTENSOR_ROOT /usr/local/cutensor)
include_directories(${CUTENSOR_ROOT}/include)

# 2) Find cuTENSOR -- this will pick up libcutensor.so
find_library(CUTENSOR_LIB
  NAMES cutensor
  HINTS ${CUTENSOR_ROOT}/lib
)
if (NOT CUTENSOR_LIB)
  message(FATAL_ERROR "Could not locate libcutensor in ${CUTENSOR_ROOT}/lib")
endif()

# 3) Also pull in the CUDA runtime
find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} first_contraction.cpp)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    CUDA::cudart
    ${CUTENSOR_LIB}
)

# 4) Ensure the loader can find it at runtime
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${CUTENSOR_ROOT}/lib"
)
