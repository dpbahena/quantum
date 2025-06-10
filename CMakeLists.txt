
cmake_minimum_required(VERSION 3.18)
# Define a variable for the project name
set(PROJECT_NAME walk_particle)

project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES C CUDA)

find_package(CUDAToolkit REQUIRED)

# Location of libcustatevec.so
link_directories(/usr/local/cuda-12.6/targets/x86_64-linux/lib)

add_executable(${PROJECT_NAME} walk.cu)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    CUDA::cudart
    custatevec
)
# This is the “modern” preferred way:
target_compile_definitions( ## or use #define _GNU_SOURCE // enable macros  in header file
  ${PROJECT_NAME}
  PRIVATE 
    _GNU_SOURCE   ## e.g M_SQRT2
)


