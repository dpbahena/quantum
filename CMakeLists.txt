# cmake_minimum_required(VERSION 3.10.0)
# project(bell VERSION 0.1.0 LANGUAGES C CXX)

# find_package(CUDAToolkit REQUIRED)

# add_executable(bell bell.c)
# target_link_libraries(bell PRIVATE ${CUDAToolkit_LIBRARIES} )
# target_include_directories(${PROJECT_NAME} PRIVATE  ${CUDAToolkit_INCLUDE_DIRS}     # Modern CUDA include directory
# )

cmake_minimum_required(VERSION 3.18)
project(bell VERSION 0.1.0 LANGUAGES C CUDA)

find_package(CUDAToolkit REQUIRED)

# Location of libcustatevec.so
link_directories(/usr/local/cuda-12.6/targets/x86_64-linux/lib)

add_executable(bell bell.c)

target_include_directories(bell
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(bell
    PRIVATE
    CUDA::cudart
    custatevec
)




