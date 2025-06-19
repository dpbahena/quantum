#pragma once

#include <cuda_runtime.h>
#include <custatevec.h>
#include <iostream>
#include <cstdlib>


// CUDA error checker
__host__ void checkCudaError(cudaError_t err, const char* file, int line, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << msg 
                  << ": " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(err, msg) checkCudaError(err, __FILE__, __LINE__, msg)

// cuStateVec error checker
__host__ void checkStateVecError(custatevecStatus_t status, const char* file, int line, const char* msg) {
    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        std::cerr << "cuStateVec error at " << file << ":" << line << " - " << msg 
                  << ": " << custatevecGetErrorString(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_STATEVEC_ERROR(status, msg) checkStateVecError(status, __FILE__, __LINE__, msg)