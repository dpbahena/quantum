// minimal_density_trace.cpp

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cudensitymat.h>

int main() {
    // 1) Create the cuDensityMat handle
    cudensitymatHandle_t handle = nullptr;
    if (cudensitymatCreate(&handle) != CUDENSITYMAT_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuDensityMat handle\n";
        return 1;
    }

    // 2) Parameters for a 1-qubit (2×2) state
    cudensitymatStatePurity_t purity       = CUDENSITYMAT_STATE_PURITY_PURE;
    int32_t                 numSpaceModes = 1;          // one ket index
    int64_t                 spaceModeExtents[] = { 2 };  // dimension = 2
    int32_t                 batchSize      = 1;          // no batching
    cudaDataType_t          dataType       = CUDA_R_64F; // double precision                  

    // 3) Create the (empty) density-matrix state object
    cudensitymatState_t rho = nullptr;
    if (cudensitymatCreateState(
            handle,
            purity,
            numSpaceModes,
            spaceModeExtents,
            batchSize,
            dataType,
            &rho
        ) != CUDENSITYMAT_STATUS_SUCCESS)
    {
        std::cerr << "cudensitymatCreateState failed\n";
        cudensitymatDestroy(handle);
        return 1;
    }

    // 4) Query and allocate component storage
    int32_t nComp = 0;
    cudensitymatStateGetNumComponents(handle, rho, &nComp);

    std::vector<size_t> compSizes(nComp);
    cudensitymatStateGetComponentStorageSize(
        handle, rho, nComp, compSizes.data()
    );

    std::vector<void*> compBufs(nComp);
    for (int i = 0; i < nComp; ++i) {
        cudaMalloc(&compBufs[i], compSizes[i]);
    }
    cudensitymatStateAttachComponentStorage(
        handle, rho, nComp, compBufs.data(), compSizes.data()
    );

    // 5) Initialize to zero
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudensitymatStateInitializeZero(handle, rho, stream);
    cudaStreamSynchronize(stream);

    // 6) Allocate device buffer for the trace(s)
    //    one FP64 value per state in the batch
    double* d_trace = nullptr;
    cudaMalloc(&d_trace, batchSize * sizeof(double));

    // 7) Compute the trace(s) into d_trace
    if (cudensitymatStateComputeTrace(handle, rho, d_trace, stream)
        != CUDENSITYMAT_STATUS_SUCCESS)
    {
        std::cerr << "cudensitymatStateComputeTrace failed\n";
        return 1;
    }
    cudaStreamSynchronize(stream);

    // 8) Copy back to host and print
    std::vector<double> h_trace(batchSize);
    cudaMemcpy(h_trace.data(), d_trace,
               batchSize * sizeof(double),
               cudaMemcpyDeviceToHost);

    std::cout << "Trace[0] = " << h_trace[0] << "\n";  // Expect “1.0”

    // 9) Cleanup
    cudaFree(d_trace);
    cudensitymatDestroyState(rho);
    for (void* buf : compBufs) cudaFree(buf);
    cudensitymatDestroy(handle);
    cudaStreamDestroy(stream);

    return 0;
}
