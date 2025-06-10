// bell5.c
#include <stdio.h>
// #define _GNU_SOURCE // enable macros 
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>


int main() {
    // Number of qubits
    const uint32_t nIndexBits = 5;
    // Statevector length = 2^nIndexBits
    const uint32_t vectorSize  = 1u << nIndexBits;  // 32 amplitudes

    // Allocate device statevector
    cuDoubleComplex* d_sv = NULL;
    cudaError_t cudaErr = cudaMalloc((void**)&d_sv,
                            sizeof(cuDoubleComplex) * vectorSize);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Initialize host statevector to |00000>
    cuDoubleComplex h_sv[1u << 5] = { {1.0, 0.0} };
    // (remaining entries default to {0,0})
    cudaErr = cudaMemcpy(d_sv, h_sv,
                        sizeof(h_sv),
                        cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(cudaErr));
        cudaFree(d_sv);
        return 1;
    }

    // Create cuStateVec handle
    custatevecHandle_t handle = NULL;
    custatevecCreate(&handle);

    // Define single-qubit Hadamard matrix (2×2, row-major)
    const cuDoubleComplex H[4] = {
        { 1.0 / M_SQRT2, 0.0 }, {  1.0 / M_SQRT2, 0.0 },
        { 1.0 / M_SQRT2, 0.0 }, { -1.0 / M_SQRT2, 0.0 }
    };
    int32_t hadamard_target = 0;

    // Apply H on qubit 0
    custatevecApplyMatrix(
        handle,
        d_sv,
        CUDA_C_64F,
        nIndexBits,
        H,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,                  // no adjoint
        &hadamard_target,
        1,                  // one target qubit
        NULL, NULL, 0,      // no controls
        CUSTATEVEC_COMPUTE_DEFAULT,
        NULL, 0
    );

    // Define Pauli-X (NOT) matrix
    const cuDoubleComplex X[4] = {
        {0.0, 0.0}, {1.0, 0.0},
        {1.0, 0.0}, {0.0, 0.0}
    };
    int32_t controlValue = 1;

    // Build a 5-qubit GHZ: chain CNOTs from qubit 0→1, 1→2, 2→3, 3→4
    for (int32_t tgt = 1; tgt < 5; ++tgt) {
        int32_t control = tgt - 1;
        int32_t target  = tgt;

        custatevecApplyMatrix(
            handle,
            d_sv,
            CUDA_C_64F,
            nIndexBits,
            X,
            CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW,
            0,                  // no adjoint
            &target,
            1,                  // one target qubit
            &control,
            &controlValue,
            1,                  // one control qubit
            CUSTATEVEC_COMPUTE_DEFAULT,
            NULL, 0
        );
    }

    // Copy result back to host and print
    cuDoubleComplex h_result[1u << 5];
    cudaErr = cudaMemcpy(h_result, d_sv,
                        sizeof(h_result),
                        cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(cudaErr));
        custatevecDestroy(handle);
        cudaFree(d_sv);
        return 1;
    }

    printf("Final 5-qubit statevector (|00000> + |11111>)/√2:\n");
    for (uint32_t i = 0; i < vectorSize; ++i) {
        printf("Index %2u: %.6f + %.6f i\n",
               i,
               h_result[i].x,
               h_result[i].y);
    }

    // Cleanup
    custatevecDestroy(handle);
    cudaFree(d_sv);
    return 0;
}
