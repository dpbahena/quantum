#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

int main() {
    const uint32_t nIndexBits = 2;
    const uint32_t vectorSize = 1 << nIndexBits;

    cuDoubleComplex* d_sv;
    cudaMalloc((void**)&d_sv, sizeof(cuDoubleComplex) * vectorSize);

    cuDoubleComplex h_sv[] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };
    cudaMemcpy(d_sv, h_sv, sizeof(h_sv), cudaMemcpyHostToDevice);

    custatevecHandle_t handle;
    custatevecCreate(&handle);

    // Hadamard matrix
    const cuDoubleComplex H[4] = {
        {1.0 / sqrt(2), 0.0}, {1.0 / sqrt(2), 0.0},
        {1.0 / sqrt(2), 0.0}, {-1.0 / sqrt(2), 0.0}
    };
    int32_t hadamard_target = 0;

    // Apply Hadamard to qubit 0
    custatevecApplyMatrix(
        handle,
        d_sv,
        CUDA_C_64F,
        nIndexBits,
        H,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,  // adjoint
        &hadamard_target,
        1,  // nTargets
        NULL,
        NULL,
        0,  // nControls
        CUSTATEVEC_COMPUTE_DEFAULT,
        NULL,
        0
    );

    // Apply CNOT using Pauli-X and control
    const cuDoubleComplex X[4] = {
        {0, 0}, {1, 0},
        {1, 0}, {0, 0}
    };
    int32_t x_target = 1;
    int32_t control = 0;
    int32_t controlValue = 1;

    custatevecApplyMatrix(
        handle,
        d_sv,
        CUDA_C_64F,
        nIndexBits,
        X,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,  // adjoint
        &x_target,
        1,  // nTargets
        &control,
        &controlValue,
        1,  // nControls
        CUSTATEVEC_COMPUTE_DEFAULT,
        NULL,
        0
    );

    // Copy result back and print
    cuDoubleComplex h_result[4];
    cudaMemcpy(h_result, d_sv, sizeof(h_result), cudaMemcpyDeviceToHost);

    printf("Final statevector:\n");
    for (int i = 0; i < 4; ++i) {
        printf("Index %d: %.3f + %.3fi\n", i, h_result[i].x, h_result[i].y);
    }

    custatevecDestroy(handle);
    cudaFree(d_sv);
    return 0;
}
