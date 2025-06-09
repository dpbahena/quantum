// quantum_random_walk.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#define N_STEPS 20

int main(){
    // --- Setup classical particle state ---
    int position = 0;
    printf("Starting at position %d\n", position);

    // --- Prepare cuStateVec for 1 qubit ---
    const uint32_t nQubits = 1;
    const uint32_t dim     = 1u << nQubits;  // = 2

    cuDoubleComplex* d_sv;
    cudaMalloc((void**)&d_sv, sizeof(cuDoubleComplex)*dim);

    // Host buffer for amplitudes
    cuDoubleComplex h_sv[2];

    // Hadamard gate
    const cuDoubleComplex H[4] = {
        {1.0/M_SQRT2, 0.0}, {1.0/M_SQRT2, 0.0},
        {1.0/M_SQRT2, 0.0}, {-1.0/M_SQRT2, 0.0}
    };
    int32_t target = 0;

    // Create handle
    custatevecHandle_t handle;
    custatevecCreate(&handle);

    // Seed CPU RNG
    srand((unsigned)time(NULL));

    for(int step=0; step<N_STEPS; ++step){
        // 1) Initialize to |0⟩ on device
        cuDoubleComplex init[2] = {{1.0,0.0},{0.0,0.0}};
        cudaMemcpy(d_sv, init, sizeof(init), cudaMemcpyHostToDevice);

        // 2) Apply H to qubit 0
        custatevecApplyMatrix(
            handle,                       // library handle
            d_sv,                         // statevector on device
            CUDA_C_64F,                   // data type
            nQubits,                      // number of qubits
            H, CUDA_C_64F,                // Hadamard matrix
            CUSTATEVEC_MATRIX_LAYOUT_ROW, // layout
            0,                            // no adjoint
            &target, 1,                   // target = qubit 0
            NULL, NULL, 0,                // no controls
            CUSTATEVEC_COMPUTE_DEFAULT,   // default config
            NULL, 0                       // no workspace
        );

        // 3) Copy amplitudes back to host
        cudaMemcpy(h_sv, d_sv, sizeof(h_sv), cudaMemcpyDeviceToHost);

        // Compute probabilities
        double p0 = cuCreal(h_sv[0])*cuCreal(h_sv[0])
                  + cuCimag(h_sv[0])*cuCimag(h_sv[0]);
        // p1 = 1 - p0
        double r = (double)rand() / RAND_MAX;

        // 4) Sample and move
        if(r < p0){
            position -= 1;  // "0" → move left
        } else {
            position += 1;  // "1" → move right
        }

        printf("Step %2d: coin=(%.3f,%.3f)  r=%.3f  → pos=%d\n",
               step,
               cuCreal(h_sv[0]), cuCimag(h_sv[0]),
               r, position);
    }

    // Cleanup
    custatevecDestroy(handle);
    cudaFree(d_sv);
    return 0;
}
