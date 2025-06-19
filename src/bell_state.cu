/** We will create a 2-qubits system and apply :
 * 1. Hadamard gate on the first qubit: H(0)
 * 2. CNOT gate with the first qubit as control and the second qubit as target: CNOT(0, 1)
*/

#include "help.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>
#include <custatevec.h>


int main() {
    constexpr int n_qubits = 2;
    constexpr int n_states = 1 << n_qubits; // 2^n_qubits 

    // Allocate memory for the statevector
    cuDoubleComplex* d_sv;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sv, n_states * sizeof(cuDoubleComplex)), "cudaMalloc");

    // Initialize to |00> = [1, 0, 0, 0]
    cuDoubleComplex h_sv[] = {
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0)
    };
    CHECK_CUDA_ERROR(cudaMemcpy(d_sv, h_sv, n_states * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "cudaMemcpy");

    // Create custatevec handle
    custatevecHandle_t handle;
    CHECK_STATEVEC_ERROR(custatevecCreate(&handle), "custatevecCreate");

    // Apply H on qubit 0
    const cuDoubleComplex H[4] = {
        make_cuDoubleComplex( 1.0f / M_SQRT2, 0.f),
        make_cuDoubleComplex( 1.0f / M_SQRT2, 0.f),
        make_cuDoubleComplex( 1.0f / M_SQRT2, 0.f),
        make_cuDoubleComplex(-1.0f / M_SQRT2, 0.f),
    };

    // // 4 x 4 CNOT matrix for 2 qubits in row-major layout
    // const cuDoubleComplex CNOT[16] = {
    //     make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0, 0), // |00> -> |00>
    //     make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0, 0), // |01> -> |01>
    //     make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1, 0), // |10> -> |11>
    //     make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0, 0), // |11> -> |10>
    // };

    // 2 x 2 Pauli-X matrix (target gate for CNOT)
    const cuDoubleComplex X[4] = {
        make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), 
        make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)
    };


   
    int32_t hadamard_target = 0;
   

    CHECK_STATEVEC_ERROR(custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, n_qubits, H, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 
                            0, // adjount = false
                            &hadamard_target, 
                            1, 
                            nullptr, 
                            nullptr,
                            0,       // no controls
                            CUSTATEVEC_COMPUTE_DEFAULT,
                            nullptr,
                            0
    ), "Apply Hadamard");  

    // Apply CNOT with control = 0; target = 1;
    int32_t x_target = 1;
    int nTargets = 1;
    int32_t control_qubit = 0;
    int32_t controlValue = 1;

    size_t extraWorkspaceSize = 0;
    void* extraWorkspace = nullptr;
    
    
    CHECK_STATEVEC_ERROR(custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, n_qubits, X, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                            0, // normal (not adjoint)
                            &x_target,
                            nTargets,
                            &control_qubit,
                            &controlValue,
                            1,  // nControls
                            CUSTATEVEC_COMPUTE_DEFAULT,
                            extraWorkspace,
                            extraWorkspaceSize
    ), "Apply CNOT");

    // Copy results back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_sv, d_sv, n_states * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "cudaMemcpy");
    
    std::cout << "Final stateVector:\n";
    for (int i = 0; i < n_states; ++i) {
        std::cout << " " << i << ": (" << cuCreal(h_sv[i]) << ", " << cuCimag(h_sv[i]) << ")\n";

    }

    // cleanup
    custatevecDestroy(handle);
    cudaFree(d_sv);

    return 0;


}