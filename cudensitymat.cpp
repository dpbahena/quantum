#include <cuda_runtime.h>
#include <cudensitymat.h>
#include <custatevec.h>

int main() {
    //  1. Create the cuDensityMat handle (library context)
    cudensitymatHandle_t dm = NULL;
    cudensitymatCreate(&dm);
    // 2. Define a 1-qubit density matrix of dimention 2*2
    //  We'll use the "simple state" API to set it to |0><0|
    cudensitymatState_t rho = NULL;
    uint64_t dims[] = {2, 2};
    cudensitymatCreateState(dm, &rho, 1, dims, CUDENSITYMAT_COMPUTE_64F);
    
    
    




  

  // 6) Cleanup
  cudensitymatDestroy(dm);
  return 0;
}