// #include <cudensitymat.h>

// int main() {
//   cudensitymatHandle_t dm = NULL;
//   // 1) Create library context
//   cudensitymatCreate(&dm);

//   // 2) (Optional) configure multi-GPU/MPI here

//   // 3) Define your density matrix and operators in GPU memory…
//   //    (Use cudensitymatDefineTensorOperator, cudensitymatSetStateSimple, etc.)

//   // 4) Evolve or apply operator: cudensitymatApplyOperator(dm, …)

//   // 5) Query results or expectation values

//   // 6) Cleanup
//   cudensitymatDestroy(dm);
//   return 0;
// }

// #include <stdio.h>
// #include <iostream>
// #include <stdlib.h>
// #include <cuda_runtime.h>

// #include <cudensitymat.h>     // first, pull in the density-matrix API
// #include <custatevec.h>       // then the statevector API (for precision enums)

// int main() {
//   cudensitymatHandle_t handle = nullptr;
//   cudensitymatCreate(&handle);

//   uint64_t dims[] = {2,2};
//   cudensitymatState_t rho = nullptr;
//   cudensitymatDefineStateSimple(
//     handle, &rho,
//     /*rank*/1, dims,
//     /*compute*/CUDENSITYMAT_COMPUTE_64F
//   );

//   double trace = 0.0;
//   cudensitymatGetTrace(handle, rho, &trace);
//   std::cout << "Trace = " << trace << "\n";

//   cudensitymatDestroyState(rho);
//   cudensitymatDestroy(handle);
//   return 0;
// }

// #include <iostream>
// #include <cuda_runtime.h>

// // First include the density-matrix API:
// #include <cudensitymat.h>
// // Then the statevector API (for the compute-mode enum):
// #include <custatevec.h>

// int main() {
//   // 1) Create the cuDensityMat handle
//   cudensitymatHandle_t handle = nullptr;
//   cudensitymatCreate(&handle);

//   // 2) Create a 1-qubit (2×2) density matrix ρ = |0⟩⟨0|
//   uint64_t dims[] = { 2, 2 };
//   cudensitymatState_t rho = nullptr;
//   cudensitymatStateCreate(
//     handle,
//     &rho,
//     /* rank    */ 1,                    // single subsystem
//     /* dims    */ dims,                 // shape = 2×2
//     /* compute */ CUDENSITYMAT_COMPUTE_64F
//   );

//   // 3) Compute its trace
//   double trace = 0.0;
//   cudensitymatStateComputeTrace(handle, rho, &trace);

//   // 4) Print it
//   std::cout << "Trace of ρ is " << trace << "\n";

//   // 5) Clean up
//   cudensitymatStateDestroy(rho);
//   cudensitymatDestroy(handle);
//   return 0;
// }




// #include <stdlib.h>
// #include <stdio.h>
// #include <assert.h>

// #include <cuda_runtime.h>
// #include <cutensor.h>

// #include <unordered_map>
// #include <vector>

// int main(int argc, char** argv)
// {
//     // Host element type definition
//     typedef float floatTypeA;
//     typedef float floatTypeB;
//     typedef float floatTypeC;
//     typedef float floatTypeCompute;

//     // CUDA types
//     cutensorDataType_t typeA = CUTENSOR_R_32F;
//     cutensorDataType_t typeB = CUTENSOR_R_32F;
//     cutensorDataType_t typeC = CUTENSOR_R_32F;
//     cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

//     printf("Include headers and define data types\n");

//     /* ***************************** */

//     // Create vector of modes
//     std::vector<int> modeC{'m','u','n','v'};
//     std::vector<int> modeA{'m','h','k','n'};
//     std::vector<int> modeB{'u','k','v','h'};
//     int nmodeA = modeA.size();
//     int nmodeB = modeB.size();
//     int nmodeC = modeC.size();

//     // Extents
//     std::unordered_map<int, int64_t> extent;
//     extent['m'] = 96;
//     extent['n'] = 96;
//     extent['u'] = 96;
//     extent['v'] = 64;
//     extent['h'] = 64;
//     extent['k'] = 64;

//     // Create a vector of extents for each tensor
//     std::vector<int64_t> extentC;
//     for(auto mode : modeC)
//         extentC.push_back(extent[mode]);
//     std::vector<int64_t> extentA;
//     for(auto mode : modeA)
//         extentA.push_back(extent[mode]);
//     std::vector<int64_t> extentB;
//     for(auto mode : modeB)
//         extentB.push_back(extent[mode]);

//     printf("Define modes and extents\n");

//     return 0;
// }

// #include <stdlib.h>
// #include <stdio.h>
// #include <assert.h>

// #include <cuda_runtime.h>
// #include <cutensor.h>

// #include <unordered_map>
// #include <vector>

// // Handle cuTENSOR errors
// #define HANDLE_ERROR(x)                                             \
// { const auto err = x;                                               \
//     if( err != CUTENSOR_STATUS_SUCCESS )                              \
//     { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
// };

// #define HANDLE_CUDA_ERROR(x)                                      \
// { const auto err = x;                                             \
//     if( err != cudaSuccess )                                        \
//     { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
// };

// int main(int argc, char** argv)
// {
//     // Host element type definition
//     typedef float floatTypeA;
//     typedef float floatTypeB;
//     typedef float floatTypeC;
//     typedef float floatTypeCompute;

//     // CUDA types
//     cutensorDataType_t typeA = CUTENSOR_R_32F;
//     cutensorDataType_t typeB = CUTENSOR_R_32F;
//     cutensorDataType_t typeC = CUTENSOR_R_32F;
//     cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

//     printf("Include headers and define data types\n");

//     /* ***************************** */

//     // Create vector of modes
//     std::vector<int> modeC{'m','u','n','v'};
//     std::vector<int> modeA{'m','h','k','n'};
//     std::vector<int> modeB{'u','k','v','h'};
//     int nmodeA = modeA.size();
//     int nmodeB = modeB.size();
//     int nmodeC = modeC.size();

//     // Extents
//     std::unordered_map<int, int64_t> extent;
//     extent['m'] = 96;
//     extent['n'] = 96;
//     extent['u'] = 96;
//     extent['v'] = 64;
//     extent['h'] = 64;
//     extent['k'] = 64;

//     // Create a vector of extents for each tensor
//     std::vector<int64_t> extentC;
//     for(auto mode : modeC)
//         extentC.push_back(extent[mode]);
//     std::vector<int64_t> extentA;
//     for(auto mode : modeA)
//         extentA.push_back(extent[mode]);
//     std::vector<int64_t> extentB;
//     for(auto mode : modeB)
//         extentB.push_back(extent[mode]);

//     printf("Define modes and extents\n");

//     /* ***************************** */

//     // Number of elements of each tensor
//     size_t elementsA = 1;
//     for(auto mode : modeA)
//         elementsA *= extent[mode];
//     size_t elementsB = 1;
//     for(auto mode : modeB)
//         elementsB *= extent[mode];
//     size_t elementsC = 1;
//     for(auto mode : modeC)
//         elementsC *= extent[mode];

//     // Size in bytes
//     size_t sizeA = sizeof(floatTypeA) * elementsA;
//     size_t sizeB = sizeof(floatTypeB) * elementsB;
//     size_t sizeC = sizeof(floatTypeC) * elementsC;

//     // Allocate on device
//     void *A_d, *B_d, *C_d;
//     cudaMalloc((void**)&A_d, sizeA);
//     cudaMalloc((void**)&B_d, sizeB);
//     cudaMalloc((void**)&C_d, sizeC);

//     // Allocate on host
//     floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
//     floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
//     floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

//     // Initialize data on host
//     for(int64_t i = 0; i < elementsA; i++)
//         A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
//     for(int64_t i = 0; i < elementsB; i++)
//         B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
//     for(int64_t i = 0; i < elementsC; i++)
//         C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

//     // Copy to device
//     HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
//     HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
//     HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

//     const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
//     assert(uintptr_t(A_d) % kAlignment == 0);
//     assert(uintptr_t(B_d) % kAlignment == 0);
//     assert(uintptr_t(C_d) % kAlignment == 0);

//     printf("Allocate, initialize and transfer tensors\n");

//     /*************************
//      * cuTENSOR
//      *************************/ 

//     cutensorHandle_t handle;
//     HANDLE_ERROR(cutensorCreate(&handle));

//     /**********************
//      * Create Tensor Descriptors
//      **********************/

//     cutensorTensorDescriptor_t descA;
//     HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
//                 &descA,
//                 nmodeA,
//                 extentA.data(),
//                 NULL,/*stride*/
//                 typeA, kAlignment));

//     cutensorTensorDescriptor_t descB;
//     HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
//                 &descB,
//                 nmodeB,
//                 extentB.data(),
//                 NULL,/*stride*/
//                 typeB, kAlignment));

//     cutensorTensorDescriptor_t descC;
//     HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
//                 &descC,
//                 nmodeC,
//                 extentC.data(),
//                 NULL,/*stride*/
//                 typeC, kAlignment));

//     printf("Initialize cuTENSOR and tensor descriptors\n");

//     /*******************************
//      * Create Contraction Descriptor
//      *******************************/

//     cutensorOperationDescriptor_t desc;
//     HANDLE_ERROR(cutensorCreateContraction(handle, 
//                 &desc,
//                 descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
//                 descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
//                 descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
//                 descC, modeC.data(),
//                 descCompute));

//     /*****************************
//      * Optional (but recommended): ensure that the scalar type is correct.
//      *****************************/

//     cutensorDataType_t scalarType;
//     HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
//                 desc,
//                 CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
//                 (void*)&scalarType,
//                 sizeof(scalarType)));

//     assert(scalarType == CUTENSOR_R_32F);
//     typedef float floatTypeCompute;
//     floatTypeCompute alpha = (floatTypeCompute)1.1f;
//     floatTypeCompute beta  = (floatTypeCompute)0.f;

//     /**************************
//      * Set the algorithm to use
//      ***************************/

//     const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

//     cutensorPlanPreference_t planPref;
//     HANDLE_ERROR(cutensorCreatePlanPreference(
//                 handle,
//                 &planPref,
//                 algo,
//                 CUTENSOR_JIT_MODE_NONE));

//     /**********************
//      * Query workspace estimate
//      **********************/

//     uint64_t workspaceSizeEstimate = 0;
//     const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
//     HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
//                 desc,
//                 planPref,
//                 workspacePref,
//                 &workspaceSizeEstimate));

//     /**************************
//      * Create Contraction Plan
//      **************************/

//     cutensorPlan_t plan;
//     HANDLE_ERROR(cutensorCreatePlan(handle,
//                 &plan,
//                 desc,
//                 planPref,
//                 workspaceSizeEstimate));

//     /**************************
//      * Optional: Query information about the created plan
//      **************************/

//     // query actually used workspace
//     uint64_t actualWorkspaceSize = 0;
//     HANDLE_ERROR(cutensorPlanGetAttribute(handle,
//                 plan,
//                 CUTENSOR_PLAN_REQUIRED_WORKSPACE,
//                 &actualWorkspaceSize,
//                 sizeof(actualWorkspaceSize)));

//     // At this point the user knows exactly how much memory is need by the operation and
//     // only the smaller actual workspace needs to be allocated
//     assert(actualWorkspaceSize <= workspaceSizeEstimate);

//     void *work = nullptr;
//     if (actualWorkspaceSize > 0)
//     {
//         HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
//         assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
//     }

//     /**********************
//      * Execute
//      **********************/

//     cudaStream_t stream;
//     HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

//     HANDLE_ERROR(cutensorContract(handle,
//                 plan,
//                 (void*) &alpha, A_d, B_d,
//                 (void*) &beta,  C_d, C_d, 
//                 work, actualWorkspaceSize, stream));

//     /**********************
//      * Free allocated data
//      **********************/
//     HANDLE_ERROR(cutensorDestroy(handle));
//     HANDLE_ERROR(cutensorDestroyPlan(plan));
//     HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
//     HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
//     HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
//     HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
//     HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

//     if (A) free(A);
//     if (B) free(B);
//     if (C) free(C);
//     if (A_d) cudaFree(A_d);
//     if (B_d) cudaFree(B_d);
//     if (C_d) cudaFree(C_d);
//     if (work) cudaFree(work);

//     return 0;
// }


// // cutensor_minimal.cpp
// #include <iostream>
// #include <cutensor.h>

// int main()
// {
//     // 1) Create and initialize a cuTENSOR handle
//     cutensorHandle_t handle;
//     cutensorStatus_t status = cutensorInit(&handle);
//     if (status != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorInit failed: "
//                   << cutensorGetErrorString(status) << "\n";
//         return 1;
//     }

//     // 2) Query the version
//     int32_t version, patch;
    
//     status = cutensorGetVersion(&handle, &version, &patch);
//     if (status != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorGetVersion failed: "
//                   << cutensorGetErrorString(status) << "\n";
//         return 1;
//     }

//     // 3) Print out confirmation
//     std::cout << "cuTENSOR initialized successfully (v"
//               << version << "." << patch << ")\n";

//     return 0;
// }

// cutensor_minimal.cpp
// #include <iostream>
// #include <cutensor.h>

// int main()
// {
//     // 1) Create a cuTENSOR handle
//     cutensorHandle_t handle;
//     cutensorStatus_t status = cutensorCreate(&handle);
//     if (status != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorCreate failed: "
//                   << cutensorGetErrorString(status) << "\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "cuTENSOR handle created successfully.\n";

//     // 2) (optional) do real work here...

//     // 3) Destroy the handle
//     status = cutensorDestroy(handle);
//     if (status != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorDestroy failed: "
//                   << cutensorGetErrorString(status) << "\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "cuTENSOR handle destroyed successfully.\n";

//     return EXIT_SUCCESS;
// }


// cutensor_minimal.cpp
// cutensor_minimal.cpp works --->
// #include <iostream>
// #include <cstdint>
// #include <cutensor.h>

// int main()
// {
//     // 1) Create a cuTENSOR handle
//     cutensorHandle_t handle;
//     cutensorStatus_t status = cutensorCreate(&handle);
//     if (status != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorCreate failed: "
//                   << cutensorGetErrorString(status) << "\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "cuTENSOR handle created successfully.\n";

//     // 2) Query the library version (encoded in a single size_t)
//     size_t versionCode = cutensorGetVersion();                       // e.g. 0x00020002 for v2.2
//     uint32_t major     = static_cast<uint32_t>(versionCode >> 16);   // high 16 bits
//     uint32_t minor     = static_cast<uint32_t>(versionCode & 0xFFFF);// low  16 bits

//     std::cout << "cuTENSOR version: "
//               << major << "." << minor << "\n";

//     // 3) (optional) do real work here…

//     // 4) Destroy the handle
//     status = cutensorDestroy(handle);
//     if (status != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorDestroy failed: "
//                   << cutensorGetErrorString(status) << "\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "cuTENSOR handle destroyed successfully.\n";

//     return EXIT_SUCCESS;
// }

// // cutensor_minimal.cpp
// #include <iostream>
// #include <cstdint>
// #include <cutensor.h>

// int main()
// {
//     // Create handle
//     cutensorHandle_t handle;
//     if (cutensorCreate(&handle) != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorCreate failed\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "cuTENSOR handle created successfully.\n";

//     // Decode version
//     size_t ver = cutensorGetVersion();     // e.g. 20200
//     int major = ver / 10000;               // 2
//     int minor = (ver % 10000) / 100;       // 2
//     int patch = ver % 100;                 // 0
//     std::cout << "cuTENSOR version: "
//               << major << "." << minor << "." << patch << "\n";

//     // Destroy handle
//     if (cutensorDestroy(handle) != CUTENSOR_STATUS_SUCCESS) {
//         std::cerr << "cutensorDestroy failed\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "cuTENSOR handle destroyed successfully.\n";
//     return EXIT_SUCCESS;
// }

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                             \
{ const auto err = x;                                               \
    if( err != CUTENSOR_STATUS_SUCCESS )                              \
    { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

int main(int argc, char** argv)
{
    // Host element type definition
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    // CUDA types
    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeB = CUTENSOR_R_32F;
    cutensorDataType_t typeC = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    printf("Include headers and define data types\n");

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");

    /* ***************************** */

    // Number of elements of each tensor
    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    // Allocate on device
    void *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sizeA);
    cudaMalloc((void**)&B_d, sizeB);
    cudaMalloc((void**)&C_d, sizeC);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    // Initialize data on host
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    printf("Allocate, initialize and transfer tensors\n");

    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, kAlignment));

    printf("Initialize cuTENSOR and tensor descriptors\n");

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, 
                &desc,
                descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                descC, modeC.data(),
                descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);
    typedef float floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /**************************
     * Set the algorithm to use
     ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                handle,
                &planPref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                desc,
                planPref,
                workspacePref,
                &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                &plan,
                desc,
                planPref,
                workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Execute
     **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cutensorContract(handle,
                plan,
                (void*) &alpha, A_d, B_d,
                (void*) &beta,  C_d, C_d, 
                work, actualWorkspaceSize, stream));

    /**********************
     * Free allocated data
     **********************/
    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}