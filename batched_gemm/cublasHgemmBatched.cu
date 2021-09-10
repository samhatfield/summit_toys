//
// Wrapper for cublasHgemm function. 
//
// Alan Gray, NVIDIA
//

#include <stdio.h>
#include "cublas_v2.h" 


bool alreadyAllocated_hgemm = false;
bool alreadyAllocated_hgemm_handle = false;

half **d_Aarray_hgemm;
half **d_Barray_hgemm;
half **d_Carray_hgemm;

half **Aarray_hgemm;
half **Barray_hgemm;
half **Carray_hgemm;

cublasHandle_t handle_hgemm;	

extern "C" void cublasHgemmBatched_wrapper(
  char transa, char transb,
  int m, int n, int k,
  half alpha,
  const half *A, int lda, int tda,
  const half *B, int ldb, int tdb,
  half beta,
  half *C, int ldc, int tdc,
  int batchCount
){
  // Define CUBLAS operation handles
  cublasOperation_t op_t1, op_t2;

  // Decide whether to transpose matrices or not
  op_t1 = (transa == 'T' || transa == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  op_t2 = (transb == 'T' || transb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Initialize CUBLAS handle
  if (!alreadyAllocated_hgemm_handle) {
    cublasCreate(&handle_hgemm);
    alreadyAllocated_hgemm_handle = true;
  }

  // Allocate host arrays
  if (!alreadyAllocated_hgemm) {
    cudaMallocHost(&Aarray_hgemm, batchCount*sizeof(half*));
    cudaMallocHost(&Barray_hgemm, batchCount*sizeof(half*));
    cudaMallocHost(&Carray_hgemm, batchCount*sizeof(half*));
    alreadyAllocated_hgemm = true;
  }

  // Allocate device arrays
  cudaMalloc(&d_Aarray_hgemm, batchCount*sizeof(half*));
  cudaMalloc(&d_Barray_hgemm, batchCount*sizeof(half*));
  cudaMalloc(&d_Carray_hgemm, batchCount*sizeof(half*));

  // Transfer data from input arrays to host arrays
  for (int i = 0; i < batchCount; i++) {
    Aarray_hgemm[i] = (half*) &(A[i*lda*tda]);
    Barray_hgemm[i] = (half*) &(B[i*ldb*tdb]);
    Carray_hgemm[i] = (half*) &(C[i*ldc*tdc]);
  }

  // Transfer data from host arrays to device arrays
  cudaMemcpy(d_Aarray_hgemm, Aarray_hgemm, batchCount*sizeof(half*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Barray_hgemm, Barray_hgemm, batchCount*sizeof(half*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Carray_hgemm, Carray_hgemm, batchCount*sizeof(half*), cudaMemcpyHostToDevice);

  // Perform batched SGEMM
  cublasGemmBatchedEx(handle_hgemm,
    op_t1, op_t2,
    m, n, k,
    (const void*)&alpha,
    (const void**)d_Aarray_hgemm, CUDA_R_16F, lda,
    (const void**)d_Barray_hgemm, CUDA_R_16F, ldb,
    (const void*)&beta,
    (void**)d_Carray_hgemm, CUDA_R_16F, ldc,
    batchCount,
    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

  cudaDeviceSynchronize();
  
  // Free device arrays
  cudaFree(d_Aarray_hgemm);
  cudaFree(d_Barray_hgemm);
  cudaFree(d_Carray_hgemm);
}

extern "C" void cublasHgemmBatched_finalize ()
{

  if (alreadyAllocated_hgemm){
  
    cudaFree(Aarray_hgemm);
    cudaFree(Barray_hgemm);
    cudaFree(Carray_hgemm);
    
    cudaFree(d_Aarray_hgemm);
    cudaFree(d_Barray_hgemm);
    cudaFree(d_Carray_hgemm);

  }

  if (alreadyAllocated_hgemm_handle){
    cublasDestroy(handle_hgemm);
  }
  
}
