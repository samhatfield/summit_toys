//
// Wrapper for cublasSgemm function. 
//
// Alan Gray, NVIDIA
//

#include <stdio.h>
#include "cublas_v2.h" 


bool alreadyAllocated_sgemm = false;
bool alreadyAllocated_sgemm_handle = false;

float **d_Aarray_sgemm;
float **d_Barray_sgemm;
float **d_Carray_sgemm;

float **Aarray_sgemm;
float **Barray_sgemm;
float **Carray_sgemm;

cublasHandle_t handle_sgemm;	

extern "C" void cublasSgemmBatched_wrapper(
  char transa, char transb,
  int m, int n, int k,
  float alpha,
  const float *A, int lda, int tda,
  const float *B, int ldb, int tdb,
  float beta,
  float *C, int ldc, int tdc,
  int batchCount
){
  // Define CUBLAS operation handles
  cublasOperation_t op_t1, op_t2;

  // Decide whether to transpose matrices or not
  op_t1 = (transa == 'T' || transa == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  op_t2 = (transb == 'T' || transb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Initialize CUBLAS handle
  if (!alreadyAllocated_sgemm_handle) {
    cublasCreate(&handle_sgemm);
    alreadyAllocated_sgemm_handle = true;
  }

  // Allocate host arrays
  if (!alreadyAllocated_sgemm) {
    cudaMallocHost(&Aarray_sgemm, batchCount*sizeof(float*));
    cudaMallocHost(&Barray_sgemm, batchCount*sizeof(float*));
    cudaMallocHost(&Carray_sgemm, batchCount*sizeof(float*));
    alreadyAllocated_sgemm = true;
  }

  // Allocate device arrays
  cudaMalloc(&d_Aarray_sgemm, batchCount*sizeof(float*));
  cudaMalloc(&d_Barray_sgemm, batchCount*sizeof(float*));
  cudaMalloc(&d_Carray_sgemm, batchCount*sizeof(float*));

  // Transfer data from input arrays to host arrays
  for (int i = 0; i < batchCount; i++) {
    Aarray_sgemm[i] = (float*) &(A[i*lda*tda]);
    Barray_sgemm[i] = (float*) &(B[i*ldb*tdb]);
    Carray_sgemm[i] = (float*) &(C[i*ldc*tdc]);
  }

  // Transfer data from host arrays to device arrays
  cudaMemcpy(d_Aarray_sgemm,Aarray_sgemm,batchCount*sizeof(float*),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Barray_sgemm,Barray_sgemm,batchCount*sizeof(float*),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Carray_sgemm,Carray_sgemm,batchCount*sizeof(float*),cudaMemcpyHostToDevice);

  // Perform batched SGEMM
  cublasGemmBatchedEx(handle_sgemm,
    op_t1, op_t2,
    m, n, k,
    (const void*)&alpha,
    (const void**)d_Aarray_sgemm, CUDA_R_32F, lda,
    (const void**)d_Barray_sgemm, CUDA_R_32F, ldb,
    (const void*)&beta,
    (void**)d_Carray_sgemm, CUDA_R_32F, ldc,
    batchCount,
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  cudaDeviceSynchronize();
  
  // Free device arrays
  cudaFree(d_Aarray_sgemm);
  cudaFree(d_Barray_sgemm);
  cudaFree(d_Carray_sgemm);
}

extern "C" void cublasSgemmBatched_finalize ()
{

  if (alreadyAllocated_sgemm){
  
    cudaFree(Aarray_sgemm);
    cudaFree(Barray_sgemm);
    cudaFree(Carray_sgemm);
    
    cudaFree(d_Aarray_sgemm);
    cudaFree(d_Barray_sgemm);
    cudaFree(d_Carray_sgemm);

  }

  if (alreadyAllocated_sgemm_handle){
    cublasDestroy(handle_sgemm);
  }
  
}
