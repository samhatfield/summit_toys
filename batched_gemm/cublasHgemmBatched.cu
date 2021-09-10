//
// Wrapper for cublasHgemm function. 
//
// Alan Gray, NVIDIA
//

#include <stdio.h>
#include "cublas_v2.h" 


bool alreadyAllocated_hgemm=false;
bool alreadyAllocated_hgemm_handle=false;

half **d_Aarray_hgemm;
half **d_Barray_hgemm;
half **d_Carray_hgemm;

half **Aarray_hgemm;
half **Barray_hgemm;
half **Carray_hgemm;

cublasHandle_t handle_hgemm;	

extern "C" void cublasHgemmBatched_wrapper (char transa, char transb, int m, int n,int k, half alpha, const half *A, int lda, int tda, const half *B, int ldb, int tdb, half beta, half *C, int ldc, int tdc, int batchCount)
{

  // printf("CUBLAS m=%d,n=%d,k=%d,batchcount=%d\n",m,n,k,batchCount);

  cublasOperation_t op_t1=CUBLAS_OP_N, op_t2=CUBLAS_OP_N;

  if (transa=='T' || transa=='t')		
    op_t1=CUBLAS_OP_T;

  if (transb=='T' || transb=='t')		
    op_t2=CUBLAS_OP_T;

  if (!alreadyAllocated_hgemm_handle){
    cublasCreate(&handle_hgemm);
    alreadyAllocated_hgemm_handle=true;
  }

  if (!alreadyAllocated_hgemm){
    cudaMallocHost(&Aarray_hgemm,batchCount*sizeof(half*));
    cudaMallocHost(&Barray_hgemm,batchCount*sizeof(half*));
    cudaMallocHost(&Carray_hgemm,batchCount*sizeof(half*));
    alreadyAllocated_hgemm=true;
  }

  cudaMalloc(&d_Aarray_hgemm,batchCount*sizeof(half*));
  cudaMalloc(&d_Barray_hgemm,batchCount*sizeof(half*));
  cudaMalloc(&d_Carray_hgemm,batchCount*sizeof(half*));

  int i;
  for(i=0;i<batchCount;i++){
    Aarray_hgemm[i]=(half*) &(A[i*lda*tda]);
    Barray_hgemm[i]=(half*) &(B[i*ldb*tdb]);
    Carray_hgemm[i]=(half*) &(C[i*ldc*tdc]);
  }
  cudaMemcpy(d_Aarray_hgemm,Aarray_hgemm,batchCount*sizeof(half*),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Barray_hgemm,Barray_hgemm,batchCount*sizeof(half*),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Carray_hgemm,Carray_hgemm,batchCount*sizeof(half*),cudaMemcpyHostToDevice);

  cublasHgemmBatched(handle_hgemm,op_t1,op_t2,m,n,k,&alpha,(const half**) d_Aarray_hgemm,lda, (const half**) d_Barray_hgemm,ldb,&beta,(half**) d_Carray_hgemm,ldc,batchCount);

  //printf("after hgemm\n");
  cudaDeviceSynchronize();
  
  //cudaFree(Aarray_hgemm);
  //cudaFree(Barray_hgemm);
  //cudaFree(Carray_hgemm);
  
  cudaFree(d_Aarray_hgemm);
  cudaFree(d_Barray_hgemm);
  cudaFree(d_Carray_hgemm);
  //cublasDestroy(handle_hgemm);
  
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
