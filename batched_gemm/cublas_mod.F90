MODULE CUBLAS_MOD
!
! Define the interfaces to the NVIDIA C code
!
INTERFACE
    SUBROUTINE CUDA_SGEMM_BATCHED(&
        & CTA, CTB,               &
        & M, N, K,                &
        & ALPHA,                  &
        & A, LDA, TDA,            &
        & B, LDB, TDB,            &
        & BETA,                   &
        & C, LDC, TDC,            &
        & BATCHCOUNT              &
    &) BIND(C, NAME='cublasSgemmBatched_wrapper')
        USE ISO_C_BINDING
        CHARACTER(1,C_CHAR), VALUE            :: CTA, CTB
        INTEGER(C_INT),      VALUE            :: M, N, K, LDA, LDB, LDC, TDA, TDB, TDC, BATCHCOUNT
        REAL(4),             VALUE            :: ALPHA, BETA
        REAL(4),             DIMENSION(LDA,*) :: A
        REAL(4),             DIMENSION(LDB,*) :: B
        REAL(4),             DIMENSION(LDC,*) :: C
    END SUBROUTINE CUDA_SGEMM_BATCHED
END INTERFACE

INTERFACE
    SUBROUTINE CUDA_HGEMM_BATCHED(&
        & CTA, CTB,               &
        & M, N, K,                &
        & ALPHA,                  &
        & A, LDA, TDA,            &
        & B, LDB, TDB,            &
        & BETA,                   &
        & C, LDC, TDC,            &
        & BATCHCOUNT              &
    &) BIND(C, NAME='cublasHgemmBatched_wrapper')
        USE ISO_C_BINDING
        CHARACTER(1,C_CHAR), VALUE            :: CTA, CTB
        INTEGER(C_INT),      VALUE            :: M, N, K, LDA, LDB, LDC, TDA, TDB, TDC, BATCHCOUNT
        REAL(2),             VALUE            :: ALPHA, BETA
        REAL(2),             DIMENSION(LDA,*) :: A
        REAL(2),             DIMENSION(LDB,*) :: B
        REAL(2),             DIMENSION(LDC,*) :: C
    END SUBROUTINE CUDA_HGEMM_BATCHED
END INTERFACE

END MODULE CUBLAS_MOD
