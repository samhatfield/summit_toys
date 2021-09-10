MODULE CUDA_GEMM_BATCHED_MOD
  USE CUBLAS_MOD

  IMPLICIT NONE 

  PRIVATE
  PUBLIC CUDA_GEMM_BATCHED

  INTERFACE CUDA_GEMM_BATCHED
    MODULE PROCEDURE CUDA_SGEMM_BATCHED_OVERLOAD
    MODULE PROCEDURE CUDA_HGEMM_BATCHED_OVERLOAD
    MODULE PROCEDURE CUDA_STCGEMM_BATCHED_OVERLOAD
  END INTERFACE CUDA_GEMM_BATCHED

CONTAINS
  SUBROUTINE CUDA_SGEMM_BATCHED_OVERLOAD( &
      & TRANSA, TRANSB, &
      & M, N, K, &
      & ALPHA, &
      & AARRAY, LDA, STRIDEA, &
      & BARRAY, LDB, STRIDEB, &
      & BETA, &
      & CARRAY, LDC, STRIDEC, &
      & BATCHCOUNT)
    CHARACTER,                 INTENT(IN)  :: TRANSA
    CHARACTER,                 INTENT(IN)  :: TRANSB
    INTEGER,                   INTENT(IN)  :: M
    INTEGER,                   INTENT(IN)  :: N
    INTEGER,                   INTENT(IN)  :: K
    REAL(4),                   INTENT(IN)  :: ALPHA
    REAL(4), DIMENSION(:,:,:), INTENT(IN)  :: AARRAY
    INTEGER,                   INTENT(IN)  :: LDA
    INTEGER,                   INTENT(IN)  :: STRIDEA
    REAL(4), DIMENSION(:,:,:), INTENT(IN)  :: BARRAY
    INTEGER,                   INTENT(IN)  :: LDB
    INTEGER,                   INTENT(IN)  :: STRIDEB
    REAL(4),                   INTENT(IN)  :: BETA
    REAL(4), DIMENSION(:,:,:), INTENT(OUT) :: CARRAY
    INTEGER,                   INTENT(IN)  :: LDC
    INTEGER,                   INTENT(IN)  :: STRIDEC
    INTEGER,                   INTENT(IN)  :: BATCHCOUNT

    CALL CUDA_SGEMM_BATCHED( &
      & TRANSA, TRANSB, &
      & M, N, K, &
      & ALPHA, &
      & AARRAY, LDA, STRIDEA, &
      & BARRAY, LDB, STRIDEB, &
      & BETA, &
      & CARRAY, LDC, STRIDEC, &
      & BATCHCOUNT)
  END SUBROUTINE CUDA_SGEMM_BATCHED_OVERLOAD

  SUBROUTINE CUDA_HGEMM_BATCHED_OVERLOAD( &
      & TRANSA, TRANSB, &
      & M, N, K, &
      & ALPHA, &
      & AARRAY, LDA, STRIDEA, &
      & BARRAY, LDB, STRIDEB, &
      & BETA, &
      & CARRAY, LDC, STRIDEC, &
      & BATCHCOUNT)
    CHARACTER,                 INTENT(IN)  :: TRANSA
    CHARACTER,                 INTENT(IN)  :: TRANSB
    INTEGER,                   INTENT(IN)  :: M
    INTEGER,                   INTENT(IN)  :: N
    INTEGER,                   INTENT(IN)  :: K
    REAL(2),                   INTENT(IN)  :: ALPHA
    REAL(2), DIMENSION(:,:,:), INTENT(IN)  :: AARRAY
    INTEGER,                   INTENT(IN)  :: LDA
    INTEGER,                   INTENT(IN)  :: STRIDEA
    REAL(2), DIMENSION(:,:,:), INTENT(IN)  :: BARRAY
    INTEGER,                   INTENT(IN)  :: LDB
    INTEGER,                   INTENT(IN)  :: STRIDEB
    REAL(2),                   INTENT(IN)  :: BETA
    REAL(2), DIMENSION(:,:,:), INTENT(OUT) :: CARRAY
    INTEGER,                   INTENT(IN)  :: LDC
    INTEGER,                   INTENT(IN)  :: STRIDEC
    INTEGER,                   INTENT(IN)  :: BATCHCOUNT

    CALL CUDA_HGEMM_BATCHED( &
      & TRANSA, TRANSB, &
      & M, N, K, &
      & ALPHA, &
      & AARRAY, LDA, STRIDEA, &
      & BARRAY, LDB, STRIDEB, &
      & BETA, &
      & CARRAY, LDC, STRIDEC, &
      & BATCHCOUNT)
  END SUBROUTINE CUDA_HGEMM_BATCHED_OVERLOAD

  SUBROUTINE CUDA_STCGEMM_BATCHED_OVERLOAD( &
      & TRANSA, TRANSB, &
      & M, N, K, &
      & ALPHA, &
      & AARRAY, LDA, STRIDEA, &
      & BARRAY, LDB, STRIDEB, &
      & BETA, &
      & CARRAY, LDC, STRIDEC, &
      & BATCHCOUNT)
    CHARACTER,                 INTENT(IN)  :: TRANSA
    CHARACTER,                 INTENT(IN)  :: TRANSB
    INTEGER,                   INTENT(IN)  :: M
    INTEGER,                   INTENT(IN)  :: N
    INTEGER,                   INTENT(IN)  :: K
    REAL(2),                   INTENT(IN)  :: ALPHA
    REAL(2), DIMENSION(:,:,:), INTENT(IN)  :: AARRAY
    INTEGER,                   INTENT(IN)  :: LDA
    INTEGER,                   INTENT(IN)  :: STRIDEA
    REAL(2), DIMENSION(:,:,:), INTENT(IN)  :: BARRAY
    INTEGER,                   INTENT(IN)  :: LDB
    INTEGER,                   INTENT(IN)  :: STRIDEB
    REAL(2),                   INTENT(IN)  :: BETA
    REAL(4), DIMENSION(:,:,:), INTENT(OUT) :: CARRAY
    INTEGER,                   INTENT(IN)  :: LDC
    INTEGER,                   INTENT(IN)  :: STRIDEC
    INTEGER,                   INTENT(IN)  :: BATCHCOUNT

    CALL CUDA_STCGEMM_BATCHED( &
      & TRANSA, TRANSB, &
      & M, N, K, &
      & ALPHA, &
      & AARRAY, LDA, STRIDEA, &
      & BARRAY, LDB, STRIDEB, &
      & BETA, &
      & CARRAY, LDC, STRIDEC, &
      & BATCHCOUNT)
  END SUBROUTINE CUDA_STCGEMM_BATCHED_OVERLOAD

END MODULE CUDA_GEMM_BATCHED_MOD
