#===============================================================================
# Makefile for program testing support for HGEMM.
#
# ECMWF/RIKEN-CCS
#===============================================================================

FORTCOMP := pgfortran
NVCCCOMP := nvcc
NVCCCOMPARG := -O3

gpu : FORTCOMPARG = -O3 -acc -lcublas -lcudart -L${CUDA_DIR}/lib64
cpu : FORTCOMPARG = -O3 -lblas

# Define GPU target
gpu: cublasHgemmBatched.o cublasSgemmBatched.o cublasSTCgemmBatched.o cuda_gemm_batched_mod.o main.o
	$(FORTCOMP) $(FORTCOMPARG) -o $@ $^

# Define CPU target
cpu: main.o
	$(FORTCOMP) $(FORTCOMPARG) -o $@ $^

cuda_gemm_batched_mod.o: cublas_mod.o
cublas_mod.o: cublasHgemmBatched.o cublasSgemmBatched.o cublasSTCgemmBatched.o

# Define generic target for Fortran 90
%.o: %.F90
	$(FORTCOMP) $(FORTCOMPARG) -c -DPROBLEM_SIZE=${PROBLEM_SIZE} $< -o $(basename $<).o

%.o: %.cu
	$(NVCCCOMP) $(NVCCCOMPARG) -c $< -o $(basename $<).o

# Define clean task
.PHONY: clean
clean:
	rm -f *.o *.mod gpu cpu

