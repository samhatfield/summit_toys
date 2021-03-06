program main
    #if defined _OPENACC
    use openacc, only: acc_get_num_devices, acc_device_nvidia
    use cuda_gemm_batched_mod, only: cuda_gemm_batched
    #endif

    implicit none

    ! Define precision constants
    integer, parameter :: sp = 4
    integer, parameter :: hp = 2

    ! Number of repetitions, batch size and problem size
    integer, parameter :: n_repeat = 100
    integer, parameter :: batch = 1
    integer, parameter :: problem_size = PROBLEM_SIZE

    ! GEMM properties
    character :: transa = "N"
    character :: transb = "N"
    integer, parameter :: m = problem_size
    integer, parameter :: n = problem_size
    integer, parameter :: k = problem_size
    integer, parameter :: lda = m
    integer, parameter :: ldb = k
    integer, parameter :: ldc = m

    ! Single-precision inputs
    real(sp), parameter :: alpha_sp = 1.0
    real(sp), parameter :: beta_sp = 0.0
    real(sp), allocatable :: a_sp(:,:,:)
    real(sp), allocatable :: b_sp(:,:,:)
    real(sp), allocatable :: c_sp(:,:,:)

    ! Half-precision inputs
    real(hp), parameter :: alpha_hp = alpha_sp
    real(hp), parameter :: beta_hp = beta_sp
    real(hp), allocatable :: a_hp(:,:,:)
    real(hp), allocatable :: b_hp(:,:,:)
    real(hp), allocatable :: c_hp(:,:,:)

    ! TensorCore inputs
    real(sp), parameter :: alpha_tc = alpha_sp
    real(sp), parameter :: beta_tc = beta_sp
    real(hp), allocatable :: a_tc(:,:,:)
    real(hp), allocatable :: b_tc(:,:,:)
    real(sp), allocatable :: c_tc(:,:,:)

    integer :: i, j, l, numdevs
    real :: temp

    ! Whether to print the input/output arrays or not
    logical :: write_values = .true.

    ! Maximum number of rows/columns to print
    integer, parameter :: maxprt = 3

    ! Timing variables
    integer :: tic, toc, t_rate

    ! Allocate all arrays on host
    allocate(a_sp(lda,k,batch))
    allocate(b_sp(ldb,n,batch))
    allocate(c_sp(ldc,n,batch))
    allocate(a_hp(lda,k,batch))
    allocate(b_hp(ldb,n,batch))
    allocate(c_hp(ldc,n,batch))
    allocate(a_tc(lda,k,batch))
    allocate(b_tc(ldb,n,batch))
    allocate(c_tc(ldc,n,batch))

    ! Enter data on device
    !$acc enter data create(a_sp,b_sp,c_sp,a_hp,b_hp,c_hp,a_tc,b_tc,c_tc)

    ! Initialise double- and half-precision matrices
    do i = 1, lda
        do j = 1, k
            do l = 1, batch
                call random_number(temp)
                a_sp(i,j,l) = temp
                a_hp(i,j,l) = temp
                a_tc(i,j,l) = temp
            end do
        end do
    end do
    do i = 1, ldb
        do j = 1, n
            do l = 1, batch
                call random_number(temp)
                b_sp(i,j,l) = temp
                b_hp(i,j,l) = temp
                b_tc(i,j,l) = temp
            end do
        end do
    end do

    ! Copy data to GPU
    !$acc update device(a_sp,b_sp,a_hp,b_hp,a_tc,b_tc)

    print "(A,X,I4)", "Problem size", problem_size
    print "(A,X,I4)", "Batch size", batch

    ! Print number of devices available
    #if defined _OPENACC
    numdevs = acc_get_num_devices(acc_device_nvidia)
    print "(I4,X,A)", NUMDEVS, 'GPUs available'
    #endif

    if (write_values) then
        print *
        print "(A)", "Single-precision matrices"
        print "(A)", "A"
        do i = 1, min(maxprt,lda)
            print *, a_sp(i,:min(maxprt,k),1)
        end do
        print "(A)", "B"
        do i = 1, min(maxprt,ldb)
            print *, b_sp(i,:min(maxprt,n),1)
        end do
    
        print *, ""
        print "(A)", "Half-precision matrices"
        print "(A)", "A"
        do i = 1, min(maxprt,lda)
            print *, a_hp(i,:min(maxprt,k),1)
        end do
        print "(A)", "B"
        do i = 1, min(maxprt,ldb)
            print *, b_hp(i,:min(maxprt,n),1)
        end do
    end if

    ! Benchmark single-precision
    call system_clock(tic)
    do i = 1, n_repeat
        #if defined _OPENACC
        call cuda_gemm_batched("N", "N",     &
                             & m, n, k,      &
                             & alpha_sp,     &
                             & a_sp, lda, k, &
                             & b_sp, ldb, n, &
                             & beta_sp,      &
                             & c_sp, ldc, n, &
                             & batch)
        #else
        do j = 1, batch
            call sgemm("N", "N",         &
                     & m, n, k,          &
                     & alpha_sp,         &
                     & a_sp(:,:,j), lda, &
                     & b_sp(:,:,j), ldb, &
                     & beta_sp,          &
                     & c_sp(:,:,j), ldc)
        end do
        #endif
    end do
    call system_clock(toc, t_rate)

    print *, " "
    print "(A,X,F9.2,X,A)", "Single-precision took", (toc - tic) / real(t_rate,8), "s"

    ! Benchmark half-precision
    call system_clock(tic)
    do i = 1, n_repeat
        #if defined _OPENACC
        call cuda_gemm_batched("N", "N",     &
                             & m, n, k,      &
                             & alpha_hp,     &
                             & a_hp, lda, k, &
                             & b_hp, ldb, n, &
                             & beta_hp,      &
                             & c_hp, ldc, n, &
                             & batch)
        #else
        print "(A)", "CPU version does not support HGEMM"
        #endif
    end do
    call system_clock(toc, t_rate)

    print *, " "
    print "(A,X,F9.2,X,A)", "Half-precision took", (toc - tic) / real(t_rate,8), "s"

    ! Benchmark TensorCore
    call system_clock(tic)
    do i = 1, n_repeat
        #if defined _OPENACC
        call cuda_gemm_batched("N", "N",     &
                             & m, n, k,      &
                             & alpha_tc,     &
                             & a_tc, lda, k, &
                             & b_tc, ldb, n, &
                             & beta_tc,      &
                             & c_tc, ldc, n, &
                             & batch)
        #else
        print "(A)", "CPU version does not support TensorCore"
        #endif
    end do
    call system_clock(toc, t_rate)

    print *, " "
    print "(A,X,F9.2,X,A)", "TensorCore took", (toc - tic) / real(t_rate,8), "s"

    if (write_values) then
        print *, ""
        print "(A)", "Single-precision C"
        do i = 1, min(maxprt,ldc)
            print *, c_sp(i,:min(maxprt,n),1)
        end do
        print *, ""
        print "(A)", "Half-precision C"
        do i = 1, min(maxprt,ldc)
            print *, real(c_hp(i,:min(maxprt,n),1),sp)
        end do
        print *, ""
        print "(A)", "TensorCore C"
        do i = 1, min(maxprt,ldc)
            print *, c_tc(i,:min(maxprt,n),1)
        end do
    end if

    print *, " "
    print "(A)", "Relative errors compared with single precision (%)"
    print "(A)", "Half precision"
    print "(F10.8)", avg_rel_diff(c_sp, real(c_hp,sp))
    print "(A)", "TensorCore"
    print "(F10.8)", avg_rel_diff(c_sp, c_tc)
    
    ! Delete data from device
    !$acc exit data delete(a_sp,b_sp,c_sp,a_hp,b_hp,c_hp,a_tc,b_tc,c_tc)

contains
    real(sp) function avg_rel_diff(array_1, array_2)
        real(sp), intent(in) :: array_1(:,:,:), array_2(:,:,:)

        avg_rel_diff = sum(abs((array_1 - array_2)/array_1))/(100.0_sp*size(array_1))

!        frobenius_norm = sqrt(sum(in_array**2.0_sp))
    end function avg_rel_diff
end program main

