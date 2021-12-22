//
// Created by navin on 18/11/2021.
//
//#include "../../../../../../../../Android/Sdk/ndk/21.1.6352462/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/include/opencl-c.h"
#define KERNEL 1


#if KERNEL == 1

__kernel void gemm1(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        const int ai = k*M + globalRow,
                  bi = globalCol*K + k;
        acc += A[ai] * B[bi];

    }
    // Store the result
    C[globalCol*M + globalRow] = acc;
}

#endif