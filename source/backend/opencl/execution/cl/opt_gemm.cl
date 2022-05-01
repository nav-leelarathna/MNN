//
// Created by navin on 18/11/2021.
//
//#include "../../../../../../../../Android/Sdk/ndk/21.1.6352462/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/include/opencl-c.h"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable


__kernel void baseline(const int M, const int N, const int K,
                       const __global float* A,
                       const __global float* B,
                       __global float* C) {

    const int globalCol = get_global_id(0);
    const int globalRow = get_global_id(1);
    // Compute a single element
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
//        const int ai = k*M + globalRow,bi = globalCol*K + k;
        const int ai = globalRow * K + k;
        const int bi = globalCol + k * N;
        acc += A[ai] * B[bi];
    }
    C[globalRow*N + globalCol] = acc;
}

#define TS1 16
// Tiled and coalesced version

__kernel void gemm1(const int M, const int N, const int K,
                    const __global float* A,
                    const __global float* B,
                    __global float* C) {
//    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
//        printf("%d\n", TS1);
//    }
    // Thread identifiers
    const int row = get_local_id(1); // Local row ID (max: TS1)
    const int col = get_local_id(0); // Local col ID (max: TS1)
    const int globalRow = TS1*get_group_id(1) + row; // Row ID of C (0..M)
    const int globalCol = TS1*get_group_id(0) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS1*TS1 elements of A and B
    __local float Asub[TS1][TS1];
    __local float Bsub[TS1][TS1];
    // Initialise the accumulation register
    float acc = 0.0f;
    int numTiles = K / TS1;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        const int tiledRow = TS1*t + row;  // row for B
        const int tiledCol = TS1*t + col;  // col for A
        // globalCol and globalRow are the output elements in C
        Asub[row][col] = A[globalRow*K + tiledCol];
        Bsub[row][col] = B[tiledRow*N + globalCol];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS1; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalRow*N + globalCol] = acc;
}

#define TS2 32
#define WPT2 8
#define RTS2 4


__kernel void gemm2(const int M, const int N, const int K,
                       const __global float* A,
                       const __global float* B,
                       __global float* C) {

    // Thread identifiers
    const int row = get_local_id(1); // Local row ID (max: TS2)
    const int col = get_local_id(0); // Local col ID (max: TS2/WPT2 == RTS2)
    const int globalRow = TS2*get_group_id(1) + row; // Row ID of C (0..M)
    const int globalCol = TS2*get_group_id(0) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS2*TS2 elements of A and B
    __local float Asub[TS2][TS2];
    __local float Bsub[TS2][TS2];

    // Initialise the accumulation registers
    float acc[WPT2];
    for (int w=0; w<WPT2; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    //const int numTiles = get_num_groups(1);
    int numTiles = K / TS2;
    for (int t=0; t<numTiles; t++) {
        //PRINT_IF("LOOP0");
        // Load one tile of A and B into local memory
        for (int w=0; w<WPT2; w++) {
            const int tiledRow = TS2*t + row;
            const int tiledCol = TS2*t + col;
            Asub[row][col + w*RTS2] = A[globalRow * K + (tiledCol + w*RTS2)];
            Bsub[row][col + w*RTS2] = B[tiledRow * N  +(globalCol + w * RTS2)];
        }
        //PRINT_IF("LOOP1");

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS2; k++) {
            for (int w=0; w<WPT2; w++) {
                acc[w] += Asub[row][k] * Bsub[k][col + w * RTS2];
            }
        }
        //PRINT_IF("LOOP2");

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
    for (int w=0; w<WPT2; w++) {
        C[globalRow * N + (globalCol + w *RTS2)] = acc[w];
    }
}



#define TS3 16
#define WIDTH3 2
#if WIDTH3 == 1
    typedef float floatX;
#elif WIDTH3 == 2
    typedef float2 floatX;
#elif WIDTH3 == 4
    typedef float4 floatX;
#elif WIDTH3 == 8
    typedef float8 floatX;
#elif WIDTH3 == 16
    typedef float16 floatX;
#endif

__kernel void gemm3(const int M, const int N, const int K,
                       const __global floatX* A,
const __global floatX* B,
__global floatX* C) {

    // Thread identifiers
    const int row = get_local_id(1); // Local row ID (max: TS)
    const int col = get_local_id(0); // Local col ID (max: TS / WIDTH)
    const int globalRow = TS3*get_group_id(1) + row; // 0..M
    const int globalCol = (TS3/WIDTH3)*get_group_id(0) + col; // 0..N/WIDTH

    // Local memory to fit a tile of TS*TS elements of A and B
    __local floatX Asub[TS3][TS3/WIDTH3];
    __local floatX Bsub[TS3][TS3/WIDTH3];

    // Initialise the accumulation registers
    #if WIDTH3 == 1
    floatX acc = 0.0f;
    #elif WIDTH3 == 2
    floatX acc = { 0.0f, 0.0f };
    #elif WIDTH3 == 4
    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    #elif WIDTH3 == 8
    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f ,0.0f, 0.0f, 0.0f, 0.0f };
    #elif WIDTH3 == 16
            floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f ,0.0f, 0.0f, 0.0f, 0.0f , 0.0f, 0.0f, 0.0f, 0.0f ,0.0f, 0.0f, 0.0f, 0.0f };
    #endif

    // Loop over all tiles
    const int numTiles = K/TS3;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS3*t + row;
        const int tiledCol = (TS3/WIDTH3)*t + col;  // for Asub
//        Asub[col][row] = A[tiledCol*(M/WIDTH3) + globalRow];
        Asub[row][col] = A[globalRow * (K/WIDTH3) + tiledCol];
//        Bsub[col][row] = B[globalCol*(K/WIDTH3) + tiledRow];
        Bsub[row][col] = B[tiledRow * (N/WIDTH3) + globalCol];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        floatX vecA, vecB;
        float valA;
        for (int k=0; k<TS3/WIDTH3; k++) {
            vecA = Asub[row][k];
            for (int w=0; w<WIDTH3; w++) {
                vecB = Bsub[WIDTH3*k + w][col];
                #if WIDTH3 == 1
                    valA = vecA;
                #elif WIDTH3 == 2
                switch (w) {
                        case 0: valA = vecA.x; break;
                        case 1: valA = vecA.y; break;
                            }
                #elif WIDTH3 == 4
                switch (w) {
                    case 0: valA = vecA.x; break;
                    case 1: valA = vecA.y; break;
                    case 2: valA = vecA.z; break;
                    case 3: valA = vecA.w; break;
                }
                #elif WIDTH3 == 8
                switch (w) {
                                  case 0: valA = vecA.s0; break;
                                  case 1: valA = vecA.s1; break;
                                  case 2: valA = vecA.s2; break;
                                  case 3: valA = vecA.s3; break;
                                  case 4: valA = vecA.s4; break;
                                  case 5: valA = vecA.s5; break;
                                  case 6: valA = vecA.s6; break;
                                  case 7: valA = vecA.s7; break;
                                }

                #elif WIDTH3 == 16
                    switch (w) {
                        case 0: valA = vecA.s0; break;
                        case 1: valA = vecA.s1; break;
                        case 2: valA = vecA.s2; break;
                        case 3: valA = vecA.s3; break;
                        case 4: valA = vecA.s4; break;
                        case 5: valA = vecA.s5; break;
                        case 6: valA = vecA.s6; break;
                        case 7: valA = vecA.s7; break;
                        case 8: valA = vecA.s8; break;
                        case 9: valA = vecA.s9; break;
                        case 10: valA = vecA.sa; break;
                        case 11: valA = vecA.sb; break;
                        case 12: valA = vecA.sc; break;
                        case 13: valA = vecA.sd; break;
                        case 14: valA = vecA.se; break;
                        case 15: valA = vecA.sf; break;
                                }
                #endif
                acc += valA * vecB;
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
//    C[globalCol*(M/WIDTH3) + globalRow] = acc;
    C[globalRow * (N/WIDTH3) + globalCol] = acc;
}


#define TSM4 32                 // The tile-size in dimension M
#define TSN4 32                 // The tile-size in dimension N
#define TSK4 32                 // The tile-size in dimension K
#define WPTN4 1                // The work-per-thread in dimension N
#define WPTM4 8
#define RTSM4 (TSM4/WPTM4)
#define RTSN4 (TSN4/WPTN4)        // The reduced tile-size in dimension N
#define LPT4 ((TSK4*TSM4)/(RTSM4*RTSN4)) // The loads-per-thread for a tile

__kernel void gemm4(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int row = get_local_id(1);                  // max TSM / WPTM4
    const int col = get_local_id(0);                  // max TSN
    const int globalRow = TSM4*get_group_id(1) + row;  // max M
    const int globalCol = TSN4*get_group_id(0) + col;  // max N

    // Local memory to fit a tile of A and B
    __local float Asub[TSM4][TSK4+2];
    __local float Bsub[TSK4][TSN4];

    // Initialise the accumulation registers
    float acc[WPTM4];
    for (int w=0; w<WPTM4; w++) {
        acc[w] = 0.0f;
    }
    // Loop over all tiles
    int numTiles = K/TSK4;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        for (int l=0; l<LPT4; l++) {
            int tiledIndex = TSK4 * t + row + l*RTSM4;  // gets row of A
            int indexA = tiledIndex*M + TSN4*get_group_id(0) + col;
            int indexB = tiledIndex*N + TSN4*get_group_id(1) + col;
            Asub[col][row+ l*RTSM4]  = A[indexA];
            Bsub[row+ l*RTSM4][col] = B[indexB];
        }
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TSK4; k++) {
            for (int w=0; w<WPTM4; w++) {
                acc[w] += Asub[row + w*RTSM4][k] * Bsub[k][col];
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
    for (int w=0; w<WPTM4; w++) {
        C[(globalRow + w * RTSM4) * N + globalCol] = acc[w];
    }
}

#define TSM5 32                // The tile-size in dimension M
#define TSN5 32                // The tile-size in dimension N
#define TSK5 16                 // The tile-size in dimension K
#define WPTM5 4                 // The work-per-thread in dimension M
#define WPTN5 4                 // The work-per-thread in dimension N
#define RTSM5 (TSM5/WPTM5)        // The reduced tile-size in dimension M
#define RTSN5 (TSN5/WPTN5)        // The reduced tile-size in dimension N
#define LPTA5 ((TSK5*TSM5)/(RTSM5*RTSN5)) // Loads-per-thread for A
#define LPTB5 ((TSK5*TSN5)/(RTSM5*RTSN5)) // Loads-per-thread for B


__kernel void gemm5(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
   // Thread identifiers
    const int tidm = get_local_id(1); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(0); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM5*get_group_id(1); // Work-group offset
    const int offsetN = TSN5*get_group_id(0); // Work-group offset
    // Local memory to fit a tile of A and B
    //__local float Asub[TSK5][TSM5];
    //__local float Bsub[TSN5][TSK5];
    __local float Asub[TSM5][TSK5+2];
    __local float Bsub[TSK5][TSN5];

    // Allocate register space
    float Areg;
    float Breg[WPTM5];
    float acc[WPTN5][WPTM5];
    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN5; wn++) {
        for (int wm=0; wm<WPTM5; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }
    // Loop over all tiles
    int numTiles = K/TSK5;
    for (int t=0; t<numTiles; t++) {
        // Load one tle of A and B into local memory
        for (int la=0; la<LPTA5; la++) {
            int tid = tidm*RTSN5 + tidn;
            int id = la*RTSM5*RTSN5 + tid;
            int col = id % TSN5;
            int row = id / TSN5;
            int tiledIndex = TSK5*t + row;
            Asub[col][row] = A[tiledIndex*M + offsetM + col];
            Bsub[row][col] = B[tiledIndex*N + offsetN + col];
        }
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Loop over the values of a single tile
        for (int k=0; k<TSK5; k++) {
            // Cache the values of Bsub in registers
            for (int wm=0; wm<WPTM5; wm++) {
                int row = tidm + wm*RTSN5;
                Breg[wm] = Asub[row][k];
            }

            // Perform the computation
            for (int wn=0; wn<WPTN5; wn++) {
                int col = tidn + wn*RTSN5;
                Areg = Bsub[k][col];
                for (int wm=0; wm<WPTM5; wm++) {
                    acc[wn][wm] += Areg * Breg[wm];
                }
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
    for (int wn=0; wn<WPTN5; wn++) {
        int globalCol = offsetN + tidn + wn*RTSN5;
        for (int wm=0; wm<WPTM5; wm++) {
            int globalRow = offsetM + tidm + wm*RTSM5;
            C[globalRow*N + globalCol] = acc[wn][wm];
        }
    }
}


// Wider loads combined with 2D register blocking
#define TSM6 32                // The tile-size in dimension M
#define TSN6 32                // The tile-size in dimension N
#define TSK6 16                 // The tile-size in dimension K
#define WPTM6 4                 // The work-per-thread in dimension M
#define WPTN6 4                 // The work-per-thread in dimension N
#define RTSM6 (TSM6/WPTM6)        // The reduced tile-size in dimension M
#define RTSN6 (TSN6/WPTN6)        // The reduced tile-size in dimension N
#define LPTA6 ((TSK6*TSM6)/(RTSM6*RTSN6)) // Loads-per-thread for A
#define LPTB6 ((TSK6*TSN6)/(RTSM6*RTSN6)) // Loads-per-thread for B
#define WIDTH6 4
#if WIDTH6 == 1
    typedef float floatX6;
#elif WIDTH6 == 2
    typedef float2 floatX6;
#elif WIDTH6 == 4
    typedef float4 floatX6;
#elif WIDTH6 == 8
    typedef float8 floatX6;
#elif WIDTH6 == 16
    typedef float16 floatX6;
#endif

__kernel void gemm6(const int M, const int N, const int K,
                      const __global floatX6* A,
    const __global floatX6* B,
    __global float* C) {

    // Thread identifiers
    const int tidn = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidm = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetN = TSN6*get_group_id(0); // Work-group offset
    const int offsetM = TSM6*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    // __local float Asub[TSK6][TSM6];
    // __local float Bsub[TSK6][TSN6];   // resolves small issue with bank conflicts
    __local float Asub[TSK6][TSM6+2];
    __local float Bsub[TSK6][TSN6+2];

    // Allocate register space
    float Breg;
    float Areg[WPTM6];
    float acc[WPTN6][WPTM6];

    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN6; wn++) {
        for (int wm=0; wm<WPTM6; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }

    // Loop over all tiles
    int numTiles = K/TSK6;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int la=0; la<LPTA6/WIDTH6; la++) {
            int tid = tidm*RTSN6 + tidn;
            int id = la*RTSM6*RTSN6 + tid;
            int col = id % (TSN6/WIDTH6);
            int row = id / (TSN6/WIDTH6);

            // Load the values (wide vector load)
            int tiledIndex = TSK6*t + row;
            floatX6 vecB = B[tiledIndex*(N/WIDTH6) + offsetN/WIDTH6 + col];
            floatX6 vecA = A[tiledIndex*(M/WIDTH6) + offsetM/WIDTH6 + col];

            // Store the loaded vectors into local memory
            (*((__local floatX6*) & Asub[row][WIDTH6 * col])) = vecA;
            (*((__local floatX6*) & Bsub[row][WIDTH6 * col])) = vecB;
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK6; k++) {
            // Cache the values of Asub in registers
            for (int wm=0; wm<WPTM6; wm++) {
                int col = tidm + wm*RTSM6;
                Areg[wm] = Asub[k][col];
            }

            // Perform the computation
            for (int wn=0; wn<WPTN6; wn++) {
                int col = tidn + wn*RTSN6;
                Breg = Bsub[k][col];
                for (int wm=0; wm<WPTM6; wm++) {
                    acc[wn][wm] += Breg * Areg[wm];
                }
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wn=0; wn<WPTN6; wn++) {
        int globalCol = offsetN + tidn + wn*RTSN6;
        for (int wm=0; wm<WPTM6; wm++) {
            int globalRow = offsetM + tidm + wm*RTSM6;
            C[globalRow*N + globalCol] = acc[wn][wm];
        }
    }
}







#define TSM7 32                // The tile-size in dimension M
#define TSN7 32                // The tile-size in dimension N
#define TSK7 16                 // The tile-size in dimension K
#define WPTM7 4                 // The work-per-thread in dimension M
#define WPTN7 4                 // The work-per-thread in dimension N
#define RTSM7 (TSM7/WPTM7)        // The reduced tile-size in dimension M
#define RTSN7 (TSN7/WPTN7)        // The reduced tile-size in dimension N
#define LPTA7 ((TSK7*TSM7)/(RTSM7*RTSN7)) // Loads-per-thread for A
#define LPTB7 ((TSK7*TSN7)/(RTSM7*RTSN7)) // Loads-per-thread for B
#define WIDTH7 8
#if WIDTH7 == 1
    typedef float floatX7;
#elif WIDTH7 == 2
    typedef float2 floatX7;
#elif WIDTH7 == 4
    typedef float4 floatX7;
#elif WIDTH7 == 8
    typedef float8 floatX7;
#elif WIDTH7 == 16
    typedef float16 floatX7;
#endif

__kernel void gemm7(const int M, const int N, const int K,
                    const __global floatX7* A,
const __global floatX7* B,
__global float* C) {

    // Thread identifiers
    const int tidn = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidm = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetN = TSN7*get_group_id(0); // Work-group offset
    const int offsetM = TSM7*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[2][TSK7*TSM7];
    __local float Bsub[2][TSK7*TSN7];

    for (int la=0; la<LPTA7/WIDTH7; la++) {
        int tid = tidm*RTSN7 + tidn;
        int id = la*RTSM7*RTSN7 + tid;
        int col = id % (TSN7/WIDTH7);
        int row = id / (TSN7/WIDTH7);
        // Load the values (wide vector load)
        int tiledIndex =  row;
        floatX7 vecB = B[tiledIndex*(N/WIDTH7) + offsetN/WIDTH7 + col];
        floatX7 vecA = A[tiledIndex*(M/WIDTH7) + offsetM/WIDTH7 + col];
        // Store the loaded vectors into local memory
        (*((__local floatX7*) & Asub[0][row * TSM7 + col * WIDTH7])) = vecA;
        (*((__local floatX7*) & Bsub[0][row * TSN7 + col * WIDTH7])) = vecB;
    }

    float Breg;
    float Areg[WPTM7];
    float acc[WPTN7][WPTM7];

    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN7; wn++) {
        for (int wm=0; wm<WPTM7; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }

    // Loop over all tiles
    int numTiles = K/TSK7;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        int tt = t + 1;
        if(tt < numTiles){
            for (int la=0; la<LPTA7/WIDTH7; la++) {
                int tid = tidm*RTSN7 + tidn;
                int id = la*RTSM7*RTSN7 + tid;
                int col = id % (TSN7/WIDTH7);
                int row = id / (TSN7/WIDTH7);

                // Load the values (wide vector load)
                int tiledIndex = TSK7*tt + row;
                floatX7 vecB = B[tiledIndex*(N/WIDTH7) + offsetN/WIDTH7 + col];
                floatX7 vecA = A[tiledIndex*(M/WIDTH7) + offsetM/WIDTH7 + col];

                // Store the loaded vectors into local memory
                (*((__local floatX7*) & Asub[tt %2][row*TSM7 + WIDTH7 * col])) = vecA;
                (*((__local floatX7*) & Bsub[tt %2][row*TSN7 + WIDTH7 * col])) = vecB;
            }
        }


        // Synchronise to make sure the tile is loaded
        // barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK7; k++) {
            // Cache the values of Asub in registers
            for (int wm=0; wm<WPTM7; wm++) {
                int col = tidm + wm*RTSM7;
                Areg[wm] = Asub[t%2][k*TSM7+col];
            }

            // Perform the computation
            for (int wn=0; wn<WPTN7; wn++) {
                int col = tidn + wn*RTSN7;
                Breg = Bsub[t%2][k*TSN7 + col];
                for (int wm=0; wm<WPTM7; wm++) {
                    acc[wn][wm] += Breg * Areg[wm];
                }
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wn=0; wn<WPTN7; wn++) {
        int globalCol = offsetN + tidn + wn*RTSN7;
        for (int wm=0; wm<WPTM7; wm++) {
            int globalRow = offsetM + tidm + wm*RTSM7;
            C[globalRow*N + globalCol] = acc[wn][wm];
        }
    }
}


#define TSM8 32                // The tile-size in dimension M
#define TSN8 32                // The tile-size in dimension N
#define TSK8 16                 // The tile-size in dimension K
#define WPTM8 4                 // The work-per-thread in dimension M
#define WPTN8 4                 // The work-per-thread in dimension N
#define RTSM8 (TSM8/WPTM8)        // The reduced tile-size in dimension M
#define RTSN8 (TSN8/WPTN8)        // The reduced tile-size in dimension N
#define LPTA8 ((TSK8*TSM8)/(RTSM8*RTSN8)) // Loads-per-thread for A
#define LPTB8 ((TSK8*TSN8)/(RTSM8*RTSN8)) // Loads-per-thread for B
#define WIDTH8 8

#if WIDTH8 == 1
    typedef float floatX8;
#elif WIDTH8 == 2
    typedef float2 floatX8;
#elif WIDTH8 == 4
    typedef float4 floatX8;
#elif WIDTH8 == 8
    typedef float8 floatX8;
#elif WIDTH8 == 16
    typedef float16 floatX8;
#endif

__kernel void gemm8(const int M, const int N, const int K,
                    const __global floatX8* A,
const __global floatX8* B,
__global float* C) {

    // Thread identifiers
    const int tidn = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidm = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetN = TSN8*get_group_id(0); // Work-group offset
    const int offsetM = TSM8*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[2][TSK8*TSM8];
    __local float Bsub[2][TSK8*TSN8];

    for (int la=0; la<LPTA8/WIDTH8; la++) {
        int tid = tidm*RTSN8 + tidn;
        int id = la*RTSM8*RTSN8 + tid;
        int col = id % (TSN8/WIDTH8);
        int row = id / (TSN8/WIDTH8);
        // Load the values (wide vector load)
        int tiledIndex =  row;
        floatX8 vecB = B[tiledIndex*(N/WIDTH8) + offsetN/WIDTH8 + col];
        floatX8 vecA = A[tiledIndex*(M/WIDTH8) + offsetM/WIDTH8 + col];
        // Store the loaded vectors into local memory
        (*((__local floatX8*) & Asub[0][row * TSM8 + col * WIDTH8])) = vecA;//convert(vecA);
        (*((__local floatX8*) & Bsub[0][row * TSN8 + col * WIDTH8])) = vecB;//convert(vecB);
    }

    float Breg;
    float Areg[WPTM8];
    float acc[WPTN8][WPTM8];

    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN8; wn++) {
        for (int wm=0; wm<WPTM8; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }

    // Loop over all tiles
    int numTiles = K/TSK8;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        int tt = t + 1;
        if(tt < numTiles){
            for (int la=0; la<LPTA8/WIDTH8; la++) {
                int tid = tidm*RTSN8 + tidn;
                int id = la*RTSM8*RTSN8 + tid;
                int col = id % (TSN8/WIDTH8);
                int row = id / (TSN8/WIDTH8);

                // Load the values (wide vector load)
                int tiledIndex = TSK8*tt + row;
                floatX8 vecB = B[tiledIndex*(N/WIDTH8) + offsetN/WIDTH8 + col];
                floatX8 vecA = A[tiledIndex*(M/WIDTH8) + offsetM/WIDTH8 + col];

                // Store the loaded vectors into local memory
                (*((__local floatX8*) & Asub[tt %2][row*TSM8 + WIDTH8 * col])) =vecA;// convert(vecA);
                (*((__local floatX8*) & Bsub[tt %2][row*TSN8 + WIDTH8 * col])) = vecB;// convert(vecB);
            }
        }


        // Synchronise to make sure the tile is loaded
        // barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK8; k++) {
            // Cache the values of Asub in registers
            for (int wm=0; wm<WPTM8; wm++) {
                int col = tidm + wm*RTSM8;
                Areg[wm] = Asub[t%2][k*TSM8+col];
            }

            // Perform the computation
            for (int wn=0; wn<WPTN8; wn++) {
                int col = tidn + wn*RTSN8;
                Breg = Bsub[t%2][k*TSN8 + col];
                for (int wm=0; wm<WPTM8; wm++) {
                    acc[wn][wm] += Breg * Areg[wm];
                }
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wn=0; wn<WPTN8; wn++) {
        int globalCol = offsetN + tidn + wn*RTSN8;
        for (int wm=0; wm<WPTM8; wm++) {
            int globalRow = offsetM + tidm + wm*RTSM8;
            C[globalRow*N + globalCol] = convert_float(acc[wn][wm]);
        }
    }
}

__kernel void gemm8Old(const int M, const int N, const int K,
                    const __global floatX8* A,
const __global floatX8* B,
__global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM8*get_group_id(0); // Work-group offset
    const int offsetN = TSN8*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[2][TSK8*TSM8];
    __local float Bsub[2][TSK8*TSN8+2];

    // load the first tile
    for (int la=0; la<LPTA8/WIDTH8; la++) {
        int tid = tidn*RTSM8 + tidm;
        int id = la*RTSN8*RTSM8 + tid;
        int row = id % (TSM8/WIDTH8);
        int col = id / (TSM8/WIDTH8);

        // Load the values (wide vector load)
        int tiledIndex =  col;
        int indexA =tiledIndex*(M/WIDTH8) + offsetM/WIDTH8 + row;
        int indexB = tiledIndex*(N/WIDTH8) + offsetN/WIDTH8 + row;

        floatX8 vecA = A[indexA];
        floatX8 vecB = B[indexB];
        // Store the loaded vectors into local memory
        (*((__local floatX8*) & Asub[0][WIDTH8*row + col *TSM8])) = vecA;
        (*((__local floatX8*) & Bsub[0][WIDTH8*row + col * TSN8])) = vecB;
    }

    // Allocate register space
    float Areg;
    float Breg[WPTN8];
    float acc[WPTM8][WPTN8];

    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM8; wm++) {
        for (int wn=0; wn<WPTN8; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
    int numTiles = K/TSK8;
    for (int t=0; t<numTiles; t++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        // Load one tile of A and B into local memory
        int tt = t + 1;
        if (tt < numTiles){
        for (int la=0; la<LPTA8/WIDTH8; la++) {
            int tid = tidn*RTSM8 + tidm;
            int id = la*RTSN8*RTSM8 + tid;
            int row = id % (TSM8/WIDTH8);
            int col = id / (TSM8/WIDTH8);

            // Load the values (wide vector load)
            int tiledIndex = TSK8*tt + col;
            int indexA =tiledIndex*(M/WIDTH8) + offsetM/WIDTH8 + row;
            int indexB = tiledIndex*(N/WIDTH8) + offsetN/WIDTH8 + row;


            floatX8 vecA = A[indexA];
            floatX8 vecB = B[indexB];

            // Store the loaded vectors into local memory
            (*((__local floatX8*) & Asub[tt % 2][WIDTH8*row + col *TSM8])) = vecA;
            (*((__local floatX8*) & Bsub[tt % 2][WIDTH8*row + col * TSN8])) = vecB;
            }
        }
            // Loop over the values of a single tile
        for (int k=0; k<TSK8; k++) {
        // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN8; wn++) {
                int col = tidn + wn*RTSN8;
                Breg[wn] = Bsub[t%2][col + k * TSN8];
            }

                // Perform the computation
            for (int wm=0; wm<WPTM8; wm++) {
                int row = tidm + wm*RTSM8;
                Areg = Asub[t%2][row + k * TSM8];
                for (int wn=0; wn<WPTN8; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
    }

    // Store the final results in C
    for (int wm=0; wm<WPTM8; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM8;
        for (int wn=0; wn<WPTN8; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN8;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}



__kernel void gemm8_mad24(const int M, const int N, const int K,
                    const __global floatX8* A,
const __global floatX8* B,
__global float* C) {

// Thread identifiers
const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
//    const int offsetM = TSM8*get_group_id(0); // Work-group offset
//    const int offsetN = TSN8*get_group_id(1); // Work-group offset
const int offsetM = mul24(TSM8, get_group_id(0)); // Work-group offset
const int offsetN = mul24(TSN8, get_group_id(1)); // Work-group offset

// Local memory to fit a tile of A and B
__local float Asub[2][TSK8*TSM8];
__local float Bsub[2][TSK8*TSN8+2];

// load the first tile
for (int la=0; la<LPTA8/WIDTH8; la++) {
//        int tid = tidn*RTSM8 + tidm;
int tid = mad24(tidn, RTSM8, tidm);
//        int id = la*RTSN8*RTSM8 + tid;
int id = mad24(la, mul24(RTSN8, RTSM8), tid);
int row = id % (TSM8/WIDTH8);
int col = id / (TSM8/WIDTH8);

// Load the values (wide vector load)
int tiledIndex =  col;
//        int indexA =tiledIndex*(M/WIDTH8) + offsetM/WIDTH8 + row;
//        int indexB = tiledIndex*(N/WIDTH8) + offsetN/WIDTH8 + row;
int indexA = mad24(tiledIndex, (M/WIDTH8), row) + offsetM / WIDTH8;
int indexB = mad24(tiledIndex, (N/WIDTH8), row) + offsetN / WIDTH8;

floatX8 vecA = A[indexA];
floatX8 vecB = B[indexB];
// Store the loaded vectors into local memory
//        (*((__local floatX8*) & Asub[0][WIDTH8*row + col *TSM8])) = vecA;
//        (*((__local floatX8*) & Bsub[0][WIDTH8*row + col * TSN8])) = vecB;
(*((__local floatX8*) & Asub[0][mad24(WIDTH8, row, mul24(col, TSM8))])) = vecA;
(*((__local floatX8*) & Bsub[0][mad24(WIDTH8, row, mul24(col, TSN8))])) = vecB;
}

// Allocate register space
float Areg;
float Breg[WPTN8];
float acc[WPTM8][WPTN8];

// Initialise the accumulation registers
for (int wm=0; wm<WPTM8; wm++) {
for (int wn=0; wn<WPTN8; wn++) {
acc[wm][wn] = 0.0f;
}
}

// Loop over all tiles
int numTiles = K/TSK8;
for (int t=0; t<numTiles; t++) {
barrier(CLK_LOCAL_MEM_FENCE);
// Load one tile of A and B into local memory
int tt = t + 1;
if (tt < numTiles){
for (int la=0; la<LPTA8/WIDTH8; la++) {
//            int tid = tidn*RTSM8 + tidm;
int tid = mad24(tidn, RTSM8, tidm);
//            int id = la*RTSN8*RTSM8 + tid;
int id = mad24(la, mul24(RTSN8, RTSM8), tid);
int row = id % (TSM8/WIDTH8);
int col = id / (TSM8/WIDTH8);

// Load the values (wide vector load)
//            int tiledIndex = TSK8*tt + col;
int tiledIndex = mad24(TSK8, tt, col);
//            int indexA =tiledIndex*(M/WIDTH8) + offsetM/WIDTH8 + row;
//            int indexB = tiledIndex*(N/WIDTH8) + offsetN/WIDTH8 + row;
int indexA = mad24(tiledIndex, (M/WIDTH8), row) + offsetM / WIDTH8;
int indexB = mad24(tiledIndex, (N/WIDTH8), row) + offsetN / WIDTH8;


floatX8 vecA = A[indexA];
floatX8 vecB = B[indexB];

// Store the loaded vectors into local memory
//            (*((__local floatX8*) & Asub[tt % 2][WIDTH8*row + col *TSM8])) = vecA;
//            (*((__local floatX8*) & Bsub[tt % 2][WIDTH8*row + col * TSN8])) = vecB;
(*((__local floatX8*) & Asub[tt % 2][mad24(WIDTH8, row, mul24(col, TSM8))])) = vecA;
(*((__local floatX8*) & Bsub[tt % 2][mad24(WIDTH8, row, mul24(col, TSN8))])) = vecB;
}
}
// Loop over the values of a single tile
for (int k=0; k<TSK8; k++) {
// Cache the values of Bsub in registers
for (int wn=0; wn<WPTN8; wn++) {
//                int col = tidn + wn*RTSN8;
int col = mad24(wn, RTSN8, tidn);
//                Breg[wn] = Bsub[t%2][col + k * TSN8];
Breg[wn] = Bsub[t % 2][mad24(k, TSN8, col)];
}

// Perform the computation
for (int wm=0; wm<WPTM8; wm++) {
//                int row = tidm + wm*RTSM8;
int row = mad24(wm, RTSM8, tidm);
//                Areg = Asub[t%2][row + k * TSM8];
Areg = Asub[t%2][mad24(k, TSM8, row)];
for (int wn=0; wn<WPTN8; wn++) {
acc[wm][wn] += Areg * Breg[wn];
}
}
}
}

// Store the final results in C
for (int wm=0; wm<WPTM8; wm++) {
//        int globalRow = offsetM + tidm + wm*RTSM8;
int globalRow = mad24(wm, RTSM8, tidm) + offsetM;
for (int wn=0; wn<WPTN8; wn++) {
//            int globalCol = offsetN + tidn + wn*RTSN8;
int globalCol = mad24(wn, RTSN8, tidn) + offsetN;
//            C[globalCol*M + globalRow] = acc[wm][wn];
C[mad24(globalCol, M, globalRow)] = acc[wm][wn];
}
}
}
