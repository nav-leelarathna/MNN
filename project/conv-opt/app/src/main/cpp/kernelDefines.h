//
// Created by navin on 26/04/2022.
//

#ifndef MNN_CONVOLUTION_OPTIMISATION_KERNELDEFINES_H
#define MNN_CONVOLUTION_OPTIMISATION_KERNELDEFINES_H


#include <jni.h>
#include <string>
#include <iostream>
//#include <format>

//General
#include "MNN/Tensor.hpp"
#include "MNN_generated.h"

//opencl resources
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "MNN/expr/Executor.hpp"
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include <CL/cl.h>

//CPU resources:
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "MNN/AutoTime.hpp"

#define KERNEL 7
#define LIMIT 16

#define TS1 16 // can't go up

#define TS2 32
#define WPT2 8
#define RTS2 (TS2 / WPT2)

#define TS3 16
#define WPT3 8
#define WIDTH3 2  // 4 discovered to be best experimentally.

#define TSM4 32                 // The tile-size in dimension M
#define TSN4 32                 // The tile-size in dimension N
#define TSK4 32                 // The tile-size in dimension K
#define WPTN4 1                 // The work-per-thread in dimension N
#define WPTM4 8
#define RTSM4 (TSM4/WPTM4)
#define RTSN4 (TSN4/WPTN4)        // The reduced tile-size in dimension N
#define LPT4 ((TSK4*TSM4)/(RTSM4*RTSN4)) // The loads-per-thread for a tile

#define TSM5 32                // The tile-size in dimension M
#define TSN5 32                // The tile-size in dimension N
#define TSK5 16                 // The tile-size in dimension K
#define WPTM5 4                 // The work-per-thread in dimension M
#define WPTN5 4                 // The work-per-thread in dimension N
#define RTSM5 (TSM5/WPTM5)        // The reduced tile-size in dimension M
#define RTSN5 (TSN5/WPTN5)        // The reduced tile-size in dimension N
#define LPTA5 ((TSK5*TSM5)/(RTSM5*RTSN5)) // Loads-per-thread for A
#define LPTB5 ((TSK5*TSN5)/(RTSM5*RTSN5)) // Loads-per-thread for B

#define TSM6 32                // The tile-size in dimension M
#define TSN6 32                // The tile-size in dimension N
#define TSK6 16                 // The tile-size in dimension K
#define WPTM6 4                 // The work-per-thread in dimension M
#define WPTN6 4                 // The work-per-thread in dimension N
#define RTSM6 (TSM6/WPTM6)        // The reduced tile-size in dimension M
#define RTSN6 (TSN6/WPTN6)        // The reduced tile-size in dimension N
#define LPTA6 ((TSK6*TSM6)/(RTSM6*RTSN6)) // Loads-per-thread for A
#define LPTB6 ((TSK6*TSN6)/(RTSM6*RTSN6)) // Loads-per-thread for B
#define WIDTH6 2

#define TSM7 32                // The tile-size in dimension M
#define TSN7 32                // The tile-size in dimension N
#define TSK7 16                 // The tile-size in dimension K
#define WPTM7 4                 // The work-per-thread in dimension M
#define WPTN7 4                 // The work-per-thread in dimension N
#define RTSM7 (TSM7/WPTM7)        // The reduced tile-size in dimension M
#define RTSN7 (TSN7/WPTN7)        // The reduced tile-size in dimension N
#define LPTA7 ((TSK7*TSM7)/(RTSM7*RTSN7)) // Loads-per-thread for A
#define LPTB7 ((TSK7*TSN7)/(RTSM7*RTSN7)) // Loads-per-thread for B
#define WIDTH7 2

#define TSM8 32                // The tile-size in dimension M
#define TSN8 32                // The tile-size in dimension N
#define TSK8 16                 // The tile-size in dimension K
#define WPTM8 4                 // The work-per-thread in dimension M
#define WPTN8 4                 // The work-per-thread in dimension N
#define RTSM8 (TSM8/WPTM8)        // The reduced tile-size in dimension M
#define RTSN8 (TSN8/WPTN8)        // The reduced tile-size in dimension N
#define LPTA8 ((TSK8*TSM8)/(RTSM8*RTSN8)) // Loads-per-thread for A
#define LPTB8 ((TSK8*TSN8)/(RTSM8*RTSN8)) // Loads-per-thread for B
#define WIDTH8 2

void prepareWidthAndTiles(int& , int& , int);

struct kernelAndLaunchParameters{
    cl::Kernel kernel;
    std::vector<uint32_t> global;
    std::vector<uint32_t> local;
};

kernelAndLaunchParameters getKernelAndLaunchParameters(int, MNN::OpenCLRuntime* ,
                                                       int , int);

cl::Kernel setKernelArgs(int M, int N, int K, cl::Buffer A, cl::Buffer B,
                         cl::Buffer C, cl::Kernel klp, MNN::OpenCLRuntime *runtime);

double getKernelRuntime2D(cl::Kernel , std::vector<uint32_t> , std::vector<uint32_t> , MNN::OpenCLRuntime* );

#endif //MNN_CONVOLUTION_OPTIMISATION_KERNELDEFINES_H
