//
// Created by navin on 18/11/2021.
//
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

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;

#define KERNEL 1
#define PROFILING
#define LOG_VERBOSE

void deviceInfo(void){
    MNN_PRINT("attempting to get device info");
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
    size_t workGroupSize;

    // get all platforms
    MNN_PRINT("getting platforms");
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);

            MNN_PRINT("%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            MNN_PRINT(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            MNN_PRINT(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            MNN_PRINT(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            MNN_PRINT(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

            // print work group size
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE ,sizeof(workGroupSize), &workGroupSize, NULL);
            MNN_PRINT(" %d.%d Max work group size: %d\n", j+1,5, workGroupSize);

            // max dimensions
            cl_uint dimensions;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS ,sizeof(dimensions), &dimensions, NULL);
            MNN_PRINT(" %d.%d Max number of dimensions: %d\n", j+1,6, dimensions);

            // max work items per dimension
            size_t workItemsPerDimension [3];
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES ,sizeof(workItemsPerDimension), &workItemsPerDimension, NULL);
            MNN_PRINT(" %d.%d number of work items per dimension: (%d,%d,%d)\n", j+1,7, workItemsPerDimension[0],workItemsPerDimension[1],workItemsPerDimension[2]);
        }
        free(devices);
    }
    free(platforms);
}

void kernelInfo(void){
    // CL_KERNEL_WORK_GROUP_SIZE
//    size_t workGroupSize;
//    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,sizeof(workGroupSize),&workGroupSize,NULL);
//    MNN_PRINT("Max kernel work group size: %d\n",workGroupSize);
    std::set <std::string> buildOptions;
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    // What do these modes do?
    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
    info.mode = Backend::Info::DIRECT;
    info.user = (BackendConfig * ) & config;
    info.numThread = 1;
    // int the below function numThreads gets set to 4 if type is MNN_FORWARD_OPENCL
    executor->setGlobalExecutorConfig(info.type, config, info.numThread);
    OpenCLRuntime runtime_(config.precision, info.gpuMode);
    OpenCLRuntime *runtime = &runtime_;
    // What is this command queue?
    runtime->setCommandQueueProfileEnable();
    // what is this context?
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();
    cl::Kernel kernel = runtime->buildKernel("opt_gemm", "gemm1", buildOptions);



    MNN_PRINT("KERNEL INFO");
    uint64_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
    MNN_PRINT(" Max work group size: %d",maxWorkGroupSize);


    uint64_t * globalWorkSize = runtime->getMaxGlobalWorkSize (kernel);
//    kernel.getWorkGroupInfo(CL_KERNEL_GLOBAL_WORK_SIZE, globalWorkSize);
//    clGetKernelWorkGroupInfo(kernel.object_, NULL, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(globalWorkSize),&globalWorkSize,NULL)
    MNN_PRINT(" Number of work items per dimension: (%d,%d,%d)\n",globalWorkSize[0],globalWorkSize[1],globalWorkSize[2]);
}

double profileCPU(int width = 32) {
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    info.user = (BackendConfig * ) & config;

    CPURuntime runtime(info);
    CPUBackend backend(&runtime, config.precision);
    // A * B = C
    float A[width][width];
    float B[width][width];
    float C[width][width];
    // Initialise arrays with values.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            A[i][j] = (float) (i * width + j);
            B[i][j] = (float) (i * width + j);
            C[i][j] = 0.0f;
        }
    }

    std::vector<int> shape({width, width});
    Tensor *tensorA = Tensor::create<float>(shape, A);
    Tensor *tensorB = Tensor::create<float>(shape, B);
    Tensor *tensorC = Tensor::create<float>(shape, C);

    Timer timer;

    std::vector < Tensor * > inputs({tensorA, tensorB}), outputs({tensorC});
    // What is this doing?
    CPUMatMul matmul(&backend, false, false, false, false);
    // What is this mode?
    MNNSetCPUThreadsMode(MNN_CPU_MODE_BIG);
    // What does onResize do?
    matmul.onResize(inputs, outputs);

    // executing matrix multiplication and averaging over multiple runs.
    // Allow some runs to "warmup" that do not contribute to the average
    // time.
    int warmup = 10, hot_run = 50, overall = 2;
    double avg_time = 0.f;
    for (int k = 0; k < overall; k++) {
        for (int i = 0; i < warmup; i++) {
            matmul.onExecute(inputs, outputs);
        }

        timer.reset();
        for (int i = 0; i < hot_run; i++) {
            matmul.onExecute(inputs, outputs);
        }

        avg_time += (double) timer.durationInUs();
    }
    timer.reset();
    return avg_time / (hot_run * overall);
}

double profileGPU(int width=32) {
    std::cout << "intializing\n";
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    // What do these modes do?
    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
    info.mode = Backend::Info::DIRECT;
    info.user = (BackendConfig * ) & config;
    info.numThread = 1;
    std::cout << "setting exeutor config\n";
    // int the below function numThreads gets set to 4 if type is MNN_FORWARD_OPENCL
    executor->setGlobalExecutorConfig(info.type, config, info.numThread);
    // what is this runtime?
    MNN_PRINT("setting runtime\n");
    OpenCLRuntime runtime_(config.precision, info.gpuMode);
    OpenCLRuntime *runtime = &runtime_;
    // What is this command queue?
    runtime->setCommandQueueProfileEnable();
    // what is this context?
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();

    std::set <std::string> buildOptions;
    std::stringstream option;

    // what is this datatype?
    uint32_t width_uint = (uint32_t) width;
    cl::NDRange offsetRange(0, 0);
    std::vector <uint32_t> offset({0, 0});
#ifdef PROFILING
    option << "-DPROFILING=1 ";
#endif

#if (KERNEL == 1)
    buildOptions.emplace("-DKERNEL=1");
    MNN_PRINT("loading kernel ...\n");
    cl::Kernel kernel = runtime->buildKernel("opt_gemm", "gemm1", buildOptions);
    MNN_PRINT("Loaded kernel successfully\n");
    std::vector<uint32_t> global({width_uint, width_uint}), local({13, 13});
//    cl::NDRange globalRange(width_uint, width_uint), localRange(16, 16);
#endif
    std::cout << "Starting timer.\n";
    double avg_time = 0.0f;
    Timer timer;

    int warmup_steps = 10, hot_runs = 50, last_runs = 5, overall_runs =2;
    int total_runs = warmup_steps + hot_runs + last_runs;
    std::vector<cl::Event> events;
    for (int k = 0; k < overall_runs; k++) {
        // warmup
        std::cout << "warmup\n";
        for (int i = 0; i < warmup_steps; i++)
            runKernel2D(kernel, global, local, runtime, nullptr);

        // hot runs
        std::cout << "hot runs\n";
        for (int i = 0; i < hot_runs; i++){
            cl::Event event;
            timer.reset();
            runKernel2D(kernel, global, local, runtime, nullptr);
//            runKernel2D(kernel, global, local, runtime, &event);
            avg_time += timer.durationInUs();
            // why is this line commented ?
            // events.push_back(event)
        };

        // cool down runs, what is the point of this?
        for (int i = 0; i < last_runs; i++)
            runKernel2D(kernel, global, local, runtime, nullptr);
        commandQueue.finish();
    }
    avg_time /= (hot_runs*overall_runs);
    return avg_time;
}
void cpu(void){
    int starting = 0;
    int start = starting/32 + 1 , offset = 15;
    std::stringstream output;

    MNN_PRINT("Starting CPU");
    std::cout << "Starting CPU" << std::endl;

    for (int i = start; i<=16; i++) {
        int mat_size = i*32;
        double time = profileCPU(mat_size);
        MNN_PRINT("%d-> %f", mat_size, time);
        output << std::to_string(time) << ",";
    }
    MNN_PRINT("%s", output.str().c_str());
    std::cout << output.str().c_str() << "\nFinished on CPU!"<< std::endl;
    MNN_PRINT("DONE!");
}

void gpu(void) {
    MNN_PRINT("Starting GPU");
    int starting = 0;
    std::stringstream output;
    int start = starting/32 + 1 , offset = 15;
    int limit = 1;
    for (int i = start; i<=limit; i++) {
        int mat_size = i*32;
        double time = profileGPU(mat_size);
        MNN_PRINT("%d-> %f", mat_size, time);
        output << std::to_string(time) << ",";
    }

    MNN_PRINT("%s", output.str().c_str());
    std::cout << output.str().c_str() << "\nFinished on GPU!"<< std::endl;
    MNN_PRINT("DONE!");
}


extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";

#ifndef PROFILING

    MNN_PRINT("starting");
    gpu(33);

#else
//    cpu();
    gpu();
    deviceInfo();
    kernelInfo();

#endif

    return env->NewStringUTF(hello.c_str());
}