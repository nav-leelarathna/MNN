//
// Created by navin on 28/04/2022.
//
#include "kernelDefines.h"
#include <chrono>

#define FETCH_PER_WI 16
#define WORK_PER_WI (2<<13)
#define MAX_LOCAL_SIZE 1024

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;

double getKernelRuntime1D(cl::Kernel kernel, int global, int local, OpenCLRuntime* runtime){
    cl::NDRange globalSize, localSize;
    globalSize = global;
    localSize = local;
    cl::CommandQueue queue = runtime->commandQueue();
    int warmup_steps = 10, hot_runs = HOT_RUNS, last_runs = 5, overall_runs =2;
    std::vector<cl::Event> events;
    double CPU_totalTime;
    std::chrono::steady_clock::time_point t;
    for (int k = 0; k < overall_runs; k++) {
        // warmup
        std::cout << "warmup\n";
        for (int i = 0; i < warmup_steps; i++){
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
            queue.finish();
        }
        // hot runs
        std::cout << "hot runs\n";
        t = std::chrono::steady_clock::now();
        for (int i = 0; i < hot_runs; i++){
            cl::Event event;
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
            queue.finish();
            events.push_back(event);
        }
        double diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();
        CPU_totalTime += diff;
        for (int i = 0; i < last_runs; i++){
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
            queue.finish();
        }
    }
    double avg_time = 0.0f;
//    std::stringstream option;
    for (cl::Event event : events){
        double time = runtime->getCostTime(&event);
//        double gbps = ((2* FETCH_PER_WI * MAX_LOCAL_SIZE * 16 * sizeof(float)) / time) / 1e3f;
//        float gFlops = static_cast<float>(MAX_LOCAL_SIZE * 2<<6) * static_cast<float>(WORK_PER_WI) / time / 1e3f;
//        option << std::to_string(gFlops) << ", ";
        avg_time += time;
    }
//    MNN_PRINT("%s", option.str().c_str());
//    return CPU_totalTime/(hot_runs * overall_runs);
    return avg_time / (hot_runs * overall_runs);
}

float getMemoryBandwidth(int floatWidth=1){
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
    OpenCLRuntime runtime_(config.precision, MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER);
    OpenCLRuntime *runtime = &runtime_;
    runtime->setCommandQueueProfileEnable();
    cl::Context context = runtime->context();

    int numItems = FETCH_PER_WI * MAX_LOCAL_SIZE * 16;
    float * arr = new float[numItems];
    for (int i = 0 ; i < numItems; i++){
        arr[i] = (float)i;
    }

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float)*numItems, arr);
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(float)*numItems);
    std::set <std::string> buildOptions;
    std::stringstream option;
    option << "-DFETCH_PER_WI=" << std::to_string(FETCH_PER_WI) << " ";
    option << "-DWORK_PER_WI=" << std::to_string(WORK_PER_WI) << " ";
    buildOptions.emplace(option.str());

    std::stringstream name;
    name << "bandwidth_float" << std::to_string(floatWidth);
    cl::Kernel kernel = runtime->buildKernel("benchmark", name.str(), buildOptions);
    int res = 0;
    res |= kernel.setArg(0, bufferA);
    res |= kernel.setArg(1, bufferB);
    MNN_CHECK_CL_SUCCESS(res, "setting args");

    uint32_t maxLocalSize = MAX_LOCAL_SIZE;
    uint32_t globalSize = numItems / FETCH_PER_WI / floatWidth;

    double timeToRun = getKernelRuntime1D(kernel, globalSize, maxLocalSize, runtime);
    float gbps = ((2* (float)numItems * sizeof(float)) / timeToRun) / 1e3f;
//    float gbps = (((float)globalSize * sizeof(float)) / timeToRun) / 1e3f;
    return gbps;
}


float getFlops(int floatWidth){
    BackendConfig config;
    OpenCLRuntime runtime_(config.precision, MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER);
    OpenCLRuntime *runtime = &runtime_;
    runtime->setCommandQueueProfileEnable();
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();

    int numItems = MAX_LOCAL_SIZE * 2<<6;
    float arr[numItems];
    for (int i = 0 ; i < numItems; i++){
        arr[i] = (float)i;
    }
    uint32_t maxLocalSize = MAX_LOCAL_SIZE;
    uint32_t globalSize = numItems;

    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(float)*numItems, arr);
    std::set <std::string> buildOptions;
    std::stringstream option;
    option << "-DWORK_PER_WI=" << std::to_string(WORK_PER_WI) << " ";
    option << "-DFETCH_PER_WI=" << std::to_string(FETCH_PER_WI) << " ";
    buildOptions.emplace(option.str());
    std::stringstream name;
    name << "compute_float" << std::to_string(floatWidth);
    cl::Kernel kernel = runtime->buildKernel("benchmark", name.str(), buildOptions);
    int res = 0;
    float A = 1.3f;
    res |= kernel.setArg(0, A);
    res |= kernel.setArg(1, bufferB);
    MNN_CHECK_CL_SUCCESS(res, "setting args");
    double timeToRun = getKernelRuntime1D(kernel, globalSize, maxLocalSize, runtime);
    float gFlops = static_cast<float>(globalSize) * static_cast<float>(WORK_PER_WI) / timeToRun / 1e3f;
    return gFlops;
}

void benchmark(){
//    float gbps_0 = getMemoryBandwidth(0);
    float gbps_1 = getMemoryBandwidth(1);
    float gbps_2 = getMemoryBandwidth(2);
    float gbps_4 = getMemoryBandwidth(4);
    float gbps_8 = getMemoryBandwidth(8);
    float gbps_16 = getMemoryBandwidth(16);
    MNN_PRINT("Global memory bandwidth (GBPS)");
    MNN_PRINT("  float   : %f", gbps_1);
    MNN_PRINT("  float2  : %f", gbps_2);
    MNN_PRINT("  float4  : %f", gbps_4);
    MNN_PRINT("  float8  : %f", gbps_8);
    MNN_PRINT("  float16 : %f", gbps_16);
    float gFlops_1 = getFlops(1);
    float gFlops_2 = getFlops(2);
    float gFlops_4 = getFlops(4);
    float gFlops_8 = getFlops(8);
    float gFlops_16 = getFlops(16);
    MNN_PRINT("Single Precision Compute (GFLOPS)");
    MNN_PRINT("  float   : %f", gFlops_1);
    MNN_PRINT("  float2  : %f", gFlops_2);
    MNN_PRINT("  float4  : %f", gFlops_4);
    MNN_PRINT("  float8  : %f", gFlops_8);
    MNN_PRINT("  float16 : %f", gFlops_16);
    float gFlops_17 = getFlops(17);
    MNN_PRINT("  float17 : %f", gFlops_17);
}


extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_benchmark(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";
    benchmark();
    return env->NewStringUTF(hello.c_str());
}