//
// Created by navin on 28/04/2022.
//

#include "kernelDefines.h"
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;

void deviceInfo(void){
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
    std::vector<cl::Platform> platforms;
    cl_int res = cl::Platform::get(&platforms);
    MNN_CHECK_CL_SUCCESS(res, "getPlatform");
    cl::Device *mFirstGPUDevicePtr;
    if(platforms.size() > 0 && res == CL_SUCCESS) {
        cl::Platform::setDefault(platforms[0]);
        std::vector<cl::Device> gpuDevices;
        res = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
        if (1 <= gpuDevices.size() && res == CL_SUCCESS) {
            mFirstGPUDevicePtr = std::make_shared<cl::Device>(gpuDevices[0]).get();
        }
    }
    std::string deviceName  = mFirstGPUDevicePtr->getInfo<CL_DEVICE_NAME>();
    uint64_t globalMemorySize = mFirstGPUDevicePtr->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    uint64_t localMemorySize = mFirstGPUDevicePtr->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    uint64_t maxWorkGroupSize = mFirstGPUDevicePtr->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::vector<cl::size_type> maxWorkItemSizes = mFirstGPUDevicePtr->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    uint64_t maxClockFrequency = mFirstGPUDevicePtr->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    uint64_t computeUnits = mFirstGPUDevicePtr->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    uint64_t preferredVectorWidth = mFirstGPUDevicePtr->getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT >();

    MNN_PRINT("device name: %s", deviceName.c_str());
    MNN_PRINT("global memory size: %d", globalMemorySize);
    MNN_PRINT("local memory size: %d", localMemorySize);
    MNN_PRINT("max work group size: %d", maxWorkGroupSize);
    MNN_PRINT("max work item size: (%d,%d,%d)", maxWorkItemSizes[0],maxWorkItemSizes[1],maxWorkItemSizes[2]);
    MNN_PRINT("max clock frequency: %d", maxClockFrequency);
    MNN_PRINT("number of compute units: %d", computeUnits);
    MNN_PRINT("preferred vector length: %d", preferredVectorWidth);
}

void kernelInfo(int kernelID){
    std::set <std::string> buildOptions;
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
    info.mode = Backend::Info::DIRECT;
    info.user = (BackendConfig * ) & config;
    info.numThread = 1;
    executor->setGlobalExecutorConfig(info.type, config, info.numThread);
    OpenCLRuntime runtime_(config.precision, info.gpuMode);
    OpenCLRuntime *runtime = &runtime_;
    runtime->setCommandQueueProfileEnable();
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();
    cl::Kernel kernel = getKernelAndLaunchParameters(kernelID, runtime, 32,32).kernel;
    uint64_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
    MNN_PRINT("Kernel %d information:", kernelID);
    MNN_PRINT("  Max #(work items) per work group in kernel : %d", maxWorkGroupSize);
}

void printKernelInfo(void){
    for (int i = 0; i < NUM_KERNELS+1; i++)
        kernelInfo(i);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_deviceInformation(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";
    deviceInfo();
    printKernelInfo();
    return env->NewStringUTF(hello.c_str());
}