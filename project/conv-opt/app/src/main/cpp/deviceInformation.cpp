//
// Created by navin on 28/04/2022.
//

#include "deviceInformation.h"

using namespace MNN;
using namespace std;

void trimString(std::string &str)
{
    size_t pos = str.find('\0');

    if (pos != std::string::npos)
    {
        str.erase(pos);
    }
}

device_info_t getDeviceInfo(cl::Device &d)
{
    device_info_t devInfo;

    devInfo.deviceName = d.getInfo<CL_DEVICE_NAME>();
    devInfo.driverVersion = d.getInfo<CL_DRIVER_VERSION>();
    trimString(devInfo.deviceName);
    trimString(devInfo.driverVersion);

    devInfo.numComputeUnits = (uint)d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    vector<size_t> maxWIPerDim;
    maxWIPerDim = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    devInfo.maxWorkGroupSize = (uint)maxWIPerDim[0];

    // Limiting max work-group size to 256
//#define MAX_WG_SIZE 256
//    devInfo.maxWGSize = std::min(devInfo.maxWGSize, (uint)MAX_WG_SIZE);

    // Kernel launch fails for workgroup size 256(CL_DEVICE_MAX_WORK_ITEM_SIZES)
//    string vendor = d.getInfo<CL_DEVICE_VENDOR>();
//    if ((vendor.find("QUALCOMM") != std::string::npos) ||
//        (vendor.find("qualcomm") != std::string::npos))
//    {
//        devInfo.maxWGSize = std::min(devInfo.maxWGSize, (uint)128);
//    }

    devInfo.maxAllocSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    devInfo.maxGlobalSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    devInfo.maxClockFrequency = static_cast<uint>(d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    devInfo.doubleSupported = false;
    devInfo.halfSupported = false;

    std::string extns = d.getInfo<CL_DEVICE_EXTENSIONS>();
    if ((extns.find("cl_khr_fp16") != std::string::npos))
        devInfo.halfSupported = true;

    if ((extns.find("cl_khr_fp64") != std::string::npos) || (extns.find("cl_amd_fp64") != std::string::npos))
        devInfo.doubleSupported = true;

//    devInfo.deviceType = d.getInfo<CL_DEVICE_TYPE>();
    return devInfo;
}

device_info_t getGPUInfo(){
    MNN_PRINT("trying to get platforms");
//    vector<cl::Platform> platforms;
//    cl::Platform::get(&platforms);
//    cl_uint platformCount;
//    cl_platform_id* platforms;
//    clGetPlatformIDs(0, NULL, &platformCount);
//    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
//    int i = 0;
//    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
//    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
//    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
//    int j = 0;
    vector<cl::Platform> platforms;
    MNN_PRINT("get Platform");
    cl::Platform::get(&platforms);
    MNN_PRINT("Number of platforms: %d", platforms.size());
    int p = 0;
    std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
    trimString(platformName);

    cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[p])(),
            0};

    cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
    vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
    MNN_PRINT("Number of devices: %d", devices.size());
    int d = 0;
    MNN_PRINT("calling getDeviceInfo");
    device_info_t devInfo = getDeviceInfo(devices[d]);
    MNN_PRINT("returning devInfo");
    return devInfo;
}


void printDeviceInfo(device_info_t deviceInfo){
    MNN_PRINT("Device Name: %s", deviceInfo.deviceName.c_str());
    MNN_PRINT("Driver Version: %s", deviceInfo.driverVersion.c_str());
    MNN_PRINT("Max Work Group Size: %d", deviceInfo.maxWorkGroupSize);
    MNN_PRINT("Max Allocation Size: %d", deviceInfo.maxAllocSize);
    MNN_PRINT("Max Global Size: %d", deviceInfo.maxGlobalSize);
    MNN_PRINT("Max Clock Frequency: %d", deviceInfo.maxClockFrequency);
    MNN_PRINT("Number of Compute Units: %d", deviceInfo.numComputeUnits);
    MNN_PRINT("Half supported: %b", deviceInfo.halfSupported);
    MNN_PRINT("Double supported: %b", deviceInfo.doubleSupported);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_deviceInformation(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";
    device_info_t deviceInformation = getGPUInfo();
    printDeviceInfo(deviceInformation);
    return env->NewStringUTF(hello.c_str());
}