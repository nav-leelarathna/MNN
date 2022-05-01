//
// Created by navin on 28/04/2022.
//

#ifndef MNN_CONVOLUTION_OPTIMISATION_DEVICEINFORMATION_H
#define MNN_CONVOLUTION_OPTIMISATION_DEVICEINFORMATION_H
#include <CL/cl.h>
#include <jni.h>
#include <string>
#include <stdlib.h>
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "backend/opencl/core/OpenCLBackend.hpp"

typedef struct {
    std::string deviceName;
    std::string driverVersion;
    uint maxWorkGroupSize;
    uint64_t maxAllocSize;
    uint64_t maxGlobalSize;
    uint maxClockFrequency;
    uint numComputeUnits;
    bool halfSupported;
    bool doubleSupported;
} device_info_t;

device_info_t getDeviceInfo(cl::Device &d);
#endif //MNN_CONVOLUTION_OPTIMISATION_DEVICEINFORMATION_H
