//
// Created by navin on 18/11/2021.
//

#include "kernelDefines.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;

#define PROFILING
//#define LOG_VERBOSE


void deviceInfo(void){
    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
    std::vector<cl::Platform> platforms;
    cl_int res = cl::Platform::get(&platforms);
    MNN_CHECK_CL_SUCCESS(res, "getPlatform");
    MNN_PRINT("hi");
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
    // What do these modes do?
    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
    info.mode = Backend::Info::DIRECT;
    info.user = (BackendConfig * ) & config;
    info.numThread = 1;
    // int the below function numThreads gets set to 4 if type is MNN_FORWARD_OPENCL so no point changing it
    executor->setGlobalExecutorConfig(info.type, config, info.numThread);
    OpenCLRuntime runtime_(config.precision, info.gpuMode);
    OpenCLRuntime *runtime = &runtime_;
    // What is this command queue?
    runtime->setCommandQueueProfileEnable();
    // what is this context?
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();
    cl::Kernel kernel = getKernelAndLaunchParameters(kernelID, runtime, 32,32).kernel;
//    MNN_PRINT("Information for kernel %d", kernelID);
    uint64_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
    MNN_PRINT("  -Max #(work items) per work group in kernel %d: %d",kernelID, maxWorkGroupSize);
//    uint32_t deviceComputeUnits = runtime->deviceComputeUnits();
//    std::vector<uint32_t> maxWorkItemSize = runtime->getMaxWorkItemSizes();
//    uint64_t maxAllocSize = runtime->maxAllocSize();
////    float flops = runtime->flops();
//    MNN_PRINT("  Device Compute Units: %d", deviceComputeUnits);
//    MNN_PRINT("  Mac work item size: (%d,%d,%d)", maxWorkItemSize[0], maxWorkItemSize[1], maxWorkItemSize[2]);
//    MNN_PRINT("  Max alloc size: %d", maxAllocSize);
////    MNN_PRINT("  flops: %f", flops);

//    uint64_t * globalWorkSize = runtime->getMaxGlobalWorkSize (kernel);
//    kernel.getWorkGroupInfo(CL_KERNEL_GLOBAL_WORK_SIZE, globalWorkSize);
//    clGetKernelWorkGroupInfo(kernel.object_, NULL, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(globalWorkSize),&globalWorkSize,NULL)
//    MNN_PRINT("  Number of work items per dimension: (%d,%d,%d)\n",globalWorkSize[0],globalWorkSize[1],globalWorkSize[2]);
}


void printKernelInfo(void){
    for (int i = 0; i < 9; i++)
        kernelInfo(i);
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
    float * A = new float[width * width];
    float * B = new float[width * width];
    float * C = new float[width * width];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            A[i*width+j] = (float) (i * width + j);
            B[i*width+j] = (float) (i * width + j);
            C[i*width+j] = 0.0f;
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
    int warmup = 10, hot_runs = 50, overall = 2;
    double avg_time = 0.f;
    for (int k = 0; k < overall; k++) {
        for (int i = 0; i < warmup; i++) {
            matmul.onExecute(inputs, outputs);
        }

        timer.reset();
        for (int i = 0; i < hot_runs; i++) {
            matmul.onExecute(inputs, outputs);
        }
        avg_time += (double) timer.durationInUs();
    }
    timer.reset();
    return avg_time / (hot_runs * overall);
}


void prepareWidthAndTiles(int& width, int& numTiles, int kernel){
    if (kernel == 2){
        numTiles = width / TS2;
        if (width % TS2 != 0)
            numTiles++;
        width = numTiles * TS2;
    }

    else if(kernel == 3){
        numTiles = width / TS3;
        if (width % TS3 != 0)
            numTiles++;
        width = numTiles * TS3;
    }
    else if (kernel == 4){
        numTiles = width / TSM4;
        if (width % TSM4 != 0)
            numTiles++;
        width = numTiles * TSM4;
    }
    else if (kernel == 5){
        int temp = width / TSM5;
        if (width % TSM5 != 0)
            temp++;
        width = temp * TSM5;
        numTiles = width / TSK5;
    }
}


kernelAndLaunchParameters getKernelAndLaunchParameters(int kernelID, OpenCLRuntime* runtime, int height, int width){
    std::set <std::string> buildOptions;
    std::stringstream option;
    uint32_t width_uint = (uint32_t) width;
    uint32_t height_uint = (uint32_t) height;
    kernelAndLaunchParameters klp;
    cl::Kernel kernel;
    std::vector<uint32_t> global;
    std::vector<uint32_t> local;
    if (kernelID == 0){
        buildOptions.emplace("-DKERNEL=0");
        kernel = runtime->buildKernel("opt_gemm", "baseline", buildOptions);
        global = {height_uint, width_uint};
        local = {16, 16};
    }

    else if (kernelID == 1){
        option << "-D TS1=" << std::to_string(TS1);
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm1", buildOptions);
        global = {height_uint, width_uint};
        local = {TS1, TS1};
    }

    else if (kernelID  == 2){
        option << "-DTS2=" << std::to_string(TS2) << " -DWPT2=" << std::to_string(WPT2) << " -DRTS2=" << std::to_string(RTS2);
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm2", buildOptions);
//        uint32_t globalColSize = width_uint / WPT2;
//        if (width_uint % WPT2 != 0) globalColSize++;
//        global ={height_uint, globalColSize};
//        local={TS2, RTS2};
        uint32_t globalColSize = width_uint / WPT2;
        if (width_uint % WPT2 != 0) globalColSize++;
        global ={globalColSize, width_uint};
        local={RTS2, TS2};
    }

    else if (kernelID == 3){
        option << " -DTS3=" << std::to_string(TS3)
               << " -DWIDTH=" << std::to_string(WIDTH3)
               << " -DKERNEL=3 ";
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm3", buildOptions);
        uint32_t global0 = (uint32_t) width / WIDTH3,
                global1 = (uint32_t) width,
                local0 = TS3/ WIDTH3,
                local1 = TS3;
        //    MNN_PRINT("global0=%d, global1=%d, local0=%d, local1=%d", global0, global1, local0, local1);
        global={global0, global1};
        local={local0, local1};
    }

    else if (kernelID  == 4){
        option << "-DTSM=" << std::to_string(TSM4) << " ";
        option << "-DTSN=" << std::to_string(TSN4) << " ";
        option << "-DTSK=" << std::to_string(TSK4) << " ";
        option << "-DWPTN=" << std::to_string(WPTN4) << " ";
        //    option << "-DWPTM=" << std::to_string(WPTM) << " ";
        option << "-DKERNEL=4 ";
        //    option << "-DRTSN=" << std::to_string(TSN/WPTN) << " ";
        //    option << "-DLPT=" << std::to_string(TSK/RTSN);
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm4", buildOptions);
        uint32_t global0 = width,
                global1 = width / WPTM4,
                local0 = TSN4,
                local1 = TSM4 / WPTM4;
//        uint32_t global0 = width / WPTN4,
//                global1 = width,
//                local0 = TSN4 /WPTN4,
//                local1 = TSM4;
//        uint32_t global0 = width,
//                global1 = width / WPTN4,
//                local0 = TSM4,
//                local1 = TSN4 / WPTN4;
//        uint32_t global0 = width / WPTM4,
//                global1 = width,
//                local0 = TSM4 / WPTM4,
//                local1 = TSN4;
//        if (width % WPTN4 != 0)
//            global1++;
        global={global0, global1};
        local={local0, local1};
    }

    else if (kernelID == 5) {
        option << "-DTSM=" << std::to_string(TSM5) << " ";
        option << "-DTSN=" << std::to_string(TSN5) << " ";
        option << "-DTSK=" << std::to_string(TSK5) << " ";
        option << "-DWPTN=" << std::to_string(WPTN5) << " ";
        option << "-DWPTM=" << std::to_string(WPTM5) << " ";
        option << "-DKERNEL=5 ";
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm5", buildOptions);
//        uint32_t global0 = width / WPTM5,
//                global1 = width / WPTN5,
//                local0 = RTSM5,
//                local1 = RTSN5;
        uint32_t global0 = width / WPTN5,
                global1 = width / WPTM5,
                local0 = RTSN5,
                local1 = RTSM5;
//        if (width % WPTN5 != 0)
//            global1++;
        global={global0, global1};
        local={local0, local1};
    }

    else if (kernelID == 6){
        option << "-DTSM=" << std::to_string(TSM6) << " ";
        option << "-DTSN=" << std::to_string(TSN6) << " ";
        option << "-DTSK=" << std::to_string(TSK6) << " ";
        option << "-DWPTN=" << std::to_string(WPTN6) << " ";
        option << "-DWPTM=" << std::to_string(WPTM6) << " ";
        option << " -DWIDTH=" << std::to_string(WIDTH6) << " ";
        option << "-DKERNEL=6 ";
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm6", buildOptions);
        uint32_t global0 = width / WPTN6,
                global1 = width / WPTM6,
                local0 = RTSN6,
                local1 = RTSM6;
//        uint32_t global0 = width / WPTM6,
//                global1 = width / WPTN6,
//                local0 = RTSM6,
//                local1 = RTSN6;
        if (width % WPTN6 != 0)
            global1++;
        global={global0, global1};
        local={local0, local1};
    }

    else if (kernelID == 7){
        option << "-DTSM=" << std::to_string(TSM7) << " ";
        option << "-DTSN=" << std::to_string(TSN7) << " ";
        option << "-DTSK=" << std::to_string(TSK7) << " ";
        option << "-DWPTN=" << std::to_string(WPTN7) << " ";
        option << "-DWPTM=" << std::to_string(WPTM7) << " ";
        option << " -DWIDTH=" << std::to_string(WIDTH7) << " ";
        option << "-DKERNEL=7 ";
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm7", buildOptions);
        uint32_t global0 = width / WPTN7,
                global1 = width / WPTM7,
                local0 = RTSN7,
                local1 = RTSM7;
        if (width % WPTN7 != 0)
            global1++;
        global={global0, global1};
        local={local0, local1};;
    }

    else if (kernelID == 8){
        option << "-DTSM=" << std::to_string(TSM8) << " ";
        option << "-DTSN=" << std::to_string(TSN8) << " ";
        option << "-DTSK=" << std::to_string(TSK8) << " ";
        option << "-DWPTN=" << std::to_string(WPTN8) << " ";
        option << "-DWPTM=" << std::to_string(WPTM8) << " ";
        option << " -DWIDTH=" << std::to_string(WIDTH8) << " ";
        option << "-DKERNEL=8";
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm8", buildOptions);
        uint32_t global0 = width / WPTN8,
                global1 = width / WPTM8,
                local0 = RTSN8,
                local1 = RTSM8;
        if (width % WPTN8 != 0)
            global1++;
        global={global0, global1};
        local={local0, local1};
    }
    else if (kernelID == 9) {
        option << "-DTSM=" << std::to_string(TSM8) << " ";
        option << "-DTSN=" << std::to_string(TSN8) << " ";
        option << "-DTSK=" << std::to_string(TSK8) << " ";
        option << "-DWPTN=" << std::to_string(WPTN8) << " ";
        option << "-DWPTM=" << std::to_string(WPTM8) << " ";
        option << " -DWIDTH=" << std::to_string(WIDTH8) << " ";
        option << "-DKERNEL=8";
        buildOptions.emplace(option.str());
        kernel = runtime->buildKernel("opt_gemm", "gemm8New", buildOptions);
        uint32_t global0 = width / WPTN8,
                global1 = width / WPTM8,
                local0 = RTSN8,
                local1 = RTSM8;
        if (width % WPTN8 != 0)
            global1++;
        global = {global0, global1};
        local={local0, local1};
    }

    klp.kernel = kernel;
    klp.global = global;
    klp.local = local;
    return klp;
}


cl::Kernel setKernelArgs(int M, int N, int K, cl::Buffer bufferA, cl::Buffer bufferB, cl::Buffer bufferC, cl::Kernel kernel, OpenCLRuntime *runtime){
    cl::Context context = runtime->context();
    int idx = 0;
    int res = 0;
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    res |= kernel.setArg(idx++, M); // M
    res |= kernel.setArg(idx++, N); // N
    res |= kernel.setArg(idx++, K); // K
    res |= kernel.setArg(idx++, bufferA);
    res |= kernel.setArg(idx++, bufferB);//openCLBuffer(tensorB));
    res |= kernel.setArg(idx++, bufferC);//openCLBuffer(tensorC));
    return kernel;
}


double getKernelRuntime2D(cl::Kernel kernel, std::vector<uint32_t> global, std::vector<uint32_t> local, OpenCLRuntime* runtime){
    int warmup_steps = 5, hot_runs = 10, last_runs = 2, overall_runs =1;
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
            runKernel2D(kernel, global, local, runtime, &event);
            events.push_back(event);
        };
        // cool down runs, what is the point of this?
        for (int i = 0; i < last_runs; i++)
            runKernel2D(kernel, global, local, runtime, nullptr);
    }
    double avg_time = 0.0f;
    for (cl::Event event : events){
        double time = runtime->getCostTime(&event);
//        MNN_PRINT("%f\n",time);
        avg_time += time;
    }
    return avg_time / (hot_runs * overall_runs);
}

double profileGPU(int width, int kernelID) {
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
    // what is this runtime?
//    MNN_PRINT("setting runtime\n");
    OpenCLRuntime runtime_(config.precision, info.gpuMode);
    OpenCLRuntime *runtime = &runtime_;
    // What is this command queue?
    runtime->setCommandQueueProfileEnable();
    // what is this context?
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();

    std::set <std::string> buildOptions;
    std::stringstream option;
    int oldWidth = width;
    int numTiles;
    prepareWidthAndTiles(width, numTiles, kernelID);
//    MNN_PRINT("width and tiles: %d, %d\n",width,numTiles);
    uint32_t width_uint = (uint32_t) width;
    kernelAndLaunchParameters klp = getKernelAndLaunchParameters(kernelID, runtime, width, width);
//    float A[width][width];
    float *A = new float[width*width];
//    float B[width][width];
    float *B = new float[width*width];
//    float C[width][width];
    float *C = new float[width*width];
    // Initialise arrays with values.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
          if (i < oldWidth && j < oldWidth){
//              B[i][j] = (float) (i * width + j);
              B[i*width+j] = (float) (i * width + j);
            if (kernelID == 4 or kernelID == 5 or kernelID == 6 or kernelID == 7){
                // A[j][i] = (float) (i * width + j);
                A[j*width+i] =(float) (i * width + j) ;
            }
            else{
                // A[i][j] = (float) (i * width + j);
                A[i*width + j] =(float) (i * width + j);
            }
//            C[i][j] = 0.0f;
            C[i*width+j] = 0.0f;
            }
          else{
              // A[i][j] = 0.f;
              A[i*width + j] = 0.f;
//              B[i][j] = 0.f;
              B[i*width+j] = 0.f;
//              C[i][j] = 0.f;
              C[i*width+j] = 0.f;
          }
        }
    }
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    cl::Buffer bufferA(context, flags, sizeof(float)*width*width, A);
    cl::Buffer bufferB(context, flags, sizeof(float)*width*width, B);
    cl::Buffer bufferC(context, flags, sizeof(float)*width*width, C);
    cl::Kernel kernel = setKernelArgs(width, width, width, bufferA, bufferB, bufferC, klp.kernel, runtime);
    std::vector<uint32_t> global = klp.global;
    std::vector<uint32_t> local = klp.local;
    return getKernelRuntime2D(kernel, global, local, runtime);
}



void cpu(void){
    int starting = 0;
    int start = starting/32 + 1; //, offset = 15;
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

void gpu(int kernelID) {
    MNN_PRINT("Running kernel %d on GPU", kernelID);
    int starting = 0;
    std::stringstream output;
    int start = starting/32 + 1; //, offset = 15;
    int limit = LIMIT;
    for (int i = start; i<=limit; i++) {
        int mat_size = i*32;
        double time = profileGPU(mat_size, kernelID);
//        MNN_PRINT("%d-> %f", mat_size, time);
        output << std::to_string(time) << ",";
    }

    MNN_PRINT("%s", output.str().c_str());
    MNN_PRINT("DONE!");
}

double getKernelGLOPS(int kernelToTestID, int size){
    double time = profileGPU(size, kernelToTestID);
    double res =  ((2 * size / time) / 1e3f) * size * size;
    return res;

}

double getCpuGFLOPS(int size){
    double time = profileCPU(size);
    double res =  ((2 * size / time) / 1e3f) * size * size;
    return res;
}

void printFlops(void){
    double cpuFLOPS = getCpuGFLOPS(1024);
    MNN_PRINT("CPU GFLOPS: %f", cpuFLOPS);
    for (int i = 0; i < 9; i++){
        double flops = getKernelGLOPS(i, 1024);
        MNN_PRINT("Kernel %d GFLOPS: %f",i,flops);
    }
}

void kernelPerformance(int kernelToTestID){
    float flops = getKernelGLOPS(kernelToTestID, 1024);
    MNN_PRINT("Kernel %d GFLOPS: %f",kernelToTestID,flops);
}


extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_profileKernel(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";
//    cpu();
//    gpu(8);
    printFlops();
//    kernelPerformance(6);
//    deviceInfo();
//    printKernelInfo();
    return env->NewStringUTF(hello.c_str());
}