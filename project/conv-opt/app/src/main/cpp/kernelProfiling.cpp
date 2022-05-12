//
// Created by navin on 18/11/2021.
//

#include "kernelDefines.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;


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
    CPUMatMul matmul(&backend, false, false, false, false);
    MNNSetCPUThreadsMode(MNN_CPU_MODE_BIG);
    matmul.onResize(inputs, outputs);

    int warmup = 10, hot_runs = HOT_RUNS, overall = 2;
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
    int warmup_steps = 5, hot_runs = HOT_RUNS, last_runs = 2, overall_runs =1;
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
        for (int i = 0; i < last_runs; i++)
            runKernel2D(kernel, global, local, runtime, nullptr);
    }
    double avg_time = 0.0f;
//    std::stringstream option;
    for (cl::Event event : events){
        double time = runtime->getCostTime(&event);
//        MNN_PRINT("%f\n",time);
//        double gflops = ((2 * 1024 / time) / 1e3f) * 1024 * 1024;
//        option << std::to_string(gflops) << ", ";
        avg_time += time;
    }
//    MNN_PRINT("%s", option.str().c_str());
    return avg_time / (hot_runs * overall_runs);
}

double profileKernelOnGPU(int width, int kernelID) {
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

    std::set <std::string> buildOptions;
    std::stringstream option;
    uint32_t width_uint = (uint32_t) width;
    kernelAndLaunchParameters klp = getKernelAndLaunchParameters(kernelID, runtime, width, width);
    float *A = new float[width*width];
    float *B = new float[width*width];
    float *C = new float[width*width];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
          B[i*width+j] = (float) (i * width + j);
        if (kernelID == 4 or kernelID == 5 or kernelID == 6 or kernelID == 7){
            A[j*width+i] =(float) (i * width + j) ;
        }
        else{
            A[i*width + j] =(float) (i * width + j);
        }
        C[i*width+j] = 0.0f;

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


void getCPUExecutionTimes(void){
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

void getGPUExecutionTimes(int kernelID) {
    MNN_PRINT("Running kernel %d on GPU", kernelID);
    int starting = 0;
    std::stringstream output;
    int start = starting/32 + 1; //, offset = 15;
    int limit = LIMIT;
    for (int i = start; i<=limit; i++) {
        int mat_size = i*32;
        double time = profileKernelOnGPU(mat_size, kernelID);
        output << std::to_string(time) << ",";
    }

    MNN_PRINT("%s", output.str().c_str());
    MNN_PRINT("DONE!");
}

double getCpuGFLOPS(int size){
    double time = profileCPU(size);
    double res =  ((2 * size / time) / 1e3f) * size * size;
    return res;
}

double getKernelGFLOPS(int kernelToTestID, int size){
    double time = profileKernelOnGPU(size, kernelToTestID);
    double res =  ((2 * size / time) / 1e3f) * size * size;
    return res;
}

void printKernelPerformance(int kernelToTestID){
    double flops = getKernelGFLOPS(kernelToTestID, 1024);
    MNN_PRINT("Kernel %d GFLOPS: %f",kernelToTestID,flops);
}

void printAllFlops(void){
    double cpuFLOPS = getCpuGFLOPS(1024);
    MNN_PRINT("CPU GFLOPS: %f", cpuFLOPS);
    for (int i = 0; i < 9; i++){
        printKernelPerformance(i);
    }
}


extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_profileKernel(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";
//    getCPUExecutionTimes();
//    getGPUExecutionTimes(8);
    printAllFlops();
//    printKernelPerformance(4);
    return env->NewStringUTF(hello.c_str());
}