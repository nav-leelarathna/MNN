//
// Created by navin on 26/04/2022.
//

#include "kernelDefines.h"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;

void print2dArray(float * arr, int rows, int cols){
    for (int i = 0; i < rows; i++){
        std::stringstream toPrint;
        for (int j = 0; j < cols; j++){
            toPrint << std::to_string(arr[i*cols + j]) << " ";
        }
        MNN_PRINT("%s\n", toPrint.str().c_str());
    }
}

void printBuffer(const cl::Buffer & buffer, OpenCLRuntime *runtime, int rows, int cols){
    cl::CommandQueue commandQueue = runtime->commandQueue();
    float output[rows][cols];
    cl_event readEvent;
    commandQueue.enqueueReadBuffer(buffer, CL_TRUE, 0, rows*cols*sizeof(float), output);
    print2dArray(*output, rows, cols);
}

float bufferDifference(const cl::Buffer & bufferA, const cl::Buffer & bufferB, OpenCLRuntime *runtime,  int rows, int cols){
    cl::CommandQueue commandQueue = runtime->commandQueue();
    float A[rows][cols];
    float B[rows][cols];
    cl_event readEvent;
    commandQueue.enqueueReadBuffer(bufferA, CL_TRUE, 0, rows*cols*sizeof(float), A);
    commandQueue.enqueueReadBuffer(bufferB, CL_TRUE, 0, rows*cols*sizeof(float), B);

    float accumulatedError;
    for (int i = 0 ; i < rows; i++){
        for (int j=0; j < cols; j++){
            accumulatedError += (A[i][j] - B[i][j]) * (A[i][j] - B[i][j]) ;
        }
    }
    return accumulatedError;
}

float getKernelError(int width, int kernelToTestID) {
    int M = width, N=width, K=width;
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

//    std::set <std::string> buildOptions;
//    std::stringstream option;
    int numTiles;
    int oldWidth = width;
    prepareWidthAndTiles(width, numTiles, kernelToTestID);
    kernelAndLaunchParameters klp = getKernelAndLaunchParameters(0, runtime, width, width);


    float A1[width][width];
    float A2[width][width];
    float B1[width][width];
    float B2[width][width];
    float C[width][width];
    // Initialise arrays with values.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (i < oldWidth && j < oldWidth){
                A1[i][j] = (float) (i * width + j) ;
                B1[i][j] = (float) (i * width + j) + 1;
                B2[i][j] = (float) (i * width + j) + 1;
//                B[i][j] = (float) (i * width + j) + 1;
                if (kernelToTestID == 4 or kernelToTestID == 5 or kernelToTestID == 6 or kernelToTestID == 7 or kernelToTestID == 8){
                    A2[j][i] = (float) (i * width + j);
                }
                else{
                    A2[i][j] = (float) (i * width + j) ;
                }
                C[i][j] = 0.0f;
            }
            else{
                A1[i][j] = 0.f;
                A2[i][j] = 0.f;
                B1[i][j] = 0.f;
                B2[i][j] = 0.f;
                C[i][j] = 0.f;
            }
        }
    }
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    cl::Buffer bufferA1(context, flags, sizeof(float)*M*K, A1);
    cl::Buffer bufferA2(context, flags, sizeof(float)*M*K, A2);
    cl::Buffer bufferB1(context, flags, sizeof(float)*K*N, B1);
    cl::Buffer bufferB2(context, flags, sizeof(float)*K*N, B2);
    cl::Buffer bufferC(context, flags, sizeof(float)*M*N, C);

    cl::Kernel kernel = setKernelArgs(width, width, width, bufferA1, bufferB1, bufferC, klp.kernel, runtime);
    runKernel2D(kernel, klp.global, klp.local, runtime, nullptr);

    kernelAndLaunchParameters klpToTest = getKernelAndLaunchParameters(kernelToTestID, runtime, width, width);
    cl::Buffer bufferC2(context, flags, sizeof(float)*M*N, C);
    cl::Kernel kernelToTest = setKernelArgs(width, width, width, bufferA2, bufferB2, bufferC2, klpToTest.kernel, runtime);
    runKernel2D(kernelToTest, klpToTest.global, klpToTest.local, runtime, nullptr);
//    MNN_PRINT("C: ");
//    printBuffer(bufferC2, runtime, width, width);
    float error = bufferDifference(bufferC, bufferC2, runtime, width, width);
    return error / (width * width);
}

bool isKernelCorrect(int kernelID, int width, float errorThreshold=1.0){
    float error = getKernelError(width, kernelID);
    MNN_PRINT("Error for kernel %d: %f", kernelID, error);
    return (error < errorThreshold);
}

void testSuite(int width=2){
    bool kernelPass;
    for (int kernelID = 0; kernelID < 1; kernelID++){
        kernelPass = isKernelCorrect(kernelID, width);
        if (kernelPass){
            MNN_PRINT("Kernel %d passes", kernelID);
        }
        else{
            MNN_PRINT("Kernel %d fails", kernelID);
        }
    }
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_mnnconvolutionoptimisation_MainActivity_functionalityTest(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Finished.";
//    testSuite();
//    isKernelCorrect(0, 32);
//    isKernelCorrect(1, 32);
//    isKernelCorrect(2, 32);
//    isKernelCorrect(3, 32);
//    isKernelCorrect(4, 32);
//    isKernelCorrect(5, 32);
//    isKernelCorrect(6, 32);
    isKernelCorrect(7, 32);
    isKernelCorrect(8, 32);
    return env->NewStringUTF(hello.c_str());
}