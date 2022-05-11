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

void print2dHalfArray(cl_half * arr, int rows, int cols){
    for (int i = 0; i < rows; i++){
        std::stringstream toPrint;
        for (int j = 0; j < cols; j++){
            toPrint << std::to_string(arr[i*cols + j]) << " ";
        }
        MNN_PRINT("%s\n", toPrint.str().c_str());
    }
}

void printBuffer(const cl::Buffer & buffer, OpenCLRuntime *runtime, int rows, int cols, int kernelID){
    cl::CommandQueue commandQueue = runtime->commandQueue();
    float output[rows][cols];
    cl_half output2[rows][cols];
    cl_event readEvent;
    commandQueue.enqueueReadBuffer(buffer, CL_TRUE, 0, rows*cols*sizeof(float), output);
    if (kernelID == 9){
        print2dHalfArray(*output2, rows, cols);
    }
    else{
        print2dArray(*output, rows, cols);
    }
}

float bufferDifference(const cl::Buffer & bufferA, const cl::Buffer & bufferB, OpenCLRuntime *runtime,  int rows, int cols, int kernelID){
    cl::CommandQueue commandQueue = runtime->commandQueue();
    float A[rows][cols];
    float B[rows][cols];
    cl_half B2[rows][cols];
    commandQueue.enqueueReadBuffer(bufferA, CL_TRUE, 0, rows*cols*sizeof(float), A);
    commandQueue.enqueueReadBuffer(bufferB, CL_TRUE, 0, rows*cols*sizeof(float), B);
    commandQueue.enqueueReadBuffer(bufferB, CL_TRUE, 0, rows*cols*sizeof(cl_half), B2);

    float accumulatedError = 0;
    if (kernelID == 8){
        for (int i = 0 ; i < rows; i++){
            for (int j=0; j < cols; j++){
                if (A[i][j] != (float)B2[i][j])
                accumulatedError += std::abs(A[i][j] - (float)B2[i][j])  ;
            }
        }
    }
    else{
        for (int i = 0 ; i < rows; i++){
            for (int j=0; j < cols; j++){
                accumulatedError += std::abs(A[i][j] - B[i][j]);
            }
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


    // float A1[width][width];
    float *A1 = new float[width*width];
    cl_half *Ahalf = new cl_half[width*width];
    float A2[width][width];
    float B1[width][width];
    float B2[width][width];
    cl_half *Bhalf = new cl_half[width*width];
    float C[width][width];
    cl_half *Chalf = new cl_half[width*width];
    // Initialise arrays with values.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
//            int val = i * width + j;
            float val = 0.5;
            if (i < oldWidth && j < oldWidth){
//                A1[i][j] = (float) (i * width + j) ;
                A1[i*width+ j] = val ;
                B1[i][j] =  val + 0.25;
                B2[i][j] = val + 0.25;
                Bhalf[i*width+j] = (cl_half) val;
//                B[i][j] = (float) (i * width + j) + 1;
                if (kernelToTestID == 4 or kernelToTestID == 5 or kernelToTestID == 6 or kernelToTestID == 7 or kernelToTestID == 8 or kernelToTestID == 9){
                    A2[j][i] =  val;
                    Ahalf[j*width + i] = (cl_half) val;
                }
                else{
                    A2[i][j] =  val;
                    Ahalf[i*width + j] = (cl_half) val;
                }
                C[i][j] = 0.0f;
                Chalf[i*width + j] = 0;
            }
            else{
//                A1[i][j] = 0.f;
                A1[i*width+ j] = 0.f;
                A2[i][j] = 0.f;
                B1[i][j] = 0.f;
                B2[i][j] = 0.f;
                C[i][j] = 0.f;
                Chalf[i*width + j] = 0;
            }
        }
    }
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    cl::Buffer bufferA2;
    cl::Buffer bufferB2;
    cl::Buffer bufferC2;
    if (kernelToTestID == 8){
        bufferA2 = cl::Buffer(context, flags, sizeof(cl_half)*M*K, Ahalf);
        bufferB2 = cl::Buffer(context, flags, sizeof(cl_half)*K*N, Bhalf);
        bufferC2 = cl::Buffer(context, flags, sizeof(cl_half)*M*N, Chalf);
    }
    else{
        bufferA2 = cl::Buffer(context, flags, sizeof(float)*M*K, A2);
        bufferB2 = cl::Buffer(context, flags, sizeof(float)*K*N, B2);
        bufferC2 = cl::Buffer(context, flags, sizeof(float)*M*N, C);
    }
    cl::Buffer bufferA1(context, flags, sizeof(float)*M*K, A1);
//    cl::Buffer bufferA2(context, flags, sizeof(float)*M*K, A2);
    cl::Buffer bufferB1(context, flags, sizeof(float)*K*N, B1);
//    cl::Buffer bufferB2(context, flags, sizeof(float)*K*N, B2);
    cl::Buffer bufferC1(context, flags, sizeof(float)*M*N, C);
//    cl::Buffer bufferC2(context, flags, sizeof(float)*M*N, C);

    cl::Kernel kernel = setKernelArgs(width, width, width, bufferA1, bufferB1, bufferC1, klp.kernel, runtime);
    runKernel2D(kernel, klp.global, klp.local, runtime, nullptr);

    kernelAndLaunchParameters klpToTest = getKernelAndLaunchParameters(kernelToTestID, runtime, width, width);

    cl::Kernel kernelToTest = setKernelArgs(width, width, width, bufferA2, bufferB2, bufferC2, klpToTest.kernel, runtime);
    runKernel2D(kernelToTest, klpToTest.global, klpToTest.local, runtime, nullptr);
//    MNN_PRINT("C: ");
//    printBuffer(bufferC1, runtime, width, width, kernelToTestID);
    float error = bufferDifference(bufferC1, bufferC2, runtime, width, width, kernelToTestID);
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
    isKernelCorrect(0, 128);
    isKernelCorrect(1, 128);
    isKernelCorrect(2, 128);
    isKernelCorrect(3, 128);
    isKernelCorrect(4, 128);
    isKernelCorrect(5, 128);
    isKernelCorrect(6, 256);
    isKernelCorrect(7, 128);
    isKernelCorrect(8, 128);
//    isKernelCorrect(9, 64);
    return env->NewStringUTF(hello.c_str());
}