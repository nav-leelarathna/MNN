//
// Created by navin on 07/02/2022.
//
#include <jni.h>
#include <sstream>
#include <string>
//#include <iostream>
// general
#include "MNN/Tensor.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
// Img2Col import
#include <geometry/GeometryConvUtils.hpp>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "MNN/expr/Executor.hpp"
#include <CL/cl.h>

#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "MNN/AutoTime.hpp"
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;
#define TS4 16
#define WPT 8
#define WIDTH 4
//void GeometryConvUtils::im2Col(Tensor* im2Col, Tensor* input, int ic, int kh, int kw, int batch, int oh, int ow, int ih, int iw, int sh, int sw, int dh, int dw, std::pair<int, int> pads, int srcKernelOffset, Tensor* padVal)

Tensor* myIm2Col(Tensor * inputTensor,int kh, int kw, int sh, int sw){
    int ih = inputTensor->height();
    int iw = inputTensor->width();
    int ic = inputTensor->channel();
//    MNN_PRINT("height,width,channels: %d,%d,%d",ih,iw,ic);
    int patchWidth = (iw - kw)/sw + 1;
    int patchHeight = (ih - kh)/sh + 1;

    auto hostData    = inputTensor->host<float>();

    int outputHeight = patchHeight * patchWidth;
    int outputWidth = kh * kw * ic;
    Tensor* outputTensor(Tensor::create<float>(std::vector<int>{outputHeight, outputWidth}));
    auto outputData = outputTensor->host<float>();
//    MNN_PRINT("address of hostData is: %p",hostData);
//    MNN_PRINT("second element of hostData: %f", hostData[1]);
    for (int c = 0; c < ic; c++){
        for (int pw=0; pw < patchWidth; pw++){
            for(int ph=0; ph < patchHeight; ph++){
                int patchNumber = ph * patchWidth + pw;
//                MNN_PRINT("starting patch %d", patchNumber);
                std::stringstream output;
                int start_x = pw * sw;
                int start_y = ph * sh;
                for (int j = 0; j < kh; j++){
                    for (int i = 0; i < kw; i++){
                        int k = (start_y+j)*iw*ic + (start_x+i)*ic + c;
                        float item = hostData[k];
//                        MNN_PRINT("index into input is %d, value is %f",k,item);
                        output << std::to_string(item) << " ";
//                        output[patchNumber][c*kh*kw+ (j*kw) + i] = item;
                        int l = patchNumber*outputWidth + c*kh*kw+ (j*kw) + i;
                        outputData[l] = item;
                    }
                }
//                MNN_PRINT("%s", output.str().c_str());
            }
        }
    }
    return outputTensor;
}


Tensor* im2ColTransform(Tensor* input, int ih, int iw, int ic, int batches=1){
    std::shared_ptr<Tensor> im2Col(new Tensor);
    int inputChannel = ic;
    int kernelHeight = 2;
    int kernelWidth = 2;
    int batch = batches;
    int inputHeight = ih;
    int inputWidth = iw;
    int sHeight = 1;
    int sWidth = 1;
    int outputHeight = ((inputHeight - kernelHeight) / sHeight) + 1;
    int outputWidth = ((inputWidth - kernelWidth) / sWidth) + 1;
    int dilateHeight = 1;
    int dilateWidth = 1;
    std::pair<int, int> pads(0,0);
    int srcKernelOffset = 0;
    Tensor* padVal = nullptr;
    MNN_PRINT("About to call im2Col");
    GeometryConvUtils::im2Col(im2Col.get(), input, inputChannel, kernelHeight, kernelWidth, batch, outputHeight, outputWidth, inputHeight, inputWidth,sHeight, sWidth, dilateHeight, dilateWidth, pads, srcKernelOffset, padVal);
    MNN_PRINT("Finished im2Col");
    return im2Col.get();
}

Tensor* initialiseInputTensor(int height, int width, int channels, int batches=1){
//    float A[batches* height * width * channels];
//    for (int i = 0 ; i < batches * height * width * channels; i++){
//        A[i] = i;
//    }
//    float A[batches][height][width][channels];
//    for (int i = 0 ; i < height; i++){
//        for (int j = 0 ; j < width ; j++){
//            for (int c = 0 ; c < channels; c++){
//                A[0][i][j][c] = i * width + j;
//            }
//        }
//    }
    std::shared_ptr<Tensor> hostTensor(Tensor::create<float>(std::vector<int>{batches, height, width, channels}));
    auto elementSize = hostTensor->elementSize();
    MNN_PRINT("element size is %d",elementSize);
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        hostData[i]    = i;
    }
    MNN_PRINT("initialiseInputTensor: address of data is %p",hostData);
    return hostTensor.get();
}

Tensor* initialiseKernelTensor(int height, int width, int channels){
    std::shared_ptr<Tensor> hostTensor(Tensor::create<float>(std::vector<int>{ 1,height, width, channels}));
    auto elementSize = hostTensor->elementSize();
    MNN_PRINT("element size is %d",elementSize);
    auto hostData    = hostTensor->host<float>();
    for (int i = 0; i < elementSize; ++i) {
        hostData[i] = 1;
    }
    return hostTensor.get();
}

void transposeTensor(Tensor* tensor){
    int height;
    int width;
    if (tensor->dimensions() != 4){
        height = tensor->length(0);
        width = tensor->length(1);
        tensor->setLength(0, width);
        tensor->setLength(1,height);
    }
    else{
        height = tensor->height();
        width = tensor->width();
        tensor->setLength(1, width);
        tensor->setLength(2, height);
    }
}

void reshapeTensor(Tensor* tensor, int height, int width){
    int size = tensor->elementSize();
    if (height * width != size){
        MNN_PRINT("Cannot convert (%d, %d) to (%d, %d)", tensor->length(0), tensor->length(1), height, width);
    }
    else if (tensor->dimensions() != 4){
        tensor->setLength(0, height);
        tensor->setLength(1,width);
    }
    else{
        tensor->setLength(1, height);
        tensor->setLength(2, width);
    }
}


void printTensor(const Tensor* result, int height = 0, int width = 0){
    int eleSize = result->elementSize();
    result->printShape();
    if (height > 0 && width > 0 && height*width==eleSize){
        for (int h = 0; h < height; h ++){
            std::stringstream output;
            for (int w = 0; w < width; w++){
                int i = h * width + w;
                output << std::to_string(result->host<float>()[i]) << " ";
            }
            MNN_PRINT("%s", output.str().c_str());
        }
    }
    else{
        for (int i = 0; i < eleSize; ++i) {
            MNN_PRINT("%f", result->host<float>()[i]);
        }
    }
}

Tensor* cpuMatMul(Tensor* im2col, Tensor* kernel2row){
    int m = im2col->length(0);
    int k = im2col->length(1);
    int n = kernel2row->length(1);
//    MNN_PRINT("(m,k,n) = (%d,%d,%d)",m,k,n );
//    float output[m][n];
    std::vector<int> shape({1, m, n, 1});
//    Tensor *outputTensor = Tensor::create<float>(shape, output);
    Tensor* outputTensor(Tensor::create<float>(shape));

    std::shared_ptr <Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    info.user = (BackendConfig * ) & config;

    CPURuntime runtime(info);
    CPUBackend backend(&runtime, config.precision);

    std::vector < Tensor * > inputs({im2col, kernel2row}), outputs({outputTensor});
    CPUMatMul matmul(&backend, false, false, false, false);
    MNNSetCPUThreadsMode(MNN_CPU_MODE_BIG);
    matmul.onResize(inputs, outputs);
    matmul.onExecute(inputs, outputs);
    printTensor(outputTensor, m, n);
    return outputTensor;
}

void gpuMatMul(const Tensor* im2col, const Tensor* kernel2row){
    int m = im2col->length(1);
    int k = im2col->length(2);
    int n = kernel2row->length(2);

    float A[m][k];
    float B[k][n];

    auto source_a = im2col->host<float>();
    auto source_b = kernel2row ->host<float>();

    for (int h = 0; h < m; h++){
        for (int w = 0; w < k; w++){
            int k = h * m + w;
            A[h][w] = source_a[k];
        }
    }
    for (int h = 0; h < k; h++){
        for (int w = 0; w < n; w++){
            int k = h * k + w;
            B[h][w] = source_b[k];
        }
    }


    MNN_PRINT("(m,k,n) = (%d,%d,%d)",m,k,n );

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
    OpenCL::CLRuntime cl_runtime(info);
    OpenCL::OpenCLBackend backend(&cl_runtime);
    OpenCLRuntime *runtime = backend.getOpenCLRuntime();
//    OpenCLRuntime runtime_(config.precision, info.gpuMode);
//    OpenCLRuntime *runtime = &runtime_;
    // What is this command queue?
    runtime->setCommandQueueProfileEnable();
    // what is this context?
    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();

    std::set <std::string> buildOptions;
    std::stringstream option;
    int numTiles;
//    int oldWidth = width;
//    prepareWidthAndTiles(width, numTiles);
//    MNN_PRINT("width and tiles: %d, %d\n",width,numTiles);

//    option << " -DTS4=" << std::to_string(TS4)
//           << " -DWIDTH=" << std::to_string(WIDTH)
//           << " -DKERNEL=4 ";
    buildOptions.emplace("-DKERNEL=1");
    cl::Kernel kernel = runtime->buildKernel("opt_gemm", "gemm1", buildOptions);
    std::vector<uint32_t> global({(uint32_t)m, (uint32_t)n}), local({16, 16});
    // buildOptions.emplace(option.str());
    // cl::Kernel kernel = runtime->buildKernel("opt_gemm", "gemm4", buildOptions);
//    uint32_t global0 = (uint32_t) m / WIDTH,
//            global1 = (uint32_t) n,
//            local1 = TS4,
//            local0 = TS4/WIDTH;
    //    MNN_PRINT("global0=%d, global1=%d, local0=%d, local1=%d", global0, global1, local0, local1);
    // std::vector<uint32_t> global({global0, global1}), local({local0, local1});
    int idx = 0;
    int res = 0;
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    MNN_PRINT("Initialising output buffers");
//    std::vector<int> shape({1, m, n, 1});
//    Tensor* outputTensor = Tensor::createDevice<float>(shape);
//    bool success = backend.onAcquireBuffer(outputTensor, Backend::DYNAMIC);
//    if (!success){
//        MNN_PRINT("unable to acquire buffer for output tensor.");
//    }
////    Tensor* outputTensor(Tensor::create<float>(shape));
//    uint64_t device_id  = outputTensor->deviceId();
//    MNN_PRINT("device id is %d", device_id);
//    cl::Buffer outputBuffer = openCLBuffer(outputTensor);
    float C[m][n];

    cl::Buffer inputBuffer(context, flags, 4*m*k, A);
    cl::Buffer kernelBuffer(context, flags, 4*n*k, B);
    cl::Buffer outputBuffer(context, flags, 4*m*n, C);
//    Tensor* a = Tensor::createDevice<float>(im2col->shape());
//    success = backend.onAcquireBuffer(a, Backend::DYNAMIC);
//    if (!success){
//        MNN_PRINT("unable to acquire buffer for im2col tensor.");
//    }
//    device_id = a->deviceId();
//    MNN_PRINT("im2col device id is %d", device_id);
////    backend.onCopyBuffer(im2col, a);
//    success = a->copyFromHostTensor(im2col);
//    if (!success){
//        MNN_PRINT("unable to copy to a");
//    }
//    MNN_PRINT("copied data");
//    Tensor * a = Tensor::createHostTensorFromDevice(im2col,true);
//    Tensor * a2 = Tensor::createDevice<float>(im2col->shape());
//    success = backend.onAcquireBuffer(a2, Backend::DYNAMIC);
//    success = a2->copyFromHostTensor(a);
//        if (!success){
//        MNN_PRINT("unable to copy to a");
//    }

//    auto destination_a = a->host<float>();
//    auto source_a = im2col->host<float>();
//    int elementSize = im2col->elementSize();
//    MNN_PRINT("setting values");
//    for (int i =0; i < elementSize; i++){
//        destination_a[i] = source_a[i];
//    }
//    cl::Buffer inputBuffer = openCLBuffer(a);
//
//    MNN_PRINT("values set");
//    Tensor* b = Tensor::createDevice<float>(kernel2row->shape());
//    success = backend.onAcquireBuffer(b, Backend::DYNAMIC);
//    backend.onCopyBuffer(kernel2row, b);
//    cl::Buffer kernelBuffer = openCLBuffer(b);
//    float output[m][n];
//    cl::Buffer outputBuffer(context, flags, 4*m*n, output);

    MNN_PRINT("setting arguments");

    res |= kernel.setArg(idx++, m); // M
    res |= kernel.setArg(idx++, n); // N
    res |= kernel.setArg(idx++, k); // K
    res |= kernel.setArg(idx++, inputBuffer);
    res |= kernel.setArg(idx++, kernelBuffer);//openCLBuffer(tensorB));
    res |= kernel.setArg(idx++, outputBuffer);//openCLBuffer(tensorC));
    MNN_CHECK_CL_SUCCESS(res, "outputBuffer");
//    res |= kernel.setArg(idx++, numTiles);
    MNN_CHECK_CL_SUCCESS(res, "setArg");
    MNN_PRINT("Running kernel.");
    runKernel2D(kernel, global, local, runtime,nullptr);
    // return outputTensor;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            MNN_PRINT("%f", C[i][j]);
        }
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_mnnconvolutionoptimisation_MainActivity_testConvolution(
        JNIEnv* env,
        jobject /* this */) {
    std::string outputString = "Im2Col + gemm";
    //    Tensor* inputTensor = initialiseInputTensor(inputHeight, inputWidth, channels);
//    printTensor(inputTensor);
//    Tensor* im2Col = im2ColTransform(inputTensor,inputHeight, inputWidth, channels);
//    MNN_PRINT("im2Col performed on input tensor: ");
//    printTensor(im2Col);
    int inputHeight = 3, inputWidth=3;
    int channels = 1;
    int kernelWidth = 2, kernelHeight = 2;
    int sh = 1, sw = 1;

    int patchWidth = (inputWidth - kernelWidth)/sw + 1;
    int patchHeight = (inputHeight - kernelHeight)/sh + 1;

    Tensor * inputTensor = initialiseInputTensor(inputHeight, inputWidth, channels);
//    printTensor(inputTensor);
    Tensor * im2col = myIm2Col(inputTensor, kernelHeight, kernelWidth,sh,sw);
    Tensor* kernel = initialiseKernelTensor(kernelHeight,kernelWidth,channels);
//    printTensor(kernel);
    Tensor* kernel2row = myIm2Col(kernel,kernelHeight,kernelWidth,1,1);
//    printTensor(kernel2row);
    transposeTensor(kernel2row);
//    MNN_PRINT("transposed kernel");
//    kernel2row->printShape();
//    Tensor* result = gpuMatMul(im2col, kernel2row);
    gpuMatMul(im2col, kernel2row);
//    Tensor* result = cpuMatMul(im2col, kernel2row);
//    reshapeTensor(result, patchHeight, patchWidth);
//    printTensor(result, patchHeight, patchWidth);
    return env->NewStringUTF(outputString.c_str());
}