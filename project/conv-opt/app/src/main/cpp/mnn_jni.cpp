#include <jni.h>
#include <string>

//General
#include "MNN/Tensor.hpp"
#include "MNN_generated.h"
#include "common/CommonCompute.hpp"
//#include "test/TestUtils.h"
//opencl resources
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "MNN/expr/Executor.hpp"
//#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "backend/opencl/execution/buffer/DepthwiseConvBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

//CPU resources:
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "MNN/AutoTime.hpp"
#include <random>

//NN resources:
#include "MNN/expr/Module.hpp"
#include "train/source/nn/NN.hpp"
#include "SGD.hpp"
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/nn/RandomGenerator.hpp"
#include "train/source/models/Lenet.hpp"
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/demo/mnistTrain.cpp"
#include "MnistDataset.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::OpenCL;

static PadMode _convertPadMode(PaddingMode mode) {
    switch (mode) {
        case CAFFE:
            return PadMode_CAFFE;
        case VALID:
            return PadMode_VALID;
        case SAME:
            return PadMode_SAME;
        default:
            break;
    }
    return PadMode_CAFFE;
}

VARP _Conv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
           PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0},
           bool relu = false, bool relu6 = false, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool sparese = false) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padMode     = _convertPadMode(pad);
    if (pads.size() == 2) {
        conv2D->common->padX        = pads[0];
        conv2D->common->padY        = pads[1];
    } else {
        conv2D->common->pads = std::move(pads);
    }
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    conv2D->common->relu6 = relu6;
    conv2D->common->relu = relu;
    if (sparese) {
        size_t weightNNZElement, weightBlockNumber = 0;
        CommonCompute::statisticWeightSparsity(weightNNZElement, weightBlockNumber, weight.data(), bias.size(), weight.size() / bias.size(), sparseBlockOC);

        std::unique_ptr<MNN::AttributeT> arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        std::unique_ptr<MNN::AttributeT> arg2(new MNN::AttributeT);;
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        std::unique_ptr<MNN::AttributeT> arg3(new MNN::AttributeT);;
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        std::unique_ptr<MNN::AttributeT> arg4(new MNN::AttributeT);;
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1.get());
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2.get());
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3.get());
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4.get());

        argsVector.emplace_back(sparseArg1);
        argsVector.emplace_back(sparseArg2);
        argsVector.emplace_back(sparseArg3);
        argsVector.emplace_back(sparseArg4);

        auto sparseArgs = builder.CreateVectorOfSortedTables<MNN::Attribute>(&argsVector);
        auto sparseCom = MNN::CreateSparseCommon(builder, sparseAlgo, sparseArgs);
        builder.Finish(sparseCom);
        auto sparseComPtr = flatbuffers::GetRoot<MNN::SparseCommon>(builder.GetBufferPointer())->UnPack();

        conv2D->sparseParameter.reset(sparseComPtr);
    }
    MNN_ASSERT(weight.size() == channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    conv2D->weight = std::move(weight);
    MNN_ASSERT(bias.size() == channel[1]);
    conv2D->bias = std::move(bias);
    return (Variable::create(Expr::create(convOp.get(), {x})));
}


void generateWeight(std::vector<float>& weightData, int ic, int oc, int kh, int kw, int dilation, int group, int sparseBlockOC) {
    for (int i = 0; i < group * (oc / group) * (ic / group) * kw * kh; i++) {
        auto data      = ((((i / kw)% 1317) * ((i / kh) % 1317)) % 1317 + i / ic + i / oc + (((oc - i) % 1317) * ic) % 1317 + i * ((oc - i) % 1317)) % 1317;
        auto floatData      = (float)(data % 255) / 255.0f;
        weightData.push_back(floatData);
    }
}

bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
          int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
          int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false) {
    using namespace MNN::Express;
    std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
    std::vector<float> weightData, biasData;

    generateWeight(weightData, ic, oc, kh, kw, dilation, group, sparseBlockOC);

    for (int i = 0; i < oc; i++) {
        auto data      = (((i / kw) % 1317) * ((i / kh) % 1317) + i / ic + i / oc + (oc - i) * ic + i * (oc - i)) % 1317;
        auto floatData = (float)(data % 255) / 255.0f;
        data           = data * data;
        biasData.push_back(floatData);
        // biasData.push_back(0.0f);
    }

    std::vector<float> inputData, outputData;
    for (int i = 0; i < ih * iw * ic * batch; ++i) {
        auto data      = ((i / kw) % 1317) * ((i / kh) % 1317) + ((i / ic)% 1317) * ((i / oc) % 1317) + ((oc - i) % 1317) * ic + (i % 1317) * ((oc - i) % 1317);
        data = data % 1317;
        data           = (data * data) % 1317;
        auto floatData = (float)(data % 255) / 255.0f;
        inputData.push_back(floatData);
    }

//    if (debug) {
//        std::vector<float> printCache(inputData.size());
//        for (int i = 0; i < inputData.size(); ++i) {
//            printCache[i] = FP32Converter[precision](inputData[i]);
//        }
//        MNN_PRINT("input:");
//        formatMatrix(printCache.data(), {batch, ic, ih, iw});
//        printCache.resize(weightData.size());
//        for (int i = 0; i < weightData.size(); ++i) {
//            printCache[i] = FP32Converter[precision](weightData[i]);
//        }
//        MNN_PRINT("weight:");
//        formatMatrix(printCache.data(), {oc, ic, kh, kw});
//        printCache.resize(biasData.size());
//        for (int i = 0; i < biasData.size(); ++i) {
//            printCache[i] = FP32Converter[precision](biasData[i]);
//        }
//        MNN_PRINT("bias:");
//        formatMatrix(printCache.data(), {oc});
//    }

    //reference_conv2d(inputData, weightData, biasData, outputData, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw,
//                     stride, dilation, group, FP32Converter[precision]);

    auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
    ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

    // Single Conv
    bool mSparse = false;
    Timer timer;
    double time = 0.0f;
    timer.reset();

    auto output = _Conv(std::move(weightData), std::move(biasData), input, {ic, oc}, {kw, kh}, padMap[mode], {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false, sparseAlgo, sparseBlockOC, mSparse);
    time += (double) timer.durationInUs();
//    MNN_PRINT("Time taken is %f", time);
    // difference below 0.5% relative error is considered correct.
    auto outputPtr = output->readMap<float>();

//    if (debug) {
//        MNN_PRINT("\ndata NCHW shape:");
//        printDims(input->getInfo()->dim);
//        MNN_PRINT("\nweight OIHW shape:");
//        printDims({oc, ic, kh, kw});
//        MNN_PRINT("\noutput NCHW shape:");
//        printDims(output->getInfo()->dim);
//        MNN_PRINT("\nexpected output:");
//        formatMatrix(outputData.data(), output->getInfo()->dim);
//        MNN_PRINT("\nreal output:");
//        formatMatrix(outputPtr, output->getInfo()->dim);
//    }
//        if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.05)) {
//            MNN_PRINT("expect:\t real:\t\n");
//            for (int i = 0; i < outputData.size(); ++i)
//            {
//                MNN_PRINT("%f\t, %f\n", outputData[i], outputPtr[i]);
//            }
//            MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
//            return false;
//        }
    return true;
}


static void printVar(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr  = x->readMap<int32_t>();
    for (int i = 0; i < size; ++i) {
        MNN_PRINT("%d, ", ptr[i]);
    }
    MNN_PRINT("\n");
}

static void printVarFloat(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr = x->readMap<float>();
    for (int i = 0; i < size; ++i) {
        MNN_PRINT("%f, ", ptr[i]);
    }
    MNN_PRINT("\n");
}

class SingleConvLayer : public Module {
public:
    std::shared_ptr<Module> convLayer;
public:
    SingleConvLayer(void){
        NN::ConvOption convOption;
        convOption.kernelSize = {3, 3};
        convOption.channel    = {1, 1};
        convOption.depthwise  = false;
        convLayer.reset(NN::Conv(convOption));
//        fcLayer.reset(NN::Linear(size,size));
        registerModel({convLayer});
    }

    std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) {
        VARP x = inputs[0];
//        MNN_PRINT("Forwarding inputs");
        x = convLayer->forward(x);
        x = _Relu(x);
        return {x};
    }
};

Convolution2DCommon* createConvolution2DCommon(flatbuffers::FlatBufferBuilder& fbb, int kernelY, int kernelX, PadMode padMode, int padY, int padX, int inputChannel, int outputChannel) {
    auto builder = Convolution2DCommonBuilder(fbb);
    builder.add_kernelX(kernelX);
    builder.add_kernelY(kernelY);
    builder.add_inputCount(inputChannel);
    builder.add_outputCount(outputChannel);
    builder.add_padX(padX);
    builder.add_padY(padY);
    builder.add_padMode(padMode);
    auto offset = builder.Finish();
    return reinterpret_cast<Convolution2DCommon*>(fbb.GetCurrentBufferPointer() + fbb.GetSize() - offset.o);
}

Op* createConvOp() {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    // Need to set the values of conv2D
    conv2D->common->padX =0;
    conv2D->common->padY =0;
    conv2D->common->padMode     = PadMode_SAME;
    conv2D->common->strideX     = 1;
    conv2D->common->strideY     = 1;
    conv2D->common->group       = 1;
// may want to change this later
    conv2D->common->outputCount = 8;
    conv2D->common->inputCount  = 3;
    conv2D->common->dilateX     = 1;
    conv2D->common->dilateY     = 1;
    conv2D->common->kernelX     = 5;
    conv2D->common->kernelY     = 5;
    conv2D->common->relu6 = false;
    conv2D->common->relu = false;
    int weight = 1;
    int bias = 0;
    conv2D->weight.resize(1 * (1 / 1) * 5 * 5);
    std::fill(conv2D->weight.begin(), conv2D->weight.end(), weight);
    conv2D->bias.resize(1);
    std::fill(conv2D->bias.begin(), conv2D->bias.end(), bias);

    OpT* op = convOp.get();
    flatbuffers::FlatBufferBuilder builder;
    auto offset = Op::Pack(builder,op);
    Op* res = reinterpret_cast<Op*>(builder.GetCurrentBufferPointer() + builder.GetSize() - offset.o);
    return res;
}



double runConvLayer(int size, bool cpu=true){
    std::random_device gDevice;
    int ic         = 3;
    int oc         = 8;
    int kw         = 5;
    int kh         = 5;
    int iw         = size;
    int ih         = size;
    int weightSize = ic * oc * kw * kh;
    std::vector<float> targetVecs(weightSize);
    for (int i = 0; i < weightSize; ++i) {
        auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        targetVecs[i] = v;
    }
    auto weightTarget = _Const(targetVecs.data(), {oc, ic, kh, kw}, NCHW);
    std::vector<float> targetVecsBias(oc);
    for (int i = 0; i < oc; ++i) {
        targetVecsBias[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
    }
    auto biasTarget = _Const(targetVecsBias.data(), {oc}, NCHW);

    NN::ConvOption convOption;
    convOption.channel    = {ic, oc};
    convOption.kernelSize = {kw, kh};
    convOption.stride     = {2, 2};
    convOption.dilate     = {1, 2};
    convOption.padMode = SAME;
    std::shared_ptr<Module> convModule(NN::Conv(convOption));

    std::shared_ptr<SGD> sgd(new SGD(convModule));
    sgd->setLearningRate(0.01f);
    std::vector<float> randomInputs(1 * ic * ih * iw);
    for (int i = 0; i < randomInputs.size(); ++i) {
        randomInputs[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
    }
    std::vector<float> outputs(1 * oc * ih * iw);

    Timer timer;
    double avg_time = 0;
    timer.reset();
    int runs = 100;

    if (cpu){
        for (int i = 0; i < runs; ++i) {
            auto input = _Input({1, ic, ih, iw}, NCHW);
            auto inputPtr = input->writeMap<float>();
            ::memcpy(inputPtr, randomInputs.data(), randomInputs.size() * sizeof(float));
            auto targetValue = _Conv(weightTarget, biasTarget, _Convert(input, NC4HW4), convOption.padMode, convOption.stride, convOption.dilate);
            // auto predictValue = convModule->forward(input);
        }
    }
    else{
        std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
        BackendConfig config;
        config.power = BackendConfig::Power_High;
        executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
        Backend::Info info;
        info.mode = Backend::Info::DIRECT;
        info.numThread = 1;
        info.user = (BackendConfig*)&config;
        info.type = MNN_FORWARD_OPENCL;
        info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
//        MNN_PRINT("Setting up runtime.");
        OpenCL::CLRuntime cl_runtime(info);
//        MNN_PRINT("Setting up backend.");
        OpenCL::OpenCLBackend backend(&cl_runtime);
        OpenCLRuntime *runtime = backend.getOpenCLRuntime();

        std::vector<int> inputShape({ih,iw,ic});
        std::vector<int> outputShape({ih,iw,oc});
//        float A[ih][iw][ic] = &randomInputs[0];
//        float B[ih][iw][oc];
        float * A = randomInputs.data();
        float * B = outputs.data();
//        std::copy(randomInputs.begin(), randomInputs.end(), A);
//        std::copy(outputs.begin(), outputs.end(), B);

//        Tensor *input = Tensor::create<float>(inputShape, randomInputs);
////        Tensor *output = Tensor::create<float>(outputShape, outputs);
        Tensor *input = Tensor::create<float>(inputShape, A);
        Tensor *output = Tensor::create<float>(outputShape, B);
        std::vector < Tensor * > inputs({input}), outputs({output});
        Op* convOp = createConvOp();

        ConvBufExecution conv(inputs, outputs, convOp, &backend);
//        DepthwiseConvBufExecution conv(inputs, convOp, &backend);
//        MNN_PRINT("beginning convolution on GPU.");
        timer.reset();
        int res = conv.onResize(inputs, outputs);
        MNN_CHECK_CL_SUCCESS(res, "ConvBufExecution.onResize");

        for (int i =0 ; i < runs; i++){
            res = conv.onExecute(inputs, outputs);
        }
//        avg_time += timer.durationInUs();

        MNN_CHECK_CL_SUCCESS(res, "ConvBufExecution.onExecute");
//        MNN_PRINT("Convolution completed.");
    }

    avg_time += timer.durationInUs();
    return avg_time / runs;
}

void profileConvLayer(bool cpu){
    int size = 32;
    int limit = 16;
    int avg_time;
    std::stringstream output;
    for (int i = 1; i <= limit; i++){
        avg_time = runConvLayer(size * i, cpu);
        MNN_PRINT("%d -> %f", size*i, avg_time);
        output << std::to_string(avg_time) << ",";
    }
    MNN_PRINT("%s", output.str().c_str());
    MNN_PRINT("profileConvLayer FINISHED");
}

void convolutionTestGPU(int width=32){
    MNN_PRINT("setting up environment.");
    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    config.power = BackendConfig::Power_High;
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
    Backend::Info info;
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    info.user = (BackendConfig*)&config;
    info.type = MNN_FORWARD_OPENCL;
    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
    MNN_PRINT("Setting up runtime.");
//    static_cast<CPUBackend*>(backend())
//    OpenCLRuntime runtime_(config.precision,
//    MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE);
//    OpenCLRuntime *runtime = &runtime_;
//    runtime->setCommandQueueProfileEnable();
    OpenCL::CLRuntime cl_runtime(info);
    MNN_PRINT("Setting up backend.");
    OpenCL::OpenCLBackend backend(&cl_runtime);
    OpenCLRuntime *runtime = backend.getOpenCLRuntime();
//    CPURuntime runtime(info);
//    CPUBackend backend(&runtime, config.precision);
    MNN_PRINT("creating input and output tensors.");
    float A[width][width];
    float B[width][width];
    // Initialise arrays with values.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            A[i][j] = (float) (i * width + j);
//            B[i][j] = (float) (i * width + j);
            B[i][j]=0;
        }
    }
    std::vector<int> shape({width,width});
    Tensor* tensorA = Tensor::create<float>(shape, A);
    Tensor* tensorB = Tensor::create<float>(shape, B);
    std::vector<Tensor * > inputs({tensorA}), outputs({tensorB});

    Op* convOp = createConvOp();

    ConvBufExecution conv(inputs, outputs, convOp, &backend);
//    MNN_PRINT("beginning convolution on GPU.");
    int res = conv.onResize(inputs, outputs);
    MNN_CHECK_CL_SUCCESS(res, "ConvBufExecution.onResize");
    res = conv.onExecute(inputs, outputs);
    MNN_CHECK_CL_SUCCESS(res, "ConvBufExecution.onExecute");
    MNN_PRINT("Convolution completed.");
//    tensorA->print();
//    tensorB->print();
//    for (int i = 0; i < width; i++) {
//        for (int j = 0; j < width; j++) {
//            MNN_PRINT("%d, ", A[i][j]);
//        }
//    }
}



void nn_run(bool runOnCPU) {
    MNN_PRINT("Evaluating a single layer.");
    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    std::string root = "/data/local/tmp/mnist";
    RandomGenerator::generator(17);
    auto fc = new SingleConvLayer();
    std::shared_ptr<Module> model(fc);
//    MnistUtils::train(model1, root, MNN_FORWARD_CPU);
//    std::shared_ptr<Module> model2(new SimpleFC(size));
//    MnistUtils::train(model2, root, MNN_FORWARD_OPENCL);
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
//    OpenCLBackend config;

    if (runOnCPU){
        MNN_PRINT("Running on CPU");
        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    }
    else{
        MNN_PRINT("Running on GPU");
        exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);
//        OpenCLRuntime runtime_(config.precision, info.gpuMode);
//        OpenCLRuntime *runtime = &runtime_;
    }
    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 64;
    const size_t numWorkers = 0;
    bool shuffle            = false;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));
    model->clearCache();
    exe->gc(Executor::FULL);
    exe->resetProfile();

    auto trainData = dataLoader->next();
    auto example = trainData[0];
    auto cast       = _Cast<float>(example.first[0]);
    example.first[0] = cast * _Const(1.0f / 255.0f);
//    moveBatchSize += example.first[0]->getInfo()->dim[0];
    int predict;
    int hotRuns = 1000;
    int warmupRuns = hotRuns/2;
    Timer timer;
    double avgTime = 0.0f;
    for (int j = 0; j < warmupRuns; j++){
//        VARP input = example.first[0];
//        printVar(input);
        model->forward(example.first[0]);
    }
    timer.reset();
    for (int i = 0; i < hotRuns; i++){
        model->forward(example.first[0]);
    }
    avgTime += timer.durationInUs();
    timer.reset();
    avgTime /= hotRuns;
//#ifdef RUN_ON_CPU
//    MnistUtils::train(model, root, MNN_FORWARD_CPU);
//#else
//    MnistUtils::train(model, root, MNN_FORWARD_OPENCL);
//#endif
    MNN_PRINT("Finished execution, avgTime: %f us",avgTime);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_mnnconvolutionoptimisation_MainActivity_profileNN(
        JNIEnv* env,
        jobject /* this */) {
    std::string outputString = "Benchmarking feedforward through fully convolutional layer";
//    nn_run(true);
//    convolutionTestGPU(128);
    profileConvLayer(false);
//    int precision = 1;
//    int size = 512;
//    bool result = test(MNN_FORWARD_CPU, "CPU", "Conv2D", 1, 32, 32, size, size, PadMode_SAME, 0, 0, 3, 3, 1, 1, 1, precision, MNN::SparseAlgo_RANDOM, 4, true);
//    if (result){
//        MNN_PRINT("Success.");
//    }
    return env->NewStringUTF(outputString.c_str());
}
