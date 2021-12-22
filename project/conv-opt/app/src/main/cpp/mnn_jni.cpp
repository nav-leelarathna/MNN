//
// Created by navin on 27/10/2021.
//

#define FCLAYER
//#define KERNEL_TRICKS

#ifdef FCLAYER
#include <jni.h>
#include <string>

//General
#include "MNN/Tensor.hpp"
#include "MNN_generated.h"

//opencl resources
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "MNN/expr/Executor.hpp"
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

//CPU resources:
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "MNN/AutoTime.hpp"

//NN resources:
#include "MNN/expr/Module.hpp"
#include "train/source/nn/NN.hpp"
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/nn/RandomGenerator.hpp"
#include "train/source/models/Lenet.hpp"
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/demo/mnistTrain.cpp"


using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;


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

class Linear {

};

class SingleConvLayer : public Module {
public:

    std::shared_ptr<Module> convLayer;

public:
    SingleConvLayer(int size){
        NN::ConvOption convOption;
        convOption.kernelSize = {3, 3};
        convOption.channel    = {1, 8};
        convOption.depthwise  = false;
        convLayer.reset(NN::Conv(convOption));
//        fcLayer.reset(NN::Linear(size,size));
//        registerModel({fcLayer});
    }

    std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) {
        VARP x = inputs[0];
//        MNN_PRINT("HERE-simpleFC");
//        x = _Reshape(x, {0, -1});
//        x = _Convert(x, NCHW);
//        MNN_PRINT("x.shape: (%d, %d, %d, %d)",
//                  x->getInfo()->dim[0], x->getInfo()->dim[1],
//                  x->getInfo()->dim[2], x->getInfo()->dim[3]);
        x = convLayer->forward(x);
        x = _Relu(x);
        return {x};
    }
};


class SimpleFC : public Module  {
public:
    std::shared_ptr<Module> fc1; // 784 -> 16
    std::shared_ptr<Module> fc2; // 16 -> 16
    std::shared_ptr<Module> fc3; // 16 -> 10

public:

    SimpleFC(int size) {

//        int size = 128;
        fc1.reset(NN::Linear(size, size));
        fc2.reset(NN::Linear(size, size));
        fc3.reset(NN::Linear(size, size));

        registerModel({fc1, fc2, fc3});
    }

    std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) {
        VARP x = inputs[0];
//        MNN_PRINT("HERE-simpleFC");
//        x = _Reshape(x, {0, -1});
//        x = _Convert(x, NCHW);
//        MNN_PRINT("x.shape: (%d, %d, %d, %d)",
//                  x->getInfo()->dim[0], x->getInfo()->dim[1],
//                  x->getInfo()->dim[2], x->getInfo()->dim[3]);
        x = fc1->forward(x);
        x = _Relu(x);
        x = fc2->forward(x);
        x = _Relu(x);
        x = fc3->forward(x);
        x = _Softmax(x, 1);

        return {x};
    }

};

#define RUN_ON_CPU

void nn_run() {
    MNN_PRINT("STARTING");

    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    std::string root = "/data/local/tmp/mnist";
    RandomGenerator::generator(17);

//    for (int size = 4; size < 10; size++) {
//        int Size = 1 << size;
//        std::shared_ptr<Module> model1(new SimpleFC(Size));
//        MnistUtils::train(model1, root, MNN_FORWARD_CPU, Size, Size);
//        std::shared_ptr<Module> model2(new SimpleFC(Size));
//        MnistUtils::train(model2, root, MNN_FORWARD_OPENCL, Size, Size);
//    }
    int size = 16;
    auto fc = new SimpleFC(size);
    std::shared_ptr<Module> model1(fc);
    MnistUtils::train(model1, root, MNN_FORWARD_CPU);

    std::shared_ptr<Module> model2(new SimpleFC(size));
    MnistUtils::train(model2, root, MNN_FORWARD_OPENCL);
//#ifdef RUN_ON_CPU
//    MnistUtils::train(model, root, MNN_FORWARD_CPU);
//#else
//    MnistUtils::train(model, root, MNN_FORWARD_OPENCL);
//#endif


    MNN_PRINT("DONE");

}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_mnnconvolutionoptimisation_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    MNN_PRINT("Here");
    nn_run();


    return env->NewStringUTF(hello.c_str());
}

#endif