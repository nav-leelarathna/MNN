//
//  onnxOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"
#include "OpCount.hpp"
#include "OnnxTmpGraph.hpp"

using namespace MNN;
static int32_t _limit(int64_t i64) {
    if (i64 > (int64_t)(1 << 30)) {
        return 1 << 30;
    }
    if (i64 < (int64_t)(-(1 << 30))) {
        return (-(1 << 30));
    }
    return i64;
}
class DefaultonnxOpConverter : public onnxOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     OnnxScope* scope) override {
        auto extra        = new ExtraT;
        dstOp->main.type  = OpParameter_Extra;
        dstOp->main.value = extra;
        extra->engine     = "ONNX";
        extra->type       = onnxNode->op_type();
        for (auto srcAttr : onnxNode->attribute()) {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = srcAttr.name();
            switch (srcAttr.type()) {
                case onnx::AttributeProto_AttributeType_INTS:
                    attr->list.reset(new ListValueT);
                    attr->list->i.resize(srcAttr.ints_size());
                    for (int i = 0; i < srcAttr.ints_size(); ++i) {
                        attr->list->i[i] = _limit(srcAttr.ints(i));
                    }
                    break;
                case onnx::AttributeProto_AttributeType_FLOATS:
                    attr->list.reset(new ListValueT);
                    attr->list->f.resize(srcAttr.floats_size());
                    for (int i = 0; i < srcAttr.floats_size(); ++i) {
                        attr->list->f[i] = srcAttr.floats(i);
                    }
                    break;
                case onnx::AttributeProto_AttributeType_TENSOR:
                    attr->tensor.reset(convertTensorToBlob(&srcAttr.t()));
                    break;
                default:
                    break;
            }
            attr->i = _limit(srcAttr.i());
            attr->s = srcAttr.s();
            attr->f = srcAttr.f();
            extra->attr.emplace_back(std::move(attr));
        }
    }
    virtual MNN::OpParameter type() override {
        return OpParameter_Extra;
    }
    virtual MNN::OpType opType() override {
        return OpType_Extra;
    }
};

onnxOpConverterSuit::onnxOpConverterSuit() {
}

onnxOpConverterSuit::~onnxOpConverterSuit() {
    for (auto& iter : mConverterContainer) {
        delete iter.second;
    }
    mConverterContainer.clear();
}

onnxOpConverterSuit* onnxOpConverterSuit::global = nullptr;

onnxOpConverterSuit* onnxOpConverterSuit::get() {
    if (global == nullptr) {
        global = new onnxOpConverterSuit;
    }
    return global;
}

void onnxOpConverterSuit::insert(onnxOpConverter* t, const char* name) {
    MNN::OpCount::get()->insertOp("ONNX", std::string(name));
    mConverterContainer.insert(std::make_pair(name, t));
}

onnxOpConverter* onnxOpConverterSuit::search(const std::string& name) {
    auto iter = mConverterContainer.find(name);
    if (iter == mConverterContainer.end()) {
        static DefaultonnxOpConverter defaultConverter;
        return &defaultConverter;
    }
    return iter->second;
}
MNN::DataType onnxOpConverter::convertDataType(::onnx::TensorProto_DataType type) {
    static std::map<::onnx::TensorProto_DataType, MNN::DataType> dataTypeMap{
        {onnx::TensorProto_DataType_FLOAT, MNN::DataType_DT_FLOAT},
        {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_INT32, MNN::DataType_DT_INT32},
        {onnx::TensorProto_DataType_INT64, MNN::DataType_DT_INT32},  // For compability, use int32 instead of int64
        {onnx::TensorProto_DataType_DOUBLE, MNN::DataType_DT_FLOAT}, // For compability, use float instead of double
        {onnx::TensorProto_DataType_UINT8, MNN::DataType_DT_UINT8},
        {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_BOOL, MNN::DataType_DT_INT32},   // For compability, use int32 instead of bool
        {onnx::TensorProto_DataType_INT16, MNN::DataType_DT_INT32},  // For compability, use int32 instead of int16
        {onnx::TensorProto_DataType_UINT16, MNN::DataType_DT_INT32}, // For compability, use int32 instead of uint16
    };
    if (dataTypeMap.find(type) != dataTypeMap.end()) {
        return dataTypeMap[type];
    }
    return MNN::DataType_DT_INVALID;
}
MNN::BlobT* onnxOpConverter::convertTensorToBlob(const onnx::TensorProto* constantTp) {
    auto constantParam = new MNN::BlobT;
    auto dataType      = convertDataType(constantTp->data_type());
    // printf("origindataType = %d, dataType = %s\n", constantTp->data_type(), MNN::EnumNameDataType(dataType));

    constantParam->dataType   = dataType;
    constantParam->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;

    size_t dimSize = constantTp->dims().size();
    constantParam->dims.resize(dimSize);
    size_t dataSize = 1;
    for (int i = 0; i < dimSize; ++i) {
        constantParam->dims[i] = constantTp->dims(i);
        dataSize               = dataSize * constantTp->dims(i);
    }
    std::vector<int64_t> alignContent((constantTp->raw_data().size() + sizeof(int64_t) - 1) / sizeof(int64_t));
    ::memcpy(alignContent.data(), constantTp->raw_data().data(), constantTp->raw_data().size());

    const void* tensor_content = (const void*)alignContent.data();

    switch (constantTp->data_type()) {
#define CASE_DATA_TYPE(src, dst)                              \
    case src:                                                 \
        if (constantTp->dst##_data_size() != 0) {             \
            tensor_content = constantTp->dst##_data().data(); \
        }                                                     \
        break;
        CASE_DATA_TYPE(onnx::TensorProto_DataType_DOUBLE, double);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_INT64, int64);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_INT32, int32);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_FLOAT, float);
        default:
            break;
    }
    if (0 == dataSize) {
        // Empty blob
        return constantParam;
    }

    if (!tensor_content) {
        DLOG(FATAL) << "Convert no data, "
                       "Please make sure ";
    }

    switch (constantTp->data_type()) {
        case onnx::TensorProto_DataType_DOUBLE: {
            constantParam->float32s.resize(dataSize);
            auto source = (double*)tensor_content;

            for (int i = 0; i < dataSize; ++i) {
                constantParam->float32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT64: {
            constantParam->int32s.resize(dataSize);
            auto source = (int64_t*)tensor_content;

            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = _limit(source[i]);
            }
            break;
        }
        case onnx::TensorProto_DataType_INT32: {
            auto source = (int32_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT16: {
            auto source = (uint16_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT16: {
            auto source = (int16_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_BOOL: {
            auto source = (bool*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT8: {
            auto source = (int8_t*)tensor_content;
            constantParam->int8s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int8s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT8: {
            auto source = (uint8_t*)tensor_content;
            constantParam->uint8s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->uint8s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_FLOAT: {
            float* tempFloatData = (float*)tensor_content;
            constantParam->float32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->float32s[i] = tempFloatData[i];
            }
            break;
        }
        default: {
            DLOG(FATAL) << "Don't support " << constantTp->data_type();
            break;
        }
    }
    return constantParam;
}

void OnnxScope::onnxInit() {
    const int initializerCount = mGraph->initializer_size();
    for (int i = 0; i < initializerCount; ++i) {
        const auto& initializer = mGraph->initializer(i);
        mInitializers.insert(std::make_pair(initializer.name(), &initializer));
    }
    const int inputCount = mGraph->input_size();
    for (int i = 0; i < inputCount; ++i) {
        const auto& input = mGraph->input(i);
        mInputs.insert(std::make_pair(input.name(), &input));
    }
    const int outputCount = mGraph->output_size();
    for (int i = 0; i < outputCount; ++i) {
        const auto& output = mGraph->output(i);
        mOutputs.insert(std::make_pair(output.name(), &output));
    }
}

int OnnxScope::lookupTensor(std::string name) {
    // onnx have optional input, which may be a placeholder when pytorch export onnx model,
    // so drop this input, but we should check it out sometimes.
    if(name == ""){
        return -1;
    }
    const auto iter = mTensorIdx.find(name);
    if (iter != mTensorIdx.end()) {
        return iter->second;
    }
    return -1;
}

std::pair<int, int> OnnxScope::buildTensorArrayOp(std::vector<int> element_shape, bool identical, const std::string& name) {
    std::unique_ptr<MNN::OpT> tensorArrayOp(new MNN::OpT);
    tensorArrayOp->name      = name;
    tensorArrayOp->type      = MNN::OpType_TensorArray;
    tensorArrayOp->main.type = MNN::OpParameter_TensorArray;
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = DataType_DT_FLOAT;
    tensorArray->dynamic_size = true;
    tensorArray->identical_element_shapes = identical;
    tensorArray->element_shape = element_shape;
    tensorArrayOp->main.value = tensorArray;
    tensorArrayOp->inputIndexes.push_back(buildIntConstOp({1}, name + "/init_size"));
    int idx_handle = declareTensor(name + "/handle");
    int idx = declareTensor(name);
    tensorArrayOp->outputIndexes.push_back(idx_handle);
    tensorArrayOp->outputIndexes.push_back(idx);
    oplists().emplace_back(std::move(tensorArrayOp));
    return std::make_pair(idx_handle, idx);
}

void OnnxScope::buildAccumulate(const std::string& name, const std::string& uName, const std::string& iName, const std::string& oName) {
    // for while_body: %user_defined_val = Add(%user_defined_val, %output)
    int idxAcc = declareTensor(name + "/accumulate_u");
    MNN::OpT* accumulateOp  = new MNN::OpT;
    accumulateOp->name      = name + "/accumulate";
    accumulateOp->type      = MNN::OpType_TensorArrayWrite;
    accumulateOp->main.type = MNN::OpParameter_TensorArray;
    auto param  = new MNN::TensorArrayT;
    param->T = MNN::DataType_DT_FLOAT;
    accumulateOp->main.value = param;
    // handle, index, value, flow_in
    addInputForOp(accumulateOp, uName + "/handle");
    addInputForOp(accumulateOp, iName);
    addInputForOp(accumulateOp, oName);
    addInputForOp(accumulateOp, uName);
    accumulateOp->outputIndexes.push_back(idxAcc);
    oplists().emplace_back(accumulateOp);
    mSubNet->outputs.push_back(idxAcc);
}

void OnnxScope::buildSubGraph(const onnx::GraphProto* graph, std::string& name, std::string& uName, bool increment) {
    std::unique_ptr<MNN::SubGraphProtoT> subgraph(new MNN::SubGraphProtoT);
    subgraph->name = name;
    std::unique_ptr<OnnxScope> scope(new OnnxScope(graph, subgraph.get(), mNet, this));
    const auto& initializers = scope->mInitializers;
    for (int i = 0; i < graph->node_size(); i++) {
        const auto& onnxNode = graph->node(i);
        const auto& opType   = onnxNode.op_type();
        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);
        auto opConverter = onnxOpConverterSuit::get()->search(opType);
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            const auto it         = initializers.find(inputName);
            if (it != initializers.end() && scope->lookupTensor(it->first) == -1) {
                // Create const Op
                MNN::OpT* constOp   = new MNN::OpT;
                constOp->type       = MNN::OpType_Const;
                constOp->main.type  = MNN::OpParameter_Blob;
                constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second);
                constOp->name    = it->first;
                constOp->outputIndexes.push_back(scope->declareTensor(it->first));
                subgraph->nodes.emplace_back(constOp);
            }
        }
        // build input and output
        for (int k = 0; k < onnxNode.input_size(); k++) {
            scope->addInputForOp(MNNOp, onnxNode.input(k));
        }
        for (int k = 0; k < onnxNode.output_size(); k++) {
            MNNOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
        }
        opConverter->run(MNNOp, &onnxNode, scope.get());
        subgraph->nodes.emplace_back(MNNOp);
    }
    scope->dealSubgraphDeps();
    for (int i = 0; i < graph->output_size(); i++) {
        int idx = scope->lookupTensor(graph->output(i).name());
        if (idx >= 0) {
            subgraph->outputs.push_back(idx);
        }
    }
    if (!uName.empty()) {
        // %user_defined_val = Add(%user_defined_val, %output)
        scope->buildAccumulate(name, uName, graph->input(0).name(), graph->output(1).name());
    }
    if (increment) {
        scope->buildIncrement(name, graph->input(0).name());
    }
    mNet->subgraphs.emplace_back(std::move(subgraph));
}
