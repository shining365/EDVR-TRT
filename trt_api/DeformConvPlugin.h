#include <NvInfer.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include <assert.h>

using namespace std;

inline std::string to_string(nvinfer1::Dims const &dim) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < dim.nbDims; i++) {
        oss << dim.d[i] << ", ";
    }
    oss << ")";
    return oss.str();
}

class DeformConvPlugin: public nvinfer1::IPluginV2IOExt {
public:
    DeformConvPlugin() {}
    
    DeformConvPlugin(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }
    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }
    
    nvinfer1::IPluginV2IOExt* clone() const override {
        return new DeformConvPlugin(&m, sizeof(m));
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    int getNbOutputs() const override {
        return 1;
    }
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* pInputDim, int nInputDim) override {
        return pInputDim[0];
    }
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
    	return nvinfer1::DataType::kFLOAT;
    }

    virtual void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput) override {
    	m.inputDim = in[0].dims;
        cout << "configurePlugin type=" << (int)out[0].type << ", inputDim=" << to_string(m.inputDim) << endl;
    }

    size_t getWorkspaceSize(int nMaxBatchSize) const override {return 0;}
    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "DeformConvPlugin";}
    const char* getPluginVersion() const override {return "0";}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {return false;}
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) {}
    void detachFromContext() {}

private:
    using nvinfer1::IPluginV2Ext::configurePlugin;
    
    struct {
        nvinfer1::Dims inputDim;
    } m;
};

class DeformConvPluginCreator : public nvinfer1::IPluginCreator {
public:
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new DeformConvPlugin(serialData, serialLength);
    }
    
    const char* getPluginName() const override {return "DeformConvPlugin";}
    const char* getPluginVersion() const override {return "0";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return nullptr;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        std::cout << __FUNCTION__ << std::endl;
        return new DeformConvPlugin();
    }
};
