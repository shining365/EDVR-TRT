CUDA_PATH = /usr/local/cuda
TRT_PATH = /usr/local/tensorrt7
GENCODE = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
BUILD_DIR = build
TORCH_PATH = /home/gji/.local/lib/python3.6/site-packages/torch

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ 
INCLUDES := -I$(CUDA_PATH)/include -isystem $(TRT_PATH)/include -isystem /usr/include/python3.6m/ -isystem $(TORCH_PATH)/include -isystem $(TORCH_PATH)/include/torch/csrc/api/include
LDFLAGS := -L/usr/local/cuda/lib64 -L$(TORCH_PATH)/lib/ -L$(TRT_PATH)/lib
LDFLAGS += -lcudart -lc10 -ltorch -ltorch_cpu -ltorch_python -lc10_cuda -ltorch_cuda -lnvinfer

OBJ = $(shell find $(BUILD_DIR) -name *.o 2>/dev/null)
DEP = $(OBJ:.o=.d)

BIN = 
SO = DeformConvPlugin.so

all: $(BIN) $(SO)

DeformConvPlugin.so: $(addprefix $(BUILD_DIR)/, DeformConvPlugin.o DeformConv.o)

-include $(DEP)

clean:
	rm -rf $(BIN) $(SO) $(OBJ) $(DEP)

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -MMD $(GENCODE) -o $@ -c $<

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(CCFLAGS) $(INCLUDES) -fPIC -MD -MP -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -fPIC -shared -o $@ $+ $(LDFLAGS)

$(BIN):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)
