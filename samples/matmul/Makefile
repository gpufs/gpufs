CXXFLAGS=-fopenmp -g

LIBGPUFSDIR=../../libgpufs/

PAYLOAD=matrixMul_kernel.cu
include $(LIBGPUFSDIR)/Makefile


CUDA_LIBS+=-L$(CUDA_ROOT)/lib64 -lcudart -L$(CUDA_SDK_ROOT)/C/lib/ -lcutil_x86_64  -L$(CUDA_SDK_ROOT)/shared/lib -lshrutil_x86_64

CUDA_INCLUDE+=-I$(CUDA_ROOT)/include -I$(CUDA_SDK_ROOT)/C/common/inc   -I$(CUDA_SDK_ROOT)/shared/inc  -I$(LIBGPUFSDIR) -I../../gpufs_con_lib.user


mm: matrixMul.o 
	$(NVCC) -link -o mm matrixMul.o  -L$(LIBGPUFSDIR) -lgpufs  $(CUDA_LIBS)  -g

mm.debug: matrixMul.o matrixMul_debug
	$(NVCC) -link -o mm matrixMul.o  -L$(LIBGPUFSDIR) -lgpufsd  $(CUDA_LIBS) -g --device-debug

matrixMul.o: matrixMul.cu matrixMul_kernel.cu
	$(NVCC) -dc matrixMul.cu -I/usr/local/cuda-5.0/samples/common/inc/ -g

matrixMul_debug: matrixMul.cu matrixMul_kernel.cu
	/usr/local/cuda-5.0//bin/nvcc  --gpu-architecture=sm_21 -I/usr/local/cuda-5.0//include -I$(CUDA_SDK_ROOT)/C/common/inc -I../../gpufs_con_lib.user  -Xptxas -v  -DDEBUG_NOINLINE="__noinline__"   -DMALLOC_STATS --debug -O0 -g   --device-debug -dc matrixMul.cu -I$(CUDA_ROOT)/include -I$(CUDA_SDK_ROOT)/C/common/inc   -I$(CUDA_SDK_ROOT)/shared/inc -lcublas -I$(LIBGPUFSDIR)
