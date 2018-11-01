CUDA_DIR=/usr/local/cuda
LIBCUDA=-L$(CUDA_DIR)/lib64 -lcudart -lcudadevrt
CUDAINC=-I$(CUDA_DIR)/include
CC=/usr/bin/g++
CCLINK = $(CC)
GPUFSDIR=../../libgpufs
FLAGS?=-DWARP -DGPUFS_VM

CXXFLAGS = -std=c++0x
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -ltbb
GPUFSINC = $(GPUFSDIR)
#TBBINC = $(GPUFSDIR)/tbb/include

OBJS = utils.o loader.o mosaic_gpu.o mosaic_gpufs.o mosaic_cpu.o main.o

MAIN = mosaic LSH preprocessor

#ifneq ($(dbg),1)
#	MACROS=-DDEBUG_NOINLINE="" -DTIMING_STATS 
#	CUFLAGS = --std=c++11 -O3 -rdc=true -maxrregcount=64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37
#	LINKFLAGS = -O3 -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37
#	NVCC=$(CUDA_DIR)/bin/nvcc $(CUDA_INCLUDE) $(MACROS)
#	GPUFSLIBS = -lgpufs -L$(GPUFSDIR)/lib
#else
#	MACROS=-DDEBUG_NOINLINE="__noinline__" -DMALLOC_STATS -DTIMING_STATS -DDEBUG
#	CUFLAGS = --std=c++11 -O0 -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37
#	LINKFLAGS = -O0 -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37
#	NVCC=$(CUDA_DIR)/bin/nvcc $(CUDA_INCLUDE) $(MACROS)
#	GPUFSLIBS = -lgpufsd -L$(GPUFSDIR)/lib
#endif

all: $(MAIN) 

include $(GPUFSDIR)/Makefile

mosaic: $(OBJS)
	$(NVCC) --relocatable-device-code=true $(LINKFLAGS) $(GPUFSLIBS) $(LDLIBS) $(LIBCUDA) $(GPUFSDIR)/release/libgpufs.a $^ -o $@ `pkg-config opencv --cflags --libs`

preprocessor: preprocessor.cpp
	$(CC) $(CXXFLAGS) $^ -o $@ 

LSH: offline.o LSH.o Random.o
	$(CCLINK) $^ -o $@

utils.o: utils.cpp utils.h
	$(CC) $(FLAGS) $(CXXFLAGS) -mavx -c $<

loader.o: loader.cpp loader.h utils.h
	$(CC) $(FLAGS) $(CXXFLAGS) -c $<

mosaic_cpu.o: mosaic_cpu.cpp loader.h utils.h
	$(CC) $(FLAGS) $(CXXFLAGS) -c $<

mosaic_gpu.o: mosaic_gpu.cu loader.h utils.h
	$(NVCC) $(CUFLAGS) -c $<

mosaic_gpufs.o: mosaic_gpufs.cu loader.h utils.h $(GPUFSINC)/host_loop.h
	$(NVCC) --relocatable-device-code=true  $(FLAGS) $(CUFLAGS) -I$(GPUFSINC) -I../../gpufs_con_lib.user/ $(GPUFSDIR)/release/libgpufs.a -c $<

main.o: main.cpp loader.h utils.h
	$(CC) $(FLAGS) $(CXXFLAGS) $(CUDAINC) -c $<

LSH.o: LSH.cpp LSH.h Random.h
	$(CC) $(CXXFLAGS) -c $<

Random.o: Random.cpp Random.h
	$(CC) $(CXXFLAGS) -c $<

offline.o: offline.cpp LSH.h Random.h
	$(CC) $(CXXFLAGS) -c $<


clean:
	\rm -f $(MAIN) *.o *~
