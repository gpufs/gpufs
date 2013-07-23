#/* 
#* This expermental software is provided AS IS. 
#* Feel free to use/modify/distribute, 
#* If used, please retain this disclaimer and cite 
#* "GPUfs: Integrating a file system with GPUs", 
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/


# This Makefile is used to build the library: 
# Run: make libgpufs.a
# For debug version: make dbg=1 libgpufsd.a 
# This file is also included in the makefiles to build user applications
# See "workloads" directory
LIBGPUFSDIR?=.

GPUFSLIB=$(LIBGPUFSDIR)/../gpufs_con_lib.user/
LIBS=
CUDA_ROOT:=/usr/local/cuda-5.0/
CUDA_LIBS=$(GPUFSLIB)/gpufs_con_lib.o -L$(CUDA_ROOT)/lib64 -lcudart 
CUDA_INCLUDE=-I$(CUDA_ROOT)/include -I$(CUDA_SDK_ROOT)/C/common/inc -I$(GPUFSLIB) -I$(LIBGPUFSDIR)


ifneq ($(dbg),1)
	MACROS=-DDEBUG_NOINLINE="" -DRELEASE=1 -O2 -DMALLOC_STATS
#	NVCC=$(CUDA_ROOT)/bin/nvcc  --gpu-architecture=sm_35 $(CUDA_INCLUDE) -Xptxas -v  $(MACROS)  -maxrregcount 32 
	NVCC=$(CUDA_ROOT)/bin/nvcc   --generate-code code=sm_21,arch=compute_20 --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE) -Xptxas -v  $(MACROS)  -maxrregcount 32 
	OUTPUTDIR=release
else
	MACROS=-DDEBUG_NOINLINE="__noinline__"   -DMALLOC_STATS --debug -O0 -g 
	NVCC=$(CUDA_ROOT)/bin/nvcc   --generate-code code=sm_21,arch=compute_20 --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE) -Xptxas -v  $(MACROS)  --device-debug
	OUTPUTDIR=debug
endif

CC=g++ $(CUDA_INCLUDE) -Wall -Wno-write-strings $(MACROS) 


SOURCES=fs.o 

GPUFS_SOURCES=$(OUTPUTDIR)/cpu_ipc.o\
$(OUTPUTDIR)/fs_calls.o\
 $(OUTPUTDIR)/fs_initializer.o\
 $(OUTPUTDIR)/fs_structures.o\
$(OUTPUTDIR)/hash_table.o\
$(OUTPUTDIR)/mallocfree.o\
$(OUTPUTDIR)/preclose_table.o\
$(OUTPUTDIR)/radix_tree.o\
$(OUTPUTDIR)/swapper.o\
$(OUTPUTDIR)/timer.o\
$(OUTPUTDIR)/fs_debug.o\
$(OUTPUTDIR)/async_ipc.o\
$(OUTPUTDIR)/generic_ringbuf.o\


$(OUTPUTDIR)/timer.o: timer.cpp timer.h
	$(CC) timer.cpp -c -o $(OUTPUTDIR)/timer.o


timer.o: timer.cpp timer.h
	$(CC) timer.cpp -c

ifneq ($(dbg),1)
fs: $(SOURCES) 
	$(NVCC) -link -o fs $(SOURCES)  -L$(LIBGPUFSDIR)  $(CUDA_LIBS) $(CUDA_INCLUDE) -lgpufs
else
fs.debug: $(SOURCES)
	$(NVCC) -link -o fs.debug $(SOURCES)  -L$(LIBGPUFSDIR)   $(CUDA_LIBS) $(CUDA_INCLUDE) -lgpufsd
endif
	


ifneq ($(dbg),1)
libgpufs.a : $(GPUFS_SOURCES)
	ar ru libgpufs.a $(GPUFS_SOURCES)
else
libgpufsd.a: $(GPUFS_SOURCES)
	ar ru libgpufsd.a $(GPUFS_SOURCES)
endif

fs.o: fs.cu  $(PAYLOAD)
	$(NVCC) -dc fs.cu -I$(LIBGPUFSDIR)

$(OUTPUTDIR)/cpu_ipc.o: cpu_ipc.cu fs_constants.h \
    util.cu.h  fs_debug.cu.h fs_structures.cu.h radix_tree.cu.h cpu_ipc.cu.h
	$(NVCC) -dc cpu_ipc.cu -o $(OUTPUTDIR)/cpu_ipc.o

$(OUTPUTDIR)/fs_calls.o: fs_calls.cu fs_debug.cu.h fs_constants.h  util.cu.h  cpu_ipc.cu.h \
    fs_structures.cu.h  mallocfree.cu.h  timer.h   hash_table.cu.h
	$(NVCC) -dc fs_calls.cu -o $(OUTPUTDIR)/fs_calls.o


$(OUTPUTDIR)/fs_initializer.o: fs_initializer.cu fs_debug.cu.h fs_constants.h  util.cu.h  cpu_ipc.cu.h \
    fs_structures.cu.h  mallocfree.cu.h  timer.h   hash_table.cu.h \
	radix_tree.cu.h preclose_table.cu.h fs_initializer.cu.h async_ipc.cu.h generic_ringbuf.cu.h
	$(NVCC) -dc fs_initializer.cu -o $(OUTPUTDIR)/fs_initializer.o

$(OUTPUTDIR)/hash_table.o: hash_table.cu hash_table.cu.h fs_constants.h util.cu.h fs_debug.cu.h radix_tree.cu.h fs_structures.cu.h
	$(NVCC) -dc hash_table.cu -o $(OUTPUTDIR)/hash_table.o

$(OUTPUTDIR)/mallocfree.o: mallocfree.cu fs_constants.h util.cu.h mallocfree.cu.h fs_structures.cu.h swapper.cu.h
	$(NVCC) -dc mallocfree.cu -o $(OUTPUTDIR)/mallocfree.o

$(OUTPUTDIR)/preclose_table.o: preclose_table.cu.h preclose_table.cu util.cu.h fs_debug.cu.h fs_constants.h
	$(NVCC) -dc preclose_table.cu -o $(OUTPUTDIR)/preclose_table.o

$(OUTPUTDIR)/radix_tree.o: radix_tree.cu radix_tree.cu.h fs_constants.h swapper.cu.h util.cu.h fs_debug.cu.h fs_structures.cu.h 
	$(NVCC) -dc radix_tree.cu -o $(OUTPUTDIR)/radix_tree.o

$(OUTPUTDIR)/swapper.o: swapper.cu.h swapper.cu fs_structures.cu.h fs_constants.h hash_table.cu.h util.cu.h fs_debug.cu.h radix_tree.cu.h\
 async_ipc.cu.h generic_ringbuf.cu.h
	$(NVCC)	-dc swapper.cu -o $(OUTPUTDIR)/swapper.o

$(OUTPUTDIR)/fs_structures.o: swapper.cu.h swapper.cu fs_structures.cu.h fs_constants.h hash_table.cu.h util.cu.h fs_debug.cu.h radix_tree.cu.h
	$(NVCC)	-dc fs_structures.cu -o $(OUTPUTDIR)/fs_structures.o

$(OUTPUTDIR)/fs_debug.o: fs_debug.cu fs_debug.cu.h
	$(NVCC) -dc fs_debug.cu -o $(OUTPUTDIR)/fs_debug.o

$(OUTPUTDIR)/async_ipc.o: util.cu.h fs_debug.cu.h generic_ringbuf.cu.h async_ipc.cu.h async_ipc.cu fs_constants.h
	$(NVCC) -dc async_ipc.cu -o $(OUTPUTDIR)/async_ipc.o

$(OUTPUTDIR)/generic_ringbuf.o: util.cu.h generic_ringbuf.cu.h generic_ringbuf.cu fs_constants.h
	$(NVCC) -dc generic_ringbuf.cu -o $(OUTPUTDIR)/generic_ringbuf.o

PHONY:  clean

clean:
	rm -f fs fs.debug $(OUTPUTDIR)/*.o *.o *.a

