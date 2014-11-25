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
INSTALLDIR?=..

GPUFSLIB=$(LIBGPUFSDIR)/../gpufs_con_lib.user/
CUDA_ROOT:=/usr/local/cuda/
CUDA_INCLUDE=-I$(CUDA_ROOT)/include -I$(GPUFSLIB) -I$(LIBGPUFSDIR)

ifneq ($(dbg),1)
	MACROS=-DDEBUG_NOINLINE="" -DRELEASE=1 -O2
	NVCC=$(CUDA_ROOT)/bin/nvcc --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE) $(MACROS) -maxrregcount 32 
	OUTPUTDIR=release
else
	MACROS=-DDEBUG_NOINLINE="__noinline__" -DDEBUG  -DMALLOC_STATS --debug -O0 -g 
	NVCC=$(CUDA_ROOT)/bin/nvcc --generate-code code=sm_35,arch=compute_35 $(CUDA_INCLUDE) $(MACROS) --device-debug
	OUTPUTDIR=debug
endif

CC=g++ $(CUDA_INCLUDE) -Wall -Wno-write-strings $(MACROS) 

GPUFS_SOURCES=\
$(OUTPUTDIR)/cpu_ipc.o\
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
$(OUTPUTDIR)/gpufs_con_lib.o\

ifneq ($(dbg),1)
$(OUTPUTDIR)/libgpufs.a : $(GPUFS_SOURCES) | $(OUTPUTDIR)
	ar ru $(OUTPUTDIR)/libgpufs.a $(GPUFS_SOURCES)
else
$(OUTPUTDIR)/libgpufsd.a: $(GPUFS_SOURCES) | $(OUTPUTDIR)
	ar ru $(OUTPUTDIR)/libgpufsd.a $(GPUFS_SOURCES)
endif

$(OUTPUTDIR)/timer.o: timer.cpp timer.h | $(OUTPUTDIR)
	$(CC) timer.cpp -c -o $(OUTPUTDIR)/timer.o

$(OUTPUTDIR)/cpu_ipc.o: cpu_ipc.cu fs_constants.h \
	util.cu.h  fs_debug.cu.h fs_structures.cu.h radix_tree.cu.h cpu_ipc.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc cpu_ipc.cu -o $(OUTPUTDIR)/cpu_ipc.o

$(OUTPUTDIR)/fs_calls.o: fs_calls.cu fs_debug.cu.h fs_constants.h  util.cu.h  cpu_ipc.cu.h fs_structures.cu.h  mallocfree.cu.h \
	timer.h   hash_table.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc fs_calls.cu -o $(OUTPUTDIR)/fs_calls.o

$(OUTPUTDIR)/fs_initializer.o: fs_initializer.cu fs_debug.cu.h fs_constants.h  util.cu.h  cpu_ipc.cu.h fs_structures.cu.h \
	mallocfree.cu.h  timer.h   hash_table.cu.h radix_tree.cu.h preclose_table.cu.h fs_initializer.cu.h async_ipc.cu.h \
	generic_ringbuf.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc fs_initializer.cu -o $(OUTPUTDIR)/fs_initializer.o

$(OUTPUTDIR)/hash_table.o: hash_table.cu hash_table.cu.h fs_constants.h util.cu.h fs_debug.cu.h radix_tree.cu.h \
	fs_structures.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc hash_table.cu -o $(OUTPUTDIR)/hash_table.o

$(OUTPUTDIR)/mallocfree.o: mallocfree.cu fs_constants.h util.cu.h mallocfree.cu.h fs_structures.cu.h swapper.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc mallocfree.cu -o $(OUTPUTDIR)/mallocfree.o

$(OUTPUTDIR)/preclose_table.o: preclose_table.cu.h preclose_table.cu util.cu.h fs_debug.cu.h fs_constants.h | $(OUTPUTDIR)
	$(NVCC) -dc preclose_table.cu -o $(OUTPUTDIR)/preclose_table.o

$(OUTPUTDIR)/radix_tree.o: radix_tree.cu radix_tree.cu.h fs_constants.h swapper.cu.h util.cu.h fs_debug.cu.h \
	fs_structures.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc radix_tree.cu -o $(OUTPUTDIR)/radix_tree.o

$(OUTPUTDIR)/swapper.o: swapper.cu.h swapper.cu fs_structures.cu.h fs_constants.h hash_table.cu.h util.cu.h \
	fs_debug.cu.h radix_tree.cu.h async_ipc.cu.h generic_ringbuf.cu.h | $(OUTPUTDIR)
	$(NVCC)	-dc swapper.cu -o $(OUTPUTDIR)/swapper.o

$(OUTPUTDIR)/fs_structures.o: swapper.cu.h swapper.cu fs_structures.cu.h fs_constants.h hash_table.cu.h util.cu.h \
	fs_debug.cu.h radix_tree.cu.h | $(OUTPUTDIR)
	$(NVCC)	-dc fs_structures.cu -o $(OUTPUTDIR)/fs_structures.o

$(OUTPUTDIR)/fs_debug.o: fs_debug.cu fs_debug.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc fs_debug.cu -o $(OUTPUTDIR)/fs_debug.o

$(OUTPUTDIR)/async_ipc.o: util.cu.h fs_debug.cu.h generic_ringbuf.cu.h async_ipc.cu.h async_ipc.cu fs_constants.h | $(OUTPUTDIR)
	$(NVCC) -dc async_ipc.cu -o $(OUTPUTDIR)/async_ipc.o

$(OUTPUTDIR)/generic_ringbuf.o: util.cu.h generic_ringbuf.cu.h generic_ringbuf.cu fs_constants.h | $(OUTPUTDIR)
	$(NVCC) -dc generic_ringbuf.cu -o $(OUTPUTDIR)/generic_ringbuf.o

$(OUTPUTDIR)/gpufs_con_lib.o: $(GPUFSLIB)/gpufs_con_lib.cpp $(GPUFSLIB)/gpufs_con_lib.h | $(OUTPUTDIR)
	$(CC) $(GPUFSLIB)/gpufs_con_lib.cpp -c -o $(OUTPUTDIR)/gpufs_con_lib.o

$(OUTPUTDIR):
	mkdir -p $(OUTPUTDIR)

install: $(OUTPUTDIR)/libgpufs.a
	mkdir -p $(INSTALLDIR)/lib
	mkdir -p $(INSTALLDIR)/include
	install -m 0755 $(OUTPUTDIR)/libgpufs.a $(INSTALLDIR)/lib
	install -m 0644 *.h $(GPUFSLIB)/*.h $(INSTALLDIR)/include


PHONY:  clean

clean:
	rm -rf $(OUTPUTDIR) *.o *.a

