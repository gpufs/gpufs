#/* 
#* This expermental software is provided AS IS. 
#* Feel free to use/modify/distribute, 
#* If used, please retain this disclaimer and cite 
#* "GPUfs: Integrating a file system with GPUs", 
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/

all: fs_gpu fs_gpu_nofs grep_text_cpu

CXXFLAGS=-fopenmp -g -O2
grep_text_cpu.o: grep_text_cpu.cpp 

grep_text_cpu: grep_text_cpu.o
	$(CXX) -o grep_text_cpu grep_text_cpu.o -O2 -fopenmp 


LIBGPUFSDIR=../../libgpufs/

include $(LIBGPUFSDIR)/Makefile

#grep_text_gpu_nofs: grep_text_gpu_nofs.cu 
#	$(NVCC) --relocatable-device-code=true  -o grep_text_gpu_nofs grep_text_gpu_nofs.cu  -I$(LIBGPUFSDIR) $(LIBGPUFSDIR)/release/libgpufs.a

fs_gpu: grep_text_gpu.o
	$(NVCC) --relocatable-device-code=true -o fs_gpu fs.cu -I$(LIBGPUFSDIR) -I../../gpufs_con_lib.user/ $(LIBGPUFSDIR)/release/libgpufs.a

fs_gpu_nofs: grep_text_gpu_nofs.o
	$(NVCC) --relocatable-device-code=true -o fs_gpu_nofs fs.cu -I$(LIBGPUFSDIR) -I../../gpufs_con_lib.user/ $(LIBGPUFSDIR)/release/libgpufs.a

#grep_text_gpu.o: grep_text_gpu.cu
#	$(NVCC) --relocatable-device-code=true -I/home/meni/tanya/gpufs/libgpufs/ /home/meni/tanya/gpufs/libgpufs/release/libgpufs.a grep_text_gpu.cu 

grep_text_gpu.o: grep_text_gpu.cu
	$(NVCC) -dc grep_text_gpu.cu -I$(LIBGPUFSDIR)

grep_text_gpu.o: grep_text_gpu_nofs.cu
	$(NVCC) -dc grep_text_gpu_nofs.cu -I$(LIBGPUFSDIR)

#grep_text_gpu.o: grep_text_gpu.cu
#	$(NVCC) -dc grep_text_gpu.cu -I/home/marks/home/gpufs/workloads/libgpufs/

#gpu: grep_text.cu
#	make -C ../../gpu debug

clean: 
	rm *.o fs_gpu fs_gpu_nofs grep_text_cpu



