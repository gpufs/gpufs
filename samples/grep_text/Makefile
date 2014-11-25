#/* 
#* This expermental software is provided AS IS. 
#* Feel free to use/modify/distribute, 
#* If used, please retain this disclaimer and cite 
#* "GPUfs: Integrating a file system with GPUs", 
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/



CXXFLAGS=-fopenmp -g -O2
grep_text_cpu.o: grep_text_cpu.cpp 

grep_text_cpu: grep_text_cpu.o
	$(CXX) -o grep_text_cpu grep_text_cpu.o -O2 -fopenmp 


LIBGPUFSDIR=../../libgpufs/

include $(LIBGPUFSDIR)/Makefile

grep_text_gpu_nofs: grep_text_gpu_nofs.cu /home/marks/home/gpufs/workloads/libgpufs/libgpufs.a
	$(NVCC) -o grep_text_gpu_nofs grep_text_gpu_nofs.cu  -I/home/marks/home/gpufs/workloads/libgpufs/

#grep_text_gpu.o: grep_text_gpu.cu
#	$(NVCC) -dc grep_text_gpu.cu -I/home/marks/home/gpufs/workloads/libgpufs/

#gpu: grep_text.cu
#	make -C ../../gpu debug


