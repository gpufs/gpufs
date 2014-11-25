#/* 
#* This expermental software is provided AS IS. 
#* Feel free to use/modify/distribute, 
#* If used, please retain this disclaimer and cite 
#* "GPUfs: Integrating a file system with GPUs", 
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/



CXXFLAGS=-fopenmp -g -O2

LIBGPUFSDIR=../../libgpufs/

PAYLOAD=bigmatrix_mmap.cu
include $(LIBGPUFSDIR)/Makefile

bigmatrix_cpu.o: bigmatrix_cpu.cpp 

bigmatrix_cpu: bigmatrix_cpu.o
	$(CXX) -o bigmatrix_cpu bigmatrix_cpu.o -O2 -fopenmp 

bigmatrix_nofs: bigmatrix_nofs.cu
	$(NVCC) -o bigmatrix_nofs bigmatrix_nofs.cu -I$(LIBGPUFSDIR)

bigmatrix_nofs_naive:bigmatrix_nofs_naive.cu
	$(NVCC) -o bigmatrix_nofs_naive bigmatrix_nofs_naive.cu -I$(LIBGPUFSDIR)



