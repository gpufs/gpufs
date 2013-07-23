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

include $(LIBGPUFSDIR)/Makefile

PAYLOAD=cp.cu

cp.o: cp.cpp 

cp_cpu: cp.o
	$(CXX) -o cp_cpu cp.o -O2




