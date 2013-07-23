CXXFLAGS=-fopenmp -g -O2

LIBGPUFSDIR=../../libgpufs/

include $(LIBGPUFSDIR)/Makefile

PAYLOAD=img_gpu.cu

img_cpu.o: img_cpu.cpp 

img_cpu: img_cpu.o
	$(CXX) -o img_cpu img_cpu.o -O2 -fopenmp 




