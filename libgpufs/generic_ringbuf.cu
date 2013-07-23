/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include "generic_ringbuf.cu.h"
#include <stdlib.h>
#include <stdio.h>
#include "util.cu.h"

__host__ void ringbuf_metadata_init(ringbuf_metadata_t** rb_cpu, ringbuf_metadata_t** rb_gpu, int num_elem)
{
	ringbuf_metadata_t* rbm=(ringbuf_metadata_t*)malloc(sizeof(ringbuf_metadata_t)); // metadata in CPU
	rbm->_size=num_elem;
	rbm->_head=rbm->_tail=0;
	(*rb_cpu)=rbm;
	// metadata in CPU shared with GPU
	CUDA_SAFE_CALL(cudaHostRegister(rbm,sizeof(ringbuf_metadata_t),cudaHostRegisterMapped));
        CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)rb_gpu,(void*)rbm,0));
}

__host__ void ringbuf_metadata_free(ringbuf_metadata_t* rb_cpu)
{
	CUDA_SAFE_CALL(cudaHostUnregister(rb_cpu));
	free(rb_cpu);
}
