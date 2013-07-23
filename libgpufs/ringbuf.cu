/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#ifndef ringbuf_gpumem
#define ringbuf_gpumem

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "util.cu.h"


struct ringbuf_metadata_t{
        volatile uint _head;
        volatile uint _tail;
        uint _size;
};


template<typename T>
struct ringbuf{
	volatile T *gpu_vals;
	ringbuf_metadata_t* cpu_meta;
	
};


template<typename T>
__host__ void rb_init_cpu(ringbuf<T>** rb_cpu, struct ringbuf<T>** rb_gpu,  uint num_elem)
{
	*rb_cpu=(ringbuf<T>*)malloc(sizeof(ringbuf<T>));

	CUDA_SAFE_CALL(cudaMalloc(&(*rb_cpu)->gpu_vals,(sizeof(T)*num_elem)));	// values are in GPU
	
	ringbuf_metadata_t* rbm=(ringbuf_metadata_t*)malloc(sizeof(ringbuf_metadata_t)); // metadata in CPU
	rbm->_size=num_elem;
	rbm->_head=rbm->_tail=0;
	(*rb_cpu)->cpu_meta=rbm;

/** host init complete **/
 
	ringbuf<T> rb_h_gpu; // initializer for GPU

	// metadata in CPU shared with GPU
	CUDA_SAFE_CALL(cudaHostRegister(rbm,sizeof(ringbuf_metdata_t),cudaHostRegisterMapped));
        CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)&rb_h_gpu.cpu_meta,(void*)rbm,0));
	
	rb_h_gpu.gpu_vals=(*rb_cpu)->gpu_vals; 

	CUDA_SAFE_CALL(rb_gpu,sizeof(ringbuf<T>)); // create GPU object
	// copy initalized rb_gpu to gpu memory 
	CUDA_SAFE_CALL(cudaMemcpy(*rb_gpu,&rb_h_gpu,sizeof(ringbuf<T>),cudaHostToDevice)); 
}

template<typename T>
__host__ void rb_free_cpu(ringbuf<T>* rb_cpu, ringbuf<T>* rb_gpu){
	CUDA_SAFE_CALL(cudaHostUnregister(rb_cpu->cpu_meta));
	CUDA_SAFE_CALL(cudaFree(rb_gpu));
	CUDA_SAFE_CALL(cudaFree(rb_cpu->gpu_vals));
	free(rb_cpu);
}
	

template<typename T>
__device__ __host__  bool rb_empty(struct ringbuf<T>* r){
        return (r->cpu_meta->_tail==r->cpu_meta->_head);
}
template<typename T>__device__ __host__ bool rb_full(struct ringbuf<T>* r){
        return ((r->cpu_meta->_head+1)%r->cpu_meta->_size)==r->cpu_meta->_tail;
}

template<typename T>
__host__ void memcpy_on_pop_cpu(T* cpu_val, volatile const T* gpu_val, cudaStream_t& s){
	CUDA_SAFE_CALL(cudaMemcpyAsync(cpu_val,gpu_val, sizeof(T),cudaMemcpyDeviceToHost,s));
	CUDA_SAFE_CALL(cudaStreamSynchronize(s));
	// this call must be synced otherwise the buffer cannot be used!
}


template<typename T>
__host__  bool rb_pop_cpu(struct ringbuf<T>* r, T* newval, cudaStream_t& s){
	 if (rb_empty(r)) return false;
	 memcpy_on_pop_cpu(newval,r->gpu_vals[r->_tail],s);
         r->cpu_meta->_tail=(r->cpu_meta->_tail+1)%r->cpu_meta_size;
         __sync_synchronize();
	 return true;
}

// inefficient implementation
template<typename T>
__device__ void memcpy_on_pop_gpu(T* dst, volatile const T* src)
{
#warning "memcpy_on_pop_gpu using inefficient implementation. Use specialization instead"
	memcpy_block(dst,src,sizeof(T));
}

template<typename T>
__device__  bool rb_pop_gpu(struct ringbuf<T>* r, T* newval){

	__shared__ bool is_empty;
	BEGIN_SINGLE_THREAD
	 is_empty=rb_empty(r);
	END_SINGLE_THREAD
         if(is_empty) return false;

	memcpy_on_pop_gpu(newval,r->gpu_vals[r->cpu_meta->_tail]);
         r->cpu_meta->_tail=(r->cpu_meta->_tail+1)%r->cpu_meta->_size;
         __threadfence_system();
	return true;
}


template<typename T>
void memcpy_on_push_cpu(volatile T* gpu_val, const T* cpu_val, cudaStream_t s)
{
	CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_val,cpu_val, sizeof(T),cudaMemcpyHostToDevice,s));
	CUDA_SAFE_CALL(cudaStreamSynchronize(s));
	// must be synchronized! 
}


template<typename T>
bool __host__ rb_push_cpu(struct ringbuf<T>* r, const T* val)
{

         if (rb_full(r)) return false;

         memcpy_on_push_cpu(r->vals[r->_head],val);

         __sync_synchronize();
         r->_head=(r->_head+1)%r->_size;
         __sync_synchronize();
        return true;
}

template<typename T>
__device__ void memcpy_on_push_gpu(volatile T* dst, const T* src){
#warning "memcpy_on_push_gpu using inefficient implementation. Use specialization instead"
	memcpy_block(dst,src,sizeof(T));
}

template<typename T>
bool __device__ rb_push_gpu(struct ringbuf<T>* r, const T* val){

	__shared__ bool is_full;
	BEGIN_SINGLE_THREAD
	is_full=rb_full(r);
	END_SINGLE_THREAD
	if (is_full) return false;

         memcpy_on_push_gpu(r->gpu_vals[r->cpu_meta->_head],val);
	__threadfence(); 
         r->cpu_meta->_head=(r->cpu_meta->_head+1)%r->cpu_meta->_size;
         __threadfence_system();
        return true;
}

#endif
