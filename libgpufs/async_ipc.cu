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
#include <stdio.h>

#include "fs_constants.h"
#include "fs_globals.cu.h"
#include "util.cu.h"
#include "generic_ringbuf.cu.h"
#include "async_ipc.cu.h"
#include <stdlib.h>

__device__ int gpu_rb_lock;

__host__ void ringbuf_page_md_init(page_md_t** cpumd, page_md_t** gpumd, int num_elem){
		*cpumd=(page_md_t*)malloc(sizeof(page_md_t)*ASYNC_CLOSE_RINGBUF_SIZE);
		CUDA_SAFE_CALL(cudaHostRegister(*cpumd,sizeof(page_md_t)*ASYNC_CLOSE_RINGBUF_SIZE,cudaHostRegisterMapped));
		CUDA_SAFE_CALL(cudaHostGetDevicePointer(gpumd,*cpumd,0));
}


	__host__ void async_close_rb_t::init_host()
	{
		ringbuf_page_md_init(&cpu_md_ar, &gpu_md_ar_ptr, ASYNC_CLOSE_RINGBUF_SIZE);
		ringbuf_metadata_init(&cpu_rb,&gpu_rb_ptr, ASYNC_CLOSE_RINGBUF_SIZE);
		CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_data_ar,sizeof(Page)*ASYNC_CLOSE_RINGBUF_SIZE));

		int zero = 0;
		cudaMemcpyToSymbol(gpu_rb_lock, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
	}

__forceinline__ __device__ void async_close_rb_t::memcpy_page_req(volatile Page* dst, const volatile Page* src, size_t size){	

	typedef volatile double cpy_type;
        size_t actual_size=size/sizeof(cpy_type);

	int tid=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	int stride=blockDim.x*blockDim.y*blockDim.z;

        for(int i=tid;i<actual_size;i+=stride){

                ((cpy_type*)dst)[i]=((cpy_type*)src)[i];
        }
	if (tid==0){
                for(int i= (actual_size*sizeof(cpy_type)); i<size;i++){
                        ((volatile char*)dst)[i]=((volatile char*)src)[i];
                }
        }
}
/*
	__forceinline__ __device__ void async_close_rb_t::memcpy_page_req(volatile Page* dst, const volatile Page* src)
	{

#define BATCH_TYPE double2
#define BATCH_SIZE 16 // 16 bytes per single fetch -- double2
#define LOG_BATCH_SIZE 4 // log 16
#define LOOP_FETCHES 4 // log

	
		GPU_ASSERT(sizeof(Page)&(BATCH_SIZE-1)==0);
		// we must be able to copy in one go
		size_t size=sizeof(Page)>>(LOG_BATCH_SIZE);
	        
		int stride=LOOP_FETCHES*blockDim.x*blockDim.y*blockDim.z;
	
	        for(int i=LOOP_FETCHES*(TID);i<size;i+=stride){
	
	                int j=i+1;
	                int k=i+2;
	                int z=i+3;
	                ((BATCH_TYPE*)dst)[i]=((BATCH_TYPE*)src)[i];
	                if (j<size)  ((BATCH_TYPE*)dst)[j]=((BATCH_TYPE*)src)[j];
	                if (k<size)  ((BATCH_TYPE*)dst)[k]=((BATCH_TYPE*)src)[k];
	                if (z<size)  ((BATCH_TYPE*)dst)[z]=((BATCH_TYPE*)src)[z];
	        }
	}
*/

	// no lock on the queue is necessary because this is done on close and global ftable lock is taken
	// this call is blocking on the CPU queue
	void __device__ async_close_rb_t::enqueue(int cpu_fd, size_t page_id, size_t file_offset, uint content_size, int type ){

		__shared__ uint head;
        
		BEGIN_SINGLE_THREAD
			MUTEX_LOCK(gpu_rb_lock);
       			while(gpu_rb_ptr->rb_full()); // lock up forever
			head=gpu_rb_ptr->rb_producer_ptr();
        	END_SINGLE_THREAD

	         memcpy_page_req(&gpu_data_ar[head],&g_ppool->rawStorage[page_id],content_size);
		__threadfence(); // push updates to the main memory

		BEGIN_SINGLE_THREAD
			page_md_t* req=&gpu_md_ar_ptr[head];
		
			req->cpu_fd=cpu_fd;
			req->file_offset=file_offset;
		        req->content_size=content_size;
			req->type=type;
			req->last_page=0;

		         __threadfence_system(); // push update to the CPU memory
			gpu_rb_ptr->rb_produce();
	        	 __threadfence_system(); // push the update to the CPU
			MUTEX_UNLOCK(gpu_rb_lock);
		END_SINGLE_THREAD;
	}

	bool  __host__ async_close_rb_t::dequeue(Page* p, page_md_t* md, cudaStream_t s){
		if (cpu_rb->rb_empty()) return false;

		page_md_t* req;

		uint tail=cpu_rb->rb_consumer_ptr();
		req=&cpu_md_ar[tail];

		md->cpu_fd=req->cpu_fd;
		md->file_offset=req->file_offset;
		md->content_size=req->content_size;
		md->type=req->type;
		md->last_page=req->last_page;
		if (!md->last_page) {
		
			CUDA_SAFE_CALL(cudaMemcpyAsync((void*)p,(void*)&gpu_data_ar[tail],md->content_size,cudaMemcpyDeviceToHost,s));
			CUDA_SAFE_CALL(cudaStreamSynchronize(s));
		}
		cpu_rb->rb_consume();
		__sync_synchronize(); // push the update to the GPU
		return true;
	}
	
	// last one
	void __device__ async_close_rb_t::enqueue_done( int cpu_fd){
        
        
		BEGIN_SINGLE_THREAD
       			while(gpu_rb_ptr->rb_full()); // lock up forever
			
			uint head=gpu_rb_ptr->rb_producer_ptr();
		
			page_md_t* req=&gpu_md_ar_ptr[head];

		        req->last_page=1;
			req->cpu_fd=cpu_fd;
	         	
			__threadfence_system(); 
			gpu_rb_ptr->rb_produce();
	        	 __threadfence_system(); // push the update to the CPU
		END_SINGLE_THREAD
	}


