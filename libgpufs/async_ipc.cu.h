/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#ifndef async_ipc_cu_h
#define async_ipc_cu_h

#include <cuda.h>
#include <cuda_runtime.h>

#include "generic_ringbuf.cu.h"

struct page_md_t{
	volatile uint cpu_fd;
	volatile size_t file_offset;
	volatile uint content_size;
	volatile int type;
	volatile uint last_page;
};


extern __device__ int gpu_rb_lock;

struct async_close_rb_t{
	page_md_t* cpu_md_ar;
	page_md_t* gpu_md_ar_ptr;
	volatile Page* gpu_data_ar;
	ringbuf_metadata_t *cpu_rb;
	ringbuf_metadata_t *gpu_rb_ptr;

	__host__ void init_host();

	__forceinline__ __device__ void memcpy_page_req(volatile Page* dst, const volatile Page* src, size_t size);

	// no lock on the queue is necessary because this is done on close and global ftable lock is taken
	// this call is blocking on the CPU queue
	void __device__ enqueue(int cpu_fd, size_t page_id, size_t file_offset, uint content_size, int type );

	bool  __host__ dequeue(Page* p, page_md_t* md, cudaStream_t s);
	
	// last one
	void __device__ enqueue_done( int cpu_fd);

};

__host__ void ringbuf_page_md_init(page_md_t** cpumd, page_md_t** gpumd, int num_elem);

#endif
