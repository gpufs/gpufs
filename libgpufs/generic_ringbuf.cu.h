/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#ifndef generic_ringbuf
#define generic_ringbuf
/* this is a generic interprocessor ring buffer
 */

struct ringbuf_metadata_t{
        volatile uint _head;
        volatile uint _tail;
        uint _size;

	__device__ __host__  bool rb_empty()
	{
	        return (_tail==_head);
	}
	
	__device__ __host__ uint rb_consumer_ptr(){
		return _tail;
	}
	
	__device__ __host__ uint rb_producer_ptr(){
		return _head;
	}
	
	__device__ __host__ bool rb_full(){
	        return ((_head+1)%_size)==_tail;
	}
	
	__device__ __host__ void rb_consume(){
	         _tail=(_tail+1)%_size;
	}
	
	__device__ __host__ void rb_produce(){
	         _head=(_head+1)%_size;
	}
};

__host__ void ringbuf_metadata_init(ringbuf_metadata_t** rb_cpu, ringbuf_metadata_t** rb_gpu, int num_elem);

__host__ void ringbuf_metadata_free(ringbuf_metadata_t* rb_cpu);
#endif
