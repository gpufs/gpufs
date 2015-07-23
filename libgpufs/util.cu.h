/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#ifndef UTIL_CU_H
#define UTIL_CU_H

#include "fs_constants.h"
#include "fs_debug.cu.h"

#define CUDA_SAFE_CALL(x) if((x)!=cudaSuccess) { fprintf(stderr,"CUDA ERROR %s: %d %s\n",__FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }

#define WAIT_ON_MEM(mem,val)  while(readNoCache(&(mem))!=val);
#define WAIT_ON_MEM_NE(mem,val)  while(readNoCache(&(mem))==val);

//#define GPU_ASSERT(x) ({WRITE_DEBUG(__FILE__,__LINE__); (*(int*)NULL)=0;})
//#define GPU_ASSERT(x) if (!(x)){WRITE_DEBUG(__FILE__,__LINE__); \
//	__threadfence_system(); asm("trap;");}; //return;}

#ifdef DEBUG
#define GPU_ASSERT(x)	assert(x);
#else
//#warning "Asserts disabled"
#define GPU_ASSERT(x)
#endif


#define MUTEX_LOCK(lock) while (atomicExch((int*)(&(lock)),1));
#define MUTEX_WAS_LOCKED(lock) atomicExch((int*)(&(lock)),1)
#define MUTEX_TRY_LOCK(lock) (!atomicExch((int*)(&(lock)),1))

#define MUTEX_UNLOCK(lock) { atomicExch((int*)(&(lock)),0);}

#define BEGIN_SINGLE_THREAD __syncthreads(); if(threadIdx.x+threadIdx.y+threadIdx.z ==0) {

#define END_SINGLE_THREAD  } __syncthreads();

#define TID (threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y)
#define LANE_ID (TID & 0x1f)
#define WARP_ID (TID >> 5)
#define NUM_WARPS ((blockDim.x * blockDim.y * blockDim.z) >> 5)
#define BLOCK_ID (blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z)

#define GET_SMID( SMID ) \
	asm volatile ("mov.u32 %0, %%smid;" : "=r"(SMID) :);
	
	
__forceinline__ __device__ void bzero_thread(volatile void* dst, uint size)
{

	int bigsteps=size>>3;
	int i=0;
	for( i=0;i<bigsteps;i++)
		((double*)dst)[i]=0;
	bigsteps=bigsteps<<3;
	for (i=bigsteps;i<size;i++){
		((char*)dst)[i]=0;
	}

}

__forceinline__ __device__ void bzero_page(volatile char* dst){
	for(int i=TID;i<FS_BLOCKSIZE>>3;i+=blockDim.x*blockDim.y){
		((volatile double*)dst)[i]=0;
	}
}

__forceinline__ __device__ void bzero_page_warp(volatile char* dst){
	for( int i=LANE_ID; i<FS_BLOCKSIZE>>3; i += 32 )
	{
		((volatile double*)dst)[i]=0;
	}
}


__forceinline__ __device__ void strcpy_thread(volatile char* dst, const volatile char* src, uint size)
{
	int i=0;
	for( i=0;i<size && src[i]!='\0';i++)
		dst[i]=src[i];

	if (src[i]=='\0') dst[i]='\0';
}

__forceinline__ __device__ char strcmp_thread(volatile const char* dst, volatile const char* src, uint size)
{
	int i=0;
	for (i=0;i<size && dst[i]==src[i] ;i++)
	{
		if(dst[i]=='\0' ) return 0;
	}
	
	if (i == size ) return 0;
	
	return 1;
}


__forceinline__ __device__ double readNoCache(const volatile double* ptr){
          double val;
      val=*ptr;       
//	asm("ld.cv.f64 %0, [%1];"  : "=d"(val):"l"(ptr));
          return val;
}

__forceinline__ __device__ float readNoCache(const volatile float* ptr){
          float val;
      val=*ptr;
//	asm("ld.cv.f64 %0, [%1];"  : "=d"(val):"l"(ptr));
          return val;
}

__forceinline__ __device__ char2 readNoCache(const volatile uchar* ptr){
	   char2 v;v.x=*ptr; v.y=*(ptr+1); return v;
//	asm("ld.cv.u16 %0, [%1];"  : "=h"(val2):"l"(ptr));
//	char2 n;
//	n.x=(char)val2; n.y=(char)val2>>8;
//          return n;
}
__forceinline__ __device__ unsigned int readNoCache(const volatile unsigned int* ptr){
          unsigned int val;
      val=*ptr;       
//	asm("ld.cv.u32 %0, [%1];"  : "=r"(val):"l"(ptr));
          return val;
}
__forceinline__ __device__ int readNoCache(const volatile int* ptr){
          int val;
      val=*ptr;       
//	asm("ld.cv.u32 %0, [%1];"  : "=r"(val):"l"(ptr));
          return val;
}
__forceinline__ __device__ char readNoCache(const volatile char* ptr){
	char v=*ptr;
	return v;
}
__forceinline__ __device__ size_t readNoCache(const volatile size_t* ptr){
          size_t val;
      val=*ptr;       
//	if (sizeof(size_t)==8)
//	asm("ld.cv.f64 %0, [%1];"  : "=d"(val):"l"(ptr));
//	else
//	asm("ld.cv.u32 %0, [%1];"  : "=r"(val):"l"(ptr));
          return val;
}

__device__ __forceinline__ void valcpy_128(double2* dst, double2* src){
	asm volatile ("{.reg .f64 t1;\n\t"
				  ".reg .f64 t2;\n\t"
				  "ld.cv.v2.f64 {t1,t2}, [%1]; \n\t"
				  "st.wt.v2.f64 [%0],{t1,t2};}"  : : "l"(dst),"l"(src):"memory");
}

__device__ __forceinline__ void valcpy_256_interleave(  double2* dst, double2* src){

	double2* d1=dst+blockDim.x*blockDim.y*blockDim.z;
	double2* s1=src+blockDim.x*blockDim.y*blockDim.z;

	asm volatile ("{.reg .f64 t1;\n\t"
				  ".reg .f64 t2;\n\t"
				  ".reg .f64 t3;\n\t"
				  ".reg .f64 t4;\n\t"
				  "ld.cv.v2.f64 {t1,t2}, [%2]; \n\t"
				  "ld.cv.v2.f64 {t3,t4}, [%3]; \n\t"
				  "st.wt.v2.f64 [%0],{t1,t2};\n\t"
				  "st.wt.v2.f64 [%1],{t3,t4};}"  : : "l"(dst),"l"(d1),"l"(src),"l"(s1):"memory");
}

__device__
void __forceinline__ copy_block_large(char* dst, char* src, uint32_t len) {
	int i = TID;

	len /= sizeof(double2);

	while (i + blockDim.x*blockDim.y*blockDim.z < len) {
		valcpy_256_interleave(((double2*)dst) + i, ((double2*)src) + i);
		i += 2 * blockDim.x*blockDim.y*blockDim.z;
	}
}

template<typename T>
__device__ void inline aligned_copy(uchar* dst, volatile uchar* src, int newsize, int tid)
{
	int stride=blockDim.x*blockDim.y*blockDim.z;
	while(tid<newsize){
		((T*)dst)[tid]=*(((T*)src)+tid);
		tid+=stride;
	}
}

template<typename T>
__device__ void inline aligned_copy_warp(uchar* dst, volatile uchar* src, int newsize)
{
	int id = threadIdx.x & 0x1f;
	int stride = 32;
	
	while( id < newsize )
	{
		((T*)dst)[id] = *(((T*)src)+id);
		id+=stride;
	}
}

__device__
void __forceinline__ copy_block_16(char* dst, char* src, uint32_t len) {
	const int block_size = blockDim.x * blockDim.y * blockDim.z;
	const int chunk_size =  (2 * sizeof(double2) * block_size);
	const int chunk_len = chunk_size * (len / chunk_size);
	int i;
	if (len >= chunk_size) {
		copy_block_large(dst, src, chunk_len);
	}

	dst += chunk_len;
	src += chunk_len;
	len -= chunk_len;
	for (i = TID; i < (len >> 4); i += block_size) {
		valcpy_128(((double2*)dst) + i, ((double2*)src) + i);
	}

	__syncthreads();
}

__forceinline__ __device__ void copy_block(uchar* dst, volatile uchar*src, int size)
{
	int tid=TID;
	int newsize;
	// get the alignment
	int shift;

	// checking whether the src/dst is 8/4/2 byte aligned
	if ((((long)dst)&0xf) == 0 && (((long)src)&0xf) == 0) {
		shift = 4;
		newsize=size>>shift;
		copy_block_16((char*)dst,(char*)src,size);
	} else
	if ((((long)dst)&0x7) == 0 && (((long)src)&0x7) == 0) {
		shift=3;
		newsize=size>>shift;
		aligned_copy<double>(dst,src,newsize,tid);
	} else
	if ((((long)dst)&0x3) == 0 && (((long)src)&0x3) == 0) {
		shift=2;
		newsize=size>>shift;
		aligned_copy<float>(dst,src,newsize,tid);
	} else
	if ((((long)dst)&0x1) == 0 && (((long)src)&0x1) == 0) {
		shift=1;
		newsize=size>>shift;
		aligned_copy<char2>(dst,src,newsize,tid);
	} else
	{
		shift=0;
		newsize=size;
		aligned_copy<char>(dst,src,newsize,tid);
	}

	newsize=newsize<<shift;
	__syncthreads();

	// copying remainders with single thread
	if ((threadIdx.x + threadIdx.y + threadIdx.z) ==0){
		while(newsize<size){
			char2 r=readNoCache(src+newsize);
			dst[newsize]=r.x;newsize++;
			if(newsize<size) dst[newsize]=r.y;
			newsize++;
		}

	}
	__syncthreads();
}

__forceinline__ __device__ void copy_block_warp(uchar* dst, volatile uchar*src, int size)
{
	int newsize;
	// get the alignment
	int shift;

	// checking whether the src/dst is 8/4/2 byte aligned
	if ((((long)dst)&0x7) == 0 && (((long)src)&0x7) == 0) {
		shift=3;
		newsize=size>>shift;
		aligned_copy_warp<double>(dst,src,newsize);
	} else
	if ((((long)dst)&0x3) == 0 && (((long)src)&0x3) == 0) {
		shift=2;
		newsize=size>>shift;
		aligned_copy_warp<float>(dst,src,newsize);
	} else
	if ((((long)dst)&0x1) == 0 && (((long)src)&0x1) == 0) {
		shift=1;
		newsize=size>>shift;
		aligned_copy_warp<char2>(dst,src,newsize);
	} else
	{
		shift=0;
		newsize=size;
		aligned_copy_warp<char>(dst,src,newsize);
	}

	newsize=newsize<<shift;

	int laneid = threadIdx.x & 0x1f;

	// copying remainders with single thread
	if( laneid ==0 )
	{
		while(newsize<size){
			char2 r=readNoCache(src+newsize);
			dst[newsize]=r.x;newsize++;
			if(newsize<size) dst[newsize]=r.y;
			newsize++;
		}

	}
}

__forceinline__ __device__ void copyNoCache_block(uchar* dst, volatile uchar*src, int size)
{
	copy_block(dst,src,size);
}
	
__forceinline__ __device__ void copyNoCache_thread(char* dst, volatile char*src, int size)
{
	for(int i=0;i<size>>2;i++){
		((int*)dst)[i]=readNoCache((int*)(src)+i);
	}
}


__forceinline__ __device__ void write_thread(uchar* dst, uchar*src, int size)
{
	for(int i=0;i<size>>2;i++){
		((int*)dst)[i]=*((int*)(src)+i);
	}
}

__device__ inline size_t offset2block(size_t offset, int log_blocksize)
{
	return offset>>log_blocksize;
}
__device__ inline uint offset2blockoffset(size_t offset, int blocksize)
{
	return offset&(blocksize-1);
}

union BroadcastHelper
{
	size_t s;
	void* v;
	long long l;
	unsigned long long ul;
	double d;
	int i[2];
};

__device__ inline BroadcastHelper broadcast(BroadcastHelper b, int leader = 0)
{
	BroadcastHelper t;

	t.i[0] = __shfl( b.i[0], leader );
	t.i[1] = __shfl( b.i[1], leader );

	return t;
}

__device__ inline int broadcast(int b, int leader = 0)
{
	b = __shfl( b, leader );

	return b;
}

__device__ inline unsigned int broadcast(unsigned int b, int leader = 0)
{
	b = __shfl( b, leader );

	return b;
}

__device__ inline float broadcast(float b, int leader = 0)
{
	b = __shfl( b, leader );

	return b;
}

struct LAST_SEMAPHORE
{
	int count;
	__device__ int is_last() volatile
	{
		atomicAdd((int*)&count,1);
		if (count== gridDim.x*gridDim.y ) return 1;
		return 0;
	}	
};
struct INIT_LOCK
{
	volatile int lock;
	__device__ int try_wait() volatile{
		int res=atomicMax((int*)&lock,1);
		if (res==0) {
			return 1; // locked now
		}
		while(lock!=2);
		return 0;
	}
	__device__ void signal() volatile{
		lock=2;
		__threadfence();
	}
	
};



#define ERROR(str) __assert_fail(str,__FILE__,__LINE__,__func__);

__device__ int getNewFileId();
#endif
