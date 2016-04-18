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


#ifndef FS_CALLS_CU
#define FS_CALLS_CU

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "fs_globals.cu.h"
#include "fat_pointer.cu.h"

// no reference counting here
__device__ int single_thread_fsync(int fd);
__device__ int single_thread_ftruncate(int fd, int size);
__device__ int single_thread_open(const char* filename, int flags);

#define READ 0
#define WRITE 1

__device__ int gclose(int fd);
__device__ int gopen(const char* filename, int flags);
__device__ int gmsync(volatile void *addr, size_t length,int flags);
__device__ int gmunmap(volatile void *addr, size_t length);
__device__ int gmunmap_warp( volatile void *addr, size_t length, int ref = 1 );
__device__ volatile void* gmmap(void *addr, size_t size,
		int prot, int flags, int fd, off_t offset);
__device__ volatile void* gmmap_warp(void *addr, size_t size,
		int prot, int flags, int fd, off_t offset, int ref = 1);
__device__ size_t gwrite(int fd,size_t offset, size_t size, uchar* buffer);
__device__ size_t gread(int fd, size_t offset, size_t size, uchar* buffer);
__device__ uint gunlink(char* filename);
__device__ size_t fstat(int fd);

template<typename T, int N>
__device__ FatPointer<T, N> gvmmap(void *addr, size_t size,
		int prot, int flags, int fd, off_t offset, TLB<N>* tlb)
{
	FatPointer<T,N> ptr(fd, offset, size, flags, tlb, (uchar*)g_ppool->rawStorage, g_ppool->frames);
	return ptr;
}


__device__ int gfsync(int fd);
__device__ int gftruncate(int fd, int size);

#endif
