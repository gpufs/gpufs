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


#ifndef SWAPPER_CU_H
#define SWAPPER_CU_H

#include "fs_structures.cu.h"

__device__ int swapout(int npages);
__device__ int flush_cpu( volatile FTable_entry* file, volatile OTable_entry* e, int flags);
__device__ int writeback_page(int fd, volatile FTable_page* p );
DEBUG_NOINLINE __device__ int writeback_page(int cpu_fd, volatile FTable_page* p,int flags, bool tolock)
;

__device__ int writeback_page_async_on_close(int cpu_fd, volatile FTable_page* p,int flags);

__device__ void writeback_page_async_on_close_done(int cpu_fd);

#endif
