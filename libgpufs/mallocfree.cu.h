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


#ifndef MALLOCFREE_CU_H
#define MALLOCFREE_CU_H
#include "fs_constants.h"
#include "fs_structures.cu.h"

#define FREE_LOCKED 1
#define FREE_UNLOCKED 0
struct PPool{

	volatile Page* 	rawStorage;
	volatile PFrame	frames[PPOOL_FRAMES];
	volatile uint 	freeList[PPOOL_FRAMES];

	volatile uint	swapLock;
	volatile uint 	head;
	volatile uint 	tail;
	volatile int 	size;

// MUST be called from a single thread
	__device__  void init_thread(volatile Page* _storage) volatile;

	__device__ volatile PFrame *allocPage() volatile;

	__device__ void freePage(volatile PFrame* frame) volatile ;
};

#endif
