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


#ifndef MALLOCFREE_CU
#define MALLOCFREE_CU

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "mallocfree.cu.h"
#include "swapper.cu.h"
#include <assert.h>


// MUST be called from a single thread
DEBUG_NOINLINE __device__  void PPool::init_thread(volatile Page* _storage) volatile
{
	rawStorage=_storage;
	head=0;
	tail=0;
	lock=0;
	size=PPOOL_FRAMES;

	for(int i=0;i<PPOOL_FRAMES;i++)
	{
		frames[i].init_thread(&rawStorage[i],i);
		freelist[i]=i;
	}
}
	
// TODO: lock free datastructure would be better
DEBUG_NOINLINE __device__ volatile PFrame* PPool::allocPage() volatile
{
	PAGE_ALLOC_START

	int oldSize = atomicSub( (int*) &size, 1 );

	if( 0 < oldSize )
	{
		MALLOC

		uint freeLoc = atomicInc( (uint*) &head, PPOOL_FRAMES - 1 );
		volatile PFrame* pFrame = &( frames[freelist[freeLoc]] );

		PAGE_ALLOC_STOP

		return pFrame;
	}

	assert( false );

	return NULL;

//	PAGE_ALLOC_START
//
//	volatile PFrame* frame;
//	MUTEX_LOCK(lock);
//	MALLOC
//
//	if (head==PPOOL_FRAMES) {
//		if (swapout(MIN_PAGES_SWAPOUT)== MIN_PAGES_SWAPOUT)
//		{
//			// TODO: error handling
//			// we failed to swap out
//			GPU_ASSERT(NULL);
//		}
//
//	}
//	frame=&frames[freelist[head]];
//	head++;
//	__threadfence();
//	MUTEX_UNLOCK(lock);
//
//	PAGE_ALLOC_STOP
//	return frame;
}

DEBUG_NOINLINE __device__ void PPool::freePage(volatile PFrame* frame, bool locked) volatile
{
	FREE

	frame->clean();
	freelist[tail] = frame->rs_offset;
	tail = ( tail + 1 ) % PPOOL_FRAMES;
	threadfence();

	atomicAdd( (int*) &size, 1 );

//	if (frame == NULL) return;
//
//	if (locked==FREE_LOCKED) MUTEX_LOCK(lock);
//	FREE
//	GPU_ASSERT(head>0);
//	head--;
//	frame->clean();
//	freelist[head]=frame->rs_offset;
//	__threadfence();
//	if (locked==FREE_LOCKED) MUTEX_UNLOCK(lock);
}

#endif
