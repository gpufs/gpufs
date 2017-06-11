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
#include "fs_globals.cu.h"
#include "util.cu.h"
#include "mallocfree.cu.h"
#include "hashMap.cu.h"
#include <assert.h>


// MUST be called from a single thread
DEBUG_NOINLINE __device__  void PPool::init_thread(volatile Page* _storage) volatile
{
	rawStorage=_storage;

	int slice = PPOOL_FRAMES / NUM_MEMORY_RINGS;

	for( int i = 0; i < NUM_MEMORY_RINGS; ++i )
	{
		subRings[i].head = 0;
		subRings[i].tail = 0;
		subRings[i].swapLock = 0;
		subRings[i].base = i * slice;
		subRings[i].size = slice;
	}

	for(int i=0;i<PPOOL_FRAMES;i++)
	{
		frames[i].init_thread(&rawStorage[i],i);
		freeList[i]=i;
	}
}
	
DEBUG_NOINLINE __device__ volatile PFrame* PPool::allocPage() volatile
{
	PAGE_ALLOC_START_WARP

	int ringID = BLOCK_ID % NUM_MEMORY_RINGS;

	volatile uint	base = subRings[ringID].base;
	volatile uint&	swapLock = subRings[ringID].swapLock;
	volatile uint& 	head = subRings[ringID].head;
	volatile uint& 	tail = subRings[ringID].tail;
	volatile int& 	size = subRings[ringID].size;

	int oldSize = atomicSub( (int*) &size, 1 );

	if( LOWER_WATER_MARK < oldSize )
	{
		uint freeLoc = atomicInc( (uint*) &head, (PPOOL_FRAMES / NUM_MEMORY_RINGS) - 1 );
		volatile PFrame* pFrame = &( frames[freeList[base + freeLoc]] );

		GPU_ASSERT( freeList[base + freeLoc] == pFrame->rs_offset );

		PAGE_ALLOC_STOP_WARP

		return pFrame;
	}

	EVICT_START_WARP
	// else, we are almost out of memory
	if( MUTEX_TRY_LOCK(swapLock) )
	{
//		GDBGV("swapping", oldSize);
//		GPRINT("%d\n", oldSize);

		// swap
		uint numSwapped = 0;
		int numRetries = 0;
		uint candLoc = tail;

//		GDBGV("candLoc", candLoc);
		while( NUM_PAGES_SWAPOUT > numSwapped )
		{
			volatile PFrame* cand = &( frames[freeList[base + candLoc]] );

			// Try to remove from the hash
			bool removed = false;

			if( cand->dirty == 0 && cand->dirtyCounter == 0 )
			{
				removed = g_hashMap->removePFrame( cand );
			}

			if( removed )
			{
//				GDBGV("candLoc", candLoc);
				if( candLoc != tail )
				{
					// swap tail and current location
					uint t = freeList[base + tail];
					freeList[base + tail] = freeList[base + candLoc];
					freeList[base + candLoc] = t;

					__threadfence();
				}

				freePage( cand, tail, base );
				numSwapped++;
				candLoc = ( candLoc + 1 ) % (PPOOL_FRAMES / NUM_MEMORY_RINGS);
				continue;
			}

			// else
			// move it down the ring buffer since it's busy
//			uint moveLoc = ( candLoc + (PPOOL_FRAMES / 4) ) % PPOOL_FRAMES;
//			uint t = freeList[moveLoc];
//			freeList[moveLoc] = freeList[candLoc];
//			freeList[candLoc] = t;
//
//			threadfence();

			// Search for another one

			// In this case we will need to swap the element in tail to prevent loosing it later
			candLoc = ( candLoc + 1 ) % (PPOOL_FRAMES / NUM_MEMORY_RINGS);

			while( (NUM_SWAP_RETRIES > numRetries) || (0 == numSwapped) )
			{
				cand = &( frames[freeList[base + candLoc]] );

				bool removed = false;

				if( cand->dirty == 0 && cand->dirtyCounter == 0 )
				{
					removed = g_hashMap->removePFrame( cand );
				}

				if( removed )
				{
//					GDBGV("candLoc", candLoc);
					// swap tail and current location
					uint t = freeList[base + tail];
					freeList[base + tail] = freeList[base + candLoc];
					freeList[base + candLoc] = t;

					__threadfence();

					freePage( cand, tail, base );
					numSwapped++;
					candLoc = ( candLoc + 1 ) % (PPOOL_FRAMES / NUM_MEMORY_RINGS);
					break;
				}

				// move it down the ring buffer since it's busy
//				uint moveLoc = ( candLoc + (PPOOL_FRAMES / 4) ) % PPOOL_FRAMES;
//				uint t = freeList[moveLoc];
//				freeList[moveLoc] = freeList[candLoc];
//				freeList[candLoc] = t;
//
//				threadfence();

				candLoc = ( candLoc + 1 ) % (PPOOL_FRAMES / NUM_MEMORY_RINGS);
				numRetries++;

				GPU_ASSERT(numRetries < ((PPOOL_FRAMES / NUM_MEMORY_RINGS) / 2))
			}

			if( NUM_SWAP_RETRIES <= numRetries  )
			{
				break;
			}
		}

		GPU_ASSERT( numSwapped > 0 );

//		GDBGV("numSwapped", numSwapped);
//		GDBGV("numRetries", numRetries);

		uint freeLoc = atomicInc( (uint*) &head, (PPOOL_FRAMES / NUM_MEMORY_RINGS) - 1 );
		volatile PFrame* pFrame = &( frames[freeList[base + freeLoc]] );

		GPU_ASSERT( freeList[base + freeLoc] == pFrame->rs_offset );

		PAGE_ALLOC_STOP_WARP

		atomicAdd( (int*) &size, numSwapped );

		MUTEX_UNLOCK( swapLock );

		EVICT_STOP_WARP
		return pFrame;
	}
	else if( LOWER_WATER_MARK < oldSize )
	{
		uint freeLoc = atomicInc( (uint*) &head, PPOOL_FRAMES - 1 );
		volatile PFrame* pFrame = &( frames[freeList[base + freeLoc]] );

		GPU_ASSERT( freeList[base + freeLoc] == pFrame->rs_offset );

		PAGE_ALLOC_STOP_WARP
		EVICT_STOP_WARP

		return pFrame;
	}
	else
	{
		// Not enough memory, and someone is already swapping
		// Abort
		int old = atomicAdd( (int*) &size, 1 );
//		GDBGV("Revert malloc", old);
		EVICT_STOP_WARP
		return NULL;
	}
}

DEBUG_NOINLINE __device__ void PPool::freePage(volatile PFrame* frame, volatile unsigned int& tail, uint base) volatile {
	GPU_ASSERT( freeList[base + tail] == frame->rs_offset );

	frame->clean();
	freeList[base + tail] = frame->rs_offset;
	tail = ( tail + 1 ) % (PPOOL_FRAMES / NUM_MEMORY_RINGS);
	__threadfence();
}

//DEBUG_NOINLINE __device__ bool PPool::tryLockSwapper() volatile
//{
//	return MUTEX_TRY_LOCK(swapLock);
//}
//
//DEBUG_NOINLINE __device__ void PPool::lockSwapper() volatile
//{
//	MUTEX_LOCK(swapLock);
//}
//
//DEBUG_NOINLINE __device__ void PPool::unlockSwapper() volatile
//{
//	MUTEX_UNLOCK(swapLock);
//}

#endif
