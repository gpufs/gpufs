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

#ifndef FS_DEBUG_CU_H
#define FS_DEBUG_CU_H

#include <stdint.h>
#include <stdio.h>

#ifdef DEBUG

struct gdebug_t;
extern            volatile struct gdebug_t *_hdbg;
extern __device__ volatile struct gdebug_t *_gdbg;
extern __device__ volatile int             _gdbg_mutex;

__host__ void _gdebug_init(void);

__device__ void _dbg(const char *s,
                       void* ptr,
                       size_t v,
                       long int t,
                       long int l,
                       const char *fname,
                       const char *func);

#define DBGT(s, ptr, v, t) _dbg(s, ptr, v, t, __LINE__, __FILE__, __func__)
#define DBG_INIT() _gdebug_init()

#else

#define DBGT(s, ptr, v, t)
#define DBG_INIT()

#endif

#define GDBG(s, ptr, v) DBGT(s, (void*)ptr, (size_t)v, 0)
#define GDBGS(s) GDBG(s, (void*)-1, (size_t)-1)
#define GDBGL() GDBG("", (void*)-1, (size_t)-1)
#define GDBGV(s, v) GDBG(s, (void*)-1, (size_t)v)

#if DEBUG
#define GPRINT(...) \
	if( (threadIdx.x + threadIdx.y + threadIdx.z) ==0 ) \
	{ \
		printf(__VA_ARGS__); \
	}
#else
#define GPRINT(...)
#endif

#define PRINT_STATS(SYMBOL) { unsigned int tmp;\
			     cudaMemcpyFromSymbol(&tmp,SYMBOL,sizeof(int),0,cudaMemcpyDeviceToHost);\
			     fprintf(stderr,"%s %u\n", #SYMBOL, tmp);}

#define INIT_STATS(SYMBOL) SYMBOL=0;

/*** malloc stats****/
#ifdef MALLOC_STATS

extern __device__ unsigned int numMallocs;
extern __device__ unsigned int numFrees;
extern __device__ unsigned int numPageAllocRetries;
extern __device__ unsigned int numLocklessSuccess;
extern __device__ unsigned int numLockedTries;
extern __device__ unsigned int numWrongFileId;

extern __device__ unsigned int numRtMallocs;
extern __device__ unsigned int numRtFrees;
extern __device__ unsigned int numHT_Miss;
extern __device__ unsigned int numHT_Hit;
extern __device__ unsigned int numPreclosePush;
extern __device__ unsigned int numPrecloseFetch;

extern __device__ unsigned int numFlushedWrites;
extern __device__ unsigned int numFlushedReads;
extern __device__ unsigned int numTrylockFailed;

extern __device__ unsigned int numKilledBufferCache;

extern __device__ unsigned int numHM_locklessSuccess;
extern __device__ unsigned int numHM_lockedSuccess;

#define INIT_MALLOC INIT_STATS(numMallocs); INIT_STATS(numFrees); INIT_STATS(numPageAllocRetries); INIT_STATS(numLocklessSuccess); INIT_STATS(numWrongFileId); INIT_STATS(numLockedTries);
#define FREE atomicAdd(&numFrees,1);
#define MALLOC atomicAdd(&numMallocs,1);
#define PAGE_ALLOC_RETRIES atomicAdd(&numPageAllocRetries,1);
#define LOCKLESS_SUCCESS atomicAdd(&numLocklessSuccess,1);
#define LOCKED_TRIES atomicAdd(&numLockedTries,1);
#define WRONG_FILE_ID atomicAdd(&numWrongFileId,1);

#define PRINT_MALLOC PRINT_STATS(numMallocs);
#define PRINT_FREE PRINT_STATS(numFrees);
#define PRINT_PAGE_ALLOC_RETRIES PRINT_STATS(numPageAllocRetries);
#define PRINT_LOCKLESS_SUCCESS PRINT_STATS(numLocklessSuccess);
#define PRINT_LOCKED_TRIES PRINT_STATS(numLockedTries);
#define PRINT_WRONG_FILE_ID  PRINT_STATS(numWrongFileId);

#define INIT_RT_MALLOC INIT_STATS(numRtMallocs); INIT_STATS(numRtFrees);
#define RT_FREE atomicAdd(&numRtFrees,1);
#define RT_MALLOC atomicAdd(&numRtMallocs,1);

#define PRINT_RT_MALLOC PRINT_STATS(numRtMallocs);
#define PRINT_RT_FREE PRINT_STATS(numRtFrees);

#define INIT_HT_STAT INIT_STATS(numHT_Miss); INIT_STATS(numHT_Hit);INIT_STATS(numPreclosePush);INIT_STATS(numPrecloseFetch);
#define HT_MISS atomicAdd(&numHT_Miss,1);
#define HT_HIT atomicAdd(&numHT_Hit,1);
#define PRECLOSE_PUSH atomicAdd(&numPreclosePush,1);
#define PRECLOSE_FETCH atomicAdd(&numPrecloseFetch,1);

#define PRINT_PRECLOSE_PUSH PRINT_STATS(numPreclosePush);
#define PRINT_PRECLOSE_FETCH PRINT_STATS(numPrecloseFetch);
#define PRINT_HT_HIT PRINT_STATS(numHT_Hit);
#define PRINT_HT_MISS PRINT_STATS(numHT_Miss);

#define INIT_SWAP_STAT INIT_STATS(numFlushedWrites); INIT_STATS(numFlushedReads); INIT_STATS(numTrylockFailed); INIT_STATS(numKilledBufferCache);
#define FLUSHED_WRITE atomicAdd(&numFlushedWrites,1);
#define FLUSHED_READ atomicAdd(&numFlushedReads,1);
#define TRY_LOCK_FAILED atomicAdd(&numTrylockFailed,1);
#define KILL_BUFFER_CACHE atomicAdd(&numKilledBufferCache,1);

#define PRINT_FLUSHED_WRITE PRINT_STATS(numFlushedWrites);
#define PRINT_FLUSHED_READ PRINT_STATS(numFlushedReads);
#define PRINT_TRY_LOCK_FAILED PRINT_STATS(numTrylockFailed);
#define PRINT_KILL_BUFFER_CACHE PRINT_STATS(numKilledBufferCache);

#define INIT_HM_STAT INIT_STATS(numHM_locklessSuccess); INIT_STATS(numHM_lockedSuccess);
#define HM_LOCKLESS atomicAdd(&numHM_locklessSuccess,1);
#define HM_LOCKED atomicAdd(&numHM_lockedSuccess,1);

#define PRINT_HM_LOCKLESS PRINT_STATS(numHM_locklessSuccess);
#define PRINT_HM_LOCKED PRINT_STATS(numHM_lockedSuccess);

#else

#define INIT_MALLOC 
#define FREE 
#define MALLOC 
#define PAGE_ALLOC_RETRIES 
#define LOCKLESS_SUCCESS 
#define LOCKED_TRIES
#define WRONG_FILE_ID 

#define PRINT_MALLOC 
#define PRINT_FREE 
#define PRINT_PAGE_ALLOC_RETRIES 
#define PRINT_LOCKLESS_SUCCESS
#define PRINT_LOCKED_TRIES
#define PRINT_WRONG_FILE_ID  

#define INIT_RT_MALLOC 
#define RT_FREE 
#define RT_MALLOC 

#define PRINT_RT_MALLOC 
#define PRINT_RT_FREE 

#define INIT_HT_STAT 
#define HT_MISS 
#define HT_HIT
#define PRECLOSE_PUSH 
#define PRECLOSE_FETCH 

#define PRINT_PRECLOSE_PUSH 
#define PRINT_PRECLOSE_FETCH 
#define PRINT_HT_HIT 
#define PRINT_HT_MISS 

#define INIT_SWAP_STAT 
#define FLUSHED_WRITE 
#define FLUSHED_READ
#define TRY_LOCK_FAILED 
#define KILL_BUFFER_CACHE

#define PRINT_FLUSHED_WRITE 
#define PRINT_FLUSHED_READ 
#define PRINT_TRY_LOCK_FAILED 
#define PRINT_KILL_BUFFER_CACHE

#define INIT_HM_STAT
#define HM_LOCKLESS
#define HM_LOCKED

#define PRINT_HM_LOCKLESS
#define PRINT_HM_LOCKED

#endif

/***timing stats****/
#ifdef TIMING_STATS

extern __device__ unsigned long long RTSearchTime;
extern __device__ unsigned long long KernelTime;
extern __device__ unsigned long long RTWaitTime;
extern __device__ unsigned long long MapTime;
extern __device__ unsigned long long CopyBlockTime;
extern __device__ unsigned long long PageReadTime;
extern __device__ unsigned long long PageAllocTime;
extern __device__ unsigned long long FileOpenTime;
extern __device__ unsigned long long CPUReadTime;
extern __device__ unsigned long long HashMapSearchTime;

#define PRINT_TIME(SYMBOL, freq, blocks) { unsigned long long tmp; \
			     cudaMemcpyFromSymbol(&tmp,SYMBOL,sizeof(unsigned long long),0,cudaMemcpyDeviceToHost); \
			     fprintf(stderr,"%s %fms\n", #SYMBOL, ((double)(tmp) / 1e6) / (double)(blocks)); }

#define GET_TIME(timer) \
	asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(timer) :);

#define START(timer) \
	unsigned long long timer##Start; \
	if( TID == 0 ) \
	{ \
		GET_TIME( timer##Start ); \
	}

#define START_WARP(timer) \
	unsigned long long timer##Start; \
	if( LANE_ID == 0 ) \
	{ \
		GET_TIME( timer##Start ); \
	}

#define STOP(timer) \
	unsigned long long timer##Stop; \
	if( TID == 0 ) \
	{ \
		GET_TIME( timer##Stop ); \
		atomicAdd(&timer##Time, timer##Stop - timer##Start); \
	}

#define STOP_WARP(timer) \
	unsigned long long timer##Stop; \
	if( LANE_ID == 0 ) \
	{ \
		GET_TIME( timer##Stop ); \
		atomicAdd(&timer##Time, timer##Stop - timer##Start); \
	}

//#define GET_TIME(timer) \
//	asm volatile ("mov.u64 %0, %%clock64;" : "=l"(timer) :);

#define INIT_RT_TIMING INIT_STATS(RTSearchTime); INIT_STATS(KernelTime);

#define KERNEL_START START( Kernel )
#define KERNEL_STOP STOP( Kernel )

#define RT_SEARCH_START START( RTSearch )
#define RT_SEARCH_STOP STOP( RTSearch )

#define RT_WAIT_START START( RTWait )
#define RT_WAIT_STOP STOP( RTWait )

#define MAP_START START( Map )
#define MAP_STOP STOP( Map )

#define COPY_BLOCK_START START( CopyBlock )
#define COPY_BLOCK_STOP STOP( CopyBlock )

#define PAGE_READ_START START( PageRead )
#define PAGE_READ_STOP STOP( PageRead )

#define PAGE_ALLOC_START START( PageAlloc )
#define PAGE_ALLOC_STOP STOP( PageAlloc )

#define PAGE_ALLOC_START_WARP START_WARP( PageAlloc )
#define PAGE_ALLOC_STOP_WARP STOP_WARP( PageAlloc )

#define FILE_OPEN_START START( FileOpen )
#define FILE_OPEN_STOP STOP( FileOpen )

#define CPU_READ_START START( CPURead )
#define CPU_READ_STOP STOP( CPURead )

#define HASH_MAP_SEARCH_START START( HashMapSearch )
#define HASH_MAP_SEARCH_STOP STOP( HashMapSearch )

#define PRINT_KERNEL_TIME(freq, blocks) PRINT_TIME(KernelTime, freq, blocks);
#define PRINT_RT_SEARCH_TIME(freq, blocks) PRINT_TIME(RTSearchTime, freq, blocks);
#define PRINT_RT_WAIT_TIME(freq, blocks) PRINT_TIME(RTWaitTime, freq, blocks);
#define PRINT_MAP_TIME(freq, blocks) PRINT_TIME(MapTime, freq, blocks);
#define PRINT_COPY_BLOCK_TIME(freq, blocks) PRINT_TIME(CopyBlockTime, freq, blocks);
#define PRINT_PAGE_READ_TIME(freq, blocks) PRINT_TIME(PageReadTime, freq, blocks);
#define PRINT_PAGE_ALLOC_TIME(freq, blocks) PRINT_TIME(PageAllocTime, freq, blocks);
#define PRINT_FILE_OPEN_TIME(freq, blocks) PRINT_TIME(FileOpenTime, freq, blocks);
#define PRINT_CPU_READ_TIME(freq, blocks) PRINT_TIME(CPUReadTime, freq, blocks);
#define PRINT_HASH_MAP_SEARCH_TIME(freq, blocks) PRINT_TIME(HashMapSearchTime, freq, blocks);

#else

#define PRINT_TIME(SYMBOL, freq, blocks)

#define GET_TIME(timer)

#define INIT_RT_TIMING

#define KERNEL_START
#define KERNEL_STOP

#define RT_SEARCH_START
#define RT_SEARCH_STOP

#define RT_WAIT_START
#define RT_WAIT_STOP

#define MAP_START
#define MAP_STOP

#define PAGE_READ_START
#define PAGE_READ_STOP

#define COPY_BLOCK_START
#define COPY_BLOCK_STOP

#define PAGE_ALLOC_START_WARP
#define PAGE_ALLOC_STOP_WARP

#define FILE_OPEN_START
#define FILE_OPEN_STOP

#define CPU_READ_START
#define CPU_READ_STOP

#define HASH_MAP_SEARCH_START
#define HASH_MAP_SEARCH_STOP

#define PRINT_KERNEL_TIME(freq, blocks)
#define PRINT_RT_SEARCH_TIME(freq, blocks)
#define PRINT_RT_WAIT_TIME(freq, blocks)
#define PRINT_MAP_TIME(freq, blocks)
#define PRINT_PAGE_READ_TIME(freq, blocks)
#define PRINT_COPY_BLOCK_TIME(freq, blocks)
#define PRINT_PAGE_ALLOC_TIME(freq, blocks)
#define PRINT_FILE_OPEN_TIME(freq, blocks)
#define PRINT_CPU_READ_TIME(freq, blocks)
#define PRINT_HASH_MAP_SEARCH_TIME(freq, blocks)

#endif

#define INIT_ALL_STATS { INIT_MALLOC; INIT_RT_MALLOC; INIT_HT_STAT; INIT_SWAP_STAT; INIT_HM_STAT;}
#define INIT_TIMING_STATS { INIT_RT_TIMING; }
#endif

