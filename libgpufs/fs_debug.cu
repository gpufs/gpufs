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



/*** malloc stats****/
#ifdef MALLOC_STATS
__device__ unsigned int numMallocs;
__device__ unsigned int numFrees;
__device__ unsigned int numPageAllocRetries;
__device__ unsigned int numLocklessSuccess;
__device__ unsigned int numLockedTries;
__device__ unsigned int numWrongFileId;

__device__ unsigned int numRtMallocs;
__device__ unsigned int numRtFrees;
__device__ unsigned int numHT_Miss;
__device__ unsigned int numHT_Hit;
__device__ unsigned int numPreclosePush;
__device__ unsigned int numPrecloseFetch;

__device__ unsigned int numFlushedWrites;
__device__ unsigned int numFlushedReads;
__device__ unsigned int numTrylockFailed;
__device__ unsigned int numKilledBufferCache;

__device__ unsigned int numHM_locklessSuccess;
__device__ unsigned int numHM_lockedSuccess;
#endif

#ifdef TIMING_STATS
__device__ unsigned long long KernelTime;
__device__ unsigned long long RTSearchTime;
__device__ unsigned long long RTWaitTime;
__device__ unsigned long long MapTime;
__device__ unsigned long long CopyBlockTime;
__device__ unsigned long long PageReadTime;
__device__ unsigned long long PageAllocTime;
__device__ unsigned long long FileOpenTime;
__device__ unsigned long long CPUReadTime;
__device__ unsigned long long HashMapSearchTime;
#endif

