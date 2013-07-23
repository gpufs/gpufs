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

#endif

