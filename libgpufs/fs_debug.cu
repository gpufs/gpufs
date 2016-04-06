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

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <pthread.h>

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
__device__ unsigned long long PageSearchTime;
__device__ unsigned long long PageSearchWaitTime;
__device__ unsigned long long MapTime;
__device__ unsigned long long CopyBlockTime;
__device__ unsigned long long PageReadTime;
__device__ unsigned long long PageAllocTime;
__device__ unsigned long long FileOpenTime;
__device__ unsigned long long FileCloseTime;
__device__ unsigned long long CPUReadTime;
__device__ unsigned long long BusyListInsertTime;
__device__ unsigned long long EvictTime;
__device__ unsigned long long EvictLockTime;
#endif

#ifdef DEBUG

#define DBG_MAX_STRING_LEN 20
#define DBG_MAX_FNAME_LEN 60
#define DBG_MAX_FUNC_LEN 20

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

struct gdebug_t {
    int valid;
    long int line;
    void* ptr;
    size_t data;
    char s[DBG_MAX_STRING_LEN];
    char fname[DBG_MAX_FNAME_LEN];
    char func[DBG_MAX_FUNC_LEN];
};

           volatile struct gdebug_t *_hdbg;
__device__ volatile struct gdebug_t *_gdbg;
__device__ volatile int             _gdbg_mutex;

__host__ void *gdebug_loop(void *arg) {
    printf("debug loop awake\n");
    while (1) {
        if (_hdbg->valid) {
        	if( ((size_t)-1 != (size_t)_hdbg->ptr) && ((size_t)-1 != _hdbg->data) && ('\0' != _hdbg->s[0]) )
        	{
        		// GDBG
        		printf("DBG :: File = %-60s :: Func = %-20s :: Line = %-6ld :: %-20s :: ptr = 0x%08lx :: val = %-6ld :: hex = 0x%08lx\n",
                    _hdbg->fname, _hdbg->func, _hdbg->line, _hdbg->s, _hdbg->ptr, _hdbg->data, _hdbg->data);
        	}
        	else if( ((size_t)-1 != _hdbg->data) && ('\0' != _hdbg->s[0]) )
			{
        		// GDBGV
				printf("DBG :: File = %-60s :: Func = %-20s :: Line = %-6ld :: %-20s :: val = %-6ld :: hex = 0x%08lx\n",
					_hdbg->fname, _hdbg->func, _hdbg->line, _hdbg->s, _hdbg->data, _hdbg->data);
			}
        	else if( ('\0' != _hdbg->s[0]) )
			{
        		// GDBGS
				printf("DBG :: File = %-60s :: Func = %-20s :: Line = %-6ld :: %-20s\n",
					_hdbg->fname, _hdbg->func, _hdbg->line, _hdbg->s);
			}
        	else
        	{
        		// GDBGL
				printf("DBG :: File = %-60s :: Func = %-20s :: Line = %-6ld\n",
					_hdbg->fname, _hdbg->func, _hdbg->line);
        	}

            fflush(stdout);
            memset((void *)_hdbg->s, 0, DBG_MAX_STRING_LEN);
            memset((void *)_hdbg->fname, 0, DBG_MAX_FNAME_LEN);
            memset((void *)_hdbg->func, 0, DBG_MAX_FUNC_LEN);
            __sync_synchronize();
            _hdbg->valid = 0;
            __sync_synchronize();
        }
    }
}

__host__ void _gdebug_init(void) {
    long int *gptr;
    int zero = 0;
    checkCudaErrors( cudaMallocHost(&_hdbg, sizeof(volatile struct gdebug_t)) );
    memset((void *)_hdbg, 0, sizeof(struct gdebug_t));
    checkCudaErrors( cudaHostGetDevicePointer(&gptr, (void *)_hdbg, 0) );
    checkCudaErrors( cudaMemcpyToSymbol(_gdbg, &gptr, sizeof(volatile struct gdebug_t *), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(_gdbg_mutex, &zero, sizeof(int), 0, cudaMemcpyHostToDevice) );
    __sync_synchronize();
    pthread_t thread;
    pthread_create(&thread, NULL, gdebug_loop, NULL);
}

__device__ void _dbg(const char *s,
                       void* ptr,
                       size_t v,
                       long int t,
                       long int l,
                       const char *fname,
                       const char *func) {
    if (threadIdx.x == t) {
        int retry = 1;
        while (retry) {
            int old;
            old = atomicExch((int *)&_gdbg_mutex, 1);
            if (old == 0) {
                if (!_gdbg->valid) {
                    _gdbg->line = l;
                    _gdbg->data = v;
                    _gdbg->ptr = ptr;
                    for (int i = 0; i < DBG_MAX_STRING_LEN; i++) {
                        if (s[i] == '\0')
                            break;
                        _gdbg->s[i] = s[i];
                    }
                    for (int i = 0; i < DBG_MAX_FNAME_LEN; i++) {
                        if (fname[i] == '\0')
                            break;
                        _gdbg->fname[i] = fname[i];
                    }
                    for (int i = 0; i < DBG_MAX_FUNC_LEN; i++) {
                        if (func[i] == '\0')
                            break;
                        _gdbg->func[i] = func[i];
                    }
                    __threadfence_system();
                    _gdbg->valid = 1;
                    __threadfence_system();
                    retry = 0;
                }
                atomicExch((int *)&_gdbg_mutex,0);
            }
        }
    }
}

#endif

