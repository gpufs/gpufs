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


#ifndef FS_INITIALIZER_H
#define  FS_INITIALIZER_H


//#ifndef MAIN_FS_FILE
//#error "This file must be included in the fs.cu"
//#endif

#include <stdio.h>

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "fs_globals.cu.h"
#include "gpufs_con_lib.h"

#define  initGpuShmemPtr(T, h_ptr,symbol)\
{\
 	CUDA_SAFE_CALL(cudaHostAlloc((void**)&(h_ptr), sizeof(T), cudaHostAllocMapped));\
	memset((void*)h_ptr,0,sizeof(T));\
	void* d_ptr;\
	CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)(&d_ptr), (void*)(h_ptr), 0));\
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((symbol),&d_ptr,sizeof(void*)));\
}

#define initGpuGlobals(T,d_ptr,symbol)\
{\
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_ptr),sizeof(T)));\
	CUDA_SAFE_CALL(cudaMemset((void*)d_ptr,0,sizeof(T)));\
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((symbol),&(d_ptr),sizeof(void*)));\
}

struct GPUStreamManager
{
// GPU streams 	
	GPUStreamManager(){
		
		CUDA_SAFE_CALL(cudaStreamCreate(&kernelStream));
		CUDA_SAFE_CALL(cudaStreamCreate(&async_close_stream));
		CUDA_SAFE_CALL(cudaHostAlloc(&async_close_scratch, sizeof(Page),cudaHostAllocDefault));

		for(int i=0;i<RW_HOST_WORKERS;i++)
		{
			CUDA_SAFE_CALL(cudaStreamCreate(&memStream[i]));

			for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
			{
				CUDA_SAFE_CALL(cudaHostAlloc(&scratch[i][j], FS_BLOCKSIZE * (RW_IPC_SIZE / RW_HOST_WORKERS), cudaHostAllocDefault));
			}
		}
		
	}

	~GPUStreamManager(){
		CUDA_SAFE_CALL(cudaStreamDestroy(kernelStream));
		for(int i=0;i<RW_HOST_WORKERS;i++){
			CUDA_SAFE_CALL(cudaStreamDestroy(memStream[i]));
			for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
			{
				CUDA_SAFE_CALL(cudaFreeHost(scratch[i][j]));
			}
		}
		CUDA_SAFE_CALL(cudaStreamDestroy(async_close_stream));
		CUDA_SAFE_CALL(cudaFreeHost(async_close_scratch));
	}
		
	cudaStream_t kernelStream;
	cudaStream_t memStream[RW_HOST_WORKERS];
	cudaStream_t async_close_stream;
	Page* async_close_scratch;

	uchar* scratch[RW_HOST_WORKERS][RW_SCRATCH_PER_WORKER];
};

struct GPUGlobals;

struct TaskData
{
	int gpuid;
	int id;
	volatile GPUGlobals* gpuGlobals;
};

#define getStagingAreaOffset(stagingArea, i, j) \
	(stagingArea + i * RW_SCRATCH_PER_WORKER * FS_BLOCKSIZE * RW_SLOTS_PER_WORKER + j * FS_BLOCKSIZE * RW_SLOTS_PER_WORKER)

struct GPUGlobals{
	volatile CPU_IPC_OPEN_Queue* cpu_ipcOpenQueue;
	
	volatile CPU_IPC_RW_Queue* cpu_ipcRWQueue;

	volatile CPU_IPC_RW_Flags* cpu_ipcRWFlags;

// RW GPU manager
	GPU_IPC_RW_Manager* ipcRWManager;
// Memory pool
	PPool* ppool;
// File table with block pointers
	FTable* ftable;
// Raw memory pool
	Page* rawStorage;
// Device memory staging area
	uchar* stagingArea;
// gpufs device file decsriptor
    int gpufs_fd;
// hashMap for the buffer cache frames
	HashMap* hashMap;

// async close ringbuffer
 	async_close_rb_t* async_close_rb;
 	async_close_rb_t* async_close_rb_gpu;

// Streams
	GPUStreamManager* streamMgr;

	GPUGlobals()
	{
		initGpuShmemPtr(CPU_IPC_OPEN_Queue,cpu_ipcOpenQueue,g_cpu_ipcOpenQueue);
		cpu_ipcOpenQueue->init_host();

		initGpuShmemPtr(CPU_IPC_RW_Queue,cpu_ipcRWQueue,g_cpu_ipcRWQueue);
		cpu_ipcRWQueue->init_host();

		initGpuShmemPtr(CPU_IPC_RW_Flags,cpu_ipcRWFlags,g_cpu_ipcRWFlags);
		cpu_ipcRWFlags->init_host();

		initGpuGlobals(GPU_IPC_RW_Manager,ipcRWManager,g_ipcRWManager);
		initGpuGlobals(PPool,ppool,g_ppool);
		initGpuGlobals(FTable,ftable,g_ftable);
		initGpuGlobals(HashMap,hashMap,g_hashMap);
	//	initGpuGlobals(preclose_table,_preclose_table,g_preclose_table);
	
		CUDA_SAFE_CALL(cudaMalloc(&rawStorage,sizeof(Page)*PPOOL_FRAMES));
		CUDA_SAFE_CALL(cudaMemset(rawStorage,0,sizeof(Page)*PPOOL_FRAMES));

		async_close_rb=new async_close_rb_t();
		async_close_rb->init_host();
		CUDA_SAFE_CALL(cudaMalloc(&async_close_rb_gpu, sizeof(async_close_rb_t)));
		CUDA_SAFE_CALL(cudaMemcpy(async_close_rb_gpu,async_close_rb,sizeof(async_close_rb_t),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_async_close_rb,&async_close_rb_gpu,sizeof(void*))); 
		
		CUDA_SAFE_CALL(cudaMalloc(&stagingArea,
				sizeof(uchar) * RW_HOST_WORKERS * RW_SCRATCH_PER_WORKER * FS_BLOCKSIZE * RW_SLOTS_PER_WORKER));

		streamMgr=new GPUStreamManager();
		gpufs_fd=-1;
		if (getenv("USE_GPUFS_DEVICE")){
			gpufs_fd=gpufs_open(GPUFS_DEV_NAME);
	                if (gpufs_fd<0) {
	                        perror("gpufs_open failed");
			}
                }else{
//			fprintf(stderr,"Warning: GPUFS device was not enabled through USE_GPUFS_DEVICE environment variable\n");
		}
	}
	
	~GPUGlobals()
	{
		if (gpufs_fd>=0){
		  if ( gpufs_close(gpufs_fd))
	          {
        	          perror("gpufs_close failed");
                  }
		}
		cudaFreeHost((void*)cpu_ipcOpenQueue);
		cudaFreeHost((void*)cpu_ipcRWQueue);
		
		cudaFree(ipcRWManager);
		cudaFree(ppool);
		cudaFree(ftable);
		cudaFree(rawStorage);
		cudaFree(hashMap);
		cudaFree(async_close_rb_gpu);
		delete streamMgr;
	}
};
typedef volatile GPUGlobals* GPUGlobals_ptr;

void initializer(GPUGlobals_ptr* globals);
#endif // FS_INITIALIZER_H
