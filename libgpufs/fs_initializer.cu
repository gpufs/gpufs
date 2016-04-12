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




//#ifndef MAIN_FS_FILE
//#error "This file must be included in the fs.cu"
//#endif

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "fs_globals.cu.h"
#include "async_ipc.cu.h"

#include "fs_initializer.cu.h"

#include <nvToolsExt.h>


/************GLOBALS********/
// CPU Write-shared memory //
__device__ volatile CPU_IPC_OPEN_Queue* g_cpu_ipcOpenQueue;
__device__ volatile CPU_IPC_RW_Queue* g_cpu_ipcRWQueue;
__device__ volatile CPU_IPC_RW_Flags* g_cpu_ipcRWFlags;
//
// manager for rw RPC queue

__device__ volatile GPU_IPC_RW_Manager* g_ipcRWManager;

// Memory pool
__device__ volatile PPool* g_ppool;

// File table with block pointers
__device__ volatile FTable* g_ftable;

// Hash table with all the previously opened files indexed by their inodes
//__device__ volatile hash_table g_closed_ftable;

// HashMap with mapping from <fd, offset> to pframes
__device__ volatile HashMap* g_hashMap;

// file_id uniq counter
__device__ int g_file_id;

// a ring buffer for write back
__device__ async_close_rb_t* g_async_close_rb;

__device__ volatile uchar* g_stagingArea[RW_HOST_WORKERS][RW_SCRATCH_PER_WORKER];

__global__ void init_fs(volatile CPU_IPC_OPEN_Queue* _ipcOpenQueue, 
			volatile CPU_IPC_RW_Queue* _ipcRWQueue, 
			volatile GPU_IPC_RW_Manager* _ipcRWManager, 
			volatile PPool* _ppool, 
			volatile Page* _rawStorage,
			volatile FTable* _ftable,
			volatile HashMap* _hashMap,
			volatile void** _stagingArea)
{
	g_cpu_ipcOpenQueue=_ipcOpenQueue;
	g_cpu_ipcRWQueue=_ipcRWQueue;

	g_ipcRWManager=_ipcRWManager;
	g_ipcRWManager->init_thread();

	g_ppool=_ppool;
	g_ppool->init_thread(_rawStorage);
	
	g_hashMap=_hashMap;
	g_hashMap->init_thread();

	g_ftable=_ftable;
	
	g_file_id=0;

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
		{
			g_stagingArea[i][j] = (volatile uchar*)_stagingArea[i * RW_SCRATCH_PER_WORKER + j];
		}
	}


	INIT_ALL_STATS
	INIT_TIMING_STATS
	//INIT_DEBUG
}

typedef volatile GPUGlobals* GPUGlobals_ptr;

void initializer(GPUGlobals_ptr* globals)
{
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	*globals=new GPUGlobals();

	//ssd_init( (*globals)->stagingArea, sizeof(uchar) * RW_HOST_WORKERS * RW_SCRATCH_PER_WORKER * FS_BLOCKSIZE * RW_SLOTS_PER_WORKER );

	volatile void** temp;
	CUDA_SAFE_CALL(cudaMalloc(&temp,sizeof(void*) * RW_HOST_WORKERS * RW_SCRATCH_PER_WORKER));

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
		{
			void* blockAddress = getStagingAreaOffset((*globals)->stagingArea, i, j);

			CUDA_SAFE_CALL(
					cudaMemcpy(&(temp[i * RW_SCRATCH_PER_WORKER + j]), &blockAddress, sizeof(void*), cudaMemcpyHostToDevice) );
		}
	}


// this must be done from a single thread!
	init_fs<<<1,1>>>((*globals)->cpu_ipcOpenQueue,
                        (*globals)->cpu_ipcRWQueue,
                        (*globals)->ipcRWManager,
                        (*globals)->ppool,
                        (*globals)->rawStorage,
                        (*globals)->ftable,
                        (*globals)->hashMap,
			            temp);
	
	cudaThreadSynchronize();
	CUDA_SAFE_CALL(cudaPeekAtLastError());

//	pthread_attr_t attr;
//	pthread_attr_init( &attr );
//	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
//
//	(*globals)->done = 0;
//
//	for( int i = 0; i < RW_HOST_WORKERS; ++i )
//	{
//		(*globals)->rwLoopTasksData[i].id = i;
//		(*globals)->rwLoopTasksData[i].gpuGlobals =  *globals;
//		(*globals)->rwLoopTasksData[i].gpuid = 0;
//
//		pthread_create( (pthread_t*)&((*globals)->rwLoopTasksIDs[i]), &attr, rw_task, (TaskData*)&((*globals)->rwLoopTasksData[i]) );
//	}
//
//	pthread_attr_destroy( &attr );
}
