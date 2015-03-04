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
#include "hash_table.cu.h"
#include "preclose_table.cu.h"
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

// Open/Close table
__device__ volatile OTable* g_otable;

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

__device__ volatile void* g_stagingArea[RW_HOST_WORKERS][RW_SCRATCH_PER_WORKER];

__global__ void init_fs(volatile CPU_IPC_OPEN_Queue* _ipcOpenQueue, 
			volatile CPU_IPC_RW_Queue* _ipcRWQueue, 
			volatile GPU_IPC_RW_Manager* _ipcRWManager, 
			volatile OTable* _otable, 
			volatile PPool* _ppool, 
			volatile Page* _rawStorage,
			volatile FTable* _ftable,
			volatile HashMap* _hashMap,
			volatile preclose_table* _preclose_table,
			volatile void** _stagingArea)
{
	g_cpu_ipcOpenQueue=_ipcOpenQueue;
	g_cpu_ipcRWQueue=_ipcRWQueue;

	g_ipcRWManager=_ipcRWManager;
	g_ipcRWManager->init_thread();

	g_otable=_otable; 
	g_otable->init_thread();

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
			g_stagingArea[i][j] = _stagingArea[i * RW_SCRATCH_PER_WORKER + j];
		}
	}

	INIT_ALL_STATS
	INIT_TIMING_STATS
	//INIT_DEBUG
}

//void* rw_task( void* param )
//{
//	TaskData* taskData = (TaskData*)param;
//
//	int id = taskData->id;
//	volatile GPUGlobals* globals = taskData->gpuGlobals;
//	volatile int* pDone = &(globals->done);
//
//	int firstSlot = id * RW_SLOTS_PER_WORKER;
//
//	// Dummy call to initialize cuda context
//	CUDA_SAFE_CALL(
//		cudaStreamSynchronize( globals->streamMgr->memStream[id] ));
//
//	while( !(*pDone) )
//	{
//		size_t scratchSize = 0;
//		int numRequests = 0;
//		int activeSlots[RW_SLOTS_PER_WORKER] = {0};
//
//		for( int i = 0; i < RW_SLOTS_PER_WORKER; i++ )
//		{
//			volatile CPU_IPC_RW_Entry* e = &globals->cpu_ipcRWQueue->entries[firstSlot + i];
//			if (e->status == CPU_IPC_PENDING)
//			{
//				int req_cpu_fd = e->cpu_fd;
//				size_t req_buffer_offset = e->buffer_offset;
//				size_t req_file_offset = e->file_offset;
//				size_t req_size = e->size;
//				int req_type = e->type;
//
//				assert(
//						req_type == RW_IPC_READ || req_type == RW_IPC_WRITE
//								|| req_type == RW_IPC_DIFF
//								|| req_type == RW_IPC_TRUNC);
//
//				if (req_type != RW_IPC_TRUNC)
//				{
//					assert(req_cpu_fd >= 0 && req_size > 0);
//				}
//
//				switch (req_type)
//				{
//				case RW_IPC_READ:
//				{
//					// read
//					int cpu_read_size = 0;
//
//					nvtxRangePushA("pread");
//					cpu_read_size = pread(req_cpu_fd,
//							globals->streamMgr->scratch[id] + scratchSize, req_size,
//							req_file_offset);
//					nvtxRangePop();
//
//					e->return_size = cpu_read_size;
//					e->return_offset = scratchSize;
//					__sync_synchronize();
//
//					if (cpu_read_size > 0)
//					{
////						scratchOffsets[id][numRequests] = scratchSize;
////						bufferOffsets[id][numRequests] = req_buffer_offset;
////						writeSizes[id][numRequests] = cpu_read_size;
//
//						activeSlots[numRequests] = firstSlot + i;
//
//						numRequests++;
//
//						scratchSize += cpu_read_size;
//					}
//				}
//					break;
//
//				default:
//					assert(NULL);
//				}
//			}
//		}
//
//		if( 0 == scratchSize )
//		{
//			// Didn't find any request, move on
//			continue;
//		}
//
////		asyncMemCpyCount[id]++;
////		asyncMemCpySize[id] += scratchSize;
////		asyncMemCpyTime[id] -= _timestamp();
//
////		pthread_mutex_lock( &rwLoopTasksLocks[id] );
//
////		writeRequests[id] = scratchSize;
//
////		pthread_cond_wait( &rwLoopTasksConds[id], &rwLoopTasksLocks[id]);
////		pthread_mutex_unlock( &rwLoopTasksLocks[id] );
//
//		CUDA_SAFE_CALL(
//			cudaMemcpyAsync( ( (char* ) globals->stagingArea[id] ),
//					         globals->streamMgr->scratch[id],
//					         scratchSize,
//					         cudaMemcpyHostToDevice,
//					         globals->streamMgr->memStream[id] ) );
//
////		while( 0 != writeRequests[id] )
////		{
////			usleep( 1 );
////		}
//
//		CUDA_SAFE_CALL(
//				cudaStreamSynchronize( globals->streamMgr->memStream[id] ));
//
////		asyncMemCpyTime[id] += _timestamp();
//
//		// complete the request
//		for( int i = 0; i < numRequests; ++i )
//		{
//			globals->cpu_ipcRWQueue->entries[activeSlots[i]].status = CPU_IPC_READY;
//
////			scratchOffsets[id][i] = 0;
////			bufferOffsets[id][i] = 0;
////			writeSizes[id][i] = 0;
//
//			activeSlots[i] = 0;
//		}
//
//		__sync_synchronize();
//	}
//
//	return NULL;
//}

typedef volatile GPUGlobals* GPUGlobals_ptr;

void initializer(GPUGlobals_ptr* globals)
{
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	*globals=new GPUGlobals();

	volatile void** temp;
	CUDA_SAFE_CALL(cudaMalloc(&temp,sizeof(void*) * RW_HOST_WORKERS * RW_SCRATCH_PER_WORKER));

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
		{
			CUDA_SAFE_CALL(
					cudaMemcpy(&(temp[i * RW_SCRATCH_PER_WORKER + j]), (void*)&((*globals)->stagingArea[i][j]), sizeof(void*), cudaMemcpyHostToDevice) );
		}
	}


// this must be done from a single thread!
	init_fs<<<1,1>>>((*globals)->cpu_ipcOpenQueue,
                        (*globals)->cpu_ipcRWQueue,
                        (*globals)->ipcRWManager,
                        (*globals)->otable,
                        (*globals)->ppool,
                        (*globals)->rawStorage,
                        (*globals)->ftable,
                        (*globals)->hashMap,
			            (*globals)->_preclose_table,
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
