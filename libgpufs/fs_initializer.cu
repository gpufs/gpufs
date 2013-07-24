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

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "hash_table.cu.h"
#include "radix_tree.cu.h"
#include "preclose_table.cu.h"
#include "fs_globals.cu.h"
#include "async_ipc.cu.h"

#include "fs_initializer.cu.h"


/************GLOBALS********/
// CPU Write-shared memory //
__device__ volatile CPU_IPC_OPEN_Queue* g_cpu_ipcOpenQueue;
__device__ volatile CPU_IPC_RW_Queue* g_cpu_ipcRWQueue;
//
// manager for rw RPC queue

__device__ volatile GPU_IPC_RW_Manager* g_ipcRWManager;

// Open/Close table
__device__ volatile OTable* g_otable;
// Memory pool
__device__ volatile PPool* g_ppool;
// File table with block pointers
__device__ volatile FTable* g_ftable;

// Radix tree memory pool for rt_nodes
__device__ volatile rt_mempool g_rtree_mempool;

// Hash table with all the previously opened files indexed by their inodes
__device__ volatile hash_table g_closed_ftable;

// file_id uniq counter
__device__ int g_file_id;

// a ring buffer for write back
__device__ async_close_rb_t* g_async_close_rb;



__global__ void init_fs(volatile CPU_IPC_OPEN_Queue* _ipcOpenQueue, 
			volatile CPU_IPC_RW_Queue* _ipcRWQueue, 
			volatile GPU_IPC_RW_Manager* _ipcRWManager, 
			volatile OTable* _otable, 
			volatile PPool* _ppool, 
			volatile Page* _rawStorage,
			volatile FTable* _ftable,
			volatile void* _rtree_raw_store,
			rtree*volatile _rtree_array,
			volatile preclose_table* _preclose_table)
{

	
	g_cpu_ipcOpenQueue=_ipcOpenQueue;
	g_cpu_ipcRWQueue=_ipcRWQueue;

	g_ipcRWManager=_ipcRWManager;
	g_ipcRWManager->init_thread();

	g_otable=_otable; 
	g_otable->init_thread();

	g_ppool=_ppool;
	g_ppool->init_thread(_rawStorage);
	
	g_rtree_mempool.init_thread(_rtree_raw_store,sizeof(rt_node));
	
	g_ftable=_ftable;
	for(int i=0;i<MAX_NUM_FILES+MAX_NUM_CLOSED_FILES;i++)
	{
		_rtree_array[i].init_thread();
	}	
	// this takes MAX_FILES from _rtree_array 
	g_ftable->init_thread(_rtree_array);

	g_closed_ftable.init_thread(_rtree_array+ MAX_NUM_FILES);

	
	g_file_id=0;
	INIT_ALL_STATS
	//INIT_DEBUG
}


typedef volatile GPUGlobals* GPUGlobals_ptr;

void initializer(GPUGlobals_ptr* globals)
{
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	*globals=new GPUGlobals();

// this must be done from a single thread!
	init_fs<<<1,1>>>((*globals)->cpu_ipcOpenQueue,
                        (*globals)->cpu_ipcRWQueue,
                        (*globals)->ipcRWManager,
                        (*globals)->otable,
                        (*globals)->ppool,
                        (*globals)->rawStorage,
                        (*globals)->ftable,
			(*globals)->rtree_pool,
		 	(*globals)->rtree_array,
			(*globals)->_preclose_table);
	
	cudaThreadSynchronize();
	CUDA_SAFE_CALL(cudaPeekAtLastError());
}
