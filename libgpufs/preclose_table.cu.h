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


#ifndef PRECLOSE_TABLE_CU_H
#define PRECLOSE_TABLE_CU_H
#include "fs_constants.h"
#include "fs_globals.cu.h"
struct preclose_node
{
	volatile int occupied;
	volatile FTable_entry f;
	volatile OTable_entry o;	
};



struct preclose_table
{
	volatile preclose_node entries[MAX_NUM_PRECLOSE_FILES];
	int _lock;
	volatile int size;
	__device__ void lock() volatile
	{
		MUTEX_LOCK(_lock);
	}
	__device__ void unlock() volatile
	{
		__threadfence();
		MUTEX_UNLOCK(_lock);
	}

	__device__ void init_thread() volatile;

	__device__ int findEntry(volatile char*filename, volatile FTable_entry* _new_f, volatile OTable_entry* _new_o) volatile;
	
	__device__ int add(volatile FTable_entry* _old_f, volatile OTable_entry* _old_o) volatile;
	
};
	
#endif
