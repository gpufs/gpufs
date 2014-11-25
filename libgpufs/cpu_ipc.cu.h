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


#ifndef CPU_IPC_H
#define CPU_IPC_H

#include "fs_structures.cu.h"

#define DO_NOT_OPEN 1
#define DO_OPEN 0


struct CPU_IPC_OPEN_Entry
{
	volatile int  status;
	volatile char filename[FILENAME_SIZE];
	volatile int cpu_fd;
	volatile size_t size;
	volatile unsigned int cpu_inode;
	volatile double cpu_timestamp;
	volatile int do_not_open;
	volatile unsigned int drop_residence_inode;
	volatile int return_value;
	volatile int flush_cache;
	volatile int flags;
	volatile bool is_dirty; // don't close right away, wait for asycn close

	__device__ void clean() volatile;

	__device__ int open(const char* reqFname, int flags, int do_not_open) volatile;
	__device__ int reopen() volatile;
	__device__ int unlink(const char* reqFname) volatile;
	__device__ int close(int cpu_fd, unsigned int _drop_residence_inode, bool _is_dirty, int did_open) volatile;
	

	__host__ void init_host() volatile;
	
};

// access to this struct is protected by the OTable. Threads wait there when being opened
struct CPU_IPC_OPEN_Queue
{
	volatile CPU_IPC_OPEN_Entry entries[FSTABLE_SIZE];
	__host__ void init_host() volatile;
};

// RW IPC queue

struct GPU_IPC_RW_Manager
{
	volatile int _locker[RW_IPC_SIZE];
	int _lock;
	__device__ int findEntry() volatile;
	__device__ void freeEntry(int entry) volatile;
	__device__ void init_thread() volatile;
};


struct CPU_IPC_RW_Entry
{
	volatile int status;
	volatile int cpu_fd;
	volatile size_t buffer_offset;
	volatile size_t file_offset;
	volatile uint size;
	volatile char type;
	volatile int return_value;
	
	
	__device__ void clean() volatile;

	__device__ int read_write(int _cpu_fd, size_t _buffer_offset, size_t _file_offset, uint _size,uint _type ) volatile;
	
	__device__ int ftruncate(int cpu_fd) volatile;

	__host__ void init_host() volatile;
};

struct CPU_IPC_RW_Queue
{
	volatile CPU_IPC_RW_Entry entries[RW_IPC_SIZE];

	__host__ void init_host() volatile;

	__device__ int  read_write_block(int cpu_fd,size_t  _buffer_offset, size_t _file_offset, int size,int type) volatile;
};

__device__ int read_cpu(int cpu_fd, volatile PFrame* frame);

__device__ int truncate_cpu(int cpu_fd);

#endif
