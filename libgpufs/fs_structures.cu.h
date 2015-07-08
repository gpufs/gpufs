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

#ifndef FS_STRUCTURES_CU_H
#define FS_STRUCTURES_CU_H

#include "fs_constants.h"

struct PFrame
{
private:
	int 			lock;

public:
	enum State {INVALID, INIT, VALID};

	volatile int 	refCount;
	volatile State 	state;

	volatile Page* 	page;
	volatile uint 	rs_offset;
	volatile uint 	file_id;
	volatile size_t file_offset;
	volatile uint 	content_size;
	volatile uint 	dirty;
	volatile int 	dirtyCounter;

	volatile PFrame* 	next;
	volatile PFrame* 	nextDirty;

	__device__ void init_thread( volatile Page* _page, int _rs_offset ) volatile;
	__device__ void clean() volatile;

	__device__ bool try_lock_init() volatile;
	__device__ void unlock_init() volatile;

	__device__ bool try_lock_rw( int fd, size_t offset ) volatile;
	__device__ void unlock_rw() volatile;

	__device__ bool try_invalidate( int fd, size_t offset ) volatile;

	__device__ void markDirty() volatile;
};

struct DirtyList
{
private:
	int 				_lock;

public:

	volatile size_t 	count;
	volatile PFrame* 	head;

	__device__ void init_thread() volatile;
	__device__ void clean() volatile;

	__device__ void lock() volatile;
	__device__ bool try_lock() volatile;
	__device__ void unlock() volatile;
};

struct FTable_entry
{
	volatile char filename[FILENAME_SIZE];
	volatile int file_id;
	volatile int status;
	volatile int refCount;
	volatile int cpu_fd;
	volatile size_t size;
	volatile int flags;
	volatile unsigned int cpu_inode;
	volatile int did_open;

	volatile int drop_cache;
	volatile int dirty;
	double cpu_timestamp;

	DirtyList dirtyList;

	__device__ void init_thread() volatile;

	__device__ void init( volatile const char* _filename, int _flags ) volatile;

	__device__ void notify( int fd, int cpu_fd, unsigned int cpu_inode, size_t size, double timestamp, int _did_open ) volatile;

	__device__ void wait_open() volatile;

	__device__ void clean() volatile;
};


struct FTable
{
	volatile FTable_entry files[FSTABLE_SIZE];
	int _lock;

	__device__ void lock() volatile;

	__device__ void unlock() volatile;

	__device__ int findEntry( const volatile char* filename, volatile bool* isNewEntry, int o_flags ) volatile;

	__device__ void init_thread() volatile;
};

#endif
