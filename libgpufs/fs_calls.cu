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

#ifndef FS_CALLS_CU
#define FS_CALLS_CU

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "hash_table.cu.h"
#include "swapper.cu.h"
#include "fs_globals.cu.h"
#include "preclose_table.cu.h"
#include "fs_calls.cu.h"
// no reference counting here

// we must have multiple threads otherwise it 
DEBUG_NOINLINE __device__ int gclose( int fd )
{
	return -1;
}

DEBUG_NOINLINE __device__ int single_thread_open( char* filename, int flags )
{
	/*
	 Lock ftable
	 find entry
	 increase ref-count
	 Unlock ftable
	 if not found -> ret E_FTABLE_FULL

	 if (new_entry) -> send CPU open req
	 else -> wait on CPU open req

	 if (req failed) ->
	 Lock ftable
	 dec ref_count
	 if last -> delete entry
	 unlock ftable
	 */

	g_otable->lock();
	bool isNewEntry = false;
	int fd = g_otable->findEntry( filename, &isNewEntry, flags );

	GPU_ASSERT(fd>=0);

	volatile OTable_entry* e = &g_otable->entries[fd];
	e->refCount++;
	__threadfence();
	g_otable->unlock();

	volatile CPU_IPC_OPEN_Entry* cpu_e = &( g_cpu_ipcOpenQueue->entries[fd] );

	if( isNewEntry )
	{
		unsigned int cpu_inode = 0;
		double timestamp = 0;

		cpu_e->open( filename, flags, DO_OPEN );

		cpu_inode = readNoCache( &cpu_e->cpu_inode );
		timestamp = readNoCache( &cpu_e->cpu_timestamp );

		g_ftable->files[fd].file_id = fd;
		g_ftable->files[fd].cpu_timestamp = timestamp;

		int cpu_fd = readNoCache( &cpu_e->cpu_fd );
		size_t size = readNoCache( &cpu_e->size );
		e->notify( cpu_fd, cpu_inode, size, 1 );

//		PRINT("File \"%s\" opened. fd=%d, cpu_fd=%d, size=%ld \n", filename, fd, cpu_fd, size);
	}
	else
	{
		e->wait_open();
	}

	return fd;
}

DEBUG_NOINLINE __device__ int gopen( char* filename, int flags )
{
	__shared__ int ret;
	BEGIN_SINGLE_THREAD
		ret = single_thread_open( filename, flags );
	END_SINGLE_THREAD;
	return ret;
}

__device__ int gLock;

#define READ 0
#define WRITE 1
DEBUG_NOINLINE __device__ volatile PFrame* getRwLockedPage( int fd, size_t block_id, int cpu_fd, int type_req )
{
	__shared__ volatile PFrame* pframe;

	BEGIN_SINGLE_THREAD

		RT_SEARCH_START

//		MUTEX_LOCK( gLock );

		pframe = NULL;

		while( NULL == pframe )
		{
			bool busy;

			pframe = g_hashMap->readPFrame( fd, block_id, busy );
			if( pframe != NULL && !busy )
			{
				HM_LOCKLESS
				break;
			}

			while( busy )
			{
				RT_WAIT_START
				pframe = g_hashMap->readPFrame( fd, block_id, busy );
				RT_WAIT_STOP
			}

			if( pframe != NULL && !busy )
			{
				HM_LOCKLESS
				break;
			}

			pframe = g_hashMap->getPFrame( fd, block_id );
			if( pframe != NULL )
			{
				HM_LOCKED
				break;
			}
		}
//PRINT("Got frame pframe=%p, state=%d\n", pframe, pframe->state);

//		MUTEX_UNLOCK( gLock );
		RT_SEARCH_STOP

	END_SINGLE_THREAD

		volatile __shared__ int entry;
		entry = -1;

	BEGIN_SINGLE_THREAD

		PAGE_READ_START
		if( pframe->state == PFrame::INIT )
		{
			// if we inited, the page is locked and we just keep going

			// check that the file has been opened
			volatile OTable_entry* e = &g_otable->entries[fd];

			FILE_OPEN_START
			if( readNoCache( &e->did_open ) == 0 || readNoCache( &e->did_open ) == 2 )
			{
				// mutual exclusion for concurrent openers
				int winner = atomicExch( (int*) &e->did_open, 2 );
				if( winner == 0 )
				{
					volatile CPU_IPC_OPEN_Entry* cpu_e = &( g_cpu_ipcOpenQueue->entries[fd] );
					e->cpu_fd = cpu_e->reopen();
					__threadfence();
					e->did_open = 1;
					__threadfence();
					GPU_ASSERT(e->cpu_fd>=0);
				}
				else
				{
					WAIT_ON_MEM( e->did_open, 1 );
				}
				cpu_fd = e->cpu_fd;
			}
			FILE_OPEN_STOP

			// cpu-fd would be less than 0 if we are opening write_once file	

			if( cpu_fd >= 0 )
			{
				int tEntry = -1;
				int datasize = read_cpu( cpu_fd, pframe, tEntry );
				if( datasize < 0 )
				{
					// TODO: error handling
					GPU_ASSERT("Failed to read data from CPU"==NULL);
				}
				pframe->content_size = datasize;
				entry = tEntry;
			}

		} PAGE_READ_STOP

		GPU_ASSERT( (pframe->state == PFrame::INIT) || ((pframe->state == PFrame::VALID) && pframe->refCount>0) );
		// if we do not need to zero out the page (cpu_fd<0)
	END_SINGLE_THREAD

		if( -1 != entry )
		{

			COPY_BLOCK_START

			volatile CPU_IPC_RW_Entry* e = &(g_cpu_ipcRWQueue->entries[entry]);
			int workerID = entry / RW_SLOTS_PER_WORKER;
//
//			int return_size = readNoCache(&(e->return_size));
//			int return_offset = readNoCache(&(e->return_offset));

//			PRINT( "entry: %8d workerID: %8d return_size: %8d return_offset: %8d\n", entry, workerID, e->return_size, e->return_offset );
			copy_block( (uchar*)pframe->page, ((uchar*)g_stagingArea[workerID][e->scratch_index]) + e->return_offset, e->return_size );

			__syncthreads();
			COPY_BLOCK_STOP
		}


	BEGIN_SINGLE_THREAD

		if( -1 != entry )
		{
			freeEntry( entry );

			int workerID = entry / RW_SLOTS_PER_WORKER;
			int scratchID = g_cpu_ipcRWQueue->entries[entry].scratch_index;

			atomicSub( (int*)&(g_cpu_ipcRWFlags->entries[workerID][scratchID]), 1 );
			__threadfence_system();

		}

		// if the page was initialized, return. Make sure to return with all threads active
		if( ( pframe->state == PFrame::INIT ) && cpu_fd >= 0 )
			pframe->unlock_init();

	END_SINGLE_THREAD
	if( ( pframe->state == PFrame::INIT && cpu_fd >= 0 ) || pframe->state == PFrame::VALID )
		return pframe;

	//fill the page with zeros - optimization for the case of write-once exclusive create owned by GPU
	bzero_page( (volatile char*) pframe->page );
	__threadfence(); // make sure all threads will see these zeros

	BEGIN_SINGLE_THREAD
		GPU_ASSERT(cpu_fd<0);

		GPU_ASSERT(pframe->state == PFrame::INIT);
		pframe->content_size = 0;
		pframe->unlock_init();

	END_SINGLE_THREAD

	return pframe;
}

DEBUG_NOINLINE __device__ int gmunmap( volatile void *addr, size_t length )
{
	size_t tmp = ( (char*) addr ) - ( (char*) g_ppool->rawStorage );
	size_t offset = tmp >> FS_LOGBLOCKSIZE;
	if( offset >= PPOOL_FRAMES )
		return -1;

	__threadfence(); // make sure all writes to the page become visible
	BEGIN_SINGLE_THREAD

		volatile PFrame* p = &( g_ppool->frames[offset] );
		p->unlock_rw();

	END_SINGLE_THREAD

	return 0;
}

DEBUG_NOINLINE __device__ volatile void* gmmap( void *addr, size_t size, int prot, int flags, int fd, off_t offset )
{
	__shared__ volatile PFrame* pframe; // the ptr is to global mem but is stored in shmem
	__shared__ size_t block_id;
	__shared__ int block_offset;

	__shared__ int cpu_fd;
	BEGIN_SINGLE_THREAD
		block_id = offset2block( offset, FS_LOGBLOCKSIZE );
		block_offset = offset2blockoffset( offset, FS_BLOCKSIZE );

		GPU_ASSERT(fd>=0 && fd<MAX_NUM_FILES);

		cpu_fd = g_otable->entries[fd].cpu_fd;
		GPU_ASSERT( g_otable->entries[fd].refCount >0 );

		if( block_offset + size > FS_BLOCKSIZE )
			GPU_ASSERT("Reading beyond the  page boundary"==0);

		// decide whether to fetch data or not
		if( g_otable->entries[fd].flags == O_GWRONCE )
			cpu_fd = -1;

	END_SINGLE_THREAD

	int purpose = ( g_otable->entries[fd].flags == O_GRDONLY ) ? PAGE_READ_ACCESS : PAGE_WRITE_ACCESS;
	pframe = getRwLockedPage( fd, block_id, cpu_fd, purpose );

	BEGIN_SINGLE_THREAD
	// page inited, just read, frame us a _shared_ mem variable

	// increase dirty counter if mapped as write
	if( purpose == PAGE_WRITE_ACCESS )
		atomicAdd( (int*) &pframe->dirtyCounter, 1 );

	//TODO: handle reading beyond eof
	if( pframe->content_size < block_offset + size && flags == O_GRDONLY )
	{
		GPU_ASSERT("Failed to map beyond the end of file"!=NULL);
	}

	if( flags != O_GRDONLY )
		atomicMax( (uint*) &( pframe->content_size ), block_offset + size );

	END_SINGLE_THREAD
	GPU_ASSERT(pframe!=NULL);

	return (void*) ( ( (uchar*) ( pframe->page ) ) + block_offset );
}

#endif
