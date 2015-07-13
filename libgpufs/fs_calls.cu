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
#include "fs_globals.cu.h"
#include "fs_calls.cu.h"
#include "fat_pointer.cu.h"
// no reference counting here

// we must have multiple threads otherwise it 
DEBUG_NOINLINE __device__ int gclose( int fd )
{
	__shared__ volatile FTable_entry* file;
	__shared__ int res;

BEGIN_SINGLE_THREAD

	GPU_ASSERT(fd>=0);
	g_ftable->lock();
	file = &g_ftable->files[fd];
	file->refCount--;
	GPU_ASSERT(file->refCount>=0);
	res=0;

	if (file->refCount>0 || file->status!=FSENTRY_OPEN)
	{
		__threadfence();
		g_ftable->unlock();
		res=1;
	}

END_SINGLE_THREAD

	if (res==1) return 0;

	__shared__ volatile CPU_IPC_OPEN_Entry* cpu_e;
	bool was_dirty;

BEGIN_SINGLE_THREAD

	// lock in the opening thread
	file->status=FSENTRY_CLOSING; // this is not used in any place in the code.. but should have been
	cpu_e = &(g_cpu_ipcOpenQueue->entries[fd]);
	was_dirty=file->dirty;

END_SINGLE_THREAD

// we flush the pages with async queue
// we have this called by all the threads in a TB, otherwise the copy inside the traverse_all
// function will be very slow

	file->traverse_all_for_close();

// we do close now: we must hold a global lock on the file table
// because otherwise the thread which is opening a file will get
// a file handle for a closed file

BEGIN_SINGLE_THREAD

	res = cpu_e->close(file->cpu_fd, true /*drop_residence_inode*/, was_dirty);

	if (res<0) {
		GPU_ASSERT(false);
	}

	file->close();

	cpu_e->clean();
	__threadfence();
	g_ftable->unlock();

END_SINGLE_THREAD

	return res;
}

DEBUG_NOINLINE __device__ int single_thread_open( const char* filename, int flags )
{
	GPRINT("GPU: Open file: %s\n", filename);
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

	g_ftable->lock();
	bool isNewEntry = false;
	int fd = g_ftable->findEntry( filename, &isNewEntry, flags );

	GPU_ASSERT(fd>=0);

	volatile FTable_entry* e = &g_ftable->files[fd];
	e->refCount++;
	__threadfence();
	g_ftable->unlock();

	volatile CPU_IPC_OPEN_Entry* cpu_e = &( g_cpu_ipcOpenQueue->entries[fd] );

	if( isNewEntry )
	{
		double timestamp = 0;

		cpu_e->open( filename, flags, DO_OPEN );

		timestamp = readNoCache( &cpu_e->cpu_timestamp );
		int cpu_fd = readNoCache( &cpu_e->cpu_fd );
		size_t size = readNoCache( &cpu_e->size );

		e->notify( fd, cpu_fd, size, timestamp, 1 );
	}
	else if( e->status == FSENTRY_CLOSED )
	{
		// We found our file but it's close, try to reopen it
		double timestamp = 0;

		cpu_e->open( filename, flags, DO_OPEN );

		timestamp = readNoCache( &cpu_e->cpu_timestamp );
		int cpu_fd = readNoCache( &cpu_e->cpu_fd );
		size_t size = readNoCache( &cpu_e->size );

		if( timestamp == e->cpu_timestamp )
		{
			e->notify( fd, cpu_fd, size, timestamp, 1 );
		}
		else
		{
			// We are reopening the file, increase it's version
			e->version++;
			e->notify( fd, cpu_fd, size, timestamp, 1 );
		}
	}
	else
	{
		e->wait_open();
	}

	return fd;
}

DEBUG_NOINLINE __device__ int gopen( const char* filename, int flags )
{
	__shared__ int ret;
	BEGIN_SINGLE_THREAD
		ret = single_thread_open( filename, flags );
	END_SINGLE_THREAD;
	return ret;
}

DEBUG_NOINLINE __device__ volatile PFrame* getRwLockedPage( int fd, int version, size_t block_id, int cpu_fd, int purpose )
{
	__shared__ volatile PFrame* pframe;

	BEGIN_SINGLE_THREAD

		RT_SEARCH_START

		pframe = NULL;

		while( NULL == pframe )
		{
			bool busy;

			pframe = g_hashMap->readPFrame( fd, version, block_id, busy );
			if( pframe != NULL && !busy )
			{
				HM_LOCKLESS
				break;
			}

			while( busy )
			{
				RT_WAIT_START
				pframe = g_hashMap->readPFrame( fd, version, block_id, busy );
				RT_WAIT_STOP
			}

			if( pframe != NULL && !busy )
			{
				HM_LOCKLESS
				break;
			}

			pframe = g_hashMap->getPFrame( fd, version, block_id );
			if( pframe != NULL )
			{
				HM_LOCKED
				break;
			}
		}

		RT_SEARCH_STOP

	END_SINGLE_THREAD

		volatile __shared__ int entry;
		entry = -1;

	BEGIN_SINGLE_THREAD

		PAGE_READ_START
		if( pframe->state == PFrame::INIT || pframe->state == PFrame::UPDATING )
		{
			// if we inited, the page is locked and we just keep going

			// Mark the file as dirty if we are adding a write enabled page
			volatile FTable_entry* file = &g_ftable->files[fd];
			if( purpose == PAGE_WRITE_ACCESS )
			{
				file->dirty = 1;
			}

			// cpu-fd would be less than 0 if we are opening write_once file

			if( cpu_fd >= 0 )
			{
				int tEntry = -1;
				int datasize = read_cpu( fd, cpu_fd, pframe, purpose, tEntry );
				if( datasize < 0 )
				{
					// TODO: error handling
					GPU_ASSERT("Failed to read data from CPU"==NULL);
				}
				pframe->content_size = datasize;
				entry = tEntry;
			}

		} PAGE_READ_STOP

		GPU_ASSERT( (pframe->state == PFrame::INIT) ||
						    (pframe->state == PFrame::UPDATING) ||
						    ((pframe->state == PFrame::VALID) && pframe->refCount>0) );
	END_SINGLE_THREAD

		if( -1 != entry )
		{

			COPY_BLOCK_START

			volatile CPU_IPC_RW_Entry* e = &(g_cpu_ipcRWQueue->entries[entry]);
			int workerID = entry / RW_SLOTS_PER_WORKER;

			copy_block( (uchar*)pframe->page, g_stagingArea[workerID][e->scratch_index] + e->return_offset, e->return_size );

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
		if( ( pframe->state == PFrame::INIT || pframe->state == PFrame::UPDATING ) && cpu_fd >= 0 )
			pframe->unlock_init();

	END_SINGLE_THREAD
	if( ( (pframe->state == PFrame::INIT || pframe->state == PFrame::UPDATING) && cpu_fd >= 0 ) || pframe->state == PFrame::VALID )
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

DEBUG_NOINLINE __device__ int gmunmap_threadblock( volatile void *addr, size_t length )
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

DEBUG_NOINLINE __device__ int gmunmap( volatile void *addr, size_t length )
{
	size_t tmp = ( (char*) addr ) - ( (char*) g_ppool->rawStorage );
	size_t offset = tmp >> FS_LOGBLOCKSIZE;
	if( offset >= PPOOL_FRAMES )
		return -1;

	__threadfence(); // make sure all writes to the page become visible
	int laneid = threadIdx.x & 0x1f;

	if( laneid == 0 )
	{
		volatile PFrame* p = &( g_ppool->frames[offset] );
		p->unlock_rw();
	}

	return 0;
}

DEBUG_NOINLINE __device__ volatile void* gmmap_threadblock( void *addr, size_t size, int prot, int flags, int fd, off_t offset )
{
	__shared__ volatile PFrame* pframe; // the ptr is to global mem but is stored in shmem
	__shared__ size_t block_id;
	__shared__ int block_offset;

	__shared__ int cpu_fd;
	BEGIN_SINGLE_THREAD
		block_id = offset2block( offset, FS_LOGBLOCKSIZE );
		block_offset = offset2blockoffset( offset, FS_BLOCKSIZE );

		GPU_ASSERT(fd>=0 && fd<MAX_NUM_FILES);

		cpu_fd = g_ftable->files[fd].cpu_fd;
		GPU_ASSERT( g_ftable->files[fd].refCount >0 );

		if( block_offset + size > FS_BLOCKSIZE )
			GPU_ASSERT("Reading beyond the  page boundary"==0);

		// decide whether to fetch data or not
		if( g_ftable->files[fd].flags == O_GWRONCE )
			cpu_fd = -1;

	END_SINGLE_THREAD

	int purpose = ( g_ftable->files[fd].flags == O_GRDONLY ) ? PAGE_READ_ACCESS : PAGE_WRITE_ACCESS;
	pframe = getRwLockedPage( fd, g_ftable->files[fd].version, block_id, cpu_fd, purpose );

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

DEBUG_NOINLINE __device__ volatile PFrame* getRwLockedPage_warp( int fd, int version, size_t block_id, int cpu_fd, int purpose )
{
	BroadcastHelper broadcastHelper;

	volatile PFrame* pframe = NULL;

	int entry = -1;

	int laneid = threadIdx.x & 0x1f;

	if( laneid == 0 )
	{
		while( NULL == pframe )
		{
			bool busy;

			// try lockless read first
			pframe = g_hashMap->readPFrame( fd, version, block_id, busy );
			if( pframe != NULL && !busy )
			{
				break;
			}

			// data is probably there but we couldn't lock the page
			// wait till it's no longer busy
			while( busy )
			{
				pframe = g_hashMap->readPFrame( fd, version, block_id, busy );

				if( pframe != NULL )
					break;
			}

			// lockless didn't work, try a more aggressive approach
			pframe = g_hashMap->getPFrame( fd, version, block_id );
			if( pframe != NULL )
			{
				break;
			}
		}

		if( pframe->state == PFrame::INIT || pframe->state == PFrame::UPDATING )
		{
			// if we inited, the page is locked and we just keep going

			// Mark the file as dirty if we are adding a write enabled page
			volatile FTable_entry* file = &g_ftable->files[fd];
			if( purpose == PAGE_WRITE_ACCESS )
			{
				file->dirty = 1;
			}

			// cpu-fd would be less than 0 if we are opening write_once file
			if( cpu_fd >= 0 )
			{
				int datasize = read_cpu( fd, cpu_fd, pframe, purpose, entry );
				if( datasize < 0 )
				{
					// TODO: error handling
					GPU_ASSERT("Failed to read data from CPU"==NULL);
				}
				pframe->content_size = datasize;
			}
			else
			{
				file->busyList.push( pframe );
			}
		}

		GPU_ASSERT( (pframe->state == PFrame::INIT) ||
				    (pframe->state == PFrame::UPDATING) ||
				    ((pframe->state == PFrame::VALID) && pframe->refCount>0) );
	}

	broadcastHelper.v = (void*)pframe;
	pframe = (typeof(pframe))broadcast(broadcastHelper).v;
	entry = broadcast( entry );

	if( -1 != entry )
	{
		volatile CPU_IPC_RW_Entry* e = &(g_cpu_ipcRWQueue->entries[entry]);
		int workerID = entry / RW_SLOTS_PER_WORKER;

		copy_block_warp( (uchar*)pframe->page, g_stagingArea[workerID][e->scratch_index] + e->return_offset, e->return_size );

		if( laneid == 0 )
		{
			freeEntry( entry );

			int workerID = entry / RW_SLOTS_PER_WORKER;
			int scratchID = g_cpu_ipcRWQueue->entries[entry].scratch_index;

			atomicSub( (int*)&(g_cpu_ipcRWFlags->entries[workerID][scratchID]), 1 );
			__threadfence_system();
		}
	}

	if( laneid == 0 )
	{
		// TODO: it looks like cpu_fd is always >=0. it is returned from reopen and there's an assert that make sure it's positive
		// if the page was initialized, return. Make sure to return with all threads active
		if( ( pframe->state == PFrame::INIT || pframe->state == PFrame::UPDATING ) && cpu_fd >= 0 )
			pframe->unlock_init();
	}

	if( ( ( pframe->state == PFrame::INIT || pframe->state == PFrame::UPDATING ) && cpu_fd >= 0 ) || pframe->state == PFrame::VALID )
		return pframe;

	//fill the page with zeros - optimization for the case of write-once exclusive create owned by GPU
	// TODO: implement a warp level version
	GPU_ASSERT(false);
	bzero_page( (volatile char*) pframe->page );
	__threadfence(); // make sure all threads will see these zeros

	if( laneid == 0 )
	{
		GPU_ASSERT(cpu_fd<0);

		GPU_ASSERT(pframe->state == PFrame::INIT);
		pframe->content_size = 0;
		pframe->unlock_init();
	}

	return pframe;
}

DEBUG_NOINLINE __device__ volatile void* gmmap( void *addr, size_t size, int prot, int flags, int fd, off_t offset )
{
	BroadcastHelper broadcastHelper;

	volatile PFrame* pframe;
	size_t block_id;
	int block_offset;
	int cpu_fd;

	int laneid = threadIdx.x & 0x1f;

	if( laneid == 0 )
	{
		block_id = offset2block( offset, FS_LOGBLOCKSIZE );
		block_offset = offset2blockoffset( offset, FS_BLOCKSIZE );

		GPU_ASSERT(fd>=0 && fd<MAX_NUM_FILES);

		cpu_fd = g_ftable->files[fd].cpu_fd;
		GPU_ASSERT( g_ftable->files[fd].refCount >0 );

		if( block_offset + size > FS_BLOCKSIZE )
			GPU_ASSERT("Reading beyond the  page boundary"==0);

		// decide whether to fetch data or not
		if( g_ftable->files[fd].flags == O_GWRONCE )
			cpu_fd = -1;
	}

	broadcastHelper.s = block_id;
	block_id = broadcast( broadcastHelper ).s;
	block_offset = broadcast( block_offset );
	cpu_fd = broadcast( cpu_fd );

	int purpose = ( g_ftable->files[fd].flags == O_GRDONLY ) ? PAGE_READ_ACCESS : PAGE_WRITE_ACCESS;
	pframe = getRwLockedPage_warp( fd, g_ftable->files[fd].version, block_id, cpu_fd, purpose );

//	printf("block_id: %ld, block_offset: %d, pframe: %lx\n", block_id, block_offset, pframe->page);

	if( laneid == 0 )
	{
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

	}

	GPU_ASSERT(pframe!=NULL);

	return (void*) ( ( (uchar*) ( pframe->page ) ) + block_offset );
}

#endif
