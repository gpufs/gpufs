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

#ifndef FS_STRUCTURES_CU
#define FS_STRUCTURES_CU

#include "fs_structures.cu.h"
#include "fs_globals.cu.h"

DEBUG_NOINLINE __device__ void PFrame::init_thread( volatile Page* _page, int _rs_offset ) volatile
{
	page = _page;
	rs_offset = _rs_offset;
	file_id = (uint) -1;
	content_size = 0;
	file_offset = (uint) -1;
	dirty = false;
	dirtyCounter = 0;

	lock = 0;
	refCount = 0;
	state = INVALID;
}

DEBUG_NOINLINE __device__ void PFrame::clean() volatile
{
	file_id = (uint) -1;
	content_size = (uint)-1;
	file_offset = (uint) -1;
	dirty = 0;
	dirtyCounter = 0;
}

DEBUG_NOINLINE __device__ bool PFrame::try_lock_init() volatile
{
	if( MUTEX_WAS_LOCKED(lock) )
	{
		// page is busy
		return false;
	}

	if( INVALID == state )
	{
		// We are the ones initiating this page
		state = INIT;
		refCount = 1;
		threadfence();
		// Keep lock
		return true;
	}
	else
	{
		// It is already initiated by someone else
		MUTEX_UNLOCK( lock );
		return false;
	}
}

DEBUG_NOINLINE __device__ void PFrame::unlock_init() volatile
{
	GPU_ASSERT( lock );
	GPU_ASSERT( state == INIT );

	state = VALID;
	MUTEX_UNLOCK( lock );
}



DEBUG_NOINLINE __device__ bool PFrame::try_lock_rw( int fd, size_t offset ) volatile
{
	if( MUTEX_WAS_LOCKED(lock) )
	{
		// page is busy
		return false;
	}

	if( state == PFrame::VALID && fd == file_id && file_offset == offset )
	{
		// This is the right one
		refCount++;
		threadfence();
		MUTEX_UNLOCK( lock );
		return true;
	}
	else
	{
		// page is either invalid or point to a different location
		MUTEX_UNLOCK( lock );
		return false;
	}
}

DEBUG_NOINLINE __device__ void PFrame::unlock_rw() volatile
{
	MUTEX_LOCK( lock );
	refCount--;
	MUTEX_UNLOCK( lock );

//	atomicSub((int*)&refCount, 1);
}

DEBUG_NOINLINE __device__ bool PFrame::try_invalidate( int fd, size_t offset ) volatile
{
	if( MUTEX_WAS_LOCKED(lock) )
	{
		// page is busy
		return false;
	}

	if( refCount == 0 && fd == file_id && offset == file_offset )
	{
		// We can safely remove this page
		clean();
		state = INVALID;
		threadfence();
		MUTEX_UNLOCK( lock );
		return true;
	}
	else
	{
		// Someone is still using it, don't invalidate
		MUTEX_UNLOCK( lock );
		return false;
	}
}

DEBUG_NOINLINE __device__ void PFrame::markDirty() volatile
{
	dirty = 1;
}

__device__ void FTable_entry::clean() volatile
{
	dirty_tree = 0;
}

//******* OPEN/CLOSE *//

DEBUG_NOINLINE __device__ void OTable_entry::init_thread() volatile
{
	status = FSENTRY_EMPTY;
	refCount = 0;
	cpu_fd = -1;
	cpu_inode = (unsigned int) -1;
}

DEBUG_NOINLINE __device__ void OTable_entry::init( const volatile char* _filename, int _flags ) volatile
{
	strcpy_thread( filename, _filename, FILENAME_SIZE );
	status = FSENTRY_PENDING;
	refCount = 0;
	cpu_fd = -1;
	cpu_inode = (unsigned int) -1;
	flags = _flags;
	did_open = 0;
}
DEBUG_NOINLINE __device__ void OTable_entry::notify( int _cpu_fd, unsigned int _cpu_inode, size_t _size,
		int _did_open ) volatile
{
	cpu_fd = _cpu_fd;
	cpu_inode = _cpu_inode;
	did_open = _did_open;
	size = _size;
	__threadfence();
	status = FSENTRY_OPEN;
	__threadfence();
}

DEBUG_NOINLINE __device__ void OTable_entry::wait_open() volatile
{
	WAIT_ON_MEM( status, FSENTRY_OPEN );
}

DEBUG_NOINLINE __device__ void OTable_entry::clean() volatile
{
	GPU_ASSERT(refCount==0);
	status = FSENTRY_EMPTY;
	did_open = 0;
}

DEBUG_NOINLINE __device__ void OTable::lock() volatile
{
	MUTEX_LOCK( _lock );
}

DEBUG_NOINLINE __device__ void OTable::unlock() volatile
{
	MUTEX_UNLOCK( _lock );
}

DEBUG_NOINLINE __device__ void OTable::init_thread() volatile
{
	for( int i = 0; i < FSTABLE_SIZE; i++ )
	{
		entries[i].init_thread();
		_lock = 0;
	}
}

DEBUG_NOINLINE __device__ int OTable::findEntry( volatile const char* filename, volatile bool* isNewEntry,
		int o_flags ) volatile
{
	// TODO: this can be sped up by creating a management GPU buffer instead of searching CPU every time

	*isNewEntry = true;
	int found = E_FSTABLE_FULL;

	for( int i = 0; i < FSTABLE_SIZE; i++ )
	{
		if( entries[i].status == FSENTRY_EMPTY )
		{
			found = i;
		}
		else if( !strcmp_thread( filename, entries[i].filename, FILENAME_SIZE ) )
		{
			// found entry
			found = i;
			*isNewEntry = false;
			break;
		}
	}
	if( found != E_FSTABLE_FULL && *isNewEntry )
	{
		entries[found].init( filename, o_flags );
		__threadfence();
	}
	return found;
}
#endif
