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

#include "hashMap.cu.h"
#include "util.cu.h"
#include "fs_globals.cu.h"

static __device__ uint calcHash( int fd, size_t block_id )
{
	const uint numBits = ilogb((float)HASH_MAP_SIZE);
	const uint blockMask = (1 << (numBits-8)) - 1;
	uint fdBits = (fd & 0xFF) << (numBits - 8);
	uint blockBits = block_id & blockMask;
	return (fdBits | blockBits);
}

DEBUG_NOINLINE __device__ void HashMap::init_thread() volatile
{
	for( int i = 0; i < HASH_MAP_SIZE; i++ )
	{
		locks[i] = 0;
		frames[i] = NULL;
	}
}

DEBUG_NOINLINE __device__ volatile PFrame* HashMap::readPFrame( int fd, int version, size_t block_id, bool& busy, int ref ) volatile
{
	busy = false;

	size_t offset = block_id << FS_LOGBLOCKSIZE;

	uint index = calcHash( fd, block_id );
	volatile PFrame* it = frames[index];

	while( NULL != it )
	{
		if( ( it->file_id == fd ) &&  (it->file_offset == offset ) )
		{
			if( it->try_lock_rw( fd, version, offset, ref ) )
			{
				return it;
			}
			else
			{
				busy = true;
				return NULL;
			}
		}

		it = it->next;
	}

	return NULL;
}


DEBUG_NOINLINE __device__ volatile PFrame* HashMap::getPFrame( int fd, int version, size_t block_id, int ref ) volatile
{
	uint index = calcHash( fd, block_id );
	size_t offset = block_id << FS_LOGBLOCKSIZE;

	MUTEX_LOCK( locks[index] );

	volatile PFrame* it = frames[index];

	// Check if list is empty
	if( NULL == it )
	{
		// Page is not found, try to add it
		volatile PFrame* newData = g_ppool->allocPage();
		while( newData == NULL )
		{
			newData = g_ppool->allocPage();
		}

		newData->try_lock_init(ref);

		newData->file_id = fd;
		newData->file_offset = offset;
		newData->version = version;
		__threadfence();

		frames[index] = newData;
		__threadfence();

		MUTEX_UNLOCK( locks[index] );
		return newData;
	}

	// Search for the data before trying to insert
	while( NULL != it )
	{
		if( ( it->file_id == fd ) &&  (it->file_offset == offset ) )
		{
			if( it->try_lock_rw( fd, version, offset, ref ) )
			{
				MUTEX_UNLOCK( locks[index] );
				return it;
			}
			else
			{
				MUTEX_UNLOCK( locks[index] );
				return NULL;
			}
		}

		it = it->next;
	}

	// Data is not found, try to add it
	volatile PFrame* newData = g_ppool->allocPage();
	while( newData == NULL )
	{
		newData = g_ppool->allocPage();
	}

	newData->try_lock_init(ref);

	newData->file_id = fd;
	newData->file_offset = offset;
	newData->version = version;

	// Make sure the pointers are always valid and seen by everyone in the correct order
	newData->next = frames[index];
	__threadfence();

	frames[index] = newData;
	__threadfence();

	MUTEX_UNLOCK( locks[index] );
	return newData;
}

DEBUG_NOINLINE __device__ bool HashMap::removePFrame( volatile PFrame* pframe ) volatile
{
	int fd = pframe->file_id;
	size_t offset = pframe->file_offset;

	size_t block_id = offset>>FS_LOGBLOCKSIZE;

	uint index = calcHash( fd, block_id );

	if( MUTEX_WAS_LOCKED( locks[index] ) )
	{
		return false;
	}

	volatile PFrame* it = frames[index];
	GPU_ASSERT( it != NULL );

	if( it == pframe )
	{
		// Try to remove head

		// First invalidate the data
		if( ! it->try_invalidate( fd, offset ) )
		{
			// Someone is probably still using it, abort
			MUTEX_UNLOCK( locks[index] );
			return false;
		}


		frames[index] = it->next;
		__threadfence();
		it->next = NULL;
		__threadfence();

		MUTEX_UNLOCK( locks[index] );
		return true;
	}

	// Else, remove from the middle of the list
	volatile PFrame* prev = it;
	it = it->next;

	while( NULL != it )
	{
		if( it == pframe )
		{
			// First invalidate the data
			if( ! it->try_invalidate( fd, offset ) )
			{
				// Someone is probably still using it, abort
				MUTEX_UNLOCK( locks[index] );
				return false;
			}

			prev->next = it->next;
			it->next = NULL;
			__threadfence();

			MUTEX_UNLOCK( locks[index] );
			return true;
		}

		prev = it;
		it = it->next;
	}

	// Data not found
	MUTEX_UNLOCK( locks[index] );
	return false;
}
