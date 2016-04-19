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

#ifndef CPU_IPC_CU
#define CPU_IPC_CU
#include <assert.h>
#include "fs_constants.h"
#include "util.cu.h"
#include "fs_structures.cu.h"
#include "cpu_ipc.cu.h"
#include "fs_globals.cu.h"
__host__ void CPU_IPC_OPEN_Queue::init_host() volatile
{
	for( int i = 0; i < FSTABLE_SIZE; i++ )
	{
		entries[i].init_host();
	}
}

__host__ void CPU_IPC_RW_Entry::init_host() volatile
{
	status = CPU_IPC_EMPTY;
	cpu_fd = -1;
	buffer_offset = 0;
	file_offset = (uint) -1;
	size = 0;
	type = 0;
	return_offset = 0;
	return_size = 0;
}

__host__ void CPU_IPC_RW_Queue::init_host() volatile
{
	for( int i = 0; i < RW_IPC_SIZE; i++ )
	{
		entries[i].init_host();
	}
}

__host__ void CPU_IPC_RW_Flags::init_host() volatile
{
	for( int i = 0; i < RW_HOST_WORKERS; i++ )
	{
		for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
		{
			entries[i][j] = 0;
		}
	}
}

__host__ void CPU_IPC_OPEN_Entry::init_host() volatile
{
	status = CPU_IPC_EMPTY;
	cpu_fd = -1;
	size = 0;
	flush_cache = 0;
	drop_residence_inode = 0;
	cpu_inode = (unsigned int) -1;
	memset( (void*) filename, 0, FILENAME_SIZE );
	return_value = 0;
	is_dirty = 0;
	do_not_open = 0;
}

__device__ void CPU_IPC_OPEN_Entry::clean() volatile
{
	filename[0] = '\0';
	cpu_fd = -1;
	cpu_inode = (unsigned) -1;
	size = 0;
	flush_cache = 0;
	drop_residence_inode = 0;
	status = CPU_IPC_EMPTY;
	is_dirty = 0;
	do_not_open = 0;
	__threadfence_system();
}

__device__ int CPU_IPC_OPEN_Entry::reopen() volatile
{

	do_not_open = 0;
	__threadfence_system();
	GPU_ASSERT(status==CPU_IPC_READY);
	status = CPU_IPC_PENDING;

	__threadfence_system();

	WAIT_ON_MEM( status, CPU_IPC_READY );

	return readNoCache( &cpu_fd );
}

__device__ int CPU_IPC_OPEN_Entry::open( const char* reqFname, int _flags, int _do_not_open ) volatile
{

	strcpy_thread( filename, reqFname, FILENAME_SIZE );
	flags = _flags;
	do_not_open = _do_not_open;
	__threadfence_system();
	GPU_ASSERT(status==CPU_IPC_EMPTY);
	status = CPU_IPC_PENDING;

	__threadfence_system();

	WAIT_ON_MEM( status, CPU_IPC_READY );

	return readNoCache( &cpu_fd );
}

__device__ int CPU_IPC_OPEN_Entry::close( int _cpu_fd, unsigned int _drop_residence_inode, bool _is_dirty ) volatile
{
	GPU_ASSERT(_cpu_fd<=0 || status==CPU_IPC_READY); // if we dont want to push cpu_fd its OK, but then make sure that its valid

	cpu_fd = _cpu_fd;
	drop_residence_inode = _drop_residence_inode;
	is_dirty = _is_dirty;

	status = CPU_IPC_PENDING;

	__threadfence_system();

	WAIT_ON_MEM( status, CPU_IPC_READY );

	return readNoCache( &cpu_fd );
}

__device__ void CPU_IPC_RW_Entry::clean() volatile
{
	cpu_fd = -1;
	status = CPU_IPC_EMPTY;
	file_offset = (uint) -1;
	__threadfence_system();
}
__device__ int CPU_IPC_RW_Entry::ftruncate( int _cpu_fd ) volatile
{
	cpu_fd = _cpu_fd;
	type = RW_IPC_TRUNC;

	__threadfence_system();
	GPU_ASSERT(status==CPU_IPC_EMPTY);
	status = CPU_IPC_PENDING;
	__threadfence_system();

	WAIT_ON_MEM( status, CPU_IPC_READY );
	return readNoCache( &return_size );
}
__device__ int CPU_IPC_RW_Entry::read_write( int fd, int _cpu_fd, volatile PFrame* frame, uint _type, bool addToDirtyList ) volatile
{
	CPU_READ_START
	cpu_fd = _cpu_fd;
	buffer_offset = frame->rs_offset << FS_LOGBLOCKSIZE;
	size = (_type == RW_IPC_READ) ? FS_BLOCKSIZE : frame->content_size;
	type = _type;
	return_size = -1;
	return_offset = -1;
	file_offset = frame->file_offset;

	__threadfence_system();
	GPU_ASSERT(status==CPU_IPC_EMPTY);
	status = CPU_IPC_PENDING;
	__threadfence_system();

	WAIT_ON_MEM( status, CPU_IPC_READY );
	CPU_READ_STOP
	return readNoCache( &return_size );
}

__device__ void GPU_IPC_RW_Manager::init_thread() volatile
{
	for( int i = 0; i < RW_IPC_SIZE; i++ )
	{
		_locker[i] = IPC_MGR_EMPTY;
	}
	_lock = 0;
}

__device__ int GPU_IPC_RW_Manager::findEntry() volatile
{
	const int init = ( blockIdx.x + threadIdx.y ) & ( RW_IPC_SIZE - 1 );
	int i;
	// TODO -> lockfree optimization, just attempt to take private TB lock trivial and will work well!!
	i = init; // assuming one concurrent call PER TB

	do
	{
		if( atomicExch( (int*) &_locker[i], IPC_MGR_BUSY ) == IPC_MGR_EMPTY )
		{
			return i;
		}
		i = ( i + 1 ) & ( RW_IPC_SIZE - 1 );
	} while( 1 );
}

__device__ void GPU_IPC_RW_Manager::freeEntry( int entry ) volatile
{
	assert( _locker[entry] == IPC_MGR_BUSY );
	_locker[entry] = IPC_MGR_EMPTY;
	//__threadfence();
}

__device__ int CPU_IPC_RW_Queue::read_write_page( int fd, int cpu_fd, volatile PFrame* frame, int type, int& entry,
		bool addToDirtyList ) volatile
{
	entry = g_ipcRWManager->findEntry();
	int ret_val = entries[entry].read_write( fd, cpu_fd, frame, type, addToDirtyList ); // this one will wait until done

	return ret_val;
}

//*****************External function to read data from CPU**//
//__device__ int truncate_cpu(int cpu_fd)
//{
//	GPU_ASSERT(cpu_fd>=0);
//	int entry=g_ipcRWManager->findEntry();
//	int ret_val=g_cpu_ipcRWQueue->entries[entry].ftruncate(cpu_fd);
//	g_cpu_ipcRWQueue->entries[entry].clean();
//	g_ipcRWManager->freeEntry(entry);
//	return ret_val;
//}

__device__ int truncate_cpu(int cpu_fd)
{
	GPU_ASSERT(cpu_fd>=0);
	int entry=g_ipcRWManager->findEntry();
	int ret_val=g_cpu_ipcRWQueue->entries[entry].ftruncate(cpu_fd);
	g_cpu_ipcRWQueue->entries[entry].clean();
	g_ipcRWManager->freeEntry(entry);
	return ret_val;
}

__device__ int read_cpu( int fd, int cpu_fd, volatile PFrame* frame, int purpose, int& entry )
{
	GPU_ASSERT(cpu_fd>=0);
	bool addToDirtyList = ( purpose == PAGE_WRITE_ACCESS );
	int ret_val = g_cpu_ipcRWQueue->read_write_page( fd, cpu_fd, frame, RW_IPC_READ, entry, addToDirtyList );
	if( ret_val <= 0 )
		return ret_val;
	frame->content_size = ret_val;
	return ret_val;
}

__device__ int write_cpu( int fd, int cpu_fd, volatile PFrame* frame, int flags )
{
	GPU_ASSERT(cpu_fd>=0);

	int type = ( flags == ( O_GWRONCE ) ) ? RW_IPC_DIFF : RW_IPC_WRITE;
	int entry;

	int ret_val = g_cpu_ipcRWQueue->read_write_page( fd, cpu_fd, frame, type, entry );
	freeEntry( entry );

	if( ret_val != frame->content_size )
	{
		// TODO: add error handling
		//	GPU_ASSERT(NULL);
		return ret_val;
	}

	return 0;
}

__device__ int writeback_page_async_on_close( int cpu_fd, volatile PFrame* frame, int flags )
{
	GPU_ASSERT(cpu_fd>=0);

	int type = ( flags == ( O_GWRONCE ) ) ? RW_IPC_DIFF : RW_IPC_WRITE;

	// TODO - CLOSE ERRORS ARE NOT COUGHT
	g_async_close_rb->enqueue( cpu_fd, frame->rs_offset, frame->file_offset, frame->content_size, type );

	return 0;
}

__device__ void writeback_page_async_on_close_done( int cpu_fd )
{
	g_async_close_rb->enqueue_done( cpu_fd );
}

__device__ void freeEntry( int entry )
{
	g_cpu_ipcRWQueue->entries[entry].clean();
	g_ipcRWManager->freeEntry( entry );
}

#endif
