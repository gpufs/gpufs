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

#ifndef HOST_LOOP_CPP
#define HOST_LOOP_CPP

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>
#include <iostream>
#include <iomanip>

#include <nvToolsExt.h>

#include "fs_initializer.cu.h"

//#define TRACE
#ifdef TRACE

#define PRINT_TRACE(...) fprintf(stderr, __VA_ARGS__);

#else

#define PRINT_TRACE(...)

#endif

#ifdef TIMING_STATS

#define PRINT_TIMES(...) fprintf(stderr, __VA_ARGS__);

#else

#define PRINT_TIMES(...)

#endif


void fd2name(const int fd, char* name, int namelen)
{

	char slink[100];
	pid_t me = getpid();
	name[0] = 0;
	sprintf(slink, "/proc/%d/fd/0", me);

	int s = readlink(slink, name, namelen - 1);
	if (s >= 0)
		name[s] = '\0';
}

Page diff_page;

uchar* diff_and_merge(const Page* page, uint req_cpu_fd, size_t req_size,
		size_t req_file_offset)
{

	struct stat s;

	if (fstat(req_cpu_fd, &s))
		s.st_size = 0;

	int data_read;
	if (s.st_size < req_file_offset)
	{
		data_read = 0;
	}
	else
	{
		data_read = pread(req_cpu_fd, &diff_page, req_size, req_file_offset);
	}

	//fprintf(stderr,"read %d, offset %d\n",data_read,req_file_offset);
	if (data_read < 0)
	{
		perror("pread failed while diff\n");
		req_size = (size_t) -1;
	}
	//if (data_read==0) fprintf(stderr,"empty read\n");

//	uchar* tmp=(uchar*)page;
//	uchar* data_ptr=(uchar*)&diff_page;
//	if (data_read==0){

//		data_ptr=tmp; // copy directly from the buffer

//	}else
	typedef char v32c __attribute__ ((vector_size (16)));
	uchar* data_ptr = (uchar*) page;
	v32c* A_v = (v32c*) data_ptr;
	v32c* B_v = (v32c*) &diff_page;
	;
	if (data_read > 0)
	{

		// perform diff-ed write
//		for(int zzz=0;zzz<data_read;zzz++)
//		{
//			if (tmp[zzz]) {
//				((uchar*)diff_page)[zzz]=tmp[zzz];
///			}
//		}

		int left = data_read % sizeof(v32c);
		for (int zzz = 0; zzz < (data_read / sizeof(v32c) + (left != 0)); zzz++)
		{
			// new is new OR old
			//data_ptr[zzz]=data_ptr[zzz]|((uchar*)diff_page)[zzz];
			A_v[zzz] = A_v[zzz] | B_v[zzz];
		}

		//memcpy(((char*)&diff_page)+data_read,tmp+data_read,req_size-data_read);
	}
	return data_ptr;
}

double asyncMemCpyTime[RW_HOST_WORKERS] = {0};
int asyncMemCpyCount[RW_HOST_WORKERS] = {0};
size_t asyncMemCpySize[RW_HOST_WORKERS] = {0};

double asyncCloseLoopTime = 0;

double total_stat = 0;
double total_stat1 = 0;

volatile int done = 0;

volatile int readRequests[RW_HOST_WORKERS];  //will be unitialized to -1
volatile int activeEntries[RW_HOST_WORKERS][RW_SCRATCH_PER_WORKER][RW_SLOTS_PER_WORKER] = {0};

pthread_mutex_t rwLoopTasksLocks[RW_HOST_WORKERS];
pthread_cond_t  rwLoopTasksConds[RW_HOST_WORKERS];

void open_loop(volatile GPUGlobals* globals, int gpuid)
{
	char* use_gpufs_lib = 0;//getenv( "USE_GPUFS_DEVICE" );

	for( int i = 0; i < FSTABLE_SIZE; i++ )
	{
		char filename[FILENAME_SIZE];
		volatile CPU_IPC_OPEN_Entry* e = &globals->cpu_ipcOpenQueue->entries[i];
		// we are doing open
		if( e->status == CPU_IPC_PENDING && e->cpu_fd < 0 )
		{
			double vvvv = _timestamp();
			memcpy( filename, (char*) e->filename, FILENAME_SIZE );
			// OPEN
			if( e->flags & O_GWRONCE )
			{
				e->flags = O_RDWR | O_CREAT;
			}
			char pageflush = 0;
			int cpu_fd = -1;
			struct stat s;
			if( e->do_not_open )
			{
				if( stat( filename, &s ) < 0 )
				{
					fprintf( stderr, " problem with STAT file %s on CPU: %s\n", filename, strerror( errno ) );
				}
			}
			else
			{
				if( use_gpufs_lib )
					cpu_fd = gpufs_file_open( globals->gpufs_fd, gpuid, filename, e->flags, S_IRUSR | S_IWUSR,
							&pageflush );
				else
				{
					// fprintf( stderr, "Open file: %s\n", filename );
					cpu_fd = open( filename, e->flags, S_IRUSR | S_IWUSR );
				}

				if( cpu_fd < 0 )
				{
					fprintf( stderr, "Problem with opening file %s on CPU: %s (e->flags = %d)\n ", filename, strerror( errno ), e->flags );
				}

				if( fstat( cpu_fd, &s ) )
				{
					fprintf( stderr, "Problem with fstat the file %s on CPU: %s \n ", filename, strerror( errno ) );
				}
			}

			e->cpu_fd = cpu_fd;
			e->flush_cache = pageflush;
			e->cpu_inode = s.st_ino;
			e->size = s.st_size;
			e->cpu_timestamp = s.st_ctime;
			__sync_synchronize();
			e->status = CPU_IPC_READY;
			__sync_synchronize();
			total_stat += ( _timestamp() - vvvv );
		}
		if( e->status == CPU_IPC_PENDING && e->cpu_fd >= 0 )
		{
			double vvvv1 = _timestamp();
			// do close
			if( use_gpufs_lib )
			{
				if( e->is_dirty )
				{ // if dirty, update gpufs device, but keep the file open
					e->cpu_fd = gpufs_file_close_stay_open( globals->gpufs_fd, gpuid, e->cpu_fd );
				}
				else
				{
					e->cpu_fd = gpufs_file_close( globals->gpufs_fd, gpuid, e->cpu_fd );
				}
				gpufs_drop_residence( globals->gpufs_fd, gpuid, e->drop_residence_inode );
			}
			else
			{
				if( !e->is_dirty )
				{
					e->cpu_fd = close( e->cpu_fd );
				}
			}
			__sync_synchronize();
			e->status = CPU_IPC_READY;
			__sync_synchronize();
			total_stat1 += _timestamp() - vvvv1;
		}

	}
}

void* open_task(void* data)
{
	TaskData* taskData = (TaskData*)data;

	volatile GPUGlobals* globals = taskData->gpuGlobals;
	int gpuid = taskData->gpuid;

	while( !done )
	{
		open_loop( globals, gpuid );
	}

	return NULL;
}

void async_close_loop(volatile GPUGlobals* globals)
{
	asyncCloseLoopTime -= _timestamp();

	async_close_rb_t* rb = globals->async_close_rb;

	char* no_files = getenv("GPU_NOFILE");
	/*
	 data must be read synchronously from GPU, but then __can be__ written asynchronously by CPU. -- TODO!
	 The goal is to read as much as possible from GPU in order to make CPU close as fast as possible
	 */

	Page* page = globals->streamMgr->async_close_scratch;
	page_md_t md;

	while (rb->dequeue(page, &md, globals->streamMgr->async_close_stream))
	{
		// drain the ringbuffer
		int res;

		if (md.last_page == 1)
		{
			// that's the last
			PRINT_TRACE("closing  dirty file %d\n", md.cpu_fd);
			res = close(md.cpu_fd);
			if (res < 0)
				perror("Async close failed, and nobody to report to:\n");
		}
		else
		{
			PRINT_TRACE("writing dirty page at offset %ld\n", md.file_offset);

			uchar* to_write;
			if (md.type == RW_IPC_DIFF)
			{
				to_write = diff_and_merge(page, md.cpu_fd, md.content_size,
						md.file_offset);
			}
			else
			{
				to_write = (uchar*) page;
			}

			int ws = pwrite(md.cpu_fd, to_write, md.content_size,
					md.file_offset);
			if (ws != md.content_size)
			{
				printf(
						"Writing while async close failed, and nobody to report to: %s\n",
						strerror( errno ));
			}
		}
	}

	asyncCloseLoopTime += _timestamp();
}

void mainLoop( volatile GPUGlobals* globals, int gpuid )
{
	int currentScratchIDs[RW_HOST_WORKERS] = {0};
	int lastScratchIDs[RW_HOST_WORKERS] = {0};

	cudaEvent_t events[RW_HOST_WORKERS][RW_SCRATCH_PER_WORKER];

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		for( int j = 0; j < RW_SCRATCH_PER_WORKER; ++j )
		{
			CUDA_SAFE_CALL( cudaEventCreate( &events[i][j] ) );
		}
	}

	while( !done )
	{
		async_close_loop(globals);

		volatile cudaError_t cuda_status = cudaStreamQuery( globals->streamMgr->kernelStream );
		if ( cudaErrorNotReady != cuda_status )
		{
			done = 1;
			return;
		}

		for( int i = 0; i < RW_HOST_WORKERS; ++i )
		{
			int firstSlot = i * RW_SLOTS_PER_WORKER;

			// check for pending writes
			if( lastScratchIDs[i] != currentScratchIDs[i] )
			{
				cudaError_t cuda_status = cudaEventQuery( events[i][lastScratchIDs[i]] );

				if( cudaSuccess == cuda_status )
				{
					asyncMemCpyTime[i] += _timestamp();

					PRINT_TRACE( "Finished event[%d][%d]\n", i, lastScratchIDs[i] );

					// complete the request
					for( int k = 0; k < RW_SLOTS_PER_WORKER; ++k )
					{
						if( activeEntries[i][lastScratchIDs[i]][k] != 0 )
						{
							PRINT_TRACE( "Notifying request complete[%d]\n", firstSlot + k );
							globals->cpu_ipcRWQueue->entries[ firstSlot + k ].status = CPU_IPC_READY;
							activeEntries[i][lastScratchIDs[i]][k] = 0;
						}
					}

					__sync_synchronize();

					lastScratchIDs[i]++;
					if( RW_SCRATCH_PER_WORKER == lastScratchIDs[i] )
					{
						lastScratchIDs[i] = 0;
					}
				}
			}

			// Check if we ran out of space in the ring buffer
			if( ( currentScratchIDs[i] + 1 == lastScratchIDs[i] ) ||
				( ( currentScratchIDs[i] + 1 == RW_SCRATCH_PER_WORKER ) && ( lastScratchIDs[i] == 0 ) )	)
			{
				// We can't handle any more requests right now
				continue;
			}

			int readReq = readRequests[i];

			if( readReq == -1 )
			{
				continue;
			}

			PRINT_TRACE( "Got request from worker: %d\n", i );

			// Handle new requests
			if( readReq >= 0 )
			{
				asyncMemCpyTime[i] -= _timestamp();

				readRequests[i] = -1;

				// Notify the rw loop that we've read the request
				pthread_mutex_lock( &rwLoopTasksLocks[i] );
				pthread_cond_signal( &rwLoopTasksConds[i] );
				pthread_mutex_unlock( &rwLoopTasksLocks[i] );

				if (readReq > 0) {
					CUDA_SAFE_CALL(
						cudaMemcpyAsync( getStagingAreaOffset( globals->stagingArea, i, currentScratchIDs[i]),
								globals->streamMgr->scratch[i][currentScratchIDs[i]], readReq,
								cudaMemcpyHostToDevice, globals->streamMgr->memStream[i] ) );
				}
				CUDA_SAFE_CALL( cudaEventRecord( events[i][currentScratchIDs[i]], globals->streamMgr->memStream[i]));

				PRINT_TRACE( "Recording event[%d][%d]\n", i, currentScratchIDs[i] );

				currentScratchIDs[i]++;
				if( RW_SCRATCH_PER_WORKER == currentScratchIDs[i] )
				{
					currentScratchIDs[i] = 0;
				}
			}
		}
	}
}

void* rw_task( void* param )
{
	TaskData* taskData = (TaskData*)param;

	int id = taskData->id;
	volatile GPUGlobals* globals = taskData->gpuGlobals;

	int firstSlot = id * RW_SLOTS_PER_WORKER;

	volatile CPU_IPC_RW_Entry* entries[RW_SLOTS_PER_WORKER];

	for( int i = 0; i < RW_SLOTS_PER_WORKER; i++ )
	{
		entries[i] = &globals->cpu_ipcRWQueue->entries[firstSlot + i];
	}

	int scratchIdx = 0;

	while( !done )
	{
		size_t scratchSize = 0;
		int numRequests = 0;

		for( int i = 0; i < RW_SLOTS_PER_WORKER; i++ )
		{
			volatile CPU_IPC_RW_Entry* e = entries[i];
			if( e->status == CPU_IPC_PENDING )
			{
				PRINT_TRACE("Handle request in worker: %d, scratch id: %d, scratch offset: %ld, request id: %d, file offset: %ld\n",
						id, scratchIdx, scratchSize, firstSlot + i, e->file_offset);

 				e->status = CPU_IPC_IN_PROCESS;

				while( globals->cpu_ipcRWFlags->entries[id][scratchIdx] != 0 )
				{
					PRINT_TRACE( "Waiting in worker: %d, scratch id: %d, status: %d\n", id, scratchIdx, globals->cpu_ipcRWFlags->entries[id][scratchIdx] );
					usleep(0);
				}

				int req_cpu_fd = e->cpu_fd;
				size_t req_buffer_offset = e->buffer_offset;
				size_t req_file_offset = e->file_offset;
				size_t req_size = e->size;
				int req_type = e->type;

				assert(
						req_type == RW_IPC_READ || req_type == RW_IPC_WRITE || req_type == RW_IPC_DIFF || req_type == RW_IPC_TRUNC );

				if( req_type != RW_IPC_TRUNC )
				{
					assert( req_cpu_fd >= 0 && req_size > 0 );
				}

				switch( req_type )
				{
				case RW_IPC_READ:
				{
					// read
					int cpu_read_size = 0;

					cpu_read_size = pread( req_cpu_fd, globals->streamMgr->scratch[id][scratchIdx] + scratchSize, req_size,
							req_file_offset );

					if (0 > cpu_read_size) {
						printf("cpu_read_size: %d, req_cpu_fd: %ld, req_size: %ld, req_file_offset: %ld\n",
								cpu_read_size, req_cpu_fd, req_size, req_file_offset);
						//pause();
					}
					assert( cpu_read_size >= 0 );

					e->return_size = cpu_read_size;
					e->return_offset = scratchSize;
					e->scratch_index = scratchIdx;
					__sync_synchronize();

					activeEntries[id][scratchIdx][i] = 1;
					scratchSize += cpu_read_size;
					numRequests++;

					break;
				}

				case RW_IPC_TRUNC:
				{
					printf("req_cpu_fd: %d\n", req_cpu_fd);
					e->return_size = ftruncate(req_cpu_fd, 0);
					__sync_synchronize();
					e->status = CPU_IPC_READY;
					__sync_synchronize();
					break;
				}

				default:
					assert( NULL );
				}
			}
		}

		if( 0 == numRequests )
		{
			// Didn't find any request, move on
			continue;
		}

		asyncMemCpyCount[id]++;
		asyncMemCpySize[id] += scratchSize;

		globals->cpu_ipcRWFlags->entries[id][scratchIdx] = numRequests;
		__sync_synchronize();

		pthread_mutex_lock( &rwLoopTasksLocks[id] );

		readRequests[id] = scratchSize;
		PRINT_TRACE( "Send request from worker: %d\n", id );

		pthread_cond_wait( &rwLoopTasksConds[id], &rwLoopTasksLocks[id]);
		pthread_mutex_unlock( &rwLoopTasksLocks[id] );

		scratchIdx++;
		if( RW_SCRATCH_PER_WORKER == scratchIdx )
		{
			scratchIdx = 0;
		}
	}

	return NULL;
}

void run_gpufs_handler(volatile GPUGlobals* gpuGlobals, int gpuid)
{
	done = 0;
	asyncCloseLoopTime = 0;

	for (int i = 0; i < RW_HOST_WORKERS; ++i) {
		readRequests[i] = -1;
	}

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		asyncMemCpyTime[i] = 0;
		asyncMemCpyCount[i] = 0;
		asyncMemCpySize[i] = 0;
	}

	double totalTime = 0;

	totalTime -= _timestamp();

	pthread_attr_t attr;
	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

	pthread_t rwLoopTasksIDs[RW_HOST_WORKERS];
	TaskData rwLoopTasksData[RW_HOST_WORKERS];

	pthread_t openLoopTasksID;
	TaskData openLoopTasksData;

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		rwLoopTasksData[i].id = i;
		rwLoopTasksData[i].gpuGlobals =  gpuGlobals;
		rwLoopTasksData[i].gpuid = 0;

		pthread_create( (pthread_t*)&(rwLoopTasksIDs[i]), &attr, rw_task, (TaskData*)&(rwLoopTasksData[i]) );
	}

	openLoopTasksData.id = 0;
	openLoopTasksData.gpuGlobals =  gpuGlobals;
	openLoopTasksData.gpuid = 0;

	pthread_create( &openLoopTasksID, &attr, open_task, &openLoopTasksData );

	pthread_attr_destroy( &attr );

	mainLoop( gpuGlobals, gpuid );

	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	{
		pthread_join( rwLoopTasksIDs[i], NULL );
	}

	pthread_join( openLoopTasksID, NULL );

	totalTime += _timestamp();

	PRINT_TIMES("Transfer time: %fms\n", totalTime / 1e3);
	PRINT_TIMES("Async close loop time: %fms\n", asyncCloseLoopTime / 1e3);

	int totalCount = 0;
	size_t totalSize = 0;
	double cpyTime = 0;

	for (int i = 0; i < RW_HOST_WORKERS; i++)
	{
		PRINT_TIMES( "Async memcpy [%d]:\n", i );
		PRINT_TIMES( "\tTime: %fms\n", asyncMemCpyTime[i] / 1e3);
		PRINT_TIMES( "\tCount: %d\n", asyncMemCpyCount[i]);
		PRINT_TIMES( "\tSize: %lluMB\n", asyncMemCpySize[i] >> 20);
		if( asyncMemCpyCount[i] > 0 )
		{
			PRINT_TIMES( "\tAverage buffer size: %lluKB\n", (asyncMemCpySize[i] >> 10) / asyncMemCpyCount[i]);
		}
		else
		{
			PRINT_TIMES( "\tAverage buffer size: 0KB\n");
		}
		PRINT_TIMES( "\tBandwidth: %fGB/s\n\n", ((float)asyncMemCpySize[i] / (1 << 30)) / (asyncMemCpyTime[i] / 1e6));

		totalCount += asyncMemCpyCount[i];
		totalSize += asyncMemCpySize[i];
	}

	PRINT_TIMES( "Async memcpy total:\n");
	PRINT_TIMES( "\tCount: %d\n", totalCount);
	PRINT_TIMES( "\tSize: %lluMB\n", totalSize >> 20);
	if( totalCount > 0 )
	{
		PRINT_TIMES( "\tAverage buffer size: %lluKB\n", (totalSize >> 10) / totalCount);
	}
	else
	{
		PRINT_TIMES( "\tAverage buffer size: 0KB\n");
	}
	PRINT_TIMES( "\tBandwidth: %fGB/s\n\n", ((float)totalSize / (1 << 30)) / (totalTime / 1e6));
}

#endif
