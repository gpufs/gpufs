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

#ifndef FS_CONSTANTS
#define FS_CONSTANTS

// needed for mmap return codes
#include <sys/mman.h>

// home many pages to swapout at once
#define NUM_PAGES_SWAPOUT (64)
#define NUM_SWAP_RETRIES (64)
#define NUM_MEMORY_RINGS (1)
#define LOWER_WATER_MARK (0)

#define NUM_BUSY_LISTS (256)

#define FS_LOGBLOCKSIZE (12)
#define FS_BLOCKSIZE ( 1 << FS_LOGBLOCKSIZE )
#define FS_BLOCKMASK (FS_BLOCKSIZE - 1);

#define PPOOL_SIZE ( 1024 * 1024 * 1024 * 2L )
#define PPOOL_FRAMES ( PPOOL_SIZE >> FS_LOGBLOCKSIZE )
#define HASH_MAP_SIZE ( PPOOL_FRAMES * 16L)

// FS constants 
// number of slots in the RB
#define ASYNC_CLOSE_RINGBUF_SIZE (64) 

#define MAX_NUM_FILES (128)
// must be power of 2
#define MAX_NUM_CLOSED_FILES ( 1 << 10 )

//** ERROR CODES **//
#define E_FSTABLE_FULL -1
#define E_IPC_OPEN_ERROR -2

//** OPEN CLOSE
#define FSTABLE_SIZE MAX_NUM_FILES
#define FILENAME_SIZE  64 // longest filename string
#define FSENTRY_EMPTY 	0
#define FSENTRY_PENDING	1
#define FSENTRY_OPEN	2
#define FSENTRY_CLOSING	3
#define FSENTRY_CLOSED	4

//** CPU IPC 
#define CPU_IPC_EMPTY 0
#define CPU_IPC_PENDING 1
#define CPU_IPC_IN_PROCESS 2
#define CPU_IPC_READY 3

//** CPU IPC R/W TABLE
#define RW_IPC_SIZE 128
#define RW_IPC_READ 0
#define RW_IPC_WRITE 1
#define RW_IPC_DIFF 2
#define RW_IPC_TRUNC 3

#define RW_HOST_WORKERS 4
#define RW_SLOTS_PER_WORKER (RW_IPC_SIZE / RW_HOST_WORKERS)
#define RW_SCRATCH_PER_WORKER 2

#define IPC_MGR_EMPTY 0
#define IPC_MGR_BUSY 1
#define IPC_TYPE_READ 0
#define IPC_TYPE_WRITE 1

#include<fcntl.h>

#define O_GRDONLY (O_RDONLY)
#define O_GWRONLY (O_WRONLY)
#define O_GCREAT (O_CREAT)
#define O_GRDWR (O_RDWR)
#define O_GWRONCE (O_GCREAT|O_GWRONLY)

#define GMAP_FAILED MAP_FAILED

#define PAGE_READ_ACCESS 0
#define PAGE_WRITE_ACCESS 1
typedef char Page[FS_BLOCKSIZE];

typedef unsigned char uchar;

#endif
