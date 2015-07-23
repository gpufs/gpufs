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


#ifndef hashMap_h_
#define hashMap_h_
#include "fs_constants.h"
#include "fs_structures.cu.h"
#include "util.cu.h"

struct HashMap
{
	int 				locks[HASH_MAP_SIZE];
	volatile PFrame*	frames[HASH_MAP_SIZE];

	__device__ void init_thread() volatile;

	__device__ volatile PFrame* readPFrame( int fd, int version, size_t block_id, bool& busy, int ref = 1 ) volatile;

	__device__ volatile PFrame* getPFrame( int fd, int version, size_t block_id, int ref = 1 ) volatile;
	
	__device__ bool removePFrame( volatile PFrame* pframe ) volatile;
};

#endif
