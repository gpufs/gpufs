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
#ifndef FAT_POINTER_CU
#define FAT_POINTER_CU

#include <limits.h>
#include "fs_constants.h"
#include "fs_structures.cu.h"

static const unsigned long long int FID_BITS = 4;

static const unsigned long long int ACCESS_BITS = 2;
static const unsigned long long int VALID_BITS = 1;
static const unsigned long long int PAGE_BITS = 64 - FS_LOGBLOCKSIZE - ACCESS_BITS - VALID_BITS;

static const unsigned long long int INVALID_VPAGE = 0x1FFFFFFFFFFFF;
static const unsigned long long int INVALID_FID = (1 << FID_BITS) - 1;

struct __align__(8) _FatPtr
{
	union
	{
		struct
		{
			unsigned int pageOffset : FS_LOGBLOCKSIZE;
			unsigned long long int physPage : PAGE_BITS;
		};

		struct
		{
			unsigned int virtPageOffset : FS_LOGBLOCKSIZE;
			unsigned long long int virtPage : PAGE_BITS;
		};

		unsigned long long int offset : FS_LOGBLOCKSIZE + PAGE_BITS;
	};
	unsigned int accBits : ACCESS_BITS;
	unsigned int valid : VALID_BITS;
};

template<int N>
struct TLB
{
};

template<typename T, int N>
class FatPointer
{
public:
	__device__ FatPointer( size_t fid, off_t start, size_t size, int flags, TLB<N>* tlb, uchar* mem, volatile PFrame* frames );

	__device__ FatPointer( const FatPointer& ptr );

	__device__ ~FatPointer();

	__device__ FatPointer& operator=( const FatPointer& ptr );

	// Move to exact offset from the beginning of the map
	__device__ FatPointer& moveTo( size_t offset );

	// Move to offset from current location
	__device__ FatPointer& move( size_t offset );

	__device__ T& operator *();

	__device__ FatPointer& operator += ( size_t offset )
	{
		return move(offset);
	}

	__device__ FatPointer& operator -= ( size_t offset )
	{
		return move(-offset);
	}

public:

	size_t m_fid;	// Should be spilled to local memory
	off_t m_start;	// Should be spilled to local memory
	size_t m_end;	// Should be spilled to local memory
	int m_flags;	// Should be spilled to local memory

	_FatPtr m_ptr;

	TLB<N> *m_tlb;
	uchar* m_mem;
	PFrame* m_frames;
};

#endif
