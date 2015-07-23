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

// Prevent circular include
__device__ volatile void* gmmap_warp(void *addr, size_t size, int prot, int flags, int fd, off_t offset, int ref);
__device__ int gmunmap_warp( volatile void *addr, size_t length, int ref );

static const unsigned long long int FID_BITS = 4;

static const unsigned long long int ACCESS_BITS = 2;
static const unsigned long long int VALID_BITS = 1;
static const unsigned long long int PAGE_BITS = 64 - FS_LOGBLOCKSIZE - ACCESS_BITS - VALID_BITS;

static const unsigned long long int INVALID_VPAGE = 0x1FFFFFFFFFFFF;
static const unsigned long long int INVALID_FID = (1 << FID_BITS) - 1;

//struct __align__(8) _PhysicalPtr
//{
//	unsigned int pageOffset : FS_LOGBLOCKSIZE;
//	unsigned int physPage : PAGE_BITS;
//	unsigned int accBits : ACCESS_BITS;
//	unsigned int valid : VALID_BITS;
//};
//
//struct __align__(8) _VirtualPtr
//{
//	unsigned int pageOffset : FS_LOGBLOCKSIZE;
//	unsigned int virtPage : PAGE_BITS;
//	unsigned int accBits : ACCESS_BITS;
//	unsigned int valid : VALID_BITS;
//};
//
//union __align__(8) _FatPtr
//{
//	int64_t i;
//	_VirtualPtr vp;
//	_PhysicalPtr pp;
//}

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
	__device__ FatPointer( size_t fid, off_t start, size_t size, int flags, TLB<N>* tlb, uchar* mem, volatile PFrame* frames ) :
	m_fid(fid), m_start(start), m_end(start + size), m_flags(flags)
	{
		m_ptr.virtPage = start >> FS_LOGBLOCKSIZE;
		m_ptr.valid = 0;
		m_ptr.pageOffset = 0;
		m_ptr.accBits = flags;

		m_tlb = tlb;

		m_mem = mem;
		m_frames = (PFrame*)frames;
	}

	__device__ FatPointer( const FatPointer& ptr ) :
	m_fid(ptr.m_fid), m_start(ptr.m_start), m_end(ptr.m_end), m_ptr(ptr.m_ptr),
	m_tlb(ptr.m_tlb), m_mem(ptr.m_mem), m_frames(ptr.m_frames), m_flags(ptr.m_flags)
	{
		// Copies are invalid by definition
		m_ptr.valid = 0;

		// TODO: update virtual address in case of physical pointer
	}

	__device__ ~FatPointer()
	{
	}

	__device__ FatPointer& operator=( const FatPointer& ptr )
	{
		m_fid = ptr.m_fid;
		m_start = ptr.m_start;
		m_end = ptr.m_end;
		m_flags = ptr.m_flags;

		m_ptr = ptr.m_ptr;

		// Copies are invalid by definition
		m_ptr.valid = 0;

		m_tlb = ptr.m_tlb;
		m_mem = ptr.m_mem;
		m_frames = ptr.m_frames;

		return *this;
	}

	// Move to exact offset from the beginning of the map
	__device__ FatPointer& moveTo( size_t offset )
	{
		if( m_ptr.valid )
		{
			m_ptr.valid = 0;

			bool resolved = false;

			while( !resolved )
			{
				// Gather every thread that needs the same page
				int invalidThreads = __ballot( 1 );

				int leader = __ffs( invalidThreads );

				// Correction because __ffs start counting from 1;
				leader -= 1;

				BroadcastHelper bHelper;
				bHelper.l = m_ptr.physPage;
				bHelper = broadcast( bHelper, leader );

				int want = (m_ptr.physPage == bHelper.l);
				int wantThreads = __ballot( want );
				int numWants = __popc( wantThreads );

				if( LANE_ID == leader )
				{
					m_frames[m_ptr.physPage].unlock_rw(numWants);
				}

				if( want )
				{
					resolved = true;
				}
			}

			// Fall through
		}

		m_ptr.virtPage = (m_start + offset) >> FS_LOGBLOCKSIZE;
		m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;

		return *this;
	}

	// Move to offset from current location
	__device__ FatPointer& move( size_t offset )
	{
		offset *= sizeof(T);

		if( m_ptr.valid )
		{
			// Try to keep it valid
			if( 0 == ((m_ptr.pageOffset + offset) >> FS_LOGBLOCKSIZE) )
			{
				// We're still in the same block, just update the physical offset
				m_ptr.pageOffset += offset;
				return *this;
			}

			// else
			m_ptr.valid = 0;

			bool resolved = false;

			while( !resolved )
			{
				// Gather every thread that needs the same page
				int invalidThreads = __ballot( 1 );

				int leader = __ffs( invalidThreads );

				// Correction because __ffs start counting from 1;
				leader -= 1;

				BroadcastHelper bHelper;
				bHelper.l = m_ptr.physPage;
				bHelper = broadcast( bHelper, leader );

				int want = (m_ptr.physPage == bHelper.l);
				int wantThreads = __ballot( want );
				int numWants = __popc( wantThreads );

				long long virtPage;

				if( LANE_ID == leader )
				{
					virtPage = m_frames[m_ptr.physPage].file_offset >> FS_LOGBLOCKSIZE;
					m_frames[m_ptr.physPage].unlock_rw(numWants);
				}

				bHelper.l = virtPage;
				bHelper = broadcast( bHelper, leader );

				if( want )
				{
					m_ptr.virtPage = bHelper.l;
					resolved = true;
				}
			}
			// Fall through
		}

		m_ptr.virtPage = m_ptr.virtPage + ((m_ptr.pageOffset + offset) >> FS_LOGBLOCKSIZE);
		m_ptr.pageOffset = (m_ptr.pageOffset + offset) & FS_BLOCKMASK;

		return *this;
	}

	__forceinline__ __device__ T& operator *()
	{
		int valid = (m_ptr.valid);
		int allValid = __all( valid );

		if( allValid )
		{
			uchar* pRet = m_mem + m_ptr.offset;
			return *((T*)pRet);
		}

		while( true )
		{
			valid = (m_ptr.valid);
			int invalidThread = __ballot( !valid );

			int leader = __ffs( invalidThread );

			if( 0 == leader )
			{
				// No invalid threads
				break;
			}

			// Correction because __ffs start counting from 1;
			leader -= 1;

			BroadcastHelper bHelper;
			bHelper.l = m_ptr.virtPage;
			bHelper = broadcast( bHelper, leader );

			int want = (m_ptr.virtPage == bHelper.l);
			int wantThreads = __ballot( want );
			int numWants = __popc( wantThreads );

			volatile void* ptr = gmmap_warp(NULL, FS_BLOCKSIZE, 0, m_flags, m_fid, bHelper.l << FS_LOGBLOCKSIZE, numWants);
			int physical = ((size_t)ptr - (size_t)m_mem) >> FS_LOGBLOCKSIZE;

			physical = __shfl( physical, 0 );

			if( want )
			{
				m_ptr.physPage = physical;
				m_ptr.valid = 1;
			}
		}

		uchar* pRet = m_mem + m_ptr.offset;
		return *((T*)pRet);
	}

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
