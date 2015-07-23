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

#include <limits.h>
#include "fat_pointer.cu.h"
#include "fs_constants.h"
#include "fs_calls.cu.h"

template<typename T, int N>
__device__ FatPointer<T,N>::FatPointer( size_t fid, off_t start, size_t size, int flags, TLB<N>* tlb, uchar* mem, volatile PFrame* frames ) :
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

template<typename T, int N>
__device__ FatPointer<T,N>::FatPointer( const FatPointer& ptr ) :
	m_fid(ptr.m_fid), m_start(ptr.m_start), m_end(ptr.m_end), m_ptr(ptr.m_ptr),
	m_tlb(ptr.m_tlb), m_mem(ptr.m_mem), m_frames(ptr.m_frames), m_flags(ptr.m_flags)
{
	// Copies are invalid by definition
	m_ptr.valid = 0;

	// TODO: update virtual address in case of physical pointer
}

template<typename T, int N>
__device__ FatPointer<T,N>::~FatPointer()
{
}

template<typename T, int N>
__device__ FatPointer<T,N>& FatPointer<T,N>::operator=( const FatPointer& ptr )
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
template<typename T, int N>
__device__ FatPointer<T,N>& FatPointer<T,N>::moveTo( size_t offset )
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
template<typename T, int N>
__device__ FatPointer<T,N>& FatPointer<T,N>::move( size_t offset )
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

template<typename T, int N>
__device__ T& FatPointer<T,N>::operator *()
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

template class FatPointer<volatile float, 16>;

