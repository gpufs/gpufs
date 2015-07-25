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
__device__ volatile void* gmmap_warp(void *addr, size_t size, int prot, int flags, int fd, off_t offset);
__device__ int gmunmap_warp( volatile void *addr, size_t length );

__forceinline__ __device__ unsigned long long getTicks()
{
	unsigned long long ticks;
	asm volatile ("mov.u64 %0, %%clock64;" : "=l"(ticks) :);
	return ticks;
}

__forceinline__ __device__ int acc2Bitfeild( int acc )
{
	switch (acc) {
		case O_GRDONLY 	: return 0x1;
		case O_GWRONLY 	: return 0x2;
		case O_GCREAT 	: return 0x4;
		case O_GRDWR 	: return 0x3;
		case O_GWRONCE 	: return 0x6;
		default 		: return 0;
	}
}

__forceinline__ __device__ int bitfeild2Acc( int bits )
{
	switch (bits) {
		case 0x1 	: return O_GRDONLY;
		case 0x2 	: return O_GWRONLY;
		case 0x4 	: return O_GCREAT;
		case 0x3 	: return O_GRDWR;
		case 0x6 	: return O_GWRONCE;
		default 	: return 0;
	}
}

static const unsigned int ACCESS_BITS = 3;
static const unsigned int VPAGE_BITS = 64 - 32 - 1 - ACCESS_BITS; // 28

static const unsigned int FID_BITS = 64 - (32 - FS_LOGBLOCKSIZE) - VPAGE_BITS;

static const unsigned int INVALID_VPAGE = (1 << VPAGE_BITS) - 1;
static const unsigned int INVALID_FID = (1 << FID_BITS) - 1;

struct __align__(8) _FatPtr
{
	unsigned int pageOffset 	: FS_LOGBLOCKSIZE;
	unsigned int physicalPage 	: (32 - FS_LOGBLOCKSIZE);
	unsigned int vpage 			: VPAGE_BITS;
	unsigned int accBits 		: ACCESS_BITS;
	unsigned int valid 			: 1;
};

struct __align__(8) _TlbLine
{
	unsigned int physicalPage : (32 - FS_LOGBLOCKSIZE);
	unsigned int vpage : VPAGE_BITS;
	unsigned int fid : FID_BITS;
};

template<int N>
struct TLB
{
	_TlbLine lines[N];
	int locks[N];

	__device__ TLB()
	{
		if( TID == 0 )
		{
			for( int i = 0; i < N; ++i )
			{
				lines[i].physicalPage = 0;
				lines[i].vpage = INVALID_VPAGE;
				lines[i].fid = INVALID_FID;
				locks[i] = 0;
			}
		}
	}

	__device__ ~TLB()
	{
		if( threadIdx.x == 0 )
		{
			for( int i = threadIdx.y; i < N; i += blockDim.y )
			{
				if( lines[i].fid != INVALID_FID )
				{
					g_ppool->frames[lines[i].physicalPage].unlock_rw();
				}
			}
		}
	}
};

template<typename T, int N>
class FatPointer
{
public:
	__device__ FatPointer( size_t fid, off_t start, size_t size, int flags, TLB<N>* tlb, uchar* mem, volatile PFrame* frames ) :
	m_fid(fid), m_start(start), m_end(start + size)
	{
		m_ptr.pageOffset 	= 0;
		m_ptr.physicalPage 	= 0;
		m_ptr.vpage 		= start >> FS_LOGBLOCKSIZE;
		m_ptr.accBits 		= acc2Bitfeild( flags );
		m_ptr.valid 		= 0;

		m_tlb = tlb;

		m_mem = mem;
		m_frames = (PFrame*)frames;

//		time1 = 0;
//		time2 = 0;
//		time3 = 0;
//		time4 = 0;
//
//		count1 = 0;
//		count2 = 0;
//		count3 = 0;
//		count4 = 0;
	}

	__device__ FatPointer( const FatPointer& ptr ) :
		m_fid(ptr.m_fid), m_start(ptr.m_start), m_end(ptr.m_end), m_ptr(ptr.m_ptr),
		m_tlb(ptr.m_tlb), m_mem(ptr.m_mem), m_frames(ptr.m_frames)
	{
		// Copies are invalid by definition
		m_ptr.valid = 0;
	}

	__device__ ~FatPointer()
	{
		if( m_ptr.valid )
		{
			// Decrease ref count since we no longer hold the page
			size_t h = hash( m_ptr.vpage );
			int old = atomicSub( (int*)&(m_tlb->locks[h]), 1 );
		}
	}

	__device__ FatPointer& operator=( const FatPointer& ptr )
	{
		m_fid = ptr.m_fid;
		m_start = ptr.m_start;
		m_end = ptr.m_end;

		m_ptr.vpage = ptr.m_ptr.vpage;
		m_ptr.physicalPage = ptr.m_ptr.physicalPage;
		m_ptr.pageOffset = ptr.m_ptr.pageOffset;

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
			// Try to keep it valid
			if( m_ptr.vpage == (m_start + offset) >> FS_LOGBLOCKSIZE )
			{
				// We're still in the same block, just update the physical offset
				m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;
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

				int query = m_ptr.physicalPage;
				query = broadcast( query, leader );

				int want = (m_ptr.physicalPage == query);
				int wantThreads = __ballot( want );
				int numWants = __popc( wantThreads );

				if( LANE_ID == leader )
				{
					// Decrease ref count since we no longer hold the page
					size_t h = hash( m_ptr.vpage );
					int old = atomicSub( (int*)&(m_tlb->locks[h]), numWants );
				}

				if( want )
				{
					resolved = true;
				}
			}

			// Fall through
		}

		m_ptr.vpage = (m_start + offset) >> FS_LOGBLOCKSIZE;
		m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;

		// We can leave physicalPage as it is cause it's invalid anyway

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

			// Decrease ref count since we no longer hold the page
			size_t h = hash( m_ptr.vpage );
			int old = atomicSub( (int*)&(m_tlb->locks[h]), 1 );

			// Fall through
		}

		m_ptr.vpage = m_ptr.vpage + ((m_ptr.pageOffset + offset) >> FS_LOGBLOCKSIZE);
		m_ptr.pageOffset = (m_ptr.pageOffset + offset) & FS_BLOCKMASK;

		// We can leave physicalPage as it is cause it's invalid anyway

		return *this;
	}

	__forceinline__ __device__ T& operator *()
	{
//		time1Start = getTicks();

		long long* t = reinterpret_cast<long long*>(&m_ptr);

		int valid = (*t < 0);
		int allValid = __all( valid );

		if( allValid )
		{
			int* ptr = reinterpret_cast<int*>(&m_ptr);
			uchar* pRet = m_mem + *ptr;

//			time1Stop = getTicks();
//			time1 += time1Stop - time1Start;

			return *((T*)pRet);
		}

		while( true )
		{
			valid = (*t < 0);
			int invalidThread = __ballot( !valid );

			int leader = __ffs( invalidThread );

			if( 0 == leader )
			{
				// No invalid threads
				break;
			}

			// Correction because __ffs start counting from 1;
			leader -= 1;

			int query = m_ptr.vpage;
			query = __shfl( query, leader );

			int want = (m_ptr.vpage == query);
			int wantThreads = __ballot( want );
			int numWants = __popc( wantThreads );

			int physical = 0;

			size_t h = hash( query );
			volatile _TlbLine &line = m_tlb->lines[h];

			int* pRefCount = &(m_tlb->locks[h]);

			int old;

			if( LANE_ID == 0 )
			{
				old = atomicAdd( pRefCount, numWants );
				DBGT( "start", WARP_ID, old, threadIdx.x );
			}
			old = __shfl( old, 0 );

			if( (old >= 0) && (line.fid = m_fid) && (line.vpage == query) )
			{
				// Found the page in the tlb
				physical = line.physicalPage;
			}
			else
			{
				// TODO: Add open addressing around here

				// Wrong page, decrease ref count
				if( LANE_ID == 0 )
				{
					old = atomicSub( pRefCount, numWants );
					DBGT( "revert", WARP_ID, old, threadIdx.x );
				}

				while( true )
				{
					if( LANE_ID == 0 )
					{
						old = atomicCAS(pRefCount, 0, INT_MIN);
						DBGT( "cas", WARP_ID, old, threadIdx.x );
					}
					old = __shfl( old, 0 );

//					DBGT( "o", WARP_ID, old, threadIdx.x );

					if( old > 0 )
					{
						DBGT( "positive", WARP_ID, *pRefCount, threadIdx.x );
						if( (line.fid == m_fid) && (line.vpage == query) )
						{
							// Someone added our line? maybe?
							if( LANE_ID == 0 )
							{
								old = atomicAdd( pRefCount, numWants );
								DBGT( "retry", WARP_ID, old, threadIdx.x );
							}
							old = __shfl( old, 0 );

							// Let's double check
							if( (old >= 0) && (line.fid == m_fid) && (line.vpage == query) )
							{
								// Found the page in the tlb
								physical = line.physicalPage;
								break;
							}
							else
							{
								// False alarm
								if( LANE_ID == 0 )
								{
									old = atomicSub( pRefCount, numWants );
									DBGT( "revert retry", WARP_ID, old, threadIdx.x );
								}
								old = __shfl( old, 0 );

								continue;
							}
						}
						else
						{
							DBGT( "positive-1", WARP_ID, *pRefCount, threadIdx.x );
							// Not our page
							continue;
						}
					}
					else if( old < 0 )
					{
						// line is locked
						continue;
					}
					else
					{
						// We locked the page, now we can do whatever we want
						// First check if we are evicting an existing map
						if( line.fid != INVALID_FID )
						{
							if( LANE_ID == 0 )
							{
								m_frames[line.physicalPage].unlock_rw();
							}
						}

//						time2Start = getTicks();
						volatile void* ptr = gmmap_warp(NULL, FS_BLOCKSIZE, 0, bitfeild2Acc( m_ptr.accBits ), m_fid, (size_t)query << FS_LOGBLOCKSIZE);
//						time2Stop = getTicks();
//						count1++;
//						time2 += time2Stop - time2Start;

						if( LANE_ID == 0 )
						{
							physical = ((size_t)ptr - (size_t)m_mem) >> FS_LOGBLOCKSIZE;


							line.fid = m_fid;
							line.vpage = query;
							line.physicalPage = physical;
						}

						threadfence();

						if( LANE_ID == 0 )
						{
							old = atomicAdd(pRefCount, numWants - INT_MIN);
							DBGT( "unlock", WARP_ID, old, threadIdx.x );
//							GDBG( "numWants", WARP_ID, numWants );
//							GDBG( "pRefCount", WARP_ID, *pRefCount );
						}

						break;
					}
				}
			}

			physical = __shfl( physical, 0 );

			if( want )
			{
				m_ptr.physicalPage = physical;
				m_ptr.valid = 1;
			}
		}

		int* ptr = reinterpret_cast<int*>(&m_ptr);
		uchar* pRet = m_mem + *ptr;

//		time1Stop = getTicks();
//		time1 += time1Stop - time1Start;

		return *((T*)pRet);
	}

	__forceinline__ __device__ T read()
	{
//		time1Start = getTicks();
		if( ( m_ptr.accBits & 0x1 ) == 0 )
		{
			GPU_ASSERT(false);
			return 0;
		}

		long long* t = reinterpret_cast<long long*>(&m_ptr);

		int valid = (*t < 0);
		int allValid = __all( valid );

		if( allValid )
		{
			int* ptr = reinterpret_cast<int*>(&m_ptr);
			uchar* pRet = m_mem + *ptr;

//			time1Stop = getTicks();
//			time1 += time1Stop - time1Start;

			return *((T*)pRet);
		}

		while( true )
		{
			valid = (*t < 0);
			int invalidThread = __ballot( !valid );

			int leader = __ffs( invalidThread );

			if( 0 == leader )
			{
				// No invalid threads
				break;
			}

			// Correction because __ffs start counting from 1;
			leader -= 1;

			int query = m_ptr.vpage;
			query = __shfl( query, leader );

			int want = (m_ptr.vpage == query);
			int wantThreads = __ballot( want );
			int numWants = __popc( wantThreads );

			int physical = 0;

			size_t h = hash( query );
			volatile _TlbLine &line = m_tlb->lines[h];

			int* pRefCount = &(m_tlb->locks[h]);

			int old;

			if( LANE_ID == 0 )
			{
				old = atomicAdd( pRefCount, numWants );
				DBGT( "start", WARP_ID, old, threadIdx.x );
			}
			old = __shfl( old, 0 );

			if( (old >= 0) && (line.fid == m_fid) && (line.vpage == query) )
			{
				// Found the page in the tlb
				physical = line.physicalPage;
			}
			else
			{
				// TODO: Add open addressing around here

				// Wrong page, decrease ref count
				if( LANE_ID == 0 )
				{
					old = atomicSub( pRefCount, numWants );
					DBGT( "revert", WARP_ID, old, threadIdx.x );
				}

				while( true )
				{
					if( LANE_ID == 0 )
					{
						old = atomicCAS(pRefCount, 0, INT_MIN);
						DBGT( "cas", WARP_ID, old, threadIdx.x );
					}
					old = __shfl( old, 0 );

//					DBGT( "o", WARP_ID, old, threadIdx.x );

					if( old > 0 )
					{
						DBGT( "positive", WARP_ID, *pRefCount, threadIdx.x );
						if( (line.fid == m_fid) && (line.vpage == query) )
						{
							// Someone added our line? maybe?
							if( LANE_ID == 0 )
							{
								old = atomicAdd( pRefCount, numWants );
								DBGT( "retry", WARP_ID, old, threadIdx.x );
							}
							old = __shfl( old, 0 );

							// Let's double check
							if( (old >= 0) && (line.fid == m_fid) && (line.vpage == query) )
							{
								// Found the page in the tlb
								physical = line.physicalPage;
								break;
							}
							else
							{
								// False alarm
								if( LANE_ID == 0 )
								{
									old = atomicSub( pRefCount, numWants );
									DBGT( "revert retry", WARP_ID, old, threadIdx.x );
								}
								old = __shfl( old, 0 );

								continue;
							}
						}
						else
						{
							DBGT( "positive-1", WARP_ID, *pRefCount, threadIdx.x );
							// Not our page
							continue;
						}
					}
					else if( old < 0 )
					{
						// line is locked
						continue;
					}
					else
					{
						// We locked the page, now we can do whatever we want
						// First check if we are evicting an existing map
						if( line.fid != INVALID_FID )
						{
							if( LANE_ID == 0 )
							{
								m_frames[line.physicalPage].unlock_rw();
							}
						}

//						time2Start = getTicks();
						volatile void* ptr = gmmap_warp(NULL, FS_BLOCKSIZE, 0, bitfeild2Acc( m_ptr.accBits ), m_fid, (size_t)query << FS_LOGBLOCKSIZE);
//						time2Stop = getTicks();
//						count1++;
//						time2 += time2Stop - time2Start;

						if( LANE_ID == 0 )
						{
							physical = ((size_t)ptr - (size_t)m_mem) >> FS_LOGBLOCKSIZE;


							line.fid = m_fid;
							line.vpage = query;
							line.physicalPage = physical;
						}

						threadfence();

						if( LANE_ID == 0 )
						{
							old = atomicAdd(pRefCount, numWants - INT_MIN);
							DBGT( "unlock", WARP_ID, old, threadIdx.x );
//							GDBG( "numWants", WARP_ID, numWants );
//							GDBG( "pRefCount", WARP_ID, *pRefCount );
						}

						break;
					}
				}
			}

			physical = __shfl( physical, 0 );

			if( want )
			{
				m_ptr.physicalPage = physical;
				m_ptr.valid = 1;
			}
		}

		int* ptr = reinterpret_cast<int*>(&m_ptr);
		uchar* pRet = m_mem + *ptr;

//		time1Stop = getTicks();
//		time1 += time1Stop - time1Start;

		return *((T*)pRet);
	}

	__forceinline__ __device__ void write(T in)
	{
//		time1Start = getTicks();
		if( ( m_ptr.accBits & 0x2 ) == 0 )
		{
			GPU_ASSERT(false);
			return;
		}

		long long* t = reinterpret_cast<long long*>(&m_ptr);

		int valid = (*t < 0);
		int allValid = __all( valid );

		if( allValid )
		{
			int* ptr = reinterpret_cast<int*>(&m_ptr);
			uchar* pRet = m_mem + *ptr;

//			time1Stop = getTicks();
//			time1 += time1Stop - time1Start;

			*((T*)pRet) = in;
			return;
		}

		while( true )
		{
			valid = (*t < 0);
			int invalidThread = __ballot( !valid );

			int leader = __ffs( invalidThread );

			if( 0 == leader )
			{
				// No invalid threads
				break;
			}

			// Correction because __ffs start counting from 1;
			leader -= 1;

			int query = m_ptr.vpage;
			query = __shfl( query, leader );

			int want = (m_ptr.vpage == query);
			int wantThreads = __ballot( want );
			int numWants = __popc( wantThreads );

			int physical = 0;

			size_t h = hash( query );
			volatile _TlbLine &line = m_tlb->lines[h];

			int* pRefCount = &(m_tlb->locks[h]);

			int old;

			if( LANE_ID == 0 )
			{
				old = atomicAdd( pRefCount, numWants );
				DBGT( "start", WARP_ID, old, threadIdx.x );
			}
			old = __shfl( old, 0 );

			if( (old >= 0) && (line.fid = m_fid) && (line.vpage == query) )
			{
				// Found the page in the tlb
				physical = line.physicalPage;
			}
			else
			{
				// TODO: Add open addressing around here

				// Wrong page, decrease ref count
				if( LANE_ID == 0 )
				{
					old = atomicSub( pRefCount, numWants );
					DBGT( "revert", WARP_ID, old, threadIdx.x );
				}

				while( true )
				{
					if( LANE_ID == 0 )
					{
						old = atomicCAS(pRefCount, 0, INT_MIN);
						DBGT( "cas", WARP_ID, old, threadIdx.x );
					}
					old = __shfl( old, 0 );

//					DBGT( "o", WARP_ID, old, threadIdx.x );

					if( old > 0 )
					{
						DBGT( "positive", WARP_ID, *pRefCount, threadIdx.x );
						if( (line.fid == m_fid) && (line.vpage == query) )
						{
							// Someone added our line? maybe?
							if( LANE_ID == 0 )
							{
								old = atomicAdd( pRefCount, numWants );
								DBGT( "retry", WARP_ID, old, threadIdx.x );
							}
							old = __shfl( old, 0 );

							// Let's double check
							if( (old >= 0) && (line.fid == m_fid) && (line.vpage == query) )
							{
								// Found the page in the tlb
								physical = line.physicalPage;
								break;
							}
							else
							{
								// False alarm
								if( LANE_ID == 0 )
								{
									old = atomicSub( pRefCount, numWants );
									DBGT( "revert retry", WARP_ID, old, threadIdx.x );
								}
								old = __shfl( old, 0 );

								continue;
							}
						}
						else
						{
							DBGT( "positive-1", WARP_ID, *pRefCount, threadIdx.x );
							// Not our page
							continue;
						}
					}
					else if( old < 0 )
					{
						// line is locked
						continue;
					}
					else
					{
						// We locked the page, now we can do whatever we want
						// First check if we are evicting an existing map
						if( line.fid != INVALID_FID )
						{
							if( LANE_ID == 0 )
							{
								m_frames[line.physicalPage].unlock_rw();
							}
						}

//						time2Start = getTicks();
						volatile void* ptr = gmmap_warp(NULL, FS_BLOCKSIZE, 0, bitfeild2Acc( m_ptr.accBits ), m_fid, (size_t)query << FS_LOGBLOCKSIZE);
//						time2Stop = getTicks();
//						count1++;
//						time2 += time2Stop - time2Start;

						if( LANE_ID == 0 )
						{
							physical = ((size_t)ptr - (size_t)m_mem) >> FS_LOGBLOCKSIZE;


							line.fid = m_fid;
							line.vpage = query;
							line.physicalPage = physical;
						}

						threadfence();

						if( LANE_ID == 0 )
						{
							old = atomicAdd(pRefCount, numWants - INT_MIN);
							DBGT( "unlock", WARP_ID, old, threadIdx.x );
//							GDBG( "numWants", WARP_ID, numWants );
//							GDBG( "pRefCount", WARP_ID, *pRefCount );
						}

						break;
					}
				}
			}

			physical = __shfl( physical, 0 );

			if( want )
			{
				m_ptr.physicalPage = physical;
				m_ptr.valid = 1;
			}
		}

		int* ptr = reinterpret_cast<int*>(&m_ptr);
		uchar* pRet = m_mem + *ptr;

//		time1Stop = getTicks();
//		time1 += time1Stop - time1Start;

		*((T*)pRet) = in;
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
	__device__ size_t hash( size_t vpage )
	{
		const uint fidBits = 2;
		const uint vPageBits = ilogb((float)N) - fidBits;

		const uint fidMask = (1 << fidBits) - 1;
		const uint vPageMask = (1 << vPageBits) - 1;

		size_t res = ((m_fid & fidMask) << vPageBits) | (vpage & vPageMask);

		return res;
	}

	size_t m_fid;	// Should be spilled to local memory
	off_t m_start;	// Should be spilled to local memory
	size_t m_end;	// Should be spilled to local memory

	_FatPtr m_ptr;

	TLB<N> *m_tlb;
	uchar* m_mem;
	PFrame* m_frames;

//	unsigned long long time1;
//	unsigned long long time1Start;
//	unsigned long long time1Stop;
//
//	unsigned long long time2;
//	unsigned long long time2Start;
//	unsigned long long time2Stop;
//
//	unsigned long long time3;
//	unsigned long long time3Start;
//	unsigned long long time3Stop;
//
//	unsigned long long time4;
//	unsigned long long time4Start;
//	unsigned long long time4Stop;
//
//	unsigned long long count1;
//	unsigned long long count2;
//	unsigned long long count3;
//	unsigned long long count4;
};

#endif
