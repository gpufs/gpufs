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

#include "fs_constants.h"

// Prevent circular include
__device__ volatile void* gmmap(void *addr, size_t size, int prot, int flags, int fd, off_t offset);
__device__ int gmunmap( volatile void *addr, size_t length );

static const int VPAGE_BITS = 28;
static const int FID_BITS = 4;
static const int REF_COUNT_BITS = 64 - FID_BITS - VPAGE_BITS - (32 - FS_LOGBLOCKSIZE);
static const int FFU_BITS = 32 - 1 - VPAGE_BITS;

struct __align__(8) _FatPtr
{
	unsigned int pageOffset : FS_LOGBLOCKSIZE;
	unsigned int physicalPage : (32 - FS_LOGBLOCKSIZE);
	unsigned int vpage : VPAGE_BITS;
	unsigned int ffu : FFU_BITS;
	unsigned int valid : 1;
};

struct __align__(8) _TlbLine
{
	unsigned int refCount : REF_COUNT_BITS;
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
		for( int i = 0; i < N; ++i )
		{
			lines[i].refCount = 0;
			lines[i].physicalPage = 0;
			lines[i].vpage = -1;
			lines[i].fid = -1;
			locks[i] = 0;
		}
	}
};

template<typename T, int N>
class FatPointer
{
public:
	__device__ FatPointer( size_t fid, off_t start, size_t size, int flags, TLB<N>* tlb, uchar* mem ) :
	m_fid(fid), m_start(start), m_end(start + size), m_flags(flags)
	{
		m_ptr.vpage = start >> FS_LOGBLOCKSIZE;
		m_ptr.valid = 0;
		m_ptr.physicalPage = 0;
		m_ptr.pageOffset = 0;

		m_tlb = tlb;

		m_mem = mem;
	}

	__device__ FatPointer( const FatPointer& ptr ) :
	m_fid(ptr.m_fid), m_start(ptr.m_start), m_end(ptr.m_end), m_ptr(ptr.m_ptr), m_tlb(ptr.m_tlb), m_mem(ptr.m_mem), m_flags(ptr.m_flags)
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
		m_flags = ptr.m_flags;

		m_ptr.vpage = ptr.m_ptr.vpage;
		m_ptr.physicalPage = ptr.m_ptr.physicalPage;
		m_ptr.pageOffset = ptr.m_ptr.pageOffset;

		// Copies are invalid by definition
		m_ptr.valid = 0;

		m_tlb = ptr.tlb;

		m_mem = ptr.mem;

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

			// Decrease ref count since we no longer hold the page
			size_t h = hash( m_ptr.vpage );
			int old = atomicSub( (int*)&(m_tlb->locks[h]), 1 );

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
			if( m_ptr.vpage == m_ptr.vpage + ((m_ptr.pageOffset + offset) >> FS_LOGBLOCKSIZE) )
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

	__forceinline__ __device__ T operator *()
	{
		long long* t = reinterpret_cast<long long*>(&m_ptr);

		int valid = (*t < 0);
		int allValid = __all( valid );

		if( allValid )
		{
			int* ptr = reinterpret_cast<int*>(&m_ptr);
			uchar* pRet = m_mem + *ptr;
			return *((T*)pRet);
		}

		int laneID = TID & 0x1f;

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

			if( laneID == leader )
			{
				size_t h = hash( query );
				volatile _TlbLine &line = m_tlb->lines[h];

				int* pRefCount = &(m_tlb->locks[h]);

				int old = atomicAdd( pRefCount, numWants );

				if( (old >= 0) && (line.fid = m_fid) && (line.vpage == query) )
				{
					// Found the page in the tlb
					physical = line.physicalPage;
				}
				else
				{
					// TODO: Add open addressing around here

					// Wrong page, decrease ref count
					atomicSub( pRefCount, numWants );

					while( true )
					{
						old = atomicCAS(pRefCount, 0, INT32_MIN);

						if( old > 0 )
						{
							if( (line.fid = m_fid) && (line.vpage == query) )
							{
								// Someone added our line? maybe?
								old = atomicAdd( pRefCount, numWants );

								// Let's double check
								if( (old >= 0) && (line.fid = m_fid) && (line.vpage == query) )
								{
									// Found the page in the tlb
									physical = line.physicalPage;
									break;
								}
								else
								{
									// False alarm
									atomicSub( pRefCount, numWants );

									continue;
								}
							}
							else
							{
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
							if( line.fid != -1 )
							{
								int oldPhysical = line.physicalPage;
								volatile void* ptr = m_mem + ((size_t)oldPhysical << FS_LOGBLOCKSIZE);
								gmunmap( ptr, FS_BLOCKSIZE );
							}

							volatile void* ptr = gmmap(NULL, FS_BLOCKSIZE, 0, m_flags, m_fid, (size_t)query << FS_LOGBLOCKSIZE);

							physical = ((size_t)ptr - (size_t)m_mem) >> FS_LOGBLOCKSIZE;

							line.fid = m_fid;
							line.vpage = query;
							line.physicalPage = physical;

							threadfence();

							atomicAdd(pRefCount, numWants - INT32_MIN);

							break;
						}
					}
				}
			}

			physical = __shfl( physical, leader );

			if( want )
			{
				m_ptr.physicalPage = physical;
				m_ptr.valid = 1;
			}
		}

		int* ptr = reinterpret_cast<int*>(&m_ptr);
		uchar* pRet = m_mem + *ptr;
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
	int m_flags;	// Should be spilled to local memory

	_FatPtr m_ptr;

	TLB<N> *m_tlb;
	uchar* m_mem;
};

#endif
