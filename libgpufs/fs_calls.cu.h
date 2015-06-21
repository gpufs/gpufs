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


#ifndef FS_CALLS_CU
#define FS_CALLS_CU

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "hash_table.cu.h"
#include "swapper.cu.h"
#include "fs_globals.cu.h"

// no reference counting here
__device__ int single_thread_fsync(int fd);
__device__ int single_thread_ftruncate(int fd, int size);
__device__ int single_thread_open(const char* filename, int flags);

#define READ 0
#define WRITE 1

__device__ int gclose(int fd);
__device__ int gopen(const char* filename, int flags);
__device__ int gmsync(volatile void *addr, size_t length,int flags);
__device__ int gmunmap(volatile void *addr, size_t length);
__device__ volatile void* gmmap(void *addr, size_t size,
		int prot, int flags, int fd, off_t offset);
__device__ volatile void* gmmap_threadblock(void *addr, size_t size,
		int prot, int flags, int fd, off_t offset);
__device__ size_t gwrite(int fd,size_t offset, size_t size, uchar* buffer);
__device__ size_t gread(int fd, size_t offset, size_t size, uchar* buffer);
__device__ uint gunlink(char* filename);
__device__ size_t fstat(int fd);


__device__ int gfsync(int fd);
__device__ int gftruncate(int fd, int size);

//struct _FatPtr
//{
//	unsigned int pageOffset : FS_LOGBLOCKSIZE;
//	unsigned int physicalPage : (32 - FS_LOGBLOCKSIZE);
//	unsigned int vpage : 28;
//	unsigned int ffu : 3;
//	unsigned int valid : 1;
//
//};
//
//struct _TlbLine
//{
//	unsigned int refCount : 12;
//	unsigned int physicalPage : (32 - FS_LOGBLOCKSIZE);
//	unsigned int fid : 4;
//	unsigned int vpage : 28;
//
//};
//
//union bytes
//{
//	_FatPtr ptr;
//	_TlbLine line;
//	int i[2];
//	char c[8];
//};
//
//static const int DEBUG_THREAD = 0;
//static const int DEBUG_BLOCK = 3;
//
//template< typename T>
//class FatPointer
//{
//public:
//	__device__ FatPointer( size_t fid, size_t start, size_t size, _TlbLine* tlb, int* leader ) :
//	m_fid(fid), m_start(start), m_end(start + size)
//	{
//		m_ptr.vpage = start >> FS_LOGBLOCKSIZE;
//		m_ptr.valid = 0;
//		m_ptr.physicalPage = 0;
//		m_ptr.pageOffset = 0;
//
//		m_tlb = tlb;
//		m_leader = leader;
//		m_mem = (uchar*)g_ppool->rawStorage;
//	}
//
//	// Move to exact offset from the beginning of the map
//	__device__ FatPointer& moveTo( size_t offset )
//	{
//		offset *= sizeof(T);
//
//		if( m_ptr.valid )
//		{
//			// Try to keep it valid
//			if( m_ptr.vpage == (m_start + offset) >> FS_LOGBLOCKSIZE )
//			{
//				// We're still in the same block, just update the physical offset
//				m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;
//				return *this;
//			}
//
//			// else
//			m_ptr.valid = 0;
//
//			// Decrease ref count since we no longer hold the page
//			size_t h = hash( m_ptr.vpage );
//			volatile _TlbLine &line = m_tlb[h];
//
//			atomicSub( (int*)&line, 1 );
//
//			// Fall through
//		}
//
//		m_ptr.vpage = (m_start + offset) >> FS_LOGBLOCKSIZE;
//		m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;
//
//		// We can leave physicalPage as it is cause it's invalid anyway
//
//		return *this;
//	}
//
//	// Move to offset from current location
//	__device__ FatPointer& move( size_t offset )
//	{
//		offset *= sizeof(T);
//
//		if( m_ptr.valid )
//		{
//			// Try to keep it valid
//			if( m_ptr.vpage == m_ptr.vpage + ((m_ptr.pageOffset + offset) >> FS_LOGBLOCKSIZE) )
//			{
//				// We're still in the same block, just update the physical offset
//				m_ptr.pageOffset += offset;
//				return *this;
//			}
//
//			// else
//			m_ptr.valid = 0;
//
//			// Decrease ref count since we no longer hold the page
//			size_t h = hash( m_ptr.vpage );
//			volatile _TlbLine &line = m_tlb[h];
//
//			atomicSub( (int*)&line, 1 );
//
//			// Fall through
//		}
//
//		m_ptr.vpage = m_ptr.vpage + ((m_ptr.pageOffset + offset) >> FS_LOGBLOCKSIZE);
//		m_ptr.pageOffset = (m_ptr.pageOffset + offset) & FS_BLOCKMASK;
//
//		// We can leave physicalPage as it is cause it's invalid anyway
//
//		return *this;
//	}
//
//	__device__ T operator *()
//	{
//		long long* t = reinterpret_cast<long long*>(&m_ptr);
//		unsigned int fault;
//
//
//		asm volatile(
//				".reg .pred p, q;\n\t"
//				"setp.lt.s64 p, %1, 0;\n\t"			// p = valid
//				"bar.red.and.pred   q, 0, p;\n\t"	// Synchronize and reduce q = all(p)
//				"selp.u32 %0, 0, 1, q;\n\t"
//				: "=r"(fault) : "l"(*t) );
//
//		if( !fault )
//		{
//			int* t = reinterpret_cast<int*>(&m_ptr);
//			uchar* pRet = ((uchar*)(m_mem) + *t);
//			return *((T*)pRet);
//		}
//
////		unsigned int valid = 0;
////		unsigned int fault;
////		unsigned int physical;
////
////		size_t h = hash( m_ptr.vpage );
////		volatile _TlbLine &line = m_tlb[h];
//////		unsigned long long t;
//////
//////		asm volatile(
//////				".reg .b64	offset, ptr, ptrs;\n\t"
//////				"cvta.to.shared.u64 ptrs, ptr;\n\t"
////////				"shl.b64	offset, %1, 3;\n\t"
////////				"add.s64	ptrs, %2, offset;\n\t"
//////				"ld.shared.u64	%0, [ptrs];\n\t"
//////				: "=l"(t) : "l"(h), "l"(m_tlb) : "memory" );
//////
//////		_TlbLine linet;
//////		linet = *(reinterpret_cast<_TlbLine*>(&t));
//////		_TlbLine linet = m_tlb[h];
////
//////		printf("(%d, %d): refCount: %d, fid: %d, vpage: %d\n", threadIdx.y, threadIdx.x, line.refCount, line.fid, m_ptr.vpage);
////
////		if( line.refCount > 0 && line.fid == m_fid && line.vpage == m_ptr.vpage )
////		{
////			// Found the page in the tlb
////			physical = line.physicalPage << FS_LOGBLOCKSIZE;
////			valid = 1;
////		}
////
////
////		asm volatile(
////				".reg .pred p, q;\n\t"
////				"setp.ne.u32 p, %1, 0;\n\t"			// p = valid
////				"bar.red.and.pred   q, 0, p;\n\t"	// Synchronize and reduce q = all(p)
////				"selp.u32 %0, 0, 1, q;\n\t"
////				: "=r"(fault) : "r"(valid) );
////
////		if( !fault )
////		{
//////			printf("no fault\n");
////			uchar* pRet = (uchar*)m_mem + physical + m_ptr.pageOffset;
////			return *((T*)pRet);
////		}
//
//		int tx = threadIdx.x;
//		int ty = threadIdx.y;
//
//		int threadID = ty * 32 + tx;
//
//		int bx = blockIdx.x;
//		int by = blockIdx.y;
//
//		int blockID = by * gridDim.x + bx;
//
//		__shared__ volatile int* leader;
//
//		// Initialize leader to non-zero so we can know when we're done
//		if( threadIdx.x + threadIdx.y + threadIdx.z == 0 )
//		{
//			leader = m_leader;
//			*leader = -2;
//		}
//		syncthreads();
//
//		size_t h = hash( m_ptr.vpage );
//		volatile _TlbLine &line = m_tlb[h];
//
//		while( *leader != -1 )
//		{
//			//			 if( 0 == tx )
//			//			 {
//			//				 printf("%d: loop\n", ty);
//			////				 printf("%d: valid: %d, refCounf: %d, fid: %d, vpage: %d\n", ty, m_ptr.valid, line.refCount, line.fid, m_ptr.valid);
//			//			 }
//
//			if( m_ptr.valid == 0 && line.refCount > 0 && line.fid == m_fid && line.vpage == m_ptr.vpage )
//			{
//				//				 if( 0 == tx )
//				//				 {
//				//					 printf("(%d,%d): found in tlb\n", ty, tx);
//				//				 }
//
//				// Found the page in the tlb
//				atomicAdd( (int*)&line, 1 );
//
//				m_ptr.physicalPage = line.physicalPage;
//				m_ptr.valid = 1;
//			}
//
//			syncthreads();
//
//			if( threadID == 0 )
//			{
//				*leader = -1;
//			}
//			syncthreads();
//
//			int old = 0;
//
//			if( !m_ptr.valid )
//				old = atomicCAS( (int*)leader, -1, m_ptr.vpage );
//
//			if( old == -1 )
//			{
//				// TODO: add collision handling
//				GPU_ASSERT( line.refCount == 0 );
//
//				line.fid = m_fid;
//				line.vpage = m_ptr.vpage;
//				line.refCount = 1;
//			}
//
//			// Found an empty slot in the TLB
//
//			syncthreads();
//
//			uchar* pRet = NULL;
//
//			if( *leader >= 0 )
//			{
//				pRet = (uchar*)gmmap(NULL, FS_BLOCKSIZE, 0, O_GRDONLY, m_fid, ((size_t)(*leader)) << FS_LOGBLOCKSIZE);
//			}
//
//			if( old == -1 )
//			{
//				line.physicalPage = ((size_t)pRet - (size_t)(g_ppool->rawStorage)) >> FS_LOGBLOCKSIZE;
//
//				 m_ptr.physicalPage = ((size_t)pRet - (size_t)(g_ppool->rawStorage)) >> FS_LOGBLOCKSIZE;
//				 m_ptr.valid = 1;
//			}
//
//			syncthreads();
//		}
//
//		GPU_ASSERT( m_ptr.valid );
//
//		uchar* pRet = ((uchar*)(m_mem) + ((size_t)(m_ptr.physicalPage) << FS_LOGBLOCKSIZE)) + m_ptr.pageOffset;
//		return *((T*)pRet);
//
//
//		//		volatile uchar* pRet = (volatile uchar*)gmmap(NULL, FS_BLOCKSIZE, 0, O_GRDONLY, m_fid, ((size_t)m_ptr.vpage) << FS_LOGBLOCKSIZE);
//		//
//		//		m_ptr.physicalPage = ((size_t)pRet - (size_t)(g_ppool->rawStorage)) >> FS_LOGBLOCKSIZE;
//		//		m_ptr.valid = 1;
//		//
//		//		pRet += m_ptr.pageOffset;
//		//
//		//		return *((T*)pRet);
//	}
//
//	__device__ FatPointer& operator += ( size_t offset )
// 							{
//		return move(offset);
// 							}
//
//	__device__ void print()
//	{
//		int bx = blockIdx.x;
//		int by = blockIdx.y;
//
//		int blockID = by * gridDim.x + bx;
//
//		int tx = threadIdx.x;
//		int ty = threadIdx.y;
//
//		int threadID = ty * 32 + tx;
//
//		if( DEBUG_THREAD == threadID )
//		{
//			printf("%d: vpage: %d, hash: %ld, valid: %d, physicalPage: %d, pageOffset: %d\n", blockID, m_ptr.vpage, hash(m_ptr.vpage), m_ptr.valid, m_ptr.physicalPage, m_ptr.pageOffset);
//		}
//	}
//
//	__device__ void printAddr()
//	{
//		int bx = blockIdx.x;
//		int by = blockIdx.y;
//
//		int blockID = by * gridDim.x + bx;
//
//		int tx = threadIdx.x;
//		int ty = threadIdx.y;
//
//		int threadID = ty * 32 + tx;
//
//		//		if( DEBUG_BLOCK == blockID &&  DEBUG_THREAD == threadID )
//		//		{
//		//			volatile uchar* pRet = ((volatile uchar*)(g_ppool->rawStorage) + (m_ptr.physicalPage << FS_LOGBLOCKSIZE)) + m_ptr.pageOffset;
//		//			printf("Address: %lx\n", pRet);
//		//		}
//	}
//
//	__device__ void printVal()
//	{
//		int bx = blockIdx.x;
//		int by = blockIdx.y;
//
//		int blockID = by * gridDim.x + bx;
//
//		int tx = threadIdx.x;
//		int ty = threadIdx.y;
//
//		int threadID = ty * 32 + tx;
//
//		//		auto f = [=] () -> void* { return storage; };
//
//		//		if( DEBUG_BLOCK == blockID &&  DEBUG_THREAD == threadID )
//		//		{
//		//			volatile uchar* pRet = ((volatile uchar*)(g_ppool->rawStorage) + (m_ptr.physicalPage << FS_LOGBLOCKSIZE)) + m_ptr.pageOffset;
//		//			printf("Val: %f\n", *((float*)pRet));
//		//		}
//	}
//
//	//	// Move to a specific offset and return the value
//	//	// Equal to *(moveTo( offset ))
//	//	__device__ T  operator [](size_t i)
//	//	{
//	//		size_t offset = i* sizeof(T);
//	//
//	//		if( m_ptr.valid )
//	//		{
//	//			// Try to keep it valid
//	//			if( m_ptr.vpage == (m_start + offset) >> FS_LOGBLOCKSIZE )
//	//			{
//	//				// We're still in the same block
//	//				m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;
//	//
//	//				volatile uchar* pRet = (volatile uchar*)(g_ppool->rawStorage) +
//	//						               (m_ptr.physicalPage << FS_LOGBLOCKSIZE) +
//	//						               m_ptr.pageOffset;
//	//				return *((T*)pRet);
//	//			}
//	//
//	//			// Fall through
//	//		}
//	//
//	//		m_ptr.vpage = (m_start + offset) >> FS_LOGBLOCKSIZE;
//	//		m_ptr.pageOffset = (m_start + offset) & FS_BLOCKMASK;
//	//
//	//		volatile uchar* pRet = (volatile uchar*)gmmap(NULL, FS_BLOCKSIZE, 0, O_GRDONLY, m_fid, m_ptr.vpage << FS_LOGBLOCKSIZE);
//	//
//	//		m_ptr.physicalPage = ((size_t)pRet - (size_t)(g_ppool->rawStorage)) >> FS_LOGBLOCKSIZE;
//	//		m_ptr.valid = 1;
//	//
//	//		pRet += m_ptr.pageOffset;
//	//
//	//		return *((T*)pRet);
//	//	}
//
//private:
//	__device__ size_t hash( size_t vpage )
//	{
//		size_t res = (m_fid << 5) & 0xE0;
//		res |= (vpage & 0x1F);
//
//		return res;
//	}
//
//	size_t m_fid;	// Should be spilled to local memory
//	size_t m_start;	// Should be spilled to local memory
//	size_t m_end;	// Should be spilled to local memory
//
//	_FatPtr m_ptr;
//
//	volatile _TlbLine *m_tlb;
//	int* m_leader;
//	uchar* m_mem;
//};


#endif
