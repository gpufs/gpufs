#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"
#include "loader.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "histLib.h"
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <immintrin.h>

#define AVX

// Needed to fix missing declaration
#define _mm256_set_m128(va, vb) \
	        _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)

#define as_float(t)\
			(*((float*)(&t)))

using namespace std;
using namespace tbb;
using namespace cv;

typedef unsigned int uint;
typedef unsigned char uchar;

static const int MAX_CAND_LIST_SIZE = 256;
static const int ALL_ONES = -1;

struct Pixel
{
	uchar r;
	uchar g;
	uchar b;
	uchar a;
};

static const int BLOCK_X = 43;
static const int BLOCK_Y = 15;

double __diff__(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec < start.tv_nsec)) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}

	double elapsed = temp.tv_sec + ((double) temp.tv_nsec / 1000000000.0);
	return elapsed;
}

static int64_t __diffn__(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec < start.tv_nsec)) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}

	int64_t elapsed = temp.tv_sec * 1000000000 + temp.tv_nsec;
	return elapsed;
}

template< class T>
class UniqListImpl
{
private:
	unordered_set<T> uniqList;
	typename unordered_set<T>::iterator it;
	bool first;

public:
	UniqListImpl() { first = true; }
	void insert( T t ) { uniqList.insert(t); }
	bool find( T t )
	{
		return (uniqList.count(t) == 1);
	}
	int size() { return uniqList.size(); }
	int next()
	{
		if( first )
		{
			it = uniqList.begin();
			first = false;
			return *it;
		}
		else
		{
			it++;
			return *it;
		}

	}
};

template< class KEY, class VAL >
class UniqMapImpl
{
private:
	unordered_map<KEY, VAL> uniqMap;

public:
	void insert( KEY key, KEY val )
	{
		uniqMap[key] = val;
	}

	bool find( KEY key )
	{
		return (uniqMap.count(key) == 1);
	}

	VAL get( KEY key )
	{
		return uniqMap[key];
	}

	int size()
	{
		return uniqMap.size();
	}
};

UniqList::UniqList()
{
	impl = new UniqListImpl<int>();
}

UniqList::~UniqList()
{
	delete impl;
}

void UniqList::insert( int t )
{
	impl->insert(t);
}

bool UniqList::find( int t )
{
	return impl->find(t);
}

int UniqList::size()
{
	return impl->size();
}

int UniqList::next()
{
	return impl->next();
}

UniqMap::UniqMap()
{
	impl = new UniqMapImpl<int, int>();
}

UniqMap::~UniqMap()
{
	delete impl;
}

void UniqMap::insert( int key, int val )
{
	impl->insert( key, val );
}

bool UniqMap::find( int key )
{
	return impl->find( key );
}

int UniqMap::get( int key )
{
	return impl->get( key );
}

int UniqMap::size()
{
	return impl->size();
}

void merge_single(int BLOCKS, uint* keys, int* cand, AlgData &data, uint* reverseGlobalImageList, size_t &numImages)
{
	UniqMap globalImageList;

//	UniqList UniqPages_12;
//	UniqList UniqPages_13;
//	UniqList UniqPages_14;
//	UniqList UniqPages_15;
//	UniqList UniqPages_16;

	for( int block = 0; block < BLOCKS; ++block )
	{
		UniqList imageList;

		for( int l = 0; l < data.L; ++l )
		{
			int key = keys[block * data.L + l];

			uint hIndex = key % data.TableSize;
			uint control = key;

			int chainIndex = data.hashTable[l][hIndex];
			if( ALL_ONES == chainIndex )
			{
				continue;
			}

			bool found = false;
			while( true )
			{
				if( data.chainTable[l][chainIndex] == control )
				{
					found = true;
					break;
				}

				if( data.chainTable[l][chainIndex + 1] < 0 )
				{
					// If MSB is set, this means it's the last node in the chain
					break;
				}

				chainIndex += 2;
			}

			if( !found )
			{
				continue;
			}

			// Remove the MSB, we no longer need it
			int idIndex = data.chainTable[l][chainIndex + 1] & 0x7FFFFFFF;
			while( true )
			{
				int id = data.indexTable[l][idIndex];

				// Insert id without the MSB
				imageList.insert( id & 0x7FFFFFFF );

				if( id < 0 )
				{
					break;
				}

				if( imageList.size() >= MAX_CAND_LIST_SIZE )
				{
					break;
				}

				idIndex++;
			}

			if( imageList.size() >= MAX_CAND_LIST_SIZE ) break;
		}

		int i = 0;
		for( ; i < imageList.size(); ++i )
		{
			int t = imageList.next();

			if( globalImageList.find(t) )
			{
				cand[block * MAX_CAND_LIST_SIZE + i] = globalImageList.get(t);
				continue;
			}

			globalImageList.insert( t, numImages );
			reverseGlobalImageList[numImages] = t;
			cand[block * MAX_CAND_LIST_SIZE + i] = numImages;
			numImages++;

			size_t offset = (size_t)t * HIST_SIZE_ON_DISK;

//			UniqPages_12.insert( offset >> 12 );
//			UniqPages_13.insert( offset >> 13 );
//			UniqPages_14.insert( offset >> 14 );
//			UniqPages_15.insert( offset >> 15 );
//			UniqPages_16.insert( offset >> 16 );
		}

		for( ; i < MAX_CAND_LIST_SIZE; ++i )
		{
			cand[block * MAX_CAND_LIST_SIZE + i] = -1;
		}
	}

	cout << "Total list size: " << numImages << endl;

//	cout << "Number of pages (12): " << UniqPages_12.size() << " (size: " << ((double)(UniqPages_12.size()) * (4096.0)) / (1024.0 * 1024.0 * 1024.0) << endl;
//	cout << "Number of pages (13): " << UniqPages_13.size() << " (size: " << ((double)(UniqPages_13.size()) * (2 * 4096.0)) / (1024.0 * 1024.0 * 1024.0) << endl;
//	cout << "Number of pages (14): " << UniqPages_14.size() << " (size: " << ((double)(UniqPages_14.size()) * (4 * 4096.0)) / (1024.0 * 1024.0 * 1024.0) << endl;
//	cout << "Number of pages (15): " << UniqPages_15.size() << " (size: " << ((double)(UniqPages_15.size()) * (8 * 4096.0)) / (1024.0 * 1024.0 * 1024.0) << endl;
//	cout << "Number of pages (16): " << UniqPages_16.size() << " (size: " << ((double)(UniqPages_16.size()) * (16 * 4096.0)) / (1024.0 * 1024.0 * 1024.0) << endl;
}

typedef tbb::concurrent_unordered_map<int, int> ConUniqMap;
tbb::task_group g;

uint computeProductModPrime(uint a[], uint key, uint K)
{
	size_t h = 0;

	for( uint i = 0; i < K; ++i )
	{
		h = h + a[i] * (( key >> i ) & 1);
		h = (h & TWO_TO_32_MINUS_1) + 5 * (h >> 32);
		if( h > UH_PRIME_DEFAULT )
		{
			h = h - UH_PRIME_DEFAULT;
		}
	}

	return h;
}

void mosaic_TBB( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS )
{
	cout << "Start mosaic_TBB" << endl;

	tbb::atomic<int64_t> calcKeys(0.);
	tbb::atomic<int64_t> readHists(0.);
	tbb::atomic<int64_t> choose(0.);

	const int COEF_SIZE = data.D * data.K * data.L * sizeof(float) * 4;

	Pixel* image = (Pixel*)inOut;

	float *coef = (float*)aligned_alloc( 32, COEF_SIZE );

	data.getCoefficents( coef );

	timespec startTotal, endTotal;

	double total = 0;

	clock_gettime(CLOCK_REALTIME, &startTotal);

	parallel_for(0, ROWS, 32, [&](int by)
	{
		timespec start, end;
		timespec startRead, endRead;

		float* hist = (float*)aligned_alloc( 32, 3 * 256 * sizeof(float) );
		float* candHist = (float*)aligned_alloc( 32, 3 * 256 * sizeof(float) );

		for( int bx = 0; bx < COLS; bx += 32 )
		{
			clock_gettime(CLOCK_REALTIME, &start);

			int block = (by / 32) * (COLS / 32) + (bx / 32);

			int base = by * COLS + bx;

			memset(hist, 0, 3 * 256 * sizeof(float));

			for( int y = 0; y < 32; ++y )
			{
				for( int x = 0; x < 32; ++x )
				{
					Pixel myPix = image[base + y * COLS + x];

					hist[0 * 256 + myPix.r ] += 1.f;
					hist[1 * 256 + myPix.g ] += 1.f;
					hist[2 * 256 + myPix.b ] += 1.f;
				}
			}

			UniqList imageList;

			for( int l = 0; l < data.L; ++l )
			{
				uint key = 0;

				for( int k = 0; k < data.K; ++k )
				{
					float value = 0;

#ifdef AVX
					for(int d = 0, i = 0; d < data.D * 4; d+= 16, i += 16)
					{
						__m256 c01 = _mm256_load_ps(&(coef[ l * data.K * data.D * 4 + k * data.D * 4 + d + 0 ]));
						__m256 c23 = _mm256_load_ps(&(coef[ l * data.K * data.D * 4 + k * data.D * 4 + d + 8 ]));

						// Load histogram values
						__m256 h0 = _mm256_load_ps(&hist[i]);
						__m256 h1 = _mm256_load_ps(&hist[i + 8]);

						// This instruction will multiply the two vectors and sum the low and high lanes
						__m256 m0 = _mm256_dp_ps(c01, h0, 0xFF);
						__m256 m1 = _mm256_dp_ps(c23, h1, 0xFF);

						// Now since we need to add the first float from each lane,
						// it's better to use scalar operations
						int i0 = _mm256_extract_epi32(m0, 0);
						int i2 = _mm256_extract_epi32(m1, 0);
						int i1 = _mm256_extract_epi32(m0, 4);
						int i3 = _mm256_extract_epi32(m1, 4);

						value += as_float(i0) + as_float(i1);
						value += as_float(i2) + as_float(i3);
					}
#else
					for(int d = 0; d < data.D; ++d)
					{
						value += (hist[d * 4 + 0] * coef[ l * data.K * data.D * 4 + k * data.D * 4 + d * 4 ]);
						value += (hist[d * 4 + 1] * coef[ l * data.K * data.D * 4 + k * data.D * 4 + d * 4 ]);
						value += (hist[d * 4 + 2] * coef[ l * data.K * data.D * 4 + k * data.D * 4 + d * 4 ]);
						value += (hist[d * 4 + 3] * coef[ l * data.K * data.D * 4 + k * data.D * 4 + d * 4 ]);
					}
#endif

					unsigned int bit = ( value > 0 ) ? 1 : 0;

					key = (key << 1) | bit;
				}

				uint hIndex = key % data.TableSize;
				uint control = key;

				int chainIndex = data.hashTable[l][hIndex];
				if( ALL_ONES == chainIndex )
				{
					continue;
				}

				bool found = false;
				while( true )
				{
					if( data.chainTable[l][chainIndex] == control )
					{
						found = true;
						break;
					}

					if( data.chainTable[l][chainIndex + 1] < 0 )
					{
						// If MSB is set, this means it's the last node in the chain
						break;
					}

					chainIndex += 2;
				}

				if( !found )
				{
					continue;
				}

//				cout << "\tchainIndex: " << chainIndex << endl;

				// Remove the MSB, we no longer need it
				int idIndex = data.chainTable[l][chainIndex + 1] & 0x7FFFFFFF;
				while( true )
				{
					int id = data.indexTable[l][idIndex];
//					cout << "\tidIndex: " << idIndex << " id: " << (id & 0x7FFFFFFF) << endl;

					// Insert id without the MSB
					imageList.insert( id & 0x7FFFFFFF );

					if( id < 0 )
					{
						break;
					}

					if( imageList.size() >= MAX_CAND_LIST_SIZE )
					{
						break;
					}

					idIndex++;
				}

				if( imageList.size() >= MAX_CAND_LIST_SIZE ) break;
			}

			clock_gettime(CLOCK_REALTIME, &end);
			calcKeys.fetch_and_add(__diffn__(start, end));

			clock_gettime(CLOCK_REALTIME, &start);

			size_t best = 0;
			float minDiff = __FLT_MAX__;

			for( int i = 0; i < imageList.size(); ++i )
			{
				uint cand = imageList.next();

				size_t offset = (size_t)cand * HIST_SIZE_ON_DISK;

				clock_gettime(CLOCK_REALTIME, &startRead);
				pread(data.histogramsFile, candHist, 3 * 256 * sizeof(float), offset);
				clock_gettime(CLOCK_REALTIME, &endRead);
				readHists.fetch_and_add(__diffn__(startRead, endRead));

				float diff = 0;

#ifdef AVX
				for( int i = 0; i < 3 * 256; i += 16 )
				{
					// Load histogram values
					__m256 h0_0 = _mm256_load_ps(hist + i + 0);
					__m256 h0_1 = _mm256_load_ps(hist + i + 8);
					__m256 h1_0 = _mm256_load_ps(candHist + i + 0);
					__m256 h1_1 = _mm256_load_ps(candHist + i + 8);

					// Calc the element-wise diff between the histograms
					__m256 s0 = _mm256_sub_ps(h0_0, h1_0);
					__m256 s1 = _mm256_sub_ps(h0_1, h1_1);

					// This instruction will multiply the two vectors and sum the low and high lanes
					__m256 m0 = _mm256_dp_ps(s0, s0, 0xFF);
					__m256 m1 = _mm256_dp_ps(s1, s1, 0xFF);

					// Now since we need to add the first float from each lane,
					// it's better to use scalar operations
					int i0 = _mm256_extract_epi32(m0, 0);
					int i2 = _mm256_extract_epi32(m1, 0);
					int i1 = _mm256_extract_epi32(m0, 4);
					int i3 = _mm256_extract_epi32(m1, 4);

					diff += as_float(i0) + as_float(i1);
					diff += as_float(i2) + as_float(i3);
				}
#else
				for( int c = 0; c < 3; ++c )
				{
					for( int j = 0; j < 256; ++j )
					{
						float myDiff = 0;

						myDiff += hist[c * 256 + j] - candHist[c * 256 + j];

						myDiff = myDiff * myDiff;

						diff += myDiff;
					}
				}
#endif

				if( diff < minDiff )
				{
					minDiff = diff;
					best = cand;
//					cout << "best: " << best << " minDiff: " << minDiff << endl;
				}
			}

			bests[block] = best;

			clock_gettime(CLOCK_REALTIME, &end);
			choose.fetch_and_add(__diffn__(start, end));
		}

		free(hist);
		free(candHist);
	} );

	clock_gettime(CLOCK_REALTIME, &endTotal);

	total = __diff__(startTotal, endTotal) * 1000.f;

	cpu_set_t tCpuSet;
	sched_getaffinity(0, sizeof(tCpuSet), &tCpuSet);

	double c = CPU_COUNT(&tCpuSet);
	cout << "Count affinity returned " << c << endl;

	cout << "calc keys: " << (double)(calcKeys) / 1000000.0 / c << "ms" << endl;
	cout << "read histograms: " << (double)(readHists) / 1000000.0 / c << "ms" << endl;
	cout << "choose best: " << (double)(choose - readHists) / 1000000.0 / c << "ms" << endl;
	cout << "total time: " << total << "ms" << endl;
}

void read(int fid, uchar* buff, size_t size, size_t offset)
{
	g.run([=]{pread(fid, buff, size, offset);});
}

void wait()
{
	g.wait();
}
