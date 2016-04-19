
#ifndef _UTILS
#define _UTILS 1

#include "loader.h"
#include <sys/time.h>

typedef unsigned int uint;
typedef unsigned char uchar;

double __diff__(timespec start, timespec end);

template< class T > class UniqListImpl;
template< class KEY, class VAL > class UniqMapImpl;

class UniqList
{
private:
	UniqListImpl<int> *impl;

public:
	UniqList();
	~UniqList();

	void insert( int t );
	bool find( int t );
	int size();
	int next();
};

class UniqMap
{
private:
	UniqMapImpl<int, int> *impl;

public:
	UniqMap();
	~UniqMap();

	void insert( int key, int val );
	bool find( int key );
	int get( int key );
	int size();
};

void merge(int BLOCKS, uint* keys, int* cand, AlgData &data, uint* reverseGlobalImageList, size_t &numImages);
void merge_single(int BLOCKS, uint* keys, int* cand, AlgData &data, uint* reverseGlobalImageList, size_t &numImages);

void mosaic_TBB( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS );

void read(int fid, uchar* buff, size_t size, size_t offset);
void wait();

#endif
