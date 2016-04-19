/*
 * LSH.h
 *
 *  Created on: Aug 7, 2014
 *      Author: sagi
 */

#ifndef LSH_H_
#define LSH_H_

#include <string.h>
#include <vector>
#include <fstream>

typedef unsigned int uint;
typedef unsigned char uchar;

static const int MAX_BUCKET_SIZE = 32;
static const int MAX_LIST_SIZE = 256;
static const int HIST_NUM_CHANNELS = 3;
static const int HIST_BINS_PER_CHANEL = 64;
static const int HIST_VALS_PER_BIN = 256 / HIST_BINS_PER_CHANEL;

// 4294967291 = 2^32-5
static const uint UH_PRIME_DEFAULT = 4294967291U;

// 2^32-1
static const uint TWO_TO_32_MINUS_1 = 4294967295U;

// 2^29
static const uint MAX_HASH_RND = 536870912U;

class CompactHistogram
{
private:
	float _data[HIST_BINS_PER_CHANEL * HIST_NUM_CHANNELS];

public:
	CompactHistogram( float* fullHist );
	void normalize(float r);
	float& operator[](int i) { return _data[i]; }
	const float& operator[](int i) const { return _data[i]; }
};

class Node
{
public:
	int 	_id;
	Node* 	_next;

	Node( int id, Node* next = NULL ) :
		_id(id), _next(next)
	{
	}
};

class Bucket
{
public:
	Bucket( unsigned controlValue, Bucket* nextBucket ) : _head(NULL), _controlValue(controlValue), _nextBucket(nextBucket), nImages(0), overflow(0)
	{
	}

	bool insert(unsigned int idx);

	Node*		_head;
	unsigned 	_controlValue;
	Bucket* 	_nextBucket;
	int 		nImages;
	int 		overflow;
};

class LSHFunction
{
private:
	static bool _loadFromStream;
	static std::istream *_stream;
	static int _K;
	static int _D;
	static uint _nImages;
	static FILE* _histogramsFile;
	float **_a;
	float *_b;

public:
	static void init(int k, int d, bool loadFromStream = false, std::istream &stream = std::cin);
	static void init(int k, int d, FILE* histogramsFile, uint nImages);
	LSHFunction();
	~LSHFunction();

	void dump(std::ostream &stream);
	void load(std::istream &stream);
	unsigned int operator() (const CompactHistogram& hist);
};

class LSH
{
private:
	int _D; 		// dimension of points.
	int _K; 		// dimension of LSH key
	int _L; 		// Number of LSH functions
	int _tableSize;	// Number of entries in the main table

	// number of images in the data set
	uint _nImages;

	FILE* _histogramsFile;

	LSHFunction *_lshFunctions;

	uint ** _chainLengths;
	Bucket ***_buckets;

	uint* r1;
	uint* r2;

	uint computeProductModPrime(uint a[], uint r[], uint K);

public:
	LSH( int D,
			int K,
			int L,
			int tableSize,
			std::string dataset );

	LSH( std::istream &stream,
			std::string dataset );

	~LSH();

	void populate();
	void dump(std::ostream &stream);
	void load(std::istream &stream);

	void dumpTables(std::string name);

	void printStatistics();
};


#endif /* LSH_H_ */
