#ifndef __LOADER_H__
#define __LOADER_H__

#include <string.h>
#include <iostream>

static const int BUCKET_SIZE = 32;

#ifdef UNALIGNED
static const int HIST_SIZE_ON_DISK = 3072;
#else
static const int HIST_SIZE_ON_DISK = 4096;
#endif

static const int IMAGE_SIZE_ON_DISK = 4096;

// 4294967291 = 2^32-5
static const uint UH_PRIME_DEFAULT = 4294967291U;

// 2^32-1
static const uint TWO_TO_32_MINUS_1 = 4294967295U;

class Bucket
{
private:
	int _images[BUCKET_SIZE];

public:
	Bucket()
	{
		memset(_images, 0xFF, BUCKET_SIZE * sizeof(int));
	}

	int& operator[](int i)
	{
		return _images[i];
	}
};

class LSHFunction
{
private:
	static int _K;
	static int _D;

public:
	float **a;
	float *b;

public:
	static void init(int k, int d) { _K = k; _D = d; }
	LSHFunction() : a(NULL), b(NULL) {}
	~LSHFunction();

	void load( std::istream &stream );
//	unsigned int operator() (const Histogram& hist);
};

class AlgData
{
public:
	int D;
	int K;
	int L;
	int TableSize;

	uint* r1;
	uint* r2;

	LSHFunction *lshFunctions;

	int** hashTable;
	int** chainTable;
	int** indexTable;

	int* hashTableSizes;
	int* chainTableSizes;
	int* indexTableSizes;

	std::string imagesFileName;
	std::string histogramsFileName;

	FILE* imagesFile;
	int histogramsFile;

public:
	AlgData() : D(0),
				K(0),
				L(0),
				TableSize(0),
				r1(NULL),
				r2(NULL),
				lshFunctions(NULL),
				imagesFileName(""),
				histogramsFileName(""),
				imagesFile(NULL),
				histogramsFile(0) {}

	~AlgData();

	void loadParams( std::istream &stream );
	void loadTables( std::string name );

	void openFiles( std::string datasetFileName, std::string histogramsFileName );

	void getCoefficents( float* coef );
	void getCoefficentsTransposed( float* coef );
};

#endif //__LOADER_H__
