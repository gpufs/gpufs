#include <sstream>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include "loader.h"

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

int LSHFunction::_K = 0;
int LSHFunction::_D = 0;

LSHFunction::~LSHFunction()
{
	for( int i = 0; i < _K; ++i )
	{
		delete[] a[i];
	}

	delete[] a;
	delete[] b;
}

void LSHFunction::load( istream &stream )
{
	a = new float*[_K];
	b = new float[_K];

	for( int i = 0; i < _K; ++i )
	{
		a[i] = new float[_D];

		for(int j = 0; j < _D; ++j)
		{
			stream >> a[i][j];
		}
	}
}

AlgData::~AlgData()
{
	for( int i = 0; i < L; ++i )
	{
		delete[] hashTable[i];
		delete[] chainTable[i];
		delete[] indexTable[i];
	}

	delete[] hashTable;
	delete[] chainTable;
	delete[] indexTable;
	delete[] lshFunctions;

	fclose(imagesFile);
	close(histogramsFile);
}

void AlgData::loadParams( istream &stream )
{
	stream >> D;
	stream >> K;
	stream >> L;
	stream >> TableSize;

	r1 = new uint[K];
	r2 = new uint[K];

	for( int i = 0; i < K; ++i )
	{
		stream >> r1[i];
	}

	for( int i = 0; i < K; ++i )
	{
		stream >> r2[i];
	}

	LSHFunction::init(K, D);
	lshFunctions = new LSHFunction[L];

	for( int i = 0; i < L; ++i )
	{
		lshFunctions[i].load( stream );
	}

	hashTable = new int*[L];
	chainTable = new int*[L];
	indexTable = new int*[L];

	hashTableSizes = new int[L];
	chainTableSizes = new int[L];
	indexTableSizes = new int[L];
}

void AlgData::loadTables( std::string name )
{
	int read = 0;

	for( int i = 0; i < L; ++i )
	{
		stringstream ss;
		ss << name << "." << i;
		string hashFileName = ss.str() + ".hash";
		string chainFileName = ss.str() + ".chain";
		string indexFileName = ss.str() + ".idx";

		FILE* hashFile = fopen(hashFileName.c_str(), "r");
		FILE* chainFile = fopen(chainFileName.c_str(), "r");
		FILE* indexFile = fopen(indexFileName.c_str(), "r");

		fseek(hashFile, 0, SEEK_END);
		size_t hashSize = ftell(hashFile) / sizeof(int);
		fseek(hashFile, 0, SEEK_SET);

		fseek(chainFile, 0, SEEK_END);
		size_t chainSize = ftell(chainFile) / sizeof(int);
		fseek(chainFile, 0, SEEK_SET);

		fseek(indexFile, 0, SEEK_END);
		size_t indexSize = ftell(indexFile) / sizeof(int);
		fseek(indexFile, 0, SEEK_SET);

		hashTable[i] = new int[hashSize];
		chainTable[i] = new int[chainSize];
		indexTable[i] = new int[indexSize];

		hashTableSizes[i] = hashSize;
		chainTableSizes[i] = chainSize;
		indexTableSizes[i] = indexSize;

		read = fread( hashTable[i], sizeof(int), hashSize, hashFile );
		assert( read == hashSize );

		read = fread( chainTable[i], sizeof(int), chainSize, chainFile );
		assert( read == chainSize );

		read = fread( indexTable[i], sizeof(int), indexSize, indexFile );
		assert( read == indexSize );

		fclose(hashFile);
		fclose(chainFile);
		fclose(indexFile);
	}
}

void AlgData::openFiles( string _datasetFileName, string _histogramsFileName )
{
	imagesFileName = _datasetFileName;
	histogramsFileName = _histogramsFileName;

	imagesFile = fopen(imagesFileName.c_str(), "r");
	histogramsFile = open(histogramsFileName.c_str(), O_RDONLY);
}

void AlgData::getCoefficents( float* coef )
{
	for( int l = 0; l < L; ++l )
	{
		for( int k = 0; k < K; ++k )
		{
			for( int d = 0; d < D; ++d )
			{
				coef[ l * K * D * 4 + k * D * 4 + d * 4 + 0 ] = lshFunctions[l].a[k][d];
				coef[ l * K * D * 4 + k * D * 4 + d * 4 + 1 ] = lshFunctions[l].a[k][d];
				coef[ l * K * D * 4 + k * D * 4 + d * 4 + 2 ] = lshFunctions[l].a[k][d];
				coef[ l * K * D * 4 + k * D * 4 + d * 4 + 3 ] = lshFunctions[l].a[k][d];
			}
		}
	}
}

void AlgData::getCoefficentsTransposed( float* coef )
{
	for( int l = 0; l < L; ++l )
	{
		for( int k = 0; k < K; ++k )
		{
			for( int d = 0; d < D; ++d )
			{
				// We use 32 instead of K for alignment
				coef[ l * 32 * D + d * 32 + k ] = lshFunctions[l].a[k][d];
			}
		}
	}
}
