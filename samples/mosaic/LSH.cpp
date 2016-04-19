/*
 * LSH.cpp
 *
 *  Created on: Aug 7, 2014
 *      Author: sagi
 */

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <vector>
#include "LSH.h"
#include "Random.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

static const int HIST_SIZE_ON_DISK = 1024 * 4;

int seed = 0;

using namespace std;

static int numInserted = 0;
static int above = 0;
static int below = 0;

bool LSHFunction::_loadFromStream = false;
istream* LSHFunction::_stream = NULL;
int LSHFunction::_K = 0;
int LSHFunction::_D = 0;
FILE* LSHFunction::_histogramsFile = 0;
uint LSHFunction::_nImages = 0;

CompactHistogram::CompactHistogram( float* fullHist )
{
	int idx = 0;

	for( int c = 0; c < HIST_NUM_CHANNELS; ++c )
	{
		for( int b = 0; b < HIST_BINS_PER_CHANEL; ++b )
		{
			_data[ c * HIST_BINS_PER_CHANEL + b ] = 0;
			for( int i = 0; i < HIST_VALS_PER_BIN; ++i )
			{
				_data[ c * HIST_BINS_PER_CHANEL + b ] += fullHist[idx++];
			}
		}
	}
}

bool Bucket::insert( unsigned int idx )
{
	if( nImages < MAX_BUCKET_SIZE )
	{
		Node* p = new Node( idx, _head);
		_head = p;
		nImages++;
		return true;
	}
	else
	{
		overflow++;
		return false;
	}
}

void LSHFunction::init(int k,
		               int d,
		               bool loadFromStream,
		               istream &stream)
{
	_loadFromStream = loadFromStream;
	_stream = &stream;
	_K = k;
	_D = d;
}

void LSHFunction::init(int k,
		               int d,
		               FILE* histogramsFile,
		               uint nImages)
{
	_K = k;
	_D = d;
	_histogramsFile = histogramsFile;
	_nImages = nImages;
}

LSHFunction::LSHFunction()
{
	if( _loadFromStream )
	{
		load( *_stream );
		return;
	}

	_a = new float*[_K];
	_b = new float[_K];

	for( int i = 0; i < _K; ++i )
	{
		_a[i] = new float[_D];

		while( true )
		{
			bool hasNeg = false;
			bool hasPos = false;

			for(int j = 0; j < _D; ++j)
			{
				_a[i][j] = genGaussianRandom();
				if( _a[i][j] < 0 ) hasNeg = true;
				if( _a[i][j] > 0 ) hasPos = true;
			}

			if( hasNeg && hasPos ) break;
		}
	}
}

LSHFunction::~LSHFunction()
{
	for( int i = 0; i < _K; ++i )
	{
		delete[] _a[i];
	}

	delete[] _a;
	delete[] _b;
}

void LSHFunction::dump(std::ostream &stream)
{
	for( int i = 0; i < _K; ++i )
	{
		stream << _a[i][0];
		for( int j = 1; j < _D; ++j )
		{
			stream << " " << _a[i][j];
		}
		stream << endl;
	}
}

void LSHFunction::load(std::istream &stream)
{
	_a = new float*[_K];
	_b = new float[_K];

	for( int i = 0; i < _K; ++i )
	{
		_a[i] = new float[_D];

		for(int j = 0; j < _D; ++j)
		{
			stream >> _a[i][j];
		}
	}
}

unsigned int LSHFunction::operator ()(const CompactHistogram &hist)
{
	unsigned int res = 0;

	for( int i = 0; i < _K; ++i )
	{
		float value = 0;
		for(int j = 0; j < _D; ++j)
		{
			value += hist[j] * _a[i][j];
		}

		unsigned int bit = ( value > 0 ) ? 1 : 0;

		if( value > 0 )
		{
			above++;
		}
		else
		{
			below++;
		}

		res = (res << 1) | bit;
	}

	return res;
}

LSH::LSH(int D, int K, int L, int tableSize, std::string dataset ) :
		_D(D), _K(K), _L(L), _tableSize(tableSize)
{
	initRandom(seed);

	string histFileName = dataset + ".hist";

	_histogramsFile = fopen(histFileName.c_str(), "r");

	fseek(_histogramsFile, 0, SEEK_END);
	size_t size = ftell(_histogramsFile);
	fseek(_histogramsFile, 0, SEEK_SET);

	_nImages = size / HIST_SIZE_ON_DISK;

	r1 = new uint[K];
	r2 = new uint[K];

	for( int i = 0; i < K; ++i )
	{
		r1[i] = genUniformRandom(1, MAX_HASH_RND);
		r2[i] = genUniformRandom(1, MAX_HASH_RND);
	}

	LSHFunction::init(K, D, _histogramsFile, _nImages);
	_lshFunctions = new LSHFunction[L];

	_buckets = new Bucket**[L];
	_chainLengths = new uint*[L];
	for( int i = 0; i < L; ++i )
	{
		_buckets[i] = new Bucket*[_tableSize];
		_chainLengths[i] = new uint[_tableSize];
		memset( _buckets[i], 0, _tableSize * sizeof(Bucket*));
		memset( _chainLengths[i], 0, _tableSize * sizeof(uint) );
	}
}

LSH::LSH( istream &stream, string dataset )
{
	string histFileName = dataset + ".hist";

	_histogramsFile = fopen(histFileName.c_str(), "r");

	fseek(_histogramsFile, 0, SEEK_END);
	size_t size = ftell(_histogramsFile);
	fseek(_histogramsFile, 0, SEEK_SET);

	_nImages = size / HIST_SIZE_ON_DISK;

	load(stream);

	LSHFunction::init(_K, _D, true, stream);
	_lshFunctions = new LSHFunction[_L];
	    
	_buckets = new Bucket**[_L];
	_chainLengths = new uint*[_L];
	for( int i = 0; i < _L; ++i )
	{
		_buckets[i] = new Bucket*[_tableSize];
		_chainLengths[i] = new uint[_tableSize];
		memset( _buckets[i], 0, _tableSize * sizeof(Bucket*));
		memset( _chainLengths[i], 0, _tableSize * sizeof(uint) );
	}
}


LSH::~LSH()
{
//	for( int i = 0; i < _L; ++i )
//	{
//		delete _buckets[i];
//	}
//
//	delete[] _buckets;
	delete[] _lshFunctions;

	fclose(_histogramsFile);
}

uint LSH::computeProductModPrime(uint a[], uint r[], uint K)
{
	size_t h = 0;

	for( uint i = 0; i < K; ++i )
	{
		h = h + a[i] * r[i];
		h = (h & TWO_TO_32_MINUS_1) + 5 * (h >> 32);
		if( h > UH_PRIME_DEFAULT )
		{
			h = h - UH_PRIME_DEFAULT;
		}
	}

	return h;
}

void LSH::populate()
{
	fseek(_histogramsFile, 0, SEEK_SET);
	uint precent = _nImages / 10;

	for( uint i = 0; i < _nImages; ++i )
	{
		if( 0 == (i % precent) )
		{
			cout << (float)i / _nImages * 100 << "%" << endl;
		}

		uint idx = i;

		uchar histData[HIST_SIZE_ON_DISK];

		int read = fread(histData, 1, HIST_SIZE_ON_DISK, _histogramsFile);
		assert( read == HIST_SIZE_ON_DISK );

		CompactHistogram hist( (float*) histData );

		bool inserted = false;

		for( int j = 0; j < _L; ++j )
		{
			uint key = _lshFunctions[j](hist);

			uint* keyBits = new uint[_K];
			for( int i = 0; i < _K; ++i )
			{
				// keyBits[0] is the LSB
				keyBits[i] = ( key >> i ) & 1;
			}

//			uint hIndex = computeProductModPrime( r1, keyBits, _K ) % _tableSize;
//			uint control = computeProductModPrime( r2, keyBits, _K );

			uint hIndex = key % _tableSize;
			uint control = key;

			Bucket* p = _buckets[j][hIndex];
			while( ( p != NULL ) && ( p->_controlValue != control ) )
			{
				p = p->_nextBucket;
			}

			if( p == NULL )
			{
				p = _buckets[j][hIndex] = new Bucket( control, _buckets[j][hIndex] );
				_chainLengths[j][hIndex]++;
			}

			if( true == p->insert(idx) )
			{
				inserted = true;
			}
		}

		if( inserted )
		{
			numInserted++;
		}
	}
}

void LSH::printStatistics()
{
	cout << "Params:" << endl;
	cout << "D: " << _D << endl;
	cout << "K: " << _K << endl;
	cout << "L: " << _L << endl;
	cout << "TAbleSize: " << _tableSize << endl;

	cout << "Diff: " << (above - below) << endl;

	for( int i = 0; i < _L; ++i )
	{
		int min = MAX_LIST_SIZE + 1;
		int max = 0;
		int numFull = 0;
		int numEmptyBins = 0;
		int numBuckets = 0;
		float sum = 0;

		cout << endl;
		cout << "L" << i << endl;

		for( int j = 0; j < _tableSize; ++j )
		{
			Bucket* p = _buckets[i][j];

			if( p == NULL )
			{
				numEmptyBins++;
			}

			while( p != NULL )
			{
				numBuckets++;

				int t = p->nImages;

				if( t > max ) max = t;
				if( t < min ) min = t;
				sum += t;

				if( t == MAX_BUCKET_SIZE) numFull++;

				p = p->_nextBucket;
			}
		}

		float average = sum / numBuckets;

		cout << "\tTable " << i << ": " << numBuckets << " buckets"<< endl;
		cout << "\tTable " << i << ": " << numEmptyBins << " empty bins"<< endl;
		cout << "\tTable " << i << ": " << sum << " images"<< endl;
		cout << "\tNum full: " << numFull << endl;
		cout << "\tMin: " << min << " Max: " << max << " Average: " << average << endl << endl;
	}

	cout << "Total inserted: " << numInserted << endl;
}

void LSH::dump(ostream &stream)
{
	stream << _D << endl;
	stream << _K << endl;
	stream << _L << endl;
	stream << _tableSize << endl;

	stream << r1[0];
	for( int i = 1; i < _K; ++i )
	{
		stream << " " << r1[i];
	}
	stream << endl;

	stream << r2[0];
	for( int i = 1; i < _K; ++i )
	{
		stream << " " << r2[i];
	}
	stream << endl;

	for( int i = 0; i < _L; ++i )
	{
		_lshFunctions[i].dump(stream);
	}
}

void LSH::load(istream &stream)
{
	stream >> _D;
	stream >> _K;
	stream >> _L;
	stream >> _tableSize;

	r1 = new uint[_K];
	r2 = new uint[_K];
	
	for( int i = 0; i < _K; ++i )
	{
		stream >> r1[i];
	}

	for( int i = 0; i < _K; ++i )
	{
		stream >> r2[i];
	}
}

void LSH::dumpTables(std::string name)
{
	int allOnes = -1;

	for( int l = 0; l < _L; ++l )
	{
		stringstream ss;
		ss << name << "." << l;
		string hashFileName = ss.str() + ".hash";
		string chainFileName = ss.str() + ".chain";
		string indexFileName = ss.str() + ".idx";

		FILE* hashFile = fopen(hashFileName.c_str(), "w");
		FILE* chainFile = fopen(chainFileName.c_str(), "w");
		FILE* indexFile = fopen(indexFileName.c_str(), "w");

		int hashOffset = 0;
		int chainOffset = 0;
		int indexOffset = 0;

		for( int h = 0; h < _tableSize; ++h )
		{
			Bucket* p = _buckets[l][h];
			if( NULL == p )
			{
				fwrite( &allOnes, sizeof(int), 1, hashFile);
				hashOffset++;
				continue;
			}

			fwrite( &chainOffset, sizeof(int), 1, hashFile);
			hashOffset++;

			while( NULL != p )
			{
				fwrite( &(p->_controlValue), sizeof(int), 1, chainFile);
				chainOffset++;

				int myIndexOffset = indexOffset;
				if( NULL == p->_nextBucket )
				{
					myIndexOffset = myIndexOffset | (1 << 31);
				}

				fwrite( &myIndexOffset, sizeof(int), 1, chainFile);
				chainOffset++;

				Node* n = p->_head;
				while( NULL != n )
				{
					int id = n->_id;

					if( NULL == n->_next )
					{
						// Mark the last node
						id = id | (1 << 31);
					}

					fwrite( &id, sizeof(int), 1, indexFile);
					indexOffset++;

					n = n->_next;
				}

				p = p->_nextBucket;
			}
		}

		fclose(hashFile);
		fclose(chainFile);
		fclose(indexFile);
	}
}

