#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include "utils.h"
#include "loader.h"

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

static const int MAX_CAND_LIST_SIZE = 256;

struct Pixel
{
	uchar r;
	uchar g;
	uchar b;
	uchar a;
};


void mosaic_CPU( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS )
{
	return mosaic_TBB( inOut, bests, data, ROWS, COLS );
}
