/*
 * Random.cpp
 *
 *  Created on: Aug 18, 2014
 *      Author: sagi
 */

#include <stdlib.h>
#include <math.h>

char rngState[256];

void initRandom(unsigned int seed)
{
	initstate(seed, rngState, 256);
}

float genUniformRandom(float rangeStart, float rangeEnd)
{
	float r;
	r = rangeStart + ((rangeEnd - rangeStart) * (float) random() / (float) RAND_MAX);
	return r;
}

int genUniformRandom(int rangeStart, int rangeEnd)
{
	int r;
	r = rangeStart + random() % (rangeEnd - rangeStart);
	return r;
}

// Generate a random real from normal distribution N(0,1).
float genGaussianRandom()
{
  // Use Box-Muller transform to generate a point from normal distribution.
  float x1, x2;
  do
  {
	  x1 = genUniformRandom(0.0f, 1.0f);
  }
  while (x1 == 0); // cannot take log of 0.
  x2 = genUniformRandom(0.0f, 1.0f);
  float z;
  z = sqrt(-2.0 * log(x1)) * cos(2.0 * M_PI * x2);
  return z;
}
