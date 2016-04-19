/*
 * Random.h
 *
 *  Created on: Aug 10, 2014
 *      Author: sagi
 */

#ifndef RANDOM_H_
#define RANDOM_H_

void initRandom(unsigned int seed);

float genUniformRandom(float rangeStart, float rangeEnd);

int genUniformRandom(int rangeStart, int rangeEnd);

float genGaussianRandom();

#endif /* RANDOM_H_ */
