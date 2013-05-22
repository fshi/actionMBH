/********************************************************************************

Filename     : matOperations.h

Description  : This file provides some functions to nomalize Mat data (every row), shuffle Mat data (among rows) etc.
			   Function "normalizeMat(const Mat &src, Mat& rst)" normalizes input Mat scr into output Mat rst
			   Function "suffleCvMat(Mat &mx)" shuffles Mat mx (shuffle among rows. no touch on cols order).
			   Function "suffleCvMat(const Mat &src, Mat &dst)" shuffle Mat src and put the shuffled results into Mat dst.

			   string fileName[6] = {"pWords1ppa.dat", "pWords2ppa.dat", "pWords3ppa.dat", "pWords4ppa.dat","pWords5ppa.dat","pWords6ppa.dat"};
			   Mat trainMat, labels;
			   getSVMData(fileName, trainMat,labels, NULL, 6, 100);//read 100 training samples for each of 6 classes from files... 
		

Typical Use  :  self-explanation

				
Author       : FengShi@Discovery lab, Oct, 2010
Version No   : 1.00 


*********************************************************************************/
#ifndef _MAT_OPERATIONS_H_
#define _MAT_OPERATIONS_H_

#include "cxcore.h"
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include "waitKeySeconds.h"
#include <iostream>
#include <fstream>
#include "biostream.h" 
#include "biistream.h" 
#include "formatBinaryStream.h"

using namespace cv;

//this function suffles Mat rows
//shuffle the array with Fisher-Yates shuffling
void shuffleCvMat(Mat &mx);
void shuffleCvMat(Mat &mx, int times);

template<class T1> void fy_shuffle(T1* first, T1* last, int stopNum = 1, int shuffleTimes = 3)
 {
	srand( (unsigned int)time(NULL) );
	T1 tmp;
	int i, id, id0;
	
	//run 3 times to make sure it is shuffled
	for(i=0; i<shuffleTimes; i++)
	{
		id0 = last - first - 1;
		while(id0 >= stopNum)
		{
			id = rand() % id0;
			tmp = first[id0]; 
			first[id0] = first[id];
			first[id] = tmp;
			
			id0--;
		}
	}
 }

void shuffleCvMat(const Mat &src, Mat &dst);

void normalizeMat1c(const Mat &src, Mat& rst);

//this function normalizes each rows of input src Mat
//the input scr Mat must be either "uchar", "int", "float" or "double"
void normalizeMat(const Mat &src, Mat& rst);

//this function normalizes each cols of input src Mat 
//the outputs are: Means, Maxs and Mins of each column and normalized rst=((src-mean)-min)/(max-min)
void normTrainData4SVM(const Mat src, Mat &rst, Mat &colMean, Mat &colMax, Mat &colMin);

//this function normalizes each cols of input src Mat by using input mean, min and max
//the output is:  normalized rst=((src-mean)-min)/(max-min)
void normTestData4SVM(const Mat src, Mat &rst, const Mat &colMean, const Mat &colMax, const Mat &colMin);

//this function read training/test data from files (fileName[numClass])
//default (readFile = -1) means read all data from the files. if 0<readFile<Data.rows, it will real a part of data from files
//the output will be the training/test data in "Data", and class label in "labels"
void getSVMData(const string * fileName, Mat &Data, Mat &labels, int *classes, const int numClass = 6, const int readFile = -1);

#endif