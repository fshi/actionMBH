//This code prepares the files for svm classificaiton. 
//It reads individual BoF file and group it into training/testing split groups of different classes.
//It needs split1.txt, split2.txt and split3.txt files.

#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "formatBinaryStream.h"
#include <windows.h> 
#include <string>
#include <cstdlib>
//#include <atlbase.h>  //for A2W(lpr);
#include "waitKeySeconds.h"
#include "matOperations.h"
#include "scaleData.h"
#include <direct.h>  // for _mkdir( "\\testtmp" ); 
//#include <opencv2/gpu/gpu.hpp>

const int classNum = 51; //number of total classes
const int nFold = 3;
using namespace cv;

int main() {

	//USES_CONVERSION;  //for A2W(lpr) and W2A(...);

	int numWords, normType;
	float maxVal = 0.f, minVal = 0.f;
	std::cout<<"How many words does \'bag of words\' has (1-29999)? \nInput: ";
	std::cin>>numWords;
	std::cout<<"\n";

	if (numWords<=0 || numWords>=30000)
	{
		std::cout<<"Wrong number of words input! please re-input number words does again (1-9999): ";
		std::cin>>numWords;
		std::cout<<"\n";
	}

	int clsNum = classNum;
	while (clsNum <= 0)
	{
		std::cout<<"Pleas input total number of classes : \n";
		cin>>clsNum;
		std::cout<<std::endl;
	}
	std::cout<<"What type of data normalization to use? \nInput negative value for no normalization, 0 for scaling,\
		1 for Standardization with midrange 0 and range 2, and 2 for standardization with 0 mean and 1 stddev: \n";
	std::cin>>normType;
	if(normType != 1 && normType != 2 && normType >= 0)
	{
		std::cout<<"please input the upper value for the scaling: \n";
		std::cin>>maxVal;
		std::cout<<"please input the lower value for the scaling: \n";
		std::cin>>minVal;
	}
	scaleData scaling(numWords, normType, minVal, maxVal);

	int trainSt = 0; //beginning point for train data and test data
	int trainEnd = clsNum;
	char tmpC[10];
	itoa(numWords, tmpC, 10);

	if (normType < 0)
	{
		std::cout<<"Please input the start number of training dir (0 - number of classes)? \n";
		std::cout<<"If \"0\", means using all training classes. If \"number of classes\", means doing no training computing. \nInput: ";
		std::cin>>trainSt;
		std::cout<<"\n";
		if (trainSt < 0)
			trainSt = 0;
		if (trainSt < clsNum)
		{
			std::cout<<"Please input the end number of training dir (0 - number of classes)? \n";
			std::cout<<"If \"0\",  means doing no training computing. \nInput: ";
			std::cin>>trainEnd;
			std::cout<<"\n";
		}
	}
	else
	{
		trainSt = 0;
		trainEnd = clsNum;
	}
	

	char tstr[5],tchr[5];

	string  dirName2 = "random";
	
	string dName1, dName2, dNm;

	string fileName;
	string fileTp = ".avi";
	char sName[1024], tmpStr[1024];
	float *mRow;

	Mat tmpMat;
	
	Mat *cluMat21s = new Mat[nFold];
	Mat *cluMat22s = new Mat[nFold];

	Mat *sMat21s = new Mat[nFold];
	Mat *sMat22s = new Mat[nFold];
	vector<int> row11, row12, row21, row22;
	
	int label;
	string *spFile = new string[nFold];

	const string dirName = "C:\\dataSets\\hmdb51\\";
	for(int i = 0; i < nFold; i++)
	{
		itoa(i+1, tstr, 10);
		spFile[i] = (string)"split"+(string)tstr+(string)".txt";
	}

	BinClusterOutStream<float> *ofile=0;
	BinClusterInStream *iFile=0;

	for(int i = 0; i < nFold; i++)
	{
		itoa(i+1, tstr, 10);
		if(normType < 0)
			dNm = (string)"split"+(string)tstr + (string)"R\\";
		else
			dNm = (string)"split"+(string)tstr  + (string)"Rs" + (string)itoa(normType, tchr, 10) + (string)"\\";
		_mkdir(dNm.c_str());
	}
	

	ifstream *spFileIn = new ifstream[nFold];

	for (int i = trainSt; i < trainEnd; i++)
	{ 
		itoa(i+1, tstr, 10);
		dNm = dirName + (string)tstr + (string)"\\";
		
		for (int i0 = 0; i0 < nFold; i0++)
			spFileIn[i0].open((dNm + spFile[i0]).c_str(), ifstream::in);

		for(int id = 0; id < nFold; id++)
		{
			cluMat21s[id] = Mat();
			cluMat22s[id] = Mat();
			while (spFileIn[id]>>sName>>label)
			{
				
				if (label == 1)
				{
					fileName = sName;
					fileName.replace(fileName.find(fileTp),fileTp.length(),".dat");
					strcpy(tmpStr, dirName2.c_str());
					strcat(tmpStr, tstr);
					strcat(tmpStr, "\\");
					strcat(tmpStr, fileName.c_str());
					iFile = new BinClusterInStream (tmpStr);
					tmpMat=Mat();
					iFile->read(tmpMat);
					if (tmpMat.cols != numWords)
					{
						cout << "The input bag of words number " << numWords << " doesn't match the bag of words computed! " << endl;
						delete iFile;
						discoverUO::wait();
						exit(-1);
					}
					delete iFile;
					cluMat21s[id].push_back(tmpMat);

				}
				if (label == 2)
				{
					fileName = sName;
					fileName.replace(fileName.find(fileTp),fileTp.length(),".dat");
					strcpy(tmpStr, dirName2.c_str());
					strcat(tmpStr, tstr);
					strcat(tmpStr, "\\");
					strcat(tmpStr, fileName.c_str());
					iFile = new BinClusterInStream (tmpStr);
					tmpMat=Mat();
					iFile->read(tmpMat);
					if (tmpMat.cols != numWords)
					{
						cout << "The input bag of words number " << numWords << " doesn't match the bag of words computed! " << endl;
						delete iFile;
						discoverUO::wait();
						exit(-1);
					}
					delete iFile;
					cluMat22s[id].push_back(tmpMat);
				}

			}
			if (normType < 0)
			{
				
				itoa(id+1, tstr, 10);
				dNm = (string)"split"+(string)tstr + (string)"R\\";
				itoa(i+1, tstr, 10);
				fileName = dNm+"pwords"+tstr+"_"+tmpC+".dat";
				ofile = new BinClusterOutStream<float> (fileName);
				for(int j0 = 0; j0 < cluMat21s[id].rows; j0++)
				{
					mRow = cluMat21s[id].ptr<float>(j0);
					ofile->write(mRow, numWords);
				}
				delete ofile;
				fileName = dNm+"bagWord"+tstr+"_"+tmpC+".dat";
				ofile = new BinClusterOutStream<float> (fileName);
				for(int j0 = 0; j0 < cluMat22s[id].rows; j0++)
				{
					mRow = cluMat22s[id].ptr<float>(j0);
					ofile->write(mRow, numWords);
				}
				delete ofile;
				
			}
			else 
			{
				sMat21s[id].push_back(cluMat21s[id]);
				if(!id)
					row21.push_back(cluMat21s[id].rows);

				sMat22s[id].push_back(cluMat22s[id]);
				if(!id)
					row22.push_back(cluMat22s[id].rows);
			}
		}
		for (int i0 = 0; i0 < nFold; i0++)
		{
			spFileIn[i0].clear();
			spFileIn[i0].close();
		}
	}


	if (normType >= 0)
	{
		for(int id = 0; id < nFold; id++)
		{
			scaling.normTrainData(sMat21s[id], sMat21s[id]);
			scaling.normTestData(sMat22s[id], sMat22s[id]);
			itoa(id+1, tstr, 10);
			dNm = (string)"split"+(string)tstr  + (string)"Rs" + (string)itoa(normType, tchr, 10) + (string)"\\";
			int count1 = 0, count2 = 0;
			for(int i = 0; i < clsNum; i++)
			{
				itoa(i+1, tstr, 10);
				fileName = dNm+"pwords"+tstr+"_"+tmpC+".dat";
				ofile = new BinClusterOutStream<float> (fileName);
				for(int i0 = 0; i0 < row21.at(i); i0++)
				{
					mRow = sMat21s[id].ptr<float>(i0+count1);
					ofile->write(mRow, numWords);
				}
				count1 += row21.at(i);
				delete ofile;

				fileName = dNm+"bagWord"+tstr+"_"+tmpC+".dat";
				ofile = new BinClusterOutStream<float> (fileName);
				for(int i0 = 0; i0 < row22.at(i); i0++)
				{
					mRow = sMat22s[id].ptr<float>(i0+count2);
					ofile->write(mRow, numWords);
				}
				count2 += row22.at(i);
				delete ofile;
			}
		}
	}

	delete []cluMat21s;
	delete []cluMat22s;
	delete []sMat21s;
	delete []sMat22s;
	delete []spFile;
	delete []spFileIn;
	//int oo=countNonZero(tt);
	//std::cout<<tt.rows<<" "<<tt.cols<<" "<<oo;
	discoverUO::wait();
	//std::cout<<discoverUO::timestamp();
	//std::cin.get();
	//cin.get();
	return 0;
}