#include "cxcore.h"
#include "cv.h"
#include "highgui.h"

//#define _USE_OCL_MATCH_
#ifdef  _USE_OCL_MATCH_
#include "bagWordsDescriptor_ocl.h"  //using gpu with opencl for brute force BoF matching
#else
#include "bagWordsDescriptor_new.h"
#endif

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "stDescriptor.h"
#include "formatBinaryStream.h"
#include <windows.h> 
#include <string>
#include <cstdlib>
//#include <atlbase.h>  //for A2W(lpr);
#include <string>
#include <time.h>
#include "waitKeySeconds.h"
//#include "opencv.hpp"
#include "i_f_toa.h"
#include <direct.h>  // for _mkdir( "\\testtmp" ); 
//#include <opencv2/gpu/gpu.hpp>

//#define _GET_PROCESS_TIME_

using namespace cv;

const int _runNum = 3; //for random feature seletion, run 3 times
const int classNum = 51; //number of total classes
const int _maxFrames = 160; //maxium frames per video. if the video has more frames, it needs to split the video to do multiple processing to avoid memory overflow and vector overflow(patcher sampling from video)

int main() {

#ifdef  _USE_OCL_MATCH_
	//initianize gpu
	vector<cv::ocl::Info> info;
	cout<<cv::ocl::getDevice(info)<<endl;
	//cv::ocl::setDevice(info[1]); //intel_gpu
	cv::ocl::setDevice(info[0]); //amd_gpu
#endif
	printf ("OpenCV version %s (%d.%d.%d)\n",
	    CV_VERSION,
	    CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);

	//USES_CONVERSION;  //for A2W(lpr) and W2A(...); 

	RNG seed[3];
	seed[0] = RNG(unsigned(time(NULL)));
	seed[1] = RNG(seed[0].next());
	seed[2] = RNG(seed[0].next());
	
	int numWords;

	std::cout<<"How many words does \'bag of words\' has (1-9999)? \nInput: ";
	std::cin>>numWords;

	while (numWords<=0 || numWords>=10000)
	{
		std::cout<<"\nWrong number of words input! please re-input number words does again (1-9999): ";
		std::cin>>numWords;
	}
	std::cout<<"\n";

	int samNum = 0;
	std::cout<<"How many samples do you want to use for random sampling?  \nInput: ";
	std::cin>>samNum;
	while (samNum<100 || samNum>30000)
	{
		std::cout<<"\nWrong number of samples input! please re-input number of samples again (100-30000): ";
		std::cin>>samNum;
	}
	std::cout<<"\n";

	MBHparam * para = new MBHparam();  //parameters for the system
	if (!para->readParam("MBH_parameters_input.txt", 0))
	{
		std::cout<<"Can't open the parameter file! Please include \"MBH_parameters_input.txt\" in the working directy.\n";
		std::cout<<"The program will exit in 10 seconds! \n";
		discoverUO::wait(10);
		exit(-5);
	}

	stDetector dscpt(para); //pass the parameters to the stDetector class
	
	int maxFrames = _maxFrames;
	para->writeParam("MBH_parameters.txt");  //writing the parameters for the reference
	delete para;

	int clsNum = classNum;
	while (clsNum <= 0)
	{
		std::cout<<"\nPleas input total number of classes :  ";
		cin>>clsNum;
	}
	std::cout<<std::endl;

	int trainSt = 0, testSt = 0; //beginning point for train data and test data
	int trainEnd = clsNum, testEnd = clsNum;

	std::cout<<"Please input the starting calss number (0 - number of classes)? \n";
	std::cin>>trainSt;
	std::cout<<"\n";
	if (trainSt < 0)
		trainSt = 0;
	if (trainSt < clsNum)
	{
		std::cout<<"Please input the ending class number (0 - number of classes)? \n";
		std::cin>>trainEnd;
		std::cout<<"\n";
	}
	
	int rootSz = dscpt.toRootFtSz();  //the root video size, decided by the input video and parameters
	int partsSz = dscpt.toPartsFtSz();   //the part video size, normally it is the 2 times the size of root

	std::fstream fRst;
	fRst.open("MBH_parameters.txt", ios::app | std::ios::out);
	if (!fRst.is_open())
	{
		std::cerr<<"Error opening file to write data dimensions!\n";
		discoverUO::wait(10);
	}
	else
	{
		fRst<<"\n\n********************************************************\n";
		fRst<<"\tRoot Size of descript feature is: "<<rootSz<<"\n";
		fRst<<"\tPart Size of descript feature is: "<<partsSz<<"\n";
		fRst<<"\tThe vector dimension of descript feature is: "<<rootSz+partsSz<<"\n";
	}

	char tstr[5], tstr1[5], tmpC[5];
	i_ftoa(tstr, numWords);
	string fName = (string)"cluster" + (string)_itoa(numWords,tmpC,10)  + (string)"ppc" + (string)".dat";  //the file name of "visual words"

	//directory for the processed video data. for simplicity, the videos are stored in a subdirectory.
	//one class, one subdirectory, marked with 1, 2, 3...
	const string dirName = "C:\\dataSets\\hmdb51\\"; 
	string fullName;
	string dNm, dName;

	bagWordsFeature bwFt(fName);  //reading the "visual words" into  BoW class
	cout<<"Done initializing Bag of Words!"<<endl;
	int wordNum = bwFt.getWordNum();

	if(numWords != wordNum)
	{
		cout<<numWords<<" "<<wordNum<<endl;
		cout<<"Bag of Word file: "<<fName<<" doesn't match the number of input words! Please input Enter key to exit!"<<endl;
		discoverUO::wait(50);
		exit(-1);
	}

	float *arr[_runNum];
	int *iarr[_runNum];
	int *iarr0[_runNum];
	
	Mat feature2[_runNum], feature1[_runNum];

	string fileName[_runNum], fileNameR;
	string fileTp = ".avi", dName1, dName2[_runNum];
	//char tmp[1024];
	char sName[1024];
	int redoNum;

	HANDLE hFind;  //for finding every file inside a subdirectory
	WIN32_FIND_DATA FileData;

	BinClusterOutStream<float> *ofile[_runNum];
	for (int i = 0; i < _runNum; i++)
	{
		ofile[i] = 0;
		arr[i] = new float[wordNum];
		iarr[i] = new int[wordNum];
		iarr0[i] = new int[wordNum];
	}
	//make dir to store computed features
	for (int i = 0; i < _runNum; i++)
	{
		itoa(i+1, tstr, 10);
		dName2[i] = (string)"run"+(string)tstr;
		_mkdir(dName2[i].c_str());
	}

	for (int i = trainSt; i < trainEnd; i++)
	{
		itoa(i+1, tstr, 10);
	
		//make dir for storing the computed features
		for (int j = 0; j < _runNum; j++)
		{
			itoa(j+1, tstr1, 10);
			dName2[j] = (string)"run"+(string)tstr1+(string)"\\random"+(string)tstr;
			_mkdir(dName2[j].c_str());
		}
	
		dNm = dirName + (string)tstr + (string)"\\";
		cout<<"Now doing folder: "<<dNm<<endl;
		hFind = FindFirstFile((dNm+"*.avi").c_str(), &FileData);  //find first .avi file inside the directory. you may need to change .avi into other video names, such as .mpg

		while (hFind != INVALID_HANDLE_VALUE)
		{
			strcpy(sName, dNm.c_str());
			strcat(sName, FileData.cFileName);
			std::cout<<"Processing file: "<<sName<<endl;
		
			//comute integral videos
			if(!dscpt.preProcessing(sName, maxFrames))  
			{
				std::cout<<"Unable to process loaded video for  computing training features 2!\n";
				discoverUO::wait();
				exit(-1);
			}

			//random sampling
			for (int j = 0; j < _runNum; j++)
			{
				dscpt.getRandomFeatures(feature2[j], samNum, seed[j]);
			}
				
			if(redoNum = dscpt.reProcessNum())   //if input video frames > _maxFrames,   it needs to split the video to do multiple processing to avoid memory overflow
			{
				int dSz0, dSz1;
				std::cout<<"redo...\n";
				for (int i0 = 1; i0 <= redoNum; i0++)
				{
					dSz0 = dscpt.getSamplingSz();
					#pragma omp parallel sections  //nowait//num_threads(3)
					{
						#pragma omp section 
						{
							for (int j = 0; j < _runNum; j++)
								if (i0 == 1)
								{
									bwFt.getFeatures(feature2[j], iarr0[j]);  //get bag of features
									feature2[j].release();
								}
								else
								{
									bwFt.getFeatures(feature2[j], iarr[j]);   //get bag of features
									feature2[j].release();
								}
						}
						#pragma omp  section
						{
							dscpt.re_Processing(sName, maxFrames, i0);  //compute integral video
						}
					}

					dSz1 = dscpt.getSamplingSz();  
					for (int j = 0; j < _runNum; j++)
					{
						if (i0 < redoNum)
							dscpt.getRandomFeatures(feature2[j], samNum, seed[j]);  //do random sampling
						else
							dscpt.getRandomFeatures(feature2[j], ((float)dSz1/(float)dSz0)*samNum, seed[j]);  //do random sampling

					}

					if (i0 > 1)
					{
						for (int j = 0; j < _runNum; j++)
						{
							//add word frequecy if splited the video with multiple processing
							for (int j0 = 0; j0 < wordNum; j0++)
								iarr0[j][j0] += iarr[j][j0];
						}
					}
				}

				for (int j = 0; j < _runNum; j++)
				{
					bwFt.getFeatures(feature2[j], iarr[j]);   //get bag of features
					feature2[j].release();
					for (int j0 = 0; j0 < wordNum; j0++)
						iarr0[j][j0] += iarr[j][j0];
					bwFt.normlizeFt(arr[j], iarr0[j]);   //get normlized bag of features
				}
			}
			else
			{
				for (int j = 0; j < _runNum; j++)
				{
					bwFt.getNormlizedFt(feature2[j], arr[j]);   //get normlized bag of features
					feature2[j].release();
				}
			}

			//writing the computed bag of features
			for (int j = 0; j < _runNum; j++)
			{
				fileName[j] = dName2[j] + (string)"\\" + (string)(FileData.cFileName);
				fileName[j].replace(fileName[j].find(fileTp),fileTp.length(),".dat");
				ofile[j] = new BinClusterOutStream<float> (fileName[j]);
				ofile[j]->write(arr[j], wordNum);
				delete ofile[j];
			}
			
			if (FindNextFile(hFind, &FileData) == 0) break; // stop when none left
		}
	}
	

	for (int i = 0; i < _runNum; i++)
	{
		delete []arr[i];
		delete []iarr[i];
		delete []iarr0[i];
	}
	
	discoverUO::wait();
	return 0;
}