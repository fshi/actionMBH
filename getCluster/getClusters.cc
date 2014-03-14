#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "stDescriptor.h"
#include "formatBinaryStream.h"
#include <windows.h> 
#include <string>
#include <cstdlib>
#include <atlbase.h>  //for A2W(lpr);
#include <time.h>
#include "matOperations.h"
#include "waitKeySeconds.h"

const int classNum = 51; //number of total classes
const int dataNum = 120000; //number of total features for finding clusters by kmeans. 
							//due to the too many training features(over millions), we normally random-choose "dataNum" from all training features
const int _maxFrames = 160; //maxium frames per video. if the video has more frames, it needs to split the video to do multiple processing to avoid memory overflow
const int pSamples = 60;  //this value is the number of sampled 3d patches for every video. 
							//you need to make sure: pSamples x number of videos > dataNum
							//if number of videos is too low, you need to increase the value of pSamples
const int _rootSz = 64;
const int _partsSz = 64*8;
const int _maxChnl = 4;
using namespace cv;

//This programs get clusters centers (1000, 2000, 3000 and 4000) by kmeans from training videos

int main() 
{
	RNG seed(unsigned(time(NULL)));
	MBHparam * para = new MBHparam();
	if (!para->readParam("MBH_parameters_input.txt", 1))
	{
		std::cout<<"Can't open the parameter file! Please include \"MBH_parameters_input.txt\" in the working directy.\n";
		std::cout<<"The program will exit in 10 seconds! \n";
		discoverUO::wait(10);
		exit(-5);
	}

	stDetector dscpt(para); 
	para->writeParam("MBH_parameters_clustering.txt");
	delete para;

	char tstr[5];
	int clsNum = classNum;
	if (clsNum <= 0)
	{
		std::cout<<"Pleas input total number of classes : \n";
		cin>>clsNum;
		std::cout<<std::endl;
	}

	int start[_maxChnl], end[_maxChnl], numClusters[_maxChnl];

	for(int i = 0; i < _maxChnl; i++)
	{
		std::cout<<"For channel: "<<i<<", how many clusters do you want to use (\'1\' for 1000 clusters, \'2\' for 2000 clusters, \'3\' for 3000 clusters, \'4\' for 4000 clusters, \'5\' for 5000 clusters, \'6\' for 6000 clusters, other value for(1000 - 4000)clusters)? \nInput: ";
		std::cin>>numClusters[i];
		std::cout<<"\n";
		switch (numClusters[i]){
			case 1:
			case 1000:
				start[i] = 0;
				end[i] = 1;
				break;
			case 2:
			case 2000:
				start[i] = 1;
				end[i] = 2;
				break;
			case 3:
			case 3000:
				start[i] = 2;
				end[i] = 3;
				break;
			case 4 :
			case 4000:
				start[i] = 3;
				end[i] = 4;
				break;
			case 5 :
			case 5000:
				start[i] = 4;
				end[i] = 5;
				break;
			case 6 :
			case 6000:
				start[i] = 5;
				end[i] = 6;
				break;
			default:
				start[i] = 0;
				end[i] = 4;
				break;
		}
	}
	int *rDataNum = new int[clsNum];
	//computing how many randomly chosen features from each training class
	rDataNum[0] = dataNum/classNum;
	for (int i = 1; i < classNum-1; i++)
		rDataNum[i] = rDataNum[0];
	rDataNum[classNum-1] = dataNum - rDataNum[0]*(classNum-1);

	//randomly choose 120000 features from the computed training features
	BinClusterInStream *iFile;
	int rowNo = 0;
	Mat roi;
	//int maxFrames = ((int)(para->rt2ps.z + 0.3)) ? _maxFrames : _maxFrames*1.5;
	int maxFrames = _maxFrames;

	bool kmeanCluster = 0;
	std::cout<<"Do you want use Kmeans to cluster(\'1\' for Kmeans, \'0\'for random selection)? \nInput: ";
	std::cin>>kmeanCluster;
	std::cout<<"\n";
	
	Mat feature, cluMat;
	BinClusterOutStream<float> *ofile, *rawFile = NULL;

	//delete ofile;

	string fileName = "binOut.dat";
	string dirName = "c:\\dataSets\\hmdb51\\";

	string fullName;
	string rFileNm = "randData100kc.dat";
	string dNm;

	int label;
	char sName[1024];
	string spFile = "split1.txt";
	ifstream spFileIn;

//compute the features from training videos
	for (int i = 0; i < clsNum; i++)
	{
		itoa(i+1, tstr, 10);
		dNm = dirName + (string)tstr + (string)"\\";
		fullName = dNm + spFile;
		ofile = new BinClusterOutStream<float> (fileName);

		spFileIn.open(fullName.c_str());
		if (!spFileIn.is_open())
		{
			std::cout<<"Unable to open split flie \""<<fullName<<"\" for reading the file name of training/testing split!\n";
			discoverUO::wait();
			exit(-1);
		}
//std::cout<<"folder name: "<<dNm<<"full split file name: "<<fullName<<endl;
		while (spFileIn>>sName>>label)
		{
			//cout<<sName<<"  label = "<<label<<endl;
			if (label == 1)
			{
				seed.next();
				fullName = dNm + sName;
				//cout<<fullName<<endl;
				if(!dscpt.preProcessing(fullName, maxFrames))
				{
					std::cout<<"Unable to process loaded video for computing training features!\n";
					discoverUO::wait();
					exit(-1);
				}

				dscpt.getRandomFeatures(feature, pSamples, seed);
				int height = feature.rows, width = feature.cols;
				float *data;
				for (int i0 = 0; i0 < height; i0++)
				{
					data = feature.ptr<float>(i0);
					ofile->write(data, width);
					//ofile->write((float*)feature.ptr(i), width);
				}

				int redoNum;
				if(redoNum = dscpt.reProcessNum())
				{
					for (int i0 = 1; i0 <= redoNum; i0++)
					{
						dscpt.re_Processing(fullName, maxFrames, i0);
						dscpt.getRandomFeatures(feature, pSamples, seed);
						height = feature.rows; 
						width = feature.cols;
						for (int j0 = 0; j0 < height; j0++)
						{
							data = feature.ptr<float>(j0);
							ofile->write(data, width);
							//ofile->write((float*)feature.ptr(i), width);
						}
					}
				}

				cout<<"Done video file: "<< fullName<<endl;
			}
		}
		spFileIn.clear();
		spFileIn.close();
		delete ofile;
	//cout<<"done split file! i = " <<i<<endl;
//randomly choose 120000 features from the computed training features
		iFile = new BinClusterInStream (fileName);
		iFile->read(cluMat, rDataNum[i]);
		//std::cout<<cluMat.cols<<" "<<cluMat.rows<<" "<<cluMat.type()<<"\n";
		delete iFile;
		if (i == 0)
			rawFile = new BinClusterOutStream<float> (rFileNm);

		int height = cluMat.rows, width = cluMat.cols;
		for (int j = 0; j < height; j++)
		{
			float *data = cluMat.ptr<float>(j);
			rawFile->write(data, width);
		}
		cluMat.release();
	cout<<"done the folder! i = " <<i<<endl;
	}
	delete rawFile;

//reading randomly chosen 120000 features from file
	iFile = new BinClusterInStream (rFileNm);
	iFile->read(cluMat);
	delete iFile;

	CV_Assert(2*_rootSz+2*_partsSz == cluMat.cols);
	Mat cMat[4], mCols;
	if(_maxChnl==2) //root and part are two separate channels, but MBHx and MBHy are concantenated
	{
		cMat[0] = Mat(cluMat.rows, 2*_rootSz, cluMat.type());
		cMat[1] = Mat(cluMat.rows, 2*_partsSz, cluMat.type());
		
		mCols = cluMat.colRange(0, _rootSz);
		mCols.copyTo( cMat[0].colRange(0, _rootSz) );

		mCols = cluMat.colRange(_rootSz+_partsSz, 2*_rootSz+_partsSz);
		mCols.copyTo( cMat[0].colRange(_rootSz, cMat[0].cols) );

		mCols = cluMat.colRange(_rootSz, _rootSz+_partsSz);
		mCols.copyTo( cMat[1].colRange(0, _partsSz) );

		mCols = cluMat.colRange(2*_rootSz+_partsSz, cluMat.cols);
		mCols.copyTo( cMat[1].colRange(_partsSz, cMat[1].cols) );
	}
	else if(_maxChnl==4)  //root, part, MBHx, MBHy are separate channels
	{
		cMat[0] = Mat(cluMat.rows, _rootSz, cluMat.type());
		cMat[1] = Mat(cluMat.rows, _partsSz, cluMat.type());
		cMat[2] = Mat(cluMat.rows, _rootSz, cluMat.type());
		cMat[3] = Mat(cluMat.rows, _partsSz, cluMat.type());
		
		mCols = cluMat.colRange(0, _rootSz);
		mCols.copyTo( cMat[0] );

		mCols = cluMat.colRange(_rootSz, _rootSz+_partsSz);
		mCols.copyTo( cMat[1] );

		mCols = cluMat.colRange(_rootSz+_partsSz, 2*_rootSz+_partsSz);
		mCols.copyTo( cMat[2] );

		mCols = cluMat.colRange(2*_rootSz+_partsSz, cluMat.cols);
		mCols.copyTo( cMat[3] );
	}
	else 
		exit(-1);
	
//now doing clustering...
	int numWords[6]={1000,2000,3000,4000,5000,6000}, height1[6], width1[6];
	string fName[24] = {"cluster1000ppc0.dat", "cluster2000ppc0.dat", "cluster3000ppc0.dat", "cluster4000ppc0.dat", "cluster5000ppc0.dat", "cluster6000ppc0.dat",
						"cluster1000ppc1.dat", "cluster2000ppc1.dat", "cluster3000ppc1.dat", "cluster4000ppc1.dat", "cluster5000ppc1.dat", "cluster6000ppc1.dat",
	"cluster1000ppc2.dat", "cluster2000ppc2.dat", "cluster3000ppc2.dat", "cluster4000ppc2.dat", "cluster5000ppc2.dat", "cluster6000ppc2.dat",
	"cluster1000ppc3.dat", "cluster2000ppc3.dat", "cluster3000ppc3.dat", "cluster4000ppc3.dat", "cluster5000ppc3.dat", "cluster6000ppc3.dat"};
	Mat labels[6], centers[6];
	BinClusterOutStream<float> *ofiles[24];
	float *data1[6];
	int id;

	if(kmeanCluster)
	{
		std::cout<<"\nNow clustering with kmeans...! Please waiting...\n";
		for(int i0 = 0; i0 < _maxChnl; i0++)
		{
			id = i0*6;
			for (int i = start[i0]; i < end[i0]; i++)
			{
				ofiles[i+id] = new BinClusterOutStream<float> (fName[i+id]);
			//clustering using kmeans
				labels[i] = Mat(cMat[i0].rows, 1, CV_32SC1);
				centers[i] = Mat(numWords[i], cMat[i0].cols,CV_32FC1);
				kmeans(cMat[i0], numWords[i], labels[i],  cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100000, 0.001),1,KMEANS_PP_CENTERS, centers[i]);
				height1[i] = centers[i].rows;
				width1[i] = centers[i].cols;

				std::cout<<"Now clustering with kmeans for "<<numWords[i]<<" clusters is done! Writing to the file. Please waiting...\n";
				for (int j = 0; j < height1[i]; j++)
				{
					data1[i] = centers[i].ptr<float>(j);
					ofiles[i+id]->write(data1[i], width1[i]);
				}
				std::cout<<"Now writing flie for "<<numWords[i]<<" clusters is done! \n";
				delete ofiles[i+id];
			}
		}

	}
	else
	{
		for(int i0 = 0; i0 < _maxChnl; i0++)
		{
			id = i0*6;
			for (int i = start[i0]; i < end[i0]; i++)
			{
				ofiles[i+id] = new BinClusterOutStream<float> (fName[i+id]);
				Mat sIdx = Mat::zeros(cluMat.rows, 1, CV_8UC1);
				Mat m0 = sIdx.rowRange(0, numWords[i]);
				m0 = 1;

				shuffleCvMat(sIdx);
				shuffleCvMat(sIdx);
				MatConstIterator_<uchar> it = sIdx.begin<uchar>(), it_end = sIdx.end<uchar>();
		
				std::cout<<"Now clusters for "<<numWords[i]<<"with dimension of "<<cMat[i0].cols<<" is sampling! Writing to the file. Please waiting...\n";
				std::cout<<cMat[i0].rows<<" "<<cMat[i0].cols<<endl;
				for (int j0=0; it != it_end; ++it, j0++)
				{
					if (*it)
					{
						data1[i] = cMat[i0].ptr<float>(j0);
						ofiles[i+id]->write(data1[i], cMat[i0].cols);
					}
				}
				std::cout<<"Now writing flie for "<<numWords[i]<<" clusters is done! \n";
				delete ofiles[i+id];
			}
		}
	}

	delete []rDataNum;
	discoverUO::wait();
	//std::cin.get();
	//cin.get();
	return 0;
}