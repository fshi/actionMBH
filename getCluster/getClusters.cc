#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
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
#include <atlbase.h>  //for A2W(lpr);
#include <string>
#include <time.h>
#include "waitKeySeconds.h"
#include "i_f_toa.h"

const int classNum = 51; //number of total classes
const int dataNum = 120000; //number of total features for finding clusters by kmeans. 
							//due to the too many training features(over millions), we normally random-choose "dataNum" from all training features
const int _maxFrames = 160; //maxium frames per video. if the video has more frames, it needs to split the video to do multiple processing to avoid memory overflow
const int pSamples = 60;  //this value is the number of sampled 3d patches for every video. 
							//you need to make sure: pSamples x number of videos > dataNum
							//if number of videos is too low, you need to increase the value of pSamples

using namespace cv;

//This programs get clusters centers (1000, 2000, 3000 and 4000) by kmeans from training videos

int main() 
{
	RNG seed(unsigned(time(NULL)));
	MBHparam * para = new MBHparam();
	if (!para->readParam("MBH_parameters_input.txt", 1))
	{
		std::cout<<"use default HOG3D parameters instead.\n";
		discoverUO::wait();
	}

	stDetector dscpt(para); 
	para->writeParam("MBH_parameters_clustering.txt");
	delete para;

	char tstr[5];
	int clsNum = classNum;
	while (clsNum <= 0)
	{
		std::cout<<"Pleas input total number of classes : \n";
		cin>>clsNum;
		std::cout<<std::endl;
	}

	int start, end, numClusters;

	std::cout<<"How many clusters do you want to use (\'1\' for 1000 clusters, \'2\' for 2000 clusters, \'3\' for 3000 clusters, \'4\' for 4000 clusters, \'5\' for 5000 clusters, \'6\' for 6000 clusters, other value for(1000 - 4000)clusters)? \nInput: ";
	std::cin>>numClusters;
	std::cout<<"\n";
	switch (numClusters){
		case 1:
		case 1000:
			start = 0;
			end = 1;
			break;
		case 2:
		case 2000:
			start = 1;
			end = 2;
			break;
		case 3:
		case 3000:
			start = 2;
			end = 3;
			break;
		case 4 :
		case 4000:
			start = 3;
			end = 4;
			break;
		case 5 :
		case 5000:
			start = 4;
			end = 5;
			break;
		case 6 :
		case 6000:
			start = 5;
			end = 6;
			break;
		default:
			start = 0;
			end = 4;
			break;
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

	bool flipIm4Training = 0;
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

//writing randomly chosen 120000 features to file
	iFile = new BinClusterInStream (rFileNm);
	iFile->read(cluMat);
	delete iFile;
	
//now doing clustering...
	int numWords[6]={1000,2000,3000,4000,5000,6000}, height1[6], width1[6];
	string fName[6] = {"cluster1000ppc.dat", "cluster2000ppc.dat", "cluster3000ppc.dat", "cluster4000ppc.dat", "cluster5000ppc.dat", "cluster6000ppc.dat"};
	
	BinClusterOutStream<float> *ofiles[6];
	float *data1[6];

	if(kmeanCluster)
	{
		Mat labels[6], centers[6];
		int i,j;

		std::cout<<"\nNow clustering with kmeans...! Please waiting...\n";
#pragma omp parallel for num_threads(4)
		for (i = start; i < end; i++)
		{
			ofiles[i] = new BinClusterOutStream<float> (fName[i]);
		//clustering using kmeans
			labels[i] = Mat(cluMat.rows, 1, CV_32SC1);
			centers[i] = Mat(numWords[i], cluMat.cols,CV_32FC1);
			kmeans(cluMat, numWords[i], labels[i],  cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100000, 0.001),1,KMEANS_PP_CENTERS, centers[i]);
			height1[i] = centers[i].rows;
			width1[i] = centers[i].cols;
			std::cout<<"Now clustering with kmeans for "<<numWords[i]<<" clusters is done! Writing to the file. Please waiting...\n";
			for (j = 0; j < height1[i]; j++)
			{
				data1[i] = centers[i].ptr<float>(j);
				ofiles[i]->write(data1[i], width1[i]);
			}
			std::cout<<"Now writing flie for "<<numWords[i]<<" clusters is done! \n";
			delete ofiles[i];
		}
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			ofiles[i] = new BinClusterOutStream<float> (fName[i]);
			Mat sIdx = Mat::zeros(cluMat.rows, 1, CV_8UC1);
			Mat m0 = sIdx.rowRange(0, numWords[i]);
			m0 = 1;
			shuffleCvMat(sIdx);
			MatConstIterator_<uchar> it = sIdx.begin<uchar>(), it_end = sIdx.end<uchar>();
		
			std::cout<<"Now clusters for "<<numWords[i]<<" is sampling! Writing to the file. Please waiting...\n";
			for (int i0=0; it != it_end; ++it, i0++)
			{
				if (*it)
				{
					data1[i] = cluMat.ptr<float>(i0);
					ofiles[i]->write(data1[i], cluMat.cols);
				}
			}
			std::cout<<"Now writing flie for "<<numWords[i]<<" clusters is done! \n";
			delete ofiles[i];
		}
	}

	delete []rDataNum;
	discoverUO::wait();
	//std::cin.get();
	//cin.get();
	return 0;
}