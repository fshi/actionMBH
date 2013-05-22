#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
//#include "HOG3Ddescriptor.h"
#include "formatBinaryStream.h"
#include <windows.h> 
#include <string>
#include <cstdlib>
#include <atlbase.h>  //for A2W(lpr);
#include <string>
#include <time.h>
#include "waitKeySeconds.h"
#include "matOperations.h"
using namespace cv;


int main() 
{
	bool kmeanCluster = 0;
	std::cout<<"Do you want use Kmeans to cluster(\'1\' for Kmeans, \'0\'for random selection)? \nInput: ";
	std::cin>>kmeanCluster;
	std::cout<<"\n";
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


	Mat cluMat;
	BinClusterInStream *iFile = new BinClusterInStream("randData100kc.dat");
std::cout<<"\nNow reading raw data from the flie...! Please waiting...\n";
	iFile->read(cluMat);
	delete iFile;
	//std::cout<<"\nrows: "<<cluMat.cols<<" cols: "<<cluMat.rows<<" Mat types: "<<cluMat.type()<<"\n Root size: "<<rootSz<<" Part Size"<<partsSz<<"\n";
	std::cout<<"\ncols: "<<cluMat.cols<<" rows: "<<cluMat.rows<<" "<<cluMat.at<float>(0,100)<<" "<<cluMat.at<float>(0,200)<<" "<<cluMat.at<float>(0,333)
		<<" "<<cluMat.at<float>(38,100)<<" "<<cluMat.at<float>(38,200)<<" "<<cluMat.at<float>(38,333)<<
		" "<<cluMat.at<float>(330,100)<<" "<<cluMat.at<float>(330,200)<<" "<<cluMat.at<float>(330,333)<<" "<<cluMat.at<float>(330,10)<<" "<<cluMat.at<float>(330,20)<<" "<<cluMat.at<float>(330,3)<<
		" "<<cluMat.at<float>(340,100)<<" "<<cluMat.at<float>(340,200)<<" "<<cluMat.at<float>(340,333)<<"\n";
int count = 0;
	for(int i = 0; i < cluMat.rows; i++)
	{
		
		float nozero = countNonZero(cluMat.row(i));
		if(nozero<10)
		{
			cout<<i<<" ";
			count++;
		}
	}
	
	//Mat row = cluMat.rowRange(0, 110000);
		cout<<endl<<endl<<count<<endl;
	discoverUO::wait();

	
	int numWords[6]={1000,2000,3000,4000, 5000, 6000}, height1[6], width1[6];
	string fName[6] = {"cluster1000ppc.dat", "cluster2000ppc.dat", "cluster3000ppc.dat", "cluster4000ppc.dat", "cluster5000ppc.dat", "cluster6000ppc.dat"};
	Mat labels[6], centers[6];
	BinClusterOutStream<float> *ofiles[6];
	float *data1[6];
	if(kmeanCluster)
	{
		//Mat labels[4], centers[4];
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


	discoverUO::wait();
	//std::cin.get();
	//cin.get();
	return 0;
}