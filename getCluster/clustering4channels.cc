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

const int _rootSz = 64;
const int _partsSz = 64*8;
const int _maxChnl = 4;
int main() 
{
	bool kmeanCluster = 0;
	int chnl = 1;
	std::cout<<"Do you want use Kmeans to cluster(\'1\' for Kmeans, \'0\'for random selection)? \nInput: ";
	std::cin>>kmeanCluster;
	std::cout<<"\n";
	
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

	Mat cluMat;
	BinClusterInStream *iFile = new BinClusterInStream("randData100kc.dat");
std::cout<<"\nNow reading raw data from the flie...! Please waiting...\n";
	iFile->read(cluMat);
	delete iFile;
	
	CV_Assert(2*_rootSz+2*_partsSz == cluMat.cols);
	Mat cMat[4], mCols;
	if(_maxChnl==2)
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
	else if(_maxChnl==4)
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

	int numWords[6]={1000,2000,3000,4000, 5000, 6000}, height1[6], width1[6];
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
		//Mat labels[4], centers[4];
		std::cout<<"\nNow clustering with kmeans...! Please waiting...\n";
		for(int i0 = 0; i0 < chnl; i0++)
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


	discoverUO::wait();
	//std::cin.get();
	//cin.get();
	return 0;
}