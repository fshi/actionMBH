/********************************************************************************

Filename     : matOperations.cc

Description  : This file provides some functions to nomalize Mat data (every row), shuffle Mat data (among rows) etc.
			   Function "normalizeMat(const Mat &src, Mat& rst)" normalizes input Mat scr into output Mat rst
			   Function "suffleCvMat(Mat &mx)" shuffles Mat mx (shuffle among rows. no touch on cols order).
			   Function "suffleCvMat(const Mat &src, Mat &dst)" shuffle Mat src and put the shuffled results into Mat dst.

Typical Use  : read "matOperations.h"

				
Author       : FengShi@Discovery lab, Oct, 2010
Version No   : 1.00 


*********************************************************************************/

#include "matOperations.h"


//this function suffles Mat rows
//shuffle the array with Fisher-Yates shuffling
 void shuffleCvMat(Mat &mx)
 {
	 srand( (unsigned int)time(NULL) );
	 int rowNo = mx.rows;
	 int row0 = rowNo - 1;

	 //suuffle the array with Fisher-Yates shuffling
	 while (row0 > 0)
	 {
		int row1 = rand() % row0;
		Mat m1 = mx.row(row1);
		Mat mt = m1.clone();
		mx.row(row0).copyTo(m1);
		mt.copyTo(mx.row(row0));
		row0--;
	 }
 }

 void shuffleCvMat(Mat &mx, int times)
 {
	 srand( (unsigned int)time(NULL) );
	 int rowNo, row0;
	 for(int i = 0; i < times; i++)
	 {
		 rowNo = mx.rows;
		 row0 = rowNo - 1;

		 //suuffle the array with Fisher-Yates shuffling
		 while (row0 > 0)
		 {
			int row1 = rand() % row0;
			Mat m1 = mx.row(row1);
			Mat mt = m1.clone();
			mx.row(row0).copyTo(m1);
			mt.copyTo(mx.row(row0));
			row0--;
		 }
	 }
 }


  void shuffleCvMat(const Mat &src, Mat &dst)
 {
	 dst = src.clone();
	 shuffleCvMat(dst);
 }

void normalizeMat1c(const Mat &src, Mat& rst)
{
	Mat tmpMat;
	if ((src.depth() == CV_64F) && (rst.depth() == CV_64F)) //if input and output data are all double, then normalized with double
		tmpMat.create(src.rows, src.cols, CV_64FC1); 
	else
		tmpMat.create(src.rows, src.cols, CV_32FC1); 
	for (int i = 0; i < src.rows; i++)
	{
		float sum = (float)(cv::sum(src.row(i))[0]);
		//std::cout<<sum<<"\n";

		switch (src.depth())
		{
		case CV_8U:
			for (int j = 0; j < tmpMat.cols; j++)
				tmpMat.at<float>(i,j) = (float)src.at<uchar>(i,j)/sum;
			break;
		case CV_32S:
			for (int j = 0; j < tmpMat.cols; j++)
				tmpMat.at<float>(i,j) = (float)src.at<int>(i,j)/sum;
			break;
		case CV_32F:
			for (int j = 0; j < tmpMat.cols; j++)
				tmpMat.at<float>(i,j) = (float)src.at<float>(i,j)/sum;
			break;
		case CV_64F:
			if (rst.depth() == CV_64F)
				for (int j = 0; j < tmpMat.cols; j++)
					tmpMat.at<double>(i,j) = (float)src.at<double>(i,j)/sum;
			else
				for (int j = 0; j < tmpMat.cols; j++)
					tmpMat.at<float>(i,j) = (float)src.at<double>(i,j)/sum;
			break;
		default:
			printf("normalizeMat() only supports data type \"uchar\", \"int\", \"float\", \"double\". Press Enter to continue...");
			discoverUO::wait();
			//system("PAUSE"); //this one only works on windows system
			exit(-1);
		}
	
	}
	rst = tmpMat;
}

//this function normalizes each rows of input src Mat
//the input scr Mat must be either "uchar", "int", "float" or "double"
void normalizeMat(const Mat &src, Mat& rst)
{
	if (src.channels() == 1)
		normalizeMat1c(src, rst);
	else
	{
		vector<Mat> tmp;
		vector<Mat> tmpR;
		split(src, tmp);
		for (int i = 0; i < src.channels(); i++)
		{
			Mat tmpMat;
			normalizeMat1c(tmp[i], tmpMat);
			tmpR.push_back(tmpMat);
		}
		merge(tmpR, rst);
	}
}


void normTrainData4SVM(const Mat src, Mat &rst, Mat &colMean, Mat &colMax, Mat &colMin)
{

	Mat rstTmp(src.rows, src.cols, src.type());

	/*
	Mat col;
	colMax.create(src.rows, 1, CV_64FC1);
	colMin.create(src.rows, 1, CV_64FC1);
	colMean.create(src.rows, 1, CV_64FC1);
	
	for (int i = 0; i < src.cols; i++)
	{
		col = src.col(i);
		minMaxLoc(col, &max, &min);
		colMax.at<double>(1, i) = Max;
		colMin.at<double>(1, i) = Min;
		colMean.at<double>(1, i) = mean(col);
	}*/
	reduce(src, colMean, 0, CV_REDUCE_AVG, CV_32F); 
	reduce(src, colMax, 0, CV_REDUCE_MAX, CV_32F); 
	reduce(src, colMin, 0, CV_REDUCE_MIN, CV_32F); 
	Mat tmp; //Mat tmp(colMin.rows, colMin.cols, colMin.type());
	subtract(colMax, colMin, tmp);  //tmp = colMax - colMin;

	//Mat tmpMin, tmpMean;
	//repeat(colMin, 1, src.cols, tmpMin);
	//repeat(colMean, 1, src.cols, tmpMean);
	//rst = src - tmpMin - tmpMean;
	for (int i = 0; i < rst.cols; i++)
	{
		float val = colMean.at<float>(0, i) + colMin.at<float>(0,i);
		rstTmp.col(i) = src.col(i) - val;
		val = tmp.at<float>(0, i);
		if (val > FLT_EPSILON)
			rstTmp.col(i) = rstTmp.col(i) / val;
		else 
			rstTmp.col(i) = 0;
	}
	rst = rstTmp;
}

void normTestData4SVM(const Mat src, Mat &rst, const Mat &colMean, const Mat &colMax, const Mat &colMin)
{
	Mat row, tmpRst(src.rows, src.cols, colMin.type()), tmp; //tmp(colMin.rows,colMin.cols, colMin.type());
	tmp = colMax - colMin;
	for (int i = 0; i < src.cols; i++)
	{
		float val = colMean.at<float>(0, i) + colMin.at<float>(0,i);
		tmpRst.col(i) = src.col(i) - val;
		val = tmp.at<float>(0, i);
		if (val > FLT_EPSILON)
			tmpRst.col(i) = tmpRst.col(i) / val;
		else 
			tmpRst.col(i) = 0;
	}
	rst = tmpRst;
}

void getSVMData(const string * fileName, Mat &Data, Mat &labels, int *classes, const int numClass, const int readFile)
{
	Mat pMat;
	int count=0;
	BinClusterInStream *pFile;
	Mat *pMatF = new Mat[numClass];
	Mat *label = new Mat[numClass];

	for (int i = 0; i < numClass; i++)
	{
		pFile = new BinClusterInStream(fileName[i]);
		pFile->read(pMat,readFile);
		pMatF[i] = Mat(pMat.rows, pMat.cols, CV_32FC1);
		label[i] = Mat(pMat.rows, 1, CV_32SC1, Scalar_<int>(i));

		//cout<<label[i].at<int>(200,0)<<"\n";
		normalizeMat(pMat, pMatF[i]);
		count += pMat.rows;
		if (classes)
			classes[i] = pMat.rows;

		delete pFile;
		pMat.release();
	}


	Data = Mat(count, pMatF[0].cols, CV_32FC1);
	labels = Mat(count, 1, CV_32SC1);
	Mat roi;
	count = 0;

	for (int i = 0; i < numClass; i++)
	{
		roi = Data.rowRange(count, count + pMatF[i].rows);
		pMatF[i].copyTo(roi);
		roi = labels.rowRange(count, count + label[i].rows);
		label[i].copyTo(roi);
		//cout<<label[i].at<int>(200,0)<<" "<<labels.at<int>(i*250,0)<<"\n";
		count += pMatF[i].rows;
		
	}
	delete []label;
	delete []pMatF;

}
