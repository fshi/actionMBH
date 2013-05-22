#ifndef _BAGWORDS_DESCRIPTOR_H_
#define _BAGWORDS_DESCRIPTOR_H_

#include "cxcore.h"
#include "cv.h"

#include "formatBinaryStream.h"

#include "waitKeySeconds.h"
#include "opencv2/ocl/ocl.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

//enum matchType {BF_L2, KNN};
class bagWordsFeature {

	//Point3i			  _numCell;  //number of cells per descriptor, default is 4x4x3
	//Point3i			  _subBlocks;  //number of subblocks per cell. default is 3x3x3
	
	BinClusterInStream	*_wordsFile;  //file to input bag of words
	oclMat				_bWords;	//bag of words
	oclMat				_buffer0;	//for avoiding gpu re-allocation
	
	//matchType			_matcherTp;  //matcher types. 0 = BruteForce(default L2), 2 = FlannBased (faster)
	
	BruteForceMatcher_OCL_base *_matcher;
	//BruteForceMatcher_OCL< L2<float> >	 _matcher;
	int					*_ft;

	unsigned int		_wordNum;   //total number of words in the bag
	//unsigned int		_ftLen;		//feature/word length

	
	bagWordsFeature (const bagWordsFeature &q) {}  //fake copy 
	bagWordsFeature  &operator= (const bagWordsFeature &q) {return *this;}  //fake assignment


public:
	bagWordsFeature ():_wordsFile(NULL), _bWords(Mat()), _buffer0(Mat()), _ft(NULL), _wordNum(0), _matcher(NULL){}

	bagWordsFeature(const std::string &iFile)
    {
		/*
		if(matchTp == 1)
			_matcherTp = KNN;
		else
			_matcherTp = BF_L2;
		*/
		_wordsFile = new  BinClusterInStream(iFile);
		Mat bword;
		_wordsFile->read(bword);
		delete _wordsFile;
		_wordsFile = NULL;
		_bWords = bword;
		_matcher = new BruteForceMatcher_OCL_base(BruteForceMatcher_OCL_base::L2Dist);
		//suffleCvMat(_bWords);
		_wordNum = bword.rows;

		_ft = new int[_wordNum];
		memset(_ft, 0, sizeof(int)*_wordNum);

/*		//test data reading
		std::cout<<_bWords.at<float>(0, 100)<<" "<<_bWords.at<float>(0, 150)<<" "<<_bWords.at<float>(0, 100)<<
			" "<<_bWords.at<float>(0, 350)<<" "<<_bWords.at<float>(0, 500)<<" "
			<<_bWords.at<float>(0, 550)<<" "<<_bWords.at<float>(0, 700)<<
			" "<<_bWords.at<float>(0, 750)<<" "<<_bWords.at<float>(0, 880)<<"\n";
*/
	 }

	void operator() (const std::string &iFile)
    {
		/*
		if(matchTp == 1)
			_matcherTp = KNN;
		else
			_matcherTp = BF_L2;
		*/
		this->~bagWordsFeature();
		_wordsFile = new  BinClusterInStream(iFile);
		Mat bword;
		_wordsFile->read(bword);
		delete _wordsFile;
		_wordsFile = NULL;
		_bWords = bword;
		_matcher = new BruteForceMatcher_OCL_base(BruteForceMatcher_OCL_base::L2Dist);
		//suffleCvMat(_bWords);
		_wordNum = bword.rows;

		_ft = new int[_wordNum];
		memset(_ft, 0, sizeof(int)*_wordNum);

	 }

	void normlizeFt(float* arr, int* src)
	{
		float sum = 0.f;
		for (int i = 0; i < (int)_wordNum; i++)
			sum += src[i];
#pragma omp parallel for //num_threads(4)
		for (int j = 0; j < (int)_wordNum; j++)
			arr[j] = (float) src[j]/sum;
	}

	void getNormlizedFt(const Mat &hog3dFt, float *arr ,int splitSz = 20000)
	{
		getFeatures(hog3dFt, splitSz);
		normlizeFt(arr, _ft);
	}

	void getFeatures(const Mat &hog3dFt, int splitSz = 20000)
	{
		
		memset(_ft, 0, sizeof(int)*_wordNum);//set all words frequnce to 0 before computing new bag of word's frequency
		Mat ftRows;
		//Mat M0(7,7,CV_32SC1,Scalar(1));
		vector<DMatch> matches;
		for( int rowNo = 0; rowNo < hog3dFt.rows; rowNo += splitSz)
		{
			ftRows = hog3dFt.rows > rowNo+splitSz ? hog3dFt.rowRange(rowNo, rowNo + splitSz) : hog3dFt.rowRange(rowNo,  hog3dFt.rows);
			_buffer0 = ftRows;
			matches.clear();
			_matcher->match(_buffer0, _bWords, matches);

			//cout<<"matches size is: "<<matches.size()<<" feature size is: "<<_buffer0.rows<<" word size is: "<<_bWords.rows<<endl;
			for (int i = 0; i < matches.size(); i++)
				_ft[matches[i].trainIdx] += 1;

		}
	}

	void getFeatures(const Mat &hog3dFt, int *arr, int splitSz = 20000)
	{
		getFeatures(hog3dFt, splitSz);
		memcpy(arr, _ft, sizeof(int)*_wordNum);

	}

	inline int getWordNum() const 
	 {
		 return _wordNum;
	 }

	inline int getWordLen() const 
	 {
		 return _bWords.cols;
	 }

	Mat getWords()const
	{
		Mat words;
		_bWords.download(words);
		//Mat words = _bWords;
		return words;
	}

	 ~bagWordsFeature()
	 {
		 delete []_ft;
		 _bWords.release();
		 _buffer0.release();
		 delete _wordsFile;
		 delete _matcher;
	 }

	 //this function suffles Mat rows
	 void suffleCvMat(Mat &mx)
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

};  //end of class bagWordsFeature

#endif //_BAGWORDS_DESCRIPTOR_H_
