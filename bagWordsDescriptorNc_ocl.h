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
	oclMat				*_bWords;	//bag of words
	oclMat				*_buffer0;	//for avoiding gpu re-allocation
	
	//matchType			_matcherTp;  //matcher types. 0 = BruteForce(default L2), 2 = FlannBased (faster)
	
	BruteForceMatcher_OCL_base **_matcher;
	//BruteForceMatcher_OCL< L2<float> >	 _matcher;
	int					*_ft;
	int					_channels;  //number of channels

	int					*_wordNums;   //word beginning number  for each channel. if 3 channels, first is "0", 
									//	second is _bwords[0].col, third is _bwords[0].col+_bwords[1].col. Total number of words = _wordNums[_channels]
	int					*_step;
	int					*_ftStep;   //feature dimension for each channel,
	//unsigned int		_wordDim;   //total number of words in the bag
	//unsigned int		_ftLen;		//feature/word length

	//bool				_sameWordDims;  // =1, if every channel has same number of words. For fast normalization
	int					_samples; // total number of samples per "getFeatures(hog3dFt0)". It equals to hog3dFt0.rows.

	
	bagWordsFeature (const bagWordsFeature &q) {}  //fake copy 
	bagWordsFeature  &operator= (const bagWordsFeature &q) {return *this;}  //fake assignment


public:
	bagWordsFeature ():_wordsFile(NULL), _bWords(NULL), _buffer0(NULL), _ft(NULL), _ftStep(NULL), _matcher(NULL),_channels(0), _wordNums(NULL){}

	bagWordsFeature(const std::string *iFile, int channels):_channels(channels), _samples(0)
    {
		/*
		if(matchTp == 1)
			_matcherTp = KNN;
		else
			_matcherTp = BF_L2;
		*/
		_bWords = new oclMat[channels];
		_buffer0 = new oclMat[channels];
		_wordNums = new int[channels+1];
		_ftStep = new int[channels];
		_step = new int[channels+1];
		_matcher = new BruteForceMatcher_OCL_base*[channels];
		int wordDim = 0;
		//_sameWordDims=1;

		for (int i = 0; i < channels; i++)
		{
			_wordsFile = new  BinClusterInStream(iFile[i]);
			Mat bword=Mat();
			_wordsFile->read(bword);
			_ftStep[i]= bword.cols;
			wordDim += bword.rows;
			_wordNums[i] = wordDim - bword.rows;
			_bWords[i].upload(bword);
			delete _wordsFile;
			_matcher[i] = new BruteForceMatcher_OCL_base(BruteForceMatcher_OCL_base::L2Dist);
			//_sameWordDims = (_sameWordDims&&(_bWords[0].rows == _bWords[i].rows));
		}
		_wordsFile = NULL;
		_wordNums[_channels] = wordDim;
		//suffleCvMat(_bWords);
		_step[0] = 0;
		for(int ch = 0; ch < _channels; ch++)
			_step[ch+1] = _ftStep[ch]/2;

		_ft = new int[wordDim];
		memset(_ft, 0, sizeof(int)*wordDim);

/*		//test data reading
		std::cout<<_bWords.at<float>(0, 100)<<" "<<_bWords.at<float>(0, 150)<<" "<<_bWords.at<float>(0, 100)<<
			" "<<_bWords.at<float>(0, 350)<<" "<<_bWords.at<float>(0, 500)<<" "
			<<_bWords.at<float>(0, 550)<<" "<<_bWords.at<float>(0, 700)<<
			" "<<_bWords.at<float>(0, 750)<<" "<<_bWords.at<float>(0, 880)<<"\n";
*/
	 }

	void operator() (const std::string *iFile, int channels)
    {
		/*
		if(matchTp == 1)
			_matcherTp = KNN;
		else
			_matcherTp = BF_L2;
		*/
		this->~bagWordsFeature();
		_channels = channels;
		_bWords = new oclMat[channels];
		_buffer0 = new oclMat[channels];
		_wordNums = new int[channels+1];
		_step = new int[channels+1];
		_ftStep = new int[channels];
		_matcher = new BruteForceMatcher_OCL_base*[channels];
		int wordDim = 0;
		//_sameWordDims=1;
		_samples=0;
		for (int i = 0; i < channels; i++)
		{
			_wordsFile = new  BinClusterInStream(iFile[i]);
			Mat bword=Mat();
			_wordsFile->read(bword);
			delete _wordsFile;

			_ftStep[i]= bword.cols;
			wordDim += bword.rows;
			_wordNums[i] = wordDim - bword.rows;
			_bWords[i].upload(bword);
			_matcher[i] = new BruteForceMatcher_OCL_base(BruteForceMatcher_OCL_base::L2Dist);
			//_sameWordDims = (_sameWordDims&&(_bWords[0].rows == _bWords[i].rows));
		}
		_wordsFile = NULL;
		_wordNums[_channels] = wordDim;
		//suffleCvMat(_bWords);
		_step[0] = 0;
		for(int ch = 0; ch < _channels; ch++)
			_step[ch+1] = _ftStep[ch]/2;

		_ft = new int[wordDim];
		memset(_ft, 0, sizeof(int)*wordDim);
	 }
/*
	void normlizeFt0(float* arr, int* src)
	{
		int *pInt = src;
		float *pFt = arr;
		int st ;

		for(int i = 0; i < _channels; i++)
		{
			st = _wordNums[i+1]-_wordNums[i]; //_wordNums[0]==0
		
			Mat m0(1, st, CV_32SC1, pInt);
			Mat m1(1, st, CV_32FC1, pFt);

			normalize(m0, m1, 1.0, 0.0, NORM_L1, CV_32F);

			pInt += _wordNums[i+1];  //_wordNums[0]==0
			pFt += _wordNums[i+1];
		}
	}

	void normlizeFt(float* arr, int* src)
	{
		#pragma omp parallel for //num_threads(4)
		for(int i0 = 0; i0 < _channels; i0++)
		{
			double sum = 0.;
			for (int i = _wordNums[i0]; i < _wordNums[i0+1]; i++)
				sum += src[i];
	
			for (int j = _wordNums[i0]; j < _wordNums[i0+1]; j++)
				arr[j] = (float) src[j]/sum;
		}
	}
*/

	void normlizeFt(float* arr, int* src)
	{
		//No matter if wordDims is same or not for each channel, 
		//the total number of samples for each channel is same. 
		//Therefore, we only need to add the first channel to normalize.
		int sum = 0;
		#pragma omp parallel for reduction(+:sum )
		for (int i = _wordNums[0]; i < _wordNums[1]; i++)  
			sum += src[i];
		#pragma omp parallel for //num_threads(4)
		for (int j = 0; j < _wordNums[_channels]; j++)
			arr[j] = (float) src[j]/(float)sum;

		//cout<<"sum: "<<sum<<endl;

	}

	void normlizeFt1(float* arr, int* src)
	{
		#pragma omp parallel for 
		for (int j = 0; j < _wordNums[_channels]; j++)
			arr[j] = (float) src[j]/(float)_samples;
	}

	void getNormlizedFt(const Mat &hog3dFt, float *arr, int splitSz = 20000)
	{
		getFeatures(hog3dFt, splitSz);
		normlizeFt1(arr, _ft);
	}

	void getFeatures(const Mat &hog3dFt, int *arr, int splitSz = 20000)
	{
		getFeatures(hog3dFt, splitSz);
		memcpy(arr, _ft, sizeof(int)*_wordNums[_channels]);

	}

	void getFeatures(const Mat &hog3dFt, int splitSz = 20000)  //step[0] ==0;
	{
		_samples = hog3dFt.rows;
		
		memset(_ft, 0, sizeof(int)*_wordNums[_channels]);//set all words frequnce to 0 before computing new bag of word's frequency
		Mat ftRows;
		//Mat M0(7,7,CV_32SC1,Scalar(1));
		vector<DMatch> matches;
		Mat* buffer = new Mat[_channels];

		for( int rowNo = 0; rowNo < hog3dFt.rows; rowNo += splitSz)
		{
			ftRows = hog3dFt.rows > rowNo+splitSz ? hog3dFt.rowRange(rowNo, rowNo + splitSz) : hog3dFt.rowRange(rowNo,  hog3dFt.rows);

			for(int ch = 0; ch < _channels; ch++)
				buffer[ch] = Mat(ftRows.rows, _ftStep[ch], hog3dFt.type());

			int col0 = 0;
			for(int ch = 0; ch < _channels; ch++)
			{
				(ftRows.colRange(col0, col0 + _ftStep[ch])).copyTo(buffer[ch]);
				
				matches.clear();
				_buffer0[ch].upload( buffer[ch] );
				_matcher[ch]->match(_buffer0[ch], _bWords[ch], matches);
				for (int i = 0; i < matches.size(); i++)
					_ft[matches[i].trainIdx+_wordNums[ch]] += 1;
				//cout<<_bWords[ch].size()<<endl;
				//cout<<"sample sz: "<<_samples<<" matcher size: "<<matches.size()<<endl;
				col0 += _ftStep[ch];
			}
		}
		delete []buffer;
	}


	inline int getWordNum() const 
	 {
		 return _wordNums[_channels];
	 }

	inline int getWordLen(int i) const 
	 {
		 return _bWords[i].cols;
	 }

	inline int getFtDim() const 
	 {
		 int dim = _ftStep[0];
		 for(int i = 1; i < _channels; i++)
			 dim += _ftStep[i];
		 return dim;
	 }

	Mat getWords(int i)const
	{
		Mat words;
		_bWords[i].download(words);
		//Mat words = _bWords;
		return words;
	}

	 ~bagWordsFeature()
	 {
		 if(_matcher)
		 {
			 for(int i = 0; i < _channels; i++)
				 delete _matcher[i];
			 delete []_matcher;

		 }
		 delete []_ft;
		 delete []_ftStep;
		 delete []_bWords;
		 delete []_step;
		 delete []_buffer0;
		 delete _wordsFile;
		 delete[]_wordNums;
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
