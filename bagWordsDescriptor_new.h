/********************************************************************************

Filename     : bagWordsDescriptor_new.h

Description  : This class is used to compute "word frequency" of "bag of features" model
			   It need to be initialized with a presaved "words" file


Typical Use  :  bagWordsFeature bw("wordFile.dat", 2)  //load the precomputed "words" from file. "2"means flann words matching. 
				Mat feautres =...;
				float arr[numWords];
				bw.getNormlizedFt(features, arr);

				
Author       : FengShi@Discovery lab, May, 2010
Version No   : 1.00 


*********************************************************************************/


#ifndef _BAGWORDS_DESCRIPTOR_H_
#define _BAGWORDS_DESCRIPTOR_H_

#include "cxcore.h"
#include "cv.h"

#include "formatBinaryStream.h"

#include "waitKeySeconds.h"

using namespace cv;

enum matchType {BF_L2, BF_L1, KNN, USER};
class bagWordsFeature {

	//Point3i			  _numCell;  //number of cells per descriptor, default is 4x4x3
	//Point3i			  _subBlocks;  //number of subblocks per cell. default is 3x3x3
	
	BinClusterInStream	*_wordsFile;  //file to input bag of words
	Mat					_bWords;	//bag of words
	int					*_ft;

	unsigned int		_wordNum;   //total number of words in the bag
	//unsigned int		_ftLen;		//feature/word length

	DescriptorMatcher	*_matcher;   //opencv matcher for matching vectors to BoWs
	matchType			_matcherTp;  //matcher types. 0 = BruteForce(default L2), 1 = BruteForce-L1 or 2 = FlannBased (faster)
	bagWordsFeature (const bagWordsFeature &q) {}  //fake copy 
	bagWordsFeature  &operator= (const bagWordsFeature &q) {return *this;}  //fake assignment


public:
	bagWordsFeature ():_wordsFile(NULL), _bWords(Mat()), _ft(NULL), 
					   _wordNum(0), _matcherTp(BF_L2)
	{
		_matcher = NULL;
	}

	bagWordsFeature(const std::string &iFile, int matchTp = 2)  //initialize the class with pre-saved "words" file. the default words matching is Flann
    {
		_wordsFile = new  BinClusterInStream(iFile);
		_wordsFile->read(_bWords);
		//suffleCvMat(_bWords);
		delete _wordsFile;

		_wordsFile = NULL;
		_wordNum = _bWords.rows;

		if(matchTp == 0)  //L2 brute force matching 
		{
			_matcher = new BFMatcher(NORM_L2);
			vector<Mat> words;
			words.push_back(_bWords);
			_matcher->add(words);
			//_matcher->train();
			_matcherTp = BF_L2;
			
		}
		else if (matchTp == 1)   //L1 brute force matching 
		{
			_matcher = new BFMatcher(NORM_L1);
			vector<Mat> words;
			words.push_back(_bWords);
			_matcher->add(words);
			//_matcher->train();
			_matcherTp = BF_L1; 

		}
		else if (matchTp == 2)  //Flann matching
		{
			_matcher = new FlannBasedMatcher();
			vector<Mat> words;
			words.push_back(_bWords);
			_matcher->add(words);
			_matcher->train();
			_matcherTp = KNN;
		}
		else  
		{
			_matcher = NULL;
			_matcherTp = USER;
		}

		_ft = new int[_wordNum];
		memset(_ft, 0, sizeof(int)*_wordNum);

/*		//test data reading
		std::cout<<_bWords.at<float>(0, 100)<<" "<<_bWords.at<float>(0, 150)<<" "<<_bWords.at<float>(0, 100)<<
			" "<<_bWords.at<float>(0, 350)<<" "<<_bWords.at<float>(0, 500)<<" "
			<<_bWords.at<float>(0, 550)<<" "<<_bWords.at<float>(0, 700)<<
			" "<<_bWords.at<float>(0, 750)<<" "<<_bWords.at<float>(0, 880)<<"\n";
*/
	 }

	void operator() (const std::string &iFile, int matchTp = 2)
    {
		this->~bagWordsFeature();
		_wordsFile = new  BinClusterInStream(iFile);
		_wordsFile->read(_bWords);
		delete _wordsFile;
		_wordsFile = NULL;
		//cvAssert(!_bWords);
		//suffleCvMat(_bWords);
		_wordNum = _bWords.rows;
		
		if(matchTp == 0)
		{
			_matcher = new BFMatcher(NORM_L2);
			vector<Mat> words;
			words.push_back(_bWords);
			_matcher->add(words);
			//_matcher->train();
			_matcherTp = BF_L2;
			
		}
		else if (matchTp == 1)
		{
			_matcher = new BFMatcher(NORM_L1);
			vector<Mat> words;
			words.push_back(_bWords);
			_matcher->add(words);
			//_matcher->train();
			_matcherTp = BF_L1; 

		}
		else if (matchTp == 2)
		{
			_matcher = new FlannBasedMatcher();
			vector<Mat> words;
			words.push_back(_bWords);
			_matcher->add(words);
			_matcher->train();
			_matcherTp = KNN;
		}
		else
		{
			_matcher = NULL;
			_matcherTp = USER;
		}

		_ft = new int[_wordNum];
		memset(_ft, 0, sizeof(int)*_wordNum);

	 }

	int findWordIDbyMinDist(const Mat &sample)
	{
		double minDist = 1E10, dist;
		int id = -1;
		Mat row;
		for (int i = 0; i < (int)_wordNum; i++)
		{
			row = _bWords.row(i);
			dist = norm(row, sample, NORM_L2);
			
			if (dist < minDist)
			{
				id = i;
				minDist = dist;
				//std::cout<<minDist<<" \n";
			}
		}
		/*if (id<0 || id >=(int)_wordNum)
		{
			std::cout<<"wrong! "<<id;
			discoverUO::wait();
		}*/
		return id;
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

	void getNormlizedFt(const Mat &hog3dFt, float *arr)
	{
		getFeatures(hog3dFt);
		//cout<<_ft[0]<<" "<<_ft[10]<<" "<<_ft[90]<<" "<<_ft[1109]<<" "<<_ft[1209]<<" "<<_ft[2220]<<" "<<_ft[2309]<<" "<<_ft[3300]<<" "<<_ft[3309]<<" "<<_ft[3880]<<" "<<_ft[3990]<<endl;
		normlizeFt(arr, _ft);
		//cout<<arr[0]<<" "<<arr[10]<<" "<<arr[90]<<" "<<arr[1109]<<" "<<arr[1209]<<" "<<arr[2220]<<" "<<arr[2309]<<" "<<arr[3300]<<" "<<arr[3309]<<" "<<arr[3880]<<" "<<arr[3990]<<endl;
	}

	void getFeatures(const Mat &hog3dFt)
	{
		memset(_ft, 0, sizeof(int)*_wordNum);//set all words frequnce to 0 before computing new bag of word's frequency
		/*
		if (_matcherTp == 2)
		{
			
			vector<vector<DMatch>> matches;
			_matcher->knnMatch(hog3dFt, matches, 1);
			
			//cout<<"matches size is: "<<matches.size()<<" feature size is: "<<_buffer0.rows<<endl;
			//cout<< matches.size()<<" "<<matches[0].size()<<" "<<hog3dFt.rows<<endl;
			for (int i = 0; i < matches.size(); i++)
				_ft[matches[i][0].trainIdx] += 1;
		}
		*/
		if (_matcherTp == 3)
		{
#pragma omp parallel for //num_threads(4)
			for(int i = 0; i < hog3dFt.rows; i++)
			{
				Mat row = hog3dFt.row(i);
				int id = findWordIDbyMinDist(row);
#pragma omp critical
				_ft[id] += 1;
			}

		}
		else
		{
			vector<DMatch> matches;
			_matcher->match(hog3dFt, matches);

			//cout<<"matches size is: "<<matches.size()<<" feature size is: "<<_bWords.rows<<endl;
			for (int i = 0; i < matches.size(); i++)
				_ft[matches[i].trainIdx] += 1;
		}
	}

	void getFeatures(const Mat &hog3dFt, int *arr)
	{
		getFeatures(hog3dFt);
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
		return _bWords.clone();
	}

	 ~bagWordsFeature()
	 {
		 delete []_ft;
		 _bWords.release();
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
