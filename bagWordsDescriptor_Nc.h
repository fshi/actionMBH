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
	Mat					*_bWords;	//bag of words
	int					*_ft;
	int					_channels;  //number of channels
	int					*_wordNums;   //word beginning number  for each channel. if 3 channels, first is "0", 
									//	second is _bwords[0].col, third is _bwords[0].col+_bwords[1].col. Total number of words = _wordNums[_channels]
	int					*_ftStep0;   //feature dimension for each channel before pca projection,
	int					*_ftStep;   //feature dimension for each channel,

	int					*_step;
	DescriptorMatcher	**_matcher;   //opencv matcher for matching vectors to BoWs

	matchType			_matcherTp;  //matcher types. 0 = BruteForce(default L2), 1 = BruteForce-L1 or 2 = FlannBased (faster)
	bagWordsFeature (const bagWordsFeature &q) {}  //fake copy 
	bagWordsFeature  &operator= (const bagWordsFeature &q) {return *this;}  //fake assignment

	PCA				    **_pca;  //using pca to reduce dimension
	int					*_maxComponents; // specify how many principal components to retain
	bool				_usePca;

	int					_colDims;   //total dims of _bWords[i].col, if pca, it equal to total reduced dim

	//bool				_sameWordDims;  // =1, if every channel has same number of words. For fast normalization
	int					_samples; // total number of samples per "getFeatures(hog3dFt0)". It equals to hog3dFt0.rows.

public:
	bagWordsFeature ():_wordsFile(NULL), _bWords(NULL), _ft(NULL), 
					   _wordNums(NULL), _matcherTp(BF_L2), _ftStep(NULL)
	{
		_matcher = NULL;
		_pca = NULL;
		_maxComponents = NULL;
	}

	bagWordsFeature(const std::string *iFile, int channels, const int const *maxComponents = NULL, int mTp = 2):_channels(channels), _samples(0)
    {
		_bWords = new Mat[channels];
		_wordNums = new int[channels+1];
		_ftStep0 = new int[channels];
		_ftStep = new int[channels];
		_step = new int[channels+1];
		_matcher = new DescriptorMatcher*[channels];
		_pca = NULL;
		_maxComponents = new int[channels];
		int wordDim = 0;
		int matchTp;
		_colDims=0;
		//_sameWordDims=1;

		for (int i = 0; i < channels; i++)
		{
			_wordsFile = new  BinClusterInStream(iFile[i]);
			_wordsFile->read(_bWords[i]);
			delete _wordsFile;
			_colDims += _bWords[i].cols;
			_ftStep0[i] = _bWords[i].cols;
			wordDim += _bWords[i].rows;
			_wordNums[i] = wordDim - _bWords[i].rows;
			//_sameWordDims = (_sameWordDims&&(_bWords[0].rows == _bWords[i].rows));

		}
		int pcaDim=0;
		if(maxComponents) 
		{
			for (int i = 0; i < channels; i++)
			{
				_maxComponents[i] = maxComponents[i];
				pcaDim+=_maxComponents[i];
			}
		}

		if(!maxComponents || pcaDim  >= _colDims )  
		{
			_usePca = false;
			_pca = NULL;
			matchTp = mTp;

		}
		else
		{
			_pca = new PCA*[channels];
			matchTp = 0;
			_usePca = true;
			std::cout<<"Now initializing dimension reduction for BoW with PCA...!\n";
			for (int i = 0; i < channels; i++)
			{
				_pca[i] = new PCA(_bWords[i], Mat(), CV_PCA_DATA_AS_ROW, _maxComponents[i]);
				//reduce the bag of words dimension 
				Mat pcaWords(_bWords[i].rows, _maxComponents[i], _bWords[i].type());
			
				for(int i0 = 0; i0 < _bWords[i].rows; i0++)
				{
					Mat vec = _bWords[i].row(i0), coeffs = pcaWords.row(i0);
					_pca[i]->project(vec, coeffs);
				}
				_bWords[i] = pcaWords;
			}
			std::cout<<"Done dimension reduction for BoW with PCA...!\n";
		}

		_colDims=0;
		for (int i = 0; i < channels; i++)
		{
			_ftStep[i]= _bWords[i].cols;
			_colDims += _bWords[i].cols;
		}

		_wordsFile = NULL;
		_wordNums[_channels] = wordDim;

		_step[0] = 0;
		for(int ch = 0; ch < _channels; ch++)
			_step[ch+1] = _ftStep[ch]/2;

		vector<Mat> words;
		if(matchTp == 0)
		{
			for (int i = 0; i < channels; i++)
			{
				_matcher[i] = new BFMatcher(NORM_L2);
				words.clear();
				words.push_back(_bWords[i]);
				_matcher[i]->add(words);
				//_matcher->train();
				_matcherTp = BF_L2;
			}
			
		}
		else if (matchTp == 1)
		{
			for (int i = 0; i < channels; i++)
			{
				_matcher[i] = new BFMatcher(NORM_L1);
				words.clear();
				words.push_back(_bWords[i]);
				_matcher[i]->add(words);
				_matcherTp = BF_L1; 
			}

		}
		else if (matchTp == 2)
		{
			for (int i = 0; i < channels; i++)
			{
				_matcher[i] = new FlannBasedMatcher();
				words.clear();
				words.push_back(_bWords[i]);
				_matcher[i]->add(words);
				_matcher[i]->train();
				_matcherTp = KNN;
			}
		}
		else  //Note that user matching is not working at the moment
		{
			std::cout<<"Note that user matching is not working at the moment!\n Please input Enter to exit!\n";
			discoverUO::wait();
			exit(-1);
			for (int i = 0; i < channels; i++)
				_matcher[i] = NULL;
			_matcherTp = USER;
		}

		_ft = new int[wordDim];
		memset(_ft, 0, sizeof(int)*wordDim);
	 }

	void operator() (const std::string *iFile, int channels, const int const *maxComponents = NULL, int mTp = 2)
    {
		this->~bagWordsFeature();
		_channels = channels;
		_samples=0;

			_bWords = new Mat[channels];
		_wordNums = new int[channels+1];
		_ftStep0 = new int[channels];
		_ftStep = new int[channels];
		_step = new int[channels+1];
		_matcher = new DescriptorMatcher*[channels];
		_pca = NULL;
		_maxComponents = new int[channels];
		int wordDim = 0;
		int matchTp;
		_colDims=0;
		//_sameWordDims=1;

		for (int i = 0; i < channels; i++)
		{
			_wordsFile = new  BinClusterInStream(iFile[i]);
			_wordsFile->read(_bWords[i]);
			delete _wordsFile;
			_colDims += _bWords[i].cols;
			_ftStep0[i] = _bWords[i].cols;
			wordDim += _bWords[i].rows;
			_wordNums[i] = wordDim - _bWords[i].rows;
			//_sameWordDims = (_sameWordDims&&(_bWords[0].rows == _bWords[i].rows));

		}
		int pcaDim=0;
		if(maxComponents) 
		{
			for (int i = 0; i < channels; i++)
			{
				_maxComponents[i] = maxComponents[i];
				pcaDim+=_maxComponents[i];
			}
		}

		if(!maxComponents || pcaDim  >= _colDims )  
		{
			_usePca = false;
			_pca = NULL;
			matchTp = mTp;

		}
		else
		{
			_pca = new PCA*[channels];
			matchTp = 0;
			_usePca = true;
			std::cout<<"Now initializing dimension reduction for BoW with PCA...!\n";
			for (int i = 0; i < channels; i++)
			{
				_pca[i] = new PCA(_bWords[i], Mat(), CV_PCA_DATA_AS_ROW, _maxComponents[i]);
				//reduce the bag of words dimension 
				Mat pcaWords(_bWords[i].rows, _maxComponents[i], _bWords[i].type());
			
				for(int i0 = 0; i0 < _bWords[i].rows; i0++)
				{
					Mat vec = _bWords[i].row(i0), coeffs = pcaWords.row(i0);
					_pca[i]->project(vec, coeffs);
				}
				_bWords[i] = pcaWords;
			}
			std::cout<<"Done dimension reduction for BoW with PCA...!\n";
		}

		_colDims=0;
		for (int i = 0; i < channels; i++)
		{
			_ftStep[i]= _bWords[i].cols;
			_colDims += _bWords[i].cols;
		}

		_wordsFile = NULL;
		_wordNums[_channels] = wordDim;

		_step[0] = 0;
		for(int ch = 0; ch < _channels; ch++)
			_step[ch+1] = _ftStep[ch]/2;

		vector<Mat> words;
		if(matchTp == 0)
		{
			for (int i = 0; i < channels; i++)
			{
				_matcher[i] = new BFMatcher(NORM_L2);
				words.clear();
				words.push_back(_bWords[i]);
				_matcher[i]->add(words);
				//_matcher->train();
				_matcherTp = BF_L2;
			}
			
		}
		else if (matchTp == 1)
		{
			for (int i = 0; i < channels; i++)
			{
				_matcher[i] = new BFMatcher(NORM_L1);
				words.clear();
				words.push_back(_bWords[i]);
				_matcher[i]->add(words);
				_matcherTp = BF_L1; 
			}

		}
		else if (matchTp == 2)
		{
			for (int i = 0; i < channels; i++)
			{
				_matcher[i] = new FlannBasedMatcher();
				words.clear();
				words.push_back(_bWords[i]);
				_matcher[i]->add(words);
				_matcher[i]->train();
				_matcherTp = KNN;
			}
		}
		else
		{
			for (int i = 0; i < channels; i++)
				_matcher[i] = NULL;
			_matcherTp = USER;
		}

		_ft = new int[wordDim];
		memset(_ft, 0, sizeof(int)*wordDim);

	 }

	vector<int> findWordIDbyMinDist(const Mat &sample)
	{
		vector<int> id;
		for(int ch = 0; ch < _channels; ch++)
			id.push_back(-1);
		Mat row, col0, col1;
		int st = 0;
		for(int ch = 0; ch < _channels; ch++)
		{
			double minDist = 1E10, dist;
			for (int i = 0; i < _bWords[ch].rows; i++)
			{
				row = _bWords[ch].row(i);
				col0 = row.colRange(st, st+_ftStep[ch]) ; 
				col1 = sample.colRange(st, st+_ftStep[ch]) ; 
				dist = norm(col0, col1, NORM_L2);
			
				if (dist < minDist)
				{
					id[ch] = i;
					minDist = dist;
					//std::cout<<minDist<<" \n";
				}
			}
			st += _ftStep[ch];
		}
		/*if (id<0 || id >=(int)_wordNums[_channels])
		{
			std::cout<<"wrong! "<<id;
			discoverUO::wait();
		}*/
		return id;
	}



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

	}

	void normlizeFt1(float* arr, int* src)
	{
		#pragma omp parallel for 
		for (int j = 0; j < _wordNums[_channels]; j++)
			arr[j] = (float) src[j]/(float)_samples;
	}

	void getNormlizedFt(const Mat &hog3dFt, float *arr)
	{
		getFeatures(hog3dFt);
		
		//cout<<_ft[0]<<" "<<_ft[10]<<" "<<_ft[90]<<" "<<_ft[1109]<<" "<<_ft[1209]<<" "<<_ft[2220]<<" "<<_ft[2309]<<" "<<_ft[3300]<<" "<<_ft[3309]<<" "<<_ft[3880]<<" "<<_ft[3990]<<endl;
		normlizeFt1(arr, _ft);
		
		//cout<<arr[0]<<" "<<arr[10]<<" "<<arr[90]<<" "<<arr[1109]<<" "<<arr[1209]<<" "<<arr[2220]<<" "<<arr[2309]<<" "<<arr[3300]<<" "<<arr[3309]<<" "<<arr[3880]<<" "<<arr[3990]<<endl;
	}

	void getFeatures(const Mat &hog3dFt0)
	{
		Mat hog3dFt;
		memset(_ft, 0, sizeof(int)*_wordNums[_channels]);//set all words frequnce to 0 before computing new bag of word's frequency
		_samples = hog3dFt0.rows;

		if(!_samples)
			return;

		if(_usePca)
		{
			Mat ftCols;
			int col0 = 0, col = 0;
			hog3dFt.create(hog3dFt0.rows, _colDims, hog3dFt0.type());
			for(int ch = 0; ch < _channels; ch++)
			{
				//std::cout<<"Now doing dimension reduction for features with PCA...!\n";
				
				ftCols = hog3dFt0.colRange(col0, col0 + _ftStep0[ch]);
	#pragma omp parallel for //num_threads(4)
				for(int i = 0; i < hog3dFt.rows; i++)
				{
					Mat vec = ftCols.row(i), coeffs, row = hog3dFt.row(i);
					_pca[ch]->project(vec, coeffs);
					coeffs.copyTo(row.colRange(col, col +_ftStep[ch]));
				}
				col0 += _ftStep0[ch];
				col += _ftStep[ch];
				//std::cout<<"Done dimension reduction for features with PCA...!\n";
			}
		}
		else
			hog3dFt = hog3dFt0;

		if (_matcherTp == 3)
		{
#pragma omp parallel for //num_threads(4)
			for(int i = 0; i < hog3dFt.rows; i++)
			{
				Mat row = hog3dFt.row(i);
				vector<int> id = findWordIDbyMinDist(row);
#pragma omp critical
				for(int ch = 0; ch < _channels; ch++)
					_ft[id[ch]+_wordNums[ch]] += 1;
			}

		}
		else
		{
			vector<DMatch> matches;

			Mat *buffer0 = new Mat[_channels];
			for(int ch = 0; ch < _channels; ch++)
				buffer0[ch] = Mat(hog3dFt.rows, _ftStep[ch], hog3dFt.type());
			int col0 = 0;
			for(int ch = 0; ch < _channels; ch++)
			{
				(hog3dFt.colRange(col0, col0 + _ftStep[ch])).copyTo(buffer0[ch]);
				matches.clear();
				_matcher[ch]->match(buffer0[ch],  matches);
				for (int i = 0; i < matches.size(); i++)
					_ft[matches[i].trainIdx+_wordNums[ch]] += 1;

				col0 += _ftStep[ch];
			}
			delete []buffer0;
		}

	}

	void getFeatures(const Mat &hog3dFt, int *arr)
	{
		getFeatures(hog3dFt);
		memcpy(arr, _ft, sizeof(int)*_wordNums[_channels]);

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
	
	inline Mat getWords(int i)const
	{
		return _bWords[i];
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
		 delete []_ftStep0;
		 delete []_ftStep;
		 delete []_bWords;
		 delete _wordsFile;
		 delete[]_wordNums;
		 delete []_step;
		 delete []_pca;
		 delete []_maxComponents;
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
