#include "stDescriptor.h"


MBHparam::MBHparam(
			const Point3i&  numBloc_,
			const Point3i&  numCell_,
			bool roundTp_,
			bool normBlk_,
			float cutZ_,
			float reSzV_,
			uchar bins_,
			bool fullOri_,
			const Point3f& sRa,
			const Point3i& stSz_,
			const Point3i& scls,
			const Point3f& r2p,
			const Point3i& noPts,
			const Point3f& ovlp,
			float	  sigma,
			float	  tao	) :
			numCell(numCell_),
			numBlock(numBloc_),
			normBlk(normBlk_),
			rdTp(roundTp_),
			cutVal(cutZ_),
			reSzVideo(reSzV_),
			nBins(bins_),
			fullOri(fullOri_),
			stSz(stSz_),
			scales(scls),
			strideRatio(sRa),
			rt2ps(r2p),
			numParts(noPts),
			olRatio(ovlp),
			sigma2(sigma),
			tao2(tao)
    { }
bool MBHparam::readParam(const string& fileName, bool showPara )
	{
		ifstream inFile;
		inFile.open(fileName.c_str());
		if (!inFile.is_open())
		{
			std::cout<<"Unable to open parameters' flie \""<<fileName<<"\" for reading paramters!\n";
			//discoverUO::wait();
			return false;
		}
		string tmp;
		getline(inFile, tmp);
		getline(inFile, tmp,':');
		inFile>>numCell.x>>numCell.y>>numCell.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<numCell.x<<"  "<<numCell.y<<"  "<<numCell.z<<endl;
		getline(inFile, tmp,':');
		inFile>>numBlock.x>>numBlock.y>>numBlock.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<numBlock.x<<"  "<<numBlock.y<<"  "<<numBlock.z<<endl;
		getline(inFile, tmp,':');
		int tt;
		inFile>>tt;
		nBins = (uchar)tt;
		if (showPara)
			std::cout<<tmp<<":\t"<<(int)nBins<<endl;
		getline(inFile, tmp,':');
		inFile>>fullOri;
		if (showPara)
			std::cout<<tmp<<":\t"<<fullOri<<endl;
		getline(inFile, tmp,':');
		inFile>>cutVal;
		if (showPara)
			std::cout<<tmp<<":\t"<<cutVal<<endl;
		getline(inFile, tmp,':');
		inFile>>rdTp;
		if (showPara)
			std::cout<<tmp<<":\t"<<rdTp<<endl;
		getline(inFile, tmp,':');
		inFile>>normBlk;
		if (showPara)
			std::cout<<tmp<<":\t"<<normBlk<<endl;
		getline(inFile, tmp,':');
		inFile>>reSzVideo;
		if (showPara)
			std::cout<<tmp<<":\t"<<reSzVideo<<endl;
		getline(inFile, tmp,':');
		inFile>>rt2ps.x>>rt2ps.y>>rt2ps.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<rt2ps.x<<"  "<<rt2ps.y<<"  "<<rt2ps.z<<endl;
		getline(inFile, tmp,':');
		inFile>>numParts.x>>numParts.y>>numParts.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<numParts.x<<"  "<<numParts.y<<"  "<<numParts.z<<endl;
		getline(inFile, tmp,':');
		inFile>>olRatio.x>>olRatio.y>>olRatio.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<olRatio.x<<"  "<<olRatio.y<<"  "<<olRatio.z<<endl;
		getline(inFile, tmp,':');
		inFile>>strideRatio.x>>strideRatio.y>>strideRatio.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<strideRatio.x<<"  "<<strideRatio.y<<"  "<<strideRatio.z<<endl;

		getline(inFile, tmp,':');
		inFile>>stSz.x>>stSz.y>>stSz.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<stSz.x<<"  "<<stSz.y<<"  "<<stSz.z<<endl;

		getline(inFile, tmp,':');
		inFile>>scales.x>>scales.y>>scales.z;
		if (showPara)
			std::cout<<tmp<<":\t"<<scales.x<<"  "<<scales.y<<"  "<<scales.z<<endl;

		getline(inFile, tmp,':');
		inFile>>sigma2;
		if (showPara)
			std::cout<<tmp<<":\t"<<sigma2<<endl;
		getline(inFile, tmp,':');
		inFile>>tao2;
		if (showPara)
			std::cout<<tmp<<":\t"<<tao2<<endl;


		inFile.close();
		return true;
	}

bool MBHparam::writeParam(const string& fileName, bool app)
	{
		ofstream outFile;
		if (app)
			outFile.open(fileName.c_str(), ios_base::app);
		else
			outFile.open(fileName.c_str()) ;
		if (!outFile.is_open())
		{
			std::cout<<"Unable to open parameters' flie \""<<fileName<<"\" for writing paramters!\n";
			discoverUO::wait();
			return false;
		}

		outFile<<"Parameters for HOG3Ddescriptor:\n";

		outFile<<"\tNumber of cells per descriptor (x, y, t):  "<<numCell.x<<"  "<<numCell.y<<"  "<<numCell.z<<"\n";

		outFile<<"\tNumber of subblocks per cell (x, y, t):  "<<numBlock.x<<"  "<<numBlock.y<<"  "<<numBlock.z<<"\n";

		outFile<<"\tNumber of bins (default is 8):   "<<(int)nBins<<"\n";

		outFile<<"\tFull orientation:   "<<fullOri<<"\n";

		outFile<<"\tCut vaule for normalization:  "<<cutVal<<"\n";

		outFile<<"\tRound types:  "<<rdTp<<"\n";

		outFile<<"\tNormalize subblock with number of pixels inside:  "<<normBlk<<"\n";

		outFile<<"\tRate for resizing the input video to compute integral video:  "<<reSzVideo<<"\n";

		outFile<<"\tRatio for root video size to part video size (x, y, t):  "<<rt2ps.x<<"  "<<rt2ps.y<<"  "<<rt2ps.z<<"\n";

		outFile<<"\tNnumber of parts per root window (x, y, t):  "<<numParts.x<<"  "<<numParts.y<<"  "<<numParts.z<<"\n";

		outFile<<"\tOverlap ratio per 3d parts (x, y, t):  "<<olRatio.x<<"  "<<olRatio.y<<"  "<<olRatio.z<<"\n";

		outFile<<"\tStride overlap ratio per 3d window slide (x, y, t):  "<<strideRatio.x<<"  "<<strideRatio.y<<"  "<<strideRatio.z<<"\n";

		outFile<<"\tStart (smallest) 3D patch size (x, y, t):  "<<stSz.x<<"  "<<stSz.y<<"  "<<stSz.z<<"\n";

		outFile<<"\tNumber of total scales (x, y, t):  "<<scales.x<<"  "<<scales.y<<"  "<<scales.z<<"\n";

		outFile<<"\tSteps' factor for spatial scale:  "<<sigma2<<"\n";

		outFile<<"\tSteps' factor for temporal scale:  "<<tao2<<"\n";
	
		outFile.close();

		return true;
	}

stDetector::stDetector(const MBHparam* param):_ft(Mat())
    {
		if (param)
		{
			_dscFt = new descFeature(param->numBlock, param->numCell, param->rt2ps, param->olRatio, param->numParts, \
				 param->rdTp, param->normBlk, param->cutVal, param->reSzVideo, param->nBins, param->fullOri);
			_s3Droi = new sampled3Droi(param->strideRatio, param->stSz, param->scales, \
				param->sigma2, param->tao2); 
		}
		else
		{//use default parameters
			_dscFt = new descFeature();
			_s3Droi = new sampled3Droi();

		}
		_width = _dscFt->toFeatureSz();
	 }


bool stDetector::preProcessing(const string& fName, int fms)
	{
		_s3Droi->clear();//for memory efficient
		_ft.release();
		VideoCapture cap;
		cap.open(fName);
		_vLen = cap.get(CV_CAP_PROP_FRAME_COUNT);
		//cout<<"video length: "<<_vLen<<" frame smaple: "<<fms<<endl;
		cap.release();
		
		_redoNum = _vLen/fms;
		if ( (!(_vLen%fms)) || (_vLen == fms + 2) || (_vLen == fms + 1))
			_redoNum--;
	
		if	(!_dscFt-> preProcessor(fName, 0, fms+2))  //"+2" for border
		{
			std::cerr<<"stDetector::getFeatures(...): Wrong input video!"<<std::endl;
			return false;
		}
		Point3i vSz = _dscFt->videoSz();
		if (_redoNum )
			vSz.z -= 2; //removing border area
			//vSz.z -= strideRatio.z*stSz.z;
		_s3Droi->get3Droi(vSz);
		std::cout<<"Video length(frames) : "<<_vLen<<" Integral video size: "<<_dscFt->videoSz()<<" Redo numbers: "<<_redoNum<<endl;
		
		return true;
	}

void stDetector::clear()   //for memory efficient  _ft.release() can't be called because the _ft may be used
{
	_s3Droi->clear();
	_dscFt->clear();

}

bool stDetector::re_Processing(const string& fName, int fms, int num)
	{
		_s3Droi->clear();//for memory efficient
		_ft.release();

		int size =(int)(pow(_s3Droi->scaleStep().y, _s3Droi->scaleSz().z-1)*_s3Droi->startSz().z);
		int overlapFrames = _s3Droi->strideRt().z*size;  //the overlap frames from former processing (+ 2 is for border area)

		int stFrm = fms*num-overlapFrames;
		if	(!_dscFt-> preProcessor(fName, stFrm, stFrm+fms+overlapFrames+2))  //"+2" for border
		{
			std::cerr<<"hog3Ddetector::getFeatures(...): Wrong input video!"<<std::endl;
			return false;
		}
		Point3i vSz = _dscFt->videoSz();
		vSz.z -= 2; //removing border area
		_s3Droi->get3Droi(vSz);
		return true;
	}


void stDetector::getRandomFeaturesByRatio(Mat &feature, RNG &seed, float ratio, bool lastRedo)
	{
		int sz = _s3Droi->size();
		int nSmpls = sz*ratio;
		//if (nSmpls > 5500)
		//	nSmpls = 5500;
		if (nSmpls < 2000 && (!lastRedo))
			nSmpls = 2000;
		getRandomFeatures(feature, nSmpls, seed);
	}

void stDetector::getRandomFeatures(Mat &feature,  unsigned int nSmpls, RNG &seed)
	{
		if(nSmpls > 0 && nSmpls < (unsigned int)_s3Droi->size())  
			randomSampling(nSmpls, seed);
		else
			denseSampling();

		feature = _ft;
	}

void stDetector::getDenseFeatures(Mat &feature)
	{
		denseSampling();
		feature = _ft;
	}

void stDetector::randomSampling(unsigned int nSmpls, RNG &rng)
	{
		int num = _s3Droi->size();
		std::cout<<"Total number of 3d patches: "<<num<<" Sampled 3d patches: "<<nSmpls<<" with  "<<_width<<endl;

		_height = nSmpls ;
		_ft = Mat(_height, _width, CV_32FC1, Scalar(0.));

		//!!!be careful with Mat::zeros(...), if the size and type is same as before, it won't release the mat, it just set the value to zero.
		//_ft.release();
		//_ft = Mat::zeros(_height, _width, CV_32FC1);
		
		int n1;
		int *arr = new int[_height];

		int *pi, *p0 = arr;
		arr[0] = rng.uniform(0, num);

		for(int i = 1; i < _height; ++i)
		{
			pi = arr+i;
			do
			{
				n1 = rng.uniform(0, num);
			} while(pi != std::find(p0, pi, n1));  //generate a different number from [0, num) 
			arr[i] = n1;
		}

		roi3d roi;
		float *m1;

#pragma omp parallel for private( roi, m1)		
		for (int i = 0; i < _height;  i++)
		{
				roi=_s3Droi->operator[](arr[i]);
				m1 = _ft.ptr<float>(i);
				_dscFt->computeFeature(roi.tlp, roi.whl, m1);
		}
		delete []arr;
	}

void stDetector::denseSampling()
	{
		_height = _s3Droi->size();
		cout<<"Dense samping with 3d patches: "<<_height<<endl;
		//_ft = Mat(_height, _width, CV_32FC1);
		_ft = Mat(_height, _width, CV_32FC1, Scalar_<float>(0.f));
	
#pragma omp parallel for //num_threads(4)
		for (int i = 0; i < _height; i++)
		{
			roi3d roi((*_s3Droi)[i]);
			Mat m0 = _ft.row(i);
			_dscFt->computeFeature(roi.tlp, roi.whl, m0);
		}
	}
