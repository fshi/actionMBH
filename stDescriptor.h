/********************************************************************************

Filename     : stDescriptor.h

Description  : class for spatio-temporal descritors.
			   It contains a sampled3Droi pointer, which performs dense sampling
			   for a given video size with proper parameters.
			   It also has a descFeature pointer to compute integral video and 
			   single feature for a 3d st patch.			   

Typical Use  :  

				
Author       : FengShi@Discovery lab, May, 2010
Version No   : 1.00 


*********************************************************************************/

#ifndef _ST_DESCRIPTOR_H_
#define _ST_DESCRIPTOR_H_

#include "cxcore.h"
#include "cv.h"

#include "integralVideo.h"
#include "sampled3Droi.h"

//#include "oriQuantizer.h"  //for 3DHOG descriptor
#include "descFeature.h"
#include "waitKeySeconds.h"
#include "matOperations.h"

using namespace cv;

struct MBHparam{

	//descFeature parameters:
	Point3i				numCell;  //number of cells per descriptor, default is 4x4x3
	Point3i				numBlock;  //number of subblocks per cell. default is 3x3x3
	uchar				nBins;    //number of bins. default is 8 
	bool				fullOri;  //full orientation. default is 1 (for 360 degree)
	float				cutVal;   //cut vaule for normalization, default 0.25 (if larger than 0.25, set it to 0.25, and normalize again)
	bool				rdTp;		//round types in the case of _numCell.x(y,z)*numBlock.x(y,z) != _whl.x(y,z). default is 0\
										//if 0, _numCell.x(y,z) and numBlock.x(y,z) will be round to their nearest int, with possible overlap pixels\
										//if 1, _whl will be adjusted(floored) to its nearest int depending on _numCell.x(y,z)*numBlock.x(y,z)\
										//if 2, _numCell.x(y,z) and numBlock.x(y,z) will be adjusted to fit _whl size (not implemented)
	bool				normBlk;  //if 1, means the subblock are normalized with number of pixels inside. default value is 1.
	//bool				_hasFeature;
	float				 reSzVideo;  //resize the input video to compute integral video. default is 1.0, no video resize. (if video size too big, it is suggested to down size the video for higher speed

	//parameters for parts
	Point3f				rt2ps;	 //ratio for root video size to part video size. default is (0.5, 0.5, 0.5) {_ivWrt/_ivWps = _ivHrt/_ivWps = _ivBrt/_ivBps = 0.5)
	Point3i				numParts;  //number of parts per root window, default is 2x2x2
	Point3f				olRatio; //overlap ratio per 3d parts, default is 50% overlap. if zero, there is no overlap for different parts. if 1, only one part for one root wind
	
	//sampled3Droi parameters:
	Point3f				strideRatio; //stride overlap ratio per 3d window slide, default is 50% overlap
	Point3i				stSz;  //start (smallest) 3D patch size. default is 18x18x10
	//Point3i _endSz;   //end (largest) 3D patch size. default is 192x192x14 
	Point3i				scales;  //number of total scales. default is (8,8,2), 8 spatial and 2 temporal scales
	//Point2f				scaleStep;  //scale factor for consecutive scales, default 2^(1/2)	
	float				sigma2, tao2;			//spatial and temporal scale, sigma=sqrt(sigma2)
	

	MBHparam(
			const Point3i&  numBloc_	= Point3i( 3, 3, 3),
			const Point3i&  numCell_	= Point3i( 2, 2, 3),
			bool roundTp_				= 0,
			bool normBlk_				= 1,
			float cutZ_					= 0.25,
			float reSzV_				= 1.0,
			uchar bins_					= 1,
			bool fullOri				= true,
			const Point3f& sRa			= Point3f(0.5, 0.5, 0.5),
			const Point3i& stSz_		= Point3i(12, 12, 10),
			const Point3i& scls			= Point3i(8, 8, 2),
			const Point3f& r2p			= Point3f(0.5, 0.5, 1),
			const Point3i& noPts		= Point3i(2, 2, 2),
			const Point3f& ovlp			= Point3f(0.5, 0.5, 0.5),
			float	  sigma				= 2,
			float	  tao				= 2);

	bool readParam(const string& fileName = "MBH_parameters.txt", bool showPara = 0);

	bool writeParam(const string& fileName = "MBH_parameters.txt", bool app = 0);

}; //end of struct hog3Dpara


class stDetector {

	//Point3i			  _numCell;  //number of cells per descriptor, default is 4x4x3  for 3DHOG
	//Point3i			  _subBlocks;  //number of subblocks per cell. default is 3x3x3   for 3DHOG
		
	sampled3Droi	  *_s3Droi;  //for deciding dense sampling grid over the video
	descFeature		  *_dscFt;   //for compute the integral video and single feature from a 3d patch
	Mat				  _ft;    //save the computed feature  
	//VideoCapture	  _cap;   //to read the input video 

	int				  _width; 
	int				  _height;
	int				  _vLen;
	int				  _redoNum; //number of times for spliting a video file. if 0, means no need to split the file
								//if the video has too many frames, it needs to split the video to do multiple processing to avoid memory overflow

	stDetector (const stDetector &q) {}  //fake copy 
	stDetector  &operator= (const stDetector &q) {return *this;}  //fake assignment
	void randomSampling(unsigned int nSmpls, RNG &seed);  //perform random sampling with number of sameles = nSmpls
	void denseSampling();


public:
	stDetector(const MBHparam* param	= NULL);  //initialize the class with parameters
	
	//compute integral video
	bool stDetector::preProcessing(const string& fName, int fms);
	bool stDetector::re_Processing(const string& fName, int fms, int num);

	void getRandomFeatures(Mat &feature,  unsigned int nSmpls, RNG &seed );
	void getRandomFeaturesByRatio(Mat &feature,  RNG &seed, float ratio = 0.1, bool lastRedo = 0);
	void getDenseFeatures(Mat &feature);
	void clear();  //for memory efficient

	//bool getFeatures(Mat &feature, const string& fName, unsigned int nSmpls = 0, bool flipIm=0);

	inline void setSamplingPara( const Point3i& vSz,
								 const Point3f& sRa,
								 const Point3i& stSz,
								 const Point3i& scls,
								 const float	  sigma = 2,
								 const float	  tao = 2) 
	{
		_s3Droi->resetPara(vSz, sRa, stSz, scls, sigma, tao);
	}

	inline int reProcessNum() const
	{
		return _redoNum;
	}

	inline void setSamplingStrideSz( const Point3f& strSz)
	{
		_s3Droi->setParaStrdRt(strSz);
	}
	inline void setSamplingScale( const Point3i& strSz)
	{
		_s3Droi->setParaScale(strSz);
	}
	inline void setSamplingStSz( const Point3i& strSz)
	{
		_s3Droi->setParaStSz(strSz);
	}

	 inline Size featureSize() const 
	 {
		 return Size(_width, _height);
	 }
	 inline int getSamplingSz()const
	 {
		 return _s3Droi->size();
	 }

	 inline int toRootFtSz() const
	{
		return _dscFt->rootFtSz();
	}

	inline int toPartsFtSz() const
	{
		return _dscFt->partsFtSz();
	}

	inline int toFeatureSz()const
	{
		return _dscFt->toFeatureSz();
	}

	inline int toVideoLen() const
	{
		return _vLen;
	}

	 virtual  ~stDetector()
	 {
		 delete _dscFt;
		 delete _s3Droi;
	 }

};  //end of class rwinDetector

#endif //_ST_DESCRIPTOR_H_