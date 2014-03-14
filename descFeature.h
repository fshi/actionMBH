/********************************************************************************

Filename     : descFeature.h

Description  : Given a 3D spatio-temporal patch, this class compute its feature vector (float*). 
			   The class contains an integral video class pointer (_iv	).
			   Given an input video, in order to compute the feature, we need to compute the integral video
			   by calling member function preProcessor("filename.txt",...);.
			   Then, use computeFeature(...) to get the feature vector.
			   

Typical Use  :  

				
Author       : FengShi@Discovery lab, May, 2010
Version No   : 1.00 


*********************************************************************************/

#ifndef _3D_DESC_FEATUREOF_H
#define _3D_DESC_FEATUREOF_H

#include "cv.h"
#include "cxcore.h"
#include "integralVideo.h"

//#include "waitKeySeconds.h"
//using namespace cv;
class descFeature
{
	Point3i				_numCell;  //number of cells per descriptor, default is 2x2x3
	Point3i				_numBlock;  //number of subblocks per cell. default is 3x3x3
	Point3i				_pSz;  // minimal patch size(_numCell.x * _numBlock.x, _numCell.y * _numBlock.y, _numCell.z * _numBlock.z)
	uchar				_nbins;    //number of bins
	bool				_fullOri;

	bool				_rdTp;		//round types in the case of _numCell.x(y,z)*numBlock.x(y,z) != _whl.x(y,z). default is 0\
										//if 0, _numCell.x(y,z) and numBlock.x(y,z) will be round to their nearest int, with possible overlap pixels\
										//if 1, _whl will be adjusted(floored) to its nearest int depending on _numCell.x(y,z)*numBlock.x(y,z)\
										//if 2, _numCell.x(y,z) and numBlock.x(y,z) will be adjusted to fit _whl size (not implemented)
		
	//Point3i				_tlp;  //top left 3d point(x,y,t)
	//Point3i				_whl;  //roi size, width, height and long
	bool				_normBlk;  //if 1, means the subblock are normalized with number of pixels inside. default value is 1.
	//bool				_hasFeature;
	//oriQuantizer		*_qtzer;  //for 3DHOG features
	IntegralVideo		*_iv;

	//parameters for parts'
	Point3f				_rt2ps;	 //ratio for root video size to part video size. default is (0.5, 0.5, 0.5) {_ivWrt/_ivWps = _ivHrt/_ivWps = _ivBrt/_ivBps = 0.5)
	Point3i				_numParts;  //number of parts per root window, default is 2x2x2
	Point3f				_olRatio; //overlap ratio per 3d parts, default is 50% overlap. if zero, there is no overlap for different parts. if 1, only one part for one root window

	float				 _cutVal;   //cut vaule for normalization, default 0.25 (if larger than 0.25, set it to 0.25, and normalize again)
	float				 _reSzVideo;  //resize the input video to compute integral video. default is 1.0, no video resize. (if video size too big, it is suggested to down size the video for higher speed

	int					_feaSz;  //descriptor size, 
	int					_dim;    //dimension quantizer
	//float				*_tmpArr; //temp arr with size of _dim

	int					_rtSz;  //root feature size
	//float				*_descFeature;   //for old implementation, no suitable for openMP

	descFeature(const descFeature& q){} //fake copy 
	descFeature &operator= (const descFeature& q) {return *this;} //fake assignment

	void l2Norm(Mat arr, Mat& dst, float cutVal, bool L2Hys = 0) const;  //if  L2Hys is ture, then renormalize with cutVal
	void l2Norm(float *arr, int sz, float cutVal, bool L2Hys = 0) const;

public:

	descFeature(
			const Point3i&  numBloc_ = Point3i( 3, 3, 3),
			const Point3i&  numCell_ = Point3i( 2, 2, 3),
			const Point3f&  rt2ps_ = Point3f( 0.5, 0.5, 0.5),
			const Point3f&  olRto_ = Point3f( 0.5, 0.5, 0.5),
			const Point3i&  numParts_ = Point3i( 2, 2, 2),
			bool roundTp_	= false,
			bool normBlk_	= false,
			float cutZ_ = 0.25,
			float reSzV_ = 1.0,
			uchar bins_ = 8,
			bool ori = true);

	//compute integral video for input file
	bool descFeature::preProcessor(const string& fName, int stFrame=0, int endFrame=0);

	//compute feature of an ST patch with top left 3d point of tlp0 and roi size  of whl0
	void computeFeature(const Point3i& tlp0, const Point3i& whl0, Mat& outArr) const;
	void computeFeature(const Point3i& tlp0, const Point3i& whl0, float *const outArr) const;

	void computeFt(const Point3i& tlp, const Point3i& whl, float *ft, int sz, bool root = 1)const;

	void clear()  //for memory efficient
	{
		if (_iv)
			delete _iv;
		_iv = NULL;
	}
	
	inline int toFeatureSz() const
	{
		return _feaSz;
	}
	
	inline bool videoSz(Point3i& vSz) const
	{
		if (_iv)
		{
			vSz = _iv->toVideoSz(); 
			return true;
		}
		return false;
	}
	inline Point3i videoSz() const
	{
		if (_iv)
		{
			return _iv->toVideoSz();
		}
		return Point3i(0, 0, 0);
	}

	inline int rootFtSz() const
	{
		return _rtSz;
	}

	inline int partsFtSz() const
	{
		return _feaSz/2 - _rtSz;
	}

	virtual ~descFeature();

};
#endif//_3D_DESC_FEATURE_H

  