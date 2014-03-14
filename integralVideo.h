/********************************************************************************

Filename     : integralVideo.h

Description  : This class is used to compute integral video of spatio-temporal derivative in 3D(x,y and t) 
			   from an input video file (current implementation doesn't include video from camera). 
			   If the number of frames is large than "MAX_VIDEO_BUFFER", it just computes the first 
			   "MAX_VIDEO_BUFFER" of frames. 

			   the computed integral video is stored in "vector<Mat> _ivRt[2], _ivPs[2]".
			   _ivRt[0]/_ivPs[0] save the integral video of derivative u, its size is (image.width+1, image.height+1, frames)
			   _ivRt[1]/_ivPs[1] save the integral video of derivative v

Typical Use  :  integralVideo iv("video.avi", ...);
                 compute integral video from input video file "video.avi" 
 

				
Author       : FengShi@Discovery lab, Apr, 2010
Version No   : 1.00 


*********************************************************************************/
#ifndef _INTEGRAL_VIDEONEW_H_
#define _INTEGRAL_VIDEONEW_H_

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <vector>
#include <omp.h>
//#include <iostream>

//#define MAX_VIDEO_BUFFER   178  //Maximum image buffer for input video, for image size 320x240
//#define MAX_VIDEO_BUFFER   500  //Maximum image buffer for input video, 500 frames
//#define MAX_VIDEO_BUFFER   600  //Maximum image buffer for input video, 600 frames
#define MAX_VIDEO_BUFFER   900  //Maximum image buffer for input video, 900 frames
#ifndef M_PI
	#define M_PI	3.1415926535897932385
#endif

//#define INTEGRAL_VIDEO_DEBUG

#ifdef INTEGRAL_VIDEO_DEBUG
	#include <fstream>
#endif

using namespace cv;

class IntegralVideo
{
	int				_ivWrt;  //width for root/global integral video
	int				_ivHrt;  //height for root/global integral video
	int				_ivBrt;  //buffer (time) for root/global integral video
	bool			_hasIv;

	vector<Mat>     _ivRt[2];  //integral  video for 3D derivatives(dx, dy, dt)of root/global
	//Point3d		   _meanGradRt;  //mean gradient for root/global

	int				_ivWps;  //width for parts integral video
	int				_ivHps;  //height for parts integral video
	int				_ivBps;  //buffer (time) for parts integral video
	vector<Mat>     _ivPs[2];  //integral  video for 3D derivatives(dx, dy, dt)of parts

	//Mat				_psBf;    //buffer  for parts image
	//Mat				_rtBf;    //buffer  for root image

	Point3f		    _rt2ps;   //ratio for root video size to part video size. default is (0.5, 0.5, 0.5) {_ivWrt/_ivWps = _ivHrt/_ivWps = _ivBrt/_ivBps = 0.5) 

	bool			_fullOri;		//if full orientation, 360. if false, 180 
	int				_nbins;			//number of bins for histograms 
	int				_fullAng;
	float			_anglePerBin;   // = _fullOri ? 360/_nbins : 180/_nbins
	
	void integralHist(const Mat& src, Mat& hist) const;
	IntegralVideo(const IntegralVideo& q){}		//fake copy 	
	IntegralVideo& operator=(const IntegralVideo& q) {return *this;}    //fake assignment

#ifdef INTEGRAL_VIDEO_DEBUG
	std::fstream filed;
#endif 

public:

	//constructor
	IntegralVideo(int bins = 8, bool fullOri = true) 
	{  
		#ifdef INTEGRAL_VIDEO_DEBUG
			if (!filed.is_open())
				filed.open("IntegralVideoError.txt",std::ios::out);
		#endif
		_hasIv = 0;
		_fullOri = fullOri;
		_nbins = bins;
		_fullAng = _fullOri ? 360:180;
		_anglePerBin = (float)_fullAng/_nbins;
	
	}	

	//constructor with an input video file
	IntegralVideo(const string& fName,  int bins = 8, bool fullOri = true, 
				  Point3f rt2ps = Point3f(0.5, 0.5, 0.5), float reSzRatio = 1., int stFrame = 0, int endFrame = 0);	

	void getDesc(const vector<Mat>& iv, const Point3i& tlp, const Point3i& whl, vector<float>& dst, bool normByArea=0) const;

	inline void getRtDesc_u(const Point3i& tlp, const Point3i& whl, vector<float>& dst, bool normByArea = 0) const
	{
		getDesc(_ivRt[0], tlp, whl, dst, normByArea);
	}
	inline void getRtDesc_v(const Point3i& tlp, const Point3i& whl, vector<float>& dst, bool normByArea = 0) const
	{
		getDesc(_ivRt[1], tlp, whl, dst, normByArea);
	}
	inline void getPsDesc_u(const Point3i& tlp, const Point3i& whl, vector<float>& dst, bool normByArea = 0) const
	{
		getDesc(_ivPs[0], tlp, whl, dst, normByArea);
	}
	inline void getPsDesc_v(const Point3i& tlp, const Point3i& whl, vector<float>& dst, bool normByArea = 0) const
	{
		getDesc(_ivPs[1], tlp, whl, dst, normByArea);
	}


	//part integral video size
	inline int rtBfSz() const { return _ivBrt; /*return _ivRt[0].size();*/}
	inline int rtIvHeight() const {return _ivHrt;}
	inline int rtIvWidth() const {return _ivWrt;}
	
	//root integral video size
	inline int psBfSz() const { return _ivBps;}
	inline int psIvHeight() const {return _ivHps;}
	inline int psIvWidth() const {return _ivWps;}

	inline bool hasIv()const {return _hasIv;}
	inline int dimSz() const {return _nbins;}

	//compute integral video
	bool computeIntegVideo(const string& fName, Point3f rt2ps, 
							 float reSzRatio, int stFrame = 0, int endFrame = 0);

	//root video size. this size is need for deciding randoming sampling
	inline Point3i toVideoSz() const
	{
		return Point3i(_ivWrt-1, _ivHrt-1, _ivBrt-1);
	}

	virtual ~IntegralVideo()
	{
		#ifdef INTEGRAL_VIDEO_DEBUG
			filed.close();
		#endif
		_ivRt[0].clear();
		_ivPs[0].clear();
		_ivRt[1].clear();
		_ivPs[1].clear();
	}
};

#ifdef M_PI
	#undef M_PI
#endif

#endif
