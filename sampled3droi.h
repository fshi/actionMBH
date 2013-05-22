/********************************************************************************

Filename     : sampled3Droi.h

Description  : This class is used to define 3D spatio-temporal patches sampled from video.
			   Given a video size, it output the dense sampling grid.

Typical Use  :  
				
				i
				
				i
				
Author       : FengShi@Discover lab, Apr,02 2010
Version No   : 1.00 


*********************************************************************************/


#ifndef _SAMPLED_3D_ROI_H_
#define _SAMPLED_3D_ROI_H_

#include "cxcore.h"
#include <omp.h>
#include <vector>
#include "waitKeySeconds.h"
#include <iostream>

using namespace cv;

struct roi3d {

	Point3i tlp;  //top left 3d point(x,y,t)
	Point3i whl;  //roi size, width, height and long

	roi3d(const Point3i &p1, const Point3i &p2)
	{
		tlp = p1;
		whl = p2;
	}

	
	inline long int pixelSum() const
	{
		return whl.x*whl.y*whl.z;
	}

	roi3d& operator=(const roi3d &r0)
	{
		// Check for self-assignment!
		if (this != &r0)      // not Same object?
		{	        
			tlp = r0.tlp;
			whl = r0.whl;
		}
		return *this;
	}

	roi3d():tlp(Point3i(0,0,0)), whl(Point3i(0,0,0))
	{}
};

class sampled3Droi {

	Point3i _vSize;  //video size
	Point3f _strideRatio; //stride overlap ratio per 3d window slide, default is 50% overlap.  0 means no overlap (stride = patchSz)

	float _sigma, _tao; //spatial and temporal scale

	Point3i _stSz;  //start (smallest) 3D patch size. default is 18x18x10
	//Point3i _endSz;   //end (largest) 3D patch size. default is 192x192x14 
	Point3i _scales;  //number of total scales. default is (8,8,2), 8 spatial and 2 temporal scales
	Point2f _scaleStep;  //scale factor for consecutive scales, default 2^(1/2)

    //float _score, _scale;
    vector<roi3d> _resultRoi;
	//bool _isScaled;  //if _isScaled == 1, the sampled regions are patches from scaled images, otherwise, the sampled regions are only from original image
	//bool _isSampled; //if _isSampled == 1,  _resultRoi.size() > 0 


public:

	sampled3Droi(const Point3f& sRa	= Point3f(0.5, 0.5, 0.5),
				const Point3i& stSz	= Point3i(18, 18, 10),
				const Point3i& scls	= Point3i(8, 8, 2),
				const float	  sigma	= 2,
				const float	  tao	= 2) :
				_stSz(stSz),
				_scales(scls),
				_strideRatio(sRa),
				_sigma(sigma),
				_tao(tao)
	{
		_scaleStep.x = sqrt(_sigma);
		_scaleStep.y = sqrt(_tao);
		_resultRoi.clear();
	}


	sampled3Droi(const Point3i& vSz,
				const Point3f& sRa	= Point3f(0.5, 0.5, 0.5),
				const Point3i& stSz	= Point3i(18, 18, 10),
				const Point3i& scls	= Point3i(8, 8, 2),
				const float	  sigma	= 2,
				const float	  tao	= 2) :
				_vSize(vSz),
				_stSz(stSz),
				_scales(scls),
				_strideRatio(sRa),
				_sigma(sigma),
				_tao(tao)
	{
		_scaleStep.x = sqrt(_sigma);
		_scaleStep.y = sqrt(_tao);
		_resultRoi.clear();

		get3Droi(_vSize);
	}

	void resetPara( const Point3i& vSz,
					const Point3f& sRa,
					const Point3i& stSz,
					const Point3i& scls,
					const float	  sigma,
					const float	  tao) 
	{
		_stSz = stSz;
		_scales = scls;
		_strideRatio = sRa;
		_sigma = sigma;
		_tao = tao;

		_scaleStep.x = sqrt(_sigma);
		_scaleStep.y = sqrt(_tao);
		_resultRoi.clear();

		get3Droi(vSz);
	}

	inline void resetVideoSz( const Point3i& vSz) 
	{
		get3Droi(vSz);
	}

	inline void setParaStrdRt(const Point3f& strSz)
	{
		if(_strideRatio != strSz)
		{
			_resultRoi.clear();
			_strideRatio = strSz;
		}
	}
	inline void setParaScale(const Point3i& scale)
	{
		if(_scales != scale)
		{
			_resultRoi.clear();
			_scales = scale;
		}
	}
	inline void setParaStSz(const Point3i& stSz)
	{
		if(_stSz != stSz)
		{
			_resultRoi.clear();
			_stSz = stSz;
		}
	}

	inline bool isSampled() const
	{
		return _resultRoi.size() != 0;
		//return !_resultRoi.empty();
	}

	void operator()(const Point3i& vSz)
	{
		get3Droi(vSz);
	}

	//without range checkup, if need to check range, use at(i)
    inline roi3d operator[](int i) const {
        return _resultRoi[i];
    }
	inline roi3d operator()(int i) const {
        return _resultRoi[i];
    }

    inline roi3d at(int i) const {
		if (i < 0 || i >= (int)_resultRoi.size()){
			std::cerr<<"sampled3Droi::at() Error! Index out of range.\n";
			discoverUO::wait();
			exit(-10);
		}
        return _resultRoi[i];
    }

	//inline roi3d start3Droi() const { return _resultRoi[0]; }
    //inline roi3d end3Droi()   const { return _resultRoi[_resultRoi.size()-1]; }

	inline int size() const {return _resultRoi.size();}

	inline Point3i startSz() const {return _stSz;}
	inline Point3f strideRt() const {return _strideRatio;}
	inline Point3i scaleSz() const {return _scales;}
	inline Point2f scaleStep() const {return _scaleStep;}


	void get3Droi(const Point3i& videoSz)
	{
		if ((videoSz == _vSize)&&(!_resultRoi.empty())) //if same size video is already sampled, no work to do
			return;

		_resultRoi.clear();
		_vSize = videoSz;

		vector<Point3i> patches;
		getPatchSz(patches);

		for (int i = 0; i < (int)patches.size(); i++)
			doSampling(patches[i]);
		//cout<<_vSize<<endl;
	}

	void getPatchSz(vector<Point3i>& patches)
	{
		int pSzx = (int)(pow(_scaleStep.x, _scales.x-1)*_stSz.x); 
		int pSzy = (int)(pow(_scaleStep.x, _scales.y-1)*_stSz.y); 
		int pSzt = (int)(pow(_scaleStep.y, _scales.z-1)*_stSz.z); 

		Point3i tmpScls = _scales;
		if (_vSize.x < pSzx)
			tmpScls.x = (int)ceil(log((float)_vSize.x / (float)_stSz.x)/log(_scaleStep.x))+1;
		if (_vSize.y < pSzy)
			tmpScls.y = (int)ceil(log((float)_vSize.y / (float)_stSz.y)/log(_scaleStep.x))+1;
		if (_vSize.z < pSzt)
			tmpScls.z = (int)ceil(log((float)_vSize.z / (float)_stSz.z)/log(_scaleStep.y))+1;

		if (tmpScls.x > tmpScls.y)
			tmpScls.x = tmpScls.y;
		else 
			tmpScls.y = tmpScls.x;

		patches.clear();			
		Point3i tmp = _stSz;
		if (_sigma == 2 && _tao == 2)
		{
			//if _sigma == 2, x1=sqrt(2)*x0,
			//x2(x4,x6...)=2(4,8...)*x0
			//x3(x5,x7...)=2(4,8...)*x1
			int t0 = tmp.z, xy0 = tmp.x, ts, xys, x0, x1;
			int t1 = int(t0*_scaleStep.y+0.5), xy1 = int(xy0*_scaleStep.x+0.5), tt = t0;
			t1 = (t1 % 2) ? (t1 - 1) : t1;
			xy1 = (xy1 % 2) ? (xy1 - 1) : xy1;
			int T0 = tmpScls.z;
			for (int t = 0; t < T0; t++)
			{
				ts = t > 1 ? 2 : 1;
				if (t % 2)
				{
					t1 *= ts;
					tt = t1;
				}
				else 
				{
					t0 *= ts;
					tt = t0;
				}
				//tt = (t % 2) ? (t1 *= ts) : (t0 *= ts);
				int xy;
				for (x0 = xy0, x1 = xy1, xy = 0; xy < tmpScls.x; xy++)
				{
					xys = xy > 1 ? 2 : 1;
					
					if (xy % 2)
					{
						x1 *= xys;
						tmp = Point3i(x1, x1, tt);
					}
					else
					{
						x0 *= xys;
						tmp = Point3i(x0, x0, tt);
					}
					patches.push_back(tmp);
				}
				x0 = _vSize.x < _vSize.y ? (_vSize.x - 1) : (_vSize.y - 1);
				tmp = Point3i(x0, x0, tt); 
				patches.push_back(tmp);  //add one slide window with the same size as the video size 
			}
		}
		else
		{
			double factor = 1.;
			//tmp.z = _stSz.z;
			for (int t = 0; t < tmpScls.z; t++)
			{
				int xy = 0;
				
				for (tmp.x = _stSz.x, tmp.y = _stSz.y; xy < tmpScls.x; xy++)
				{
					patches.push_back(tmp);
					tmp.y = (int)(tmp.y * _scaleStep.x); 
					tmp.x = (tmp.y = (tmp.y % 2) ? (tmp.y - 1) : tmp.y);  
					//tmp.x = (tmp.y -= (tmp.y % 2)); 
				}
				tmp.x = (tmp.y = _vSize.x < _vSize.y ? (_vSize.x - 1) : (_vSize.y - 1));
				patches.push_back(tmp);  //add one slide window with the same size as the video size
				factor *= _scaleStep.y;
				tmp.z = (int)(_stSz.z*factor + 0.5);
				tmp.z = (tmp.z % 2) ? (tmp.z - 1) : tmp.z;
				//tmp.z -= (tmp.z % 2);		//make sure the patch's size is even
			}
		}
	}

	//for every patch's size, do smapling for a video (slide patch inside the video)
	bool doSampling(const Point3i& patchSz)
	{
		Point3i extent = _vSize - patchSz;
		if ((extent.x < 0) || (extent.y < 0) || (extent.z < 0))
			return false;

		Point3i strideSz = Point3i((int)((1-_strideRatio.x)*patchSz.x), (int)((1-_strideRatio.y)*patchSz.y), (int)((1-_strideRatio.z)*patchSz.z));

		if(strideSz.x < 1)
			strideSz.x = 1;
		if(strideSz.y < 1)
			strideSz.y = 1;
		if(strideSz.z < 1)
			strideSz.z = 1;

		bool leftoverX = 0, leftoverY = 0, leftoverT = 0;

		//add another slide window if there are some area left after all the slides
		if ((extent.x % strideSz.x) > strideSz.x/2)
		{
			//extent.width++;
			leftoverX = 1;
		}
		if ((extent.y % strideSz.y) > strideSz.y/2)
		{
			//extent.width++;
			leftoverY = 1;
		}	
		if ((extent.z % strideSz.z) > strideSz.z/2)
		{
			//extent.width++;
			leftoverT = 1;
		}
				
		//extent.x = extent.x/strideSz.x + 1;
		//extent.y = extent.y/strideSz.y + 1;
		//extent.z = extent.z/strideSz.z + 1;

		for (int t0 = 0, t1 = 0; t0 <= extent.z; t0 += strideSz.z) {
			for (int y0 = 0, y1 = 0; y0 <= extent.y; y0 += strideSz.y) {
				for (int x0 = 0, x1 = 0; x0 <= extent.x; x0 += strideSz.x) {
					_resultRoi.push_back(roi3d(Point3i(x0, y0, t0), patchSz));
					if (leftoverX && (x0 + strideSz.x > extent.x))//add extra slide at the end of the video size
					{
						x0 = extent.x - strideSz.x + x1;
						x1 = 1;//only do it(add extra slide) once
					}
				}
				if (leftoverY && (y0 + strideSz.y > extent.y))
				{
					y0 = extent.y - strideSz.y + y1;
					y1 = 1;
				}
			}
			if (leftoverT && (t0 + strideSz.z > extent.z))
			{
				t0 = extent.z - strideSz.z + t1;
				t1 = 1;
			}
		}
		return true;
	}

	inline void randomShuffle()
	{
		srand ( unsigned ( time(NULL) ) );
		random_shuffle(_resultRoi.begin(), _resultRoi.end());
	}

	// Fisher-Yates shuffling
	void fyShuffle(int stopNum = 1, int shuffleTimes = 3)
	{
		srand( (unsigned int)time(NULL) );
		int i, id0, id = _resultRoi.size() - 1;
		Point3i tlp0, whl0;
		while (id >= stopNum)
		{
			//run 3 times to make sure it is shuffled
			for(i=0; i<shuffleTimes; i++)
			{
				id0 = rand() % id;

				tlp0 = _resultRoi[id0].tlp;
				whl0 = _resultRoi[id0].whl;

				_resultRoi[id0].tlp = _resultRoi[id].tlp;
				_resultRoi[id0].whl = _resultRoi[id].whl;

				_resultRoi[id].tlp = tlp0;
				_resultRoi[id].whl = whl0;
			}
			id--;
		}
	}

	void clear()
	{
		_resultRoi.clear();
	}
	virtual ~sampled3Droi()
	{
		_resultRoi.clear();
	}

protected:
	//using operator() or operator[], following iterator method is not safe. it can change class members
	inline vector<roi3d>::iterator begin() {return _resultRoi.begin();}
	inline vector<roi3d>::iterator end() {return _resultRoi.end();}

	//this method is not safe. it can change class members. use with care
	inline vector<roi3d>* to3Droi()
	{
		return &_resultRoi;
	}

};


#endif //_SAMPLED_3D_ROI_H_.