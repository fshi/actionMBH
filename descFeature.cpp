#include "descFeature.h"
#define epislon 1e-6   //epison 
#define _threshold_  0.0001
//#define HOG3D_DESC_FEATURE_DEBUG

void opcvL2Norm(float *arr, int sz, bool useThres = 0) 
{
	Mat m1(1, sz, CV_32FC1, arr);
	/*if(!checkRange(m1))  //chech if the feature value is infinite
	{
		cout<<"not good  feature descriptor";
		cin.get();cin.get();
	}*/
	Mat m2(1, sz, CV_32FC1);
	cv::max(m1, 0., m2); //take value large than zero
	if(useThres)
		m2 += _threshold_;
	normalize(m2, m1);
}
void descFeature::l2Norm(float *arr, int sz, float cutVal, bool L2Hys) const
{
	double sum = 0, sum1, pCutVal = cutVal*cutVal;
	float *arr2 = new float[sz];
	int normCount = 0;

	for (int i = 0; i < sz; i++) 
	{
		arr2[i] = arr[i]*arr[i];
		sum += arr2[i];
	}
	if( L2Hys)
	{
		bool reNorm = 1;
		while (reNorm && (normCount <= 1)) 
		{
			reNorm = 0;
			sum1 = 0;
			double sqrSum = sqrt(sum + epislon);
			for (int j = 0; j < sz;  j++)
			{
				arr[j] /= (float)(sqrSum);
				if (arr[j] > cutVal)
				{
					reNorm = 1;
					arr[j] = cutVal;
					arr2[j] = pCutVal;
				}
				else 
					arr2[j] = arr[j]*arr[j];

				sum1 += arr2[j];
			}
			sum = sum1;
			normCount++;
		}
	}
	else
	{
		double sqrSum = sqrt(sum + epislon);
		for (int j = 0; j < sz;  j++)
			arr[j] /= (float)(sqrSum);

	}
	delete []arr2;
}


void l2NormOld(float *arr,const int sz,const float cutVal) 
{
	double sum = 0, sum1, pCutVal = cutVal*cutVal;
	float *arr2 = new float[sz];
	bool reNorm = 1;
	int normCount = 0;

	for (int i = 0; i < sz; i++) 
	{
		arr2[i] = arr[i]*arr[i];
		sum += arr2[i];
	}

	while (reNorm && (normCount <= 1)) 
	{
		reNorm = 0;
		sum1 = 0;
		double sqrSum = sqrt(sum);
		if(sqrSum < epislon)
			sqrSum = epislon;
		for (int j = 0; j < sz;  j++)
		{
			arr[j] /= (float)(sqrSum);
			if (arr[j] > cutVal)
			{
				reNorm = 1;
				arr[j] = cutVal;
				arr2[j] = pCutVal;
			}
			else 
				arr2[j] = arr[j]*arr[j];

			sum1 += arr2[j];
		}
		sum = sum1;
		normCount++;
	}
	delete []arr2;
}


void descFeature::l2Norm(Mat arr, Mat& dst, float cutVal, bool L2Hys) const 
{
	dst = arr.clone();
	float *pt;
	for (int i = 0; i < dst.rows; i++)
	{
		pt = dst.ptr<float>(i);
		l2Norm(pt, dst.cols, cutVal, L2Hys);
	}
	
	//normalize(arr, dst);
	//threshold(dst, dst, cutVal, 1, THRESH_TRUNC);
	//normalize(dst, dst);
}


descFeature::descFeature(
			const Point3i&  numBloc_,
			const Point3i&  numCell_,
			const Point3f&  rt2ps_,
			const Point3f&  olRto_,
			const Point3i&  numParts_,
			bool roundTp_,
			bool normBlk_,
			float cutZ_,
			float reSzV_,
			uchar bins_,
			bool ori_):
		_numCell(numCell_),
		_numBlock(numBloc_),
		_normBlk(normBlk_),
		_rdTp(roundTp_),
		_cutVal(cutZ_),
		_reSzVideo(reSzV_),
		_nbins(bins_),
		_fullOri(ori_),
		_numParts(numParts_),
		_olRatio(olRto_),
		_rt2ps(rt2ps_)
{
	_iv = new IntegralVideo(_nbins, _fullOri);
	_pSz = Point3i(_numCell.x * _numBlock.x, _numCell.y * _numBlock.y, _numCell.z * _numBlock.z);
	_dim = _iv->dimSz();  
	_rtSz = _dim * (_numCell.x * _numCell.y * _numCell.z);
	_feaSz = 2*_rtSz * (1 + _numParts.x * _numParts.y * _numParts.z);  //*2 is for optical flow u and v
}



bool descFeature::preProcessor(const string& fName, int stFrame, int endFrame)
{
	if (_iv)
		delete _iv;
	_iv = new IntegralVideo(fName, _nbins, _fullOri,_rt2ps, _reSzVideo, stFrame, endFrame);
	return _iv->hasIv();
}

void descFeature::computeFeature(const Point3i& tlp0, const Point3i& whl0, Mat& row0) const
{
	float *ft = row0.ptr<float>(0);
	computeFeature(tlp0, whl0, ft);
}

void descFeature::computeFeature(const Point3i& tlp0, const Point3i& whl0, float *const outArr) const
{
	float *ft = outArr;

//compute features for root's window
	computeFt(tlp0, whl0, ft, _rtSz, 1);

//compute features for parts' windows corresponding to root's window (tlp0, whl0)
	//top left point of all parts' window corresponding to root's window
	Point3i tlpP0 = Point3i(cvFloor(tlp0.x/_rt2ps.x), cvFloor(tlp0.y/_rt2ps.y), cvFloor(tlp0.z/_rt2ps.z));  
	//size of all parts' window corresponding to root's window
	Point3i whlP0 = Point3i(cvFloor(whl0.x/_rt2ps.x), cvFloor(whl0.y/_rt2ps.y), cvFloor(whl0.z/_rt2ps.z));  

	//for a window with size "w" and overlap ratio "rt", the part size "wp" should be: 
	// wp = w/(p*(1-rt)+rt), while "p" is number of parts inside the window
	Point3i whlPsz;
	//compute parts' size
	whlPsz.x = (int)(whlP0.x/(_numParts.x*(1-_olRatio.x)+_olRatio.x));
	whlPsz.y = (int)(whlP0.y/(_numParts.y*(1-_olRatio.y)+_olRatio.y));
	whlPsz.z = (int)(whlP0.z/(_numParts.z*(1-_olRatio.z)+_olRatio.z));

	//overlap size
	Point3i whlOlp = Point3i((int)(whlPsz.x*_olRatio.x), (int)(whlPsz.y*_olRatio.y), (int)(whlPsz.z*_olRatio.z));

	for (int iz = 0; iz < _numParts.z; iz++)
		for (int iy = 0; iy < _numParts.y; iy++)
			for (int ix = 0; ix < _numParts.x; ix++)
			{
				Point3i tlpTmp = Point3i(tlpP0.x + ix*(whlPsz.x-whlOlp.x), 
								tlpP0.y + iy*(whlPsz.y-whlOlp.y), 
								tlpP0.z + iz*(whlPsz.z-whlOlp.z));
				ft += _rtSz;
				computeFt(tlpTmp, whlPsz, ft, _rtSz, 0);
			}

	ft = outArr;
	for( int i = 0; i < _feaSz; i += _rtSz)
		opcvL2Norm(ft+i, _rtSz);
	opcvL2Norm(ft+_rtSz, _feaSz/2-_rtSz);  //renormalize parts' channel for MBHx
	opcvL2Norm(ft+_rtSz+_feaSz/2, _feaSz/2-_rtSz); //renormalize parts' channel for MBHy
}

void descFeature::computeFt(const Point3i& tlp, const Point3i& whl, float *ft, int sz, bool root) const 
{
	Point3i tlp0, whl0, blkSz, cellSz, cSz, tlpCell, tlpBlk;
	float* desc;

	if (_rdTp)
	{
		std::cout<<"_rdTp:"<<std::endl;

		tlp0 = tlp;
		blkSz.x = cvFloor(((float)whl.x) / _pSz.x);
		blkSz.y = cvFloor(((float)whl.y) / _pSz.y);
		blkSz.z = cvFloor(((float)whl.z) / _pSz.z);
		blkSz.x = (blkSz.x < 1) ? 1 : blkSz.x;
		blkSz.y = (blkSz.y < 1) ? 1 : blkSz.y;
		blkSz.z = (blkSz.z < 1) ? 1 : blkSz.z;

		cellSz.x = _numBlock.x * blkSz.x; 
		cellSz.y = _numBlock.y * blkSz.y; 
		cellSz.z = _numBlock.z * blkSz.z; 

		//adjust the patch size
		whl0.x = _pSz.x * blkSz.x;
		if (whl0.x >= whl.x)
		{
			whl0.x = whl.x;
			cSz.x = 1;
		}
		else
		{
			tlp0.x += ((whl.x - whl0.x)/2);
			cSz.x = cellSz.x;
		}

		whl0.y = _pSz.y * blkSz.y;
		if (whl0.y >= whl.y)
		{
			whl0.y = whl.y;
			cSz.y = 1;
		}
		else
		{
			tlp0.y += ((whl.y - whl0.y)/2);
			cSz.y = cellSz.y;
		}

		whl0.z = _pSz.z * blkSz.z;
		if (whl0.z >= whl.z)
		{
			whl0.z = whl.z;
			cSz.z = 1;
		}
		else
		{
			tlp0.z += ((whl.z - whl0.z)/2);
			cSz.z = cellSz.z;
		}

		int step = 0;
		int bx, by, bz, cx, cy, cz;

		for (cz = 0, tlpCell.z = tlp0.z; cz < _numCell.z; cz++)
		{
			for (cy = 0, tlpCell.y = tlp0.y; cy < _numCell.y; cy++)
			{
				for (cx = 0, tlpCell.x = tlp0.x; cx < _numCell.x; cx++, step += _dim)
				{
					//cout<<"computing grad "<<cx<<" "<<cy<<" "<<cz<<" "<<step<<" "<<tlp0<<whl0<<cellSz<<endl;
					for (bz = 0, tlpBlk.z = tlpCell.z; bz < _numBlock.z; bz++)
					{
						for (by = 0, tlpBlk.y = tlpCell.y; by < _numBlock.y; by++)
						{
							for (bx = 0, tlpBlk.x = tlpCell.x; bx < _numBlock.x; bx++)
							{
								if (root)
								{
									desc = &((_iv->getRtDesc_u(tlpBlk, blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step)[i] += desc[i];

									desc = &((_iv->getRtDesc_v(tlpBlk, blkSz, _normBlk)))[0];
									for (int i = 0; i < _dim; i++)
										(ft + step+ _feaSz/2)[i] += desc[i];
									
								}
								else
								{
									desc = &((_iv->getPsDesc_u(tlpBlk, blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step)[i] += desc[i];
								
									desc = &((_iv->getPsDesc_v(tlpBlk, blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step+ _feaSz/2)[i] += desc[i];
					
								}
								tlpBlk.x += blkSz.x;
							}
							tlpBlk.y += blkSz.y;
						}
						tlpBlk.z += blkSz.z;
					}
					tlpCell.x += cSz.x;
				}
				tlpCell.y += cSz.y;
			}
			tlpCell.z += cSz.z;
		}

		#ifdef HOG3D_DESC_FEATURE_DEBUG	

			if (step != _numCell.x * _numCell.y * _numCell.z*_dim)
			{
				std::cout<<"wrong!\n"<<step<<" "<<_dim<<" "<<_numCell.x * _numCell.y * _numCell.z<<"\n";
				discoverUO::wait();
			}
		#endif	
	}
	else
	{
		int olpx0, olpy0, olpz0;
		olpx0 = _pSz.x > whl.x ? _pSz.x - whl.x : 0;
		olpy0 = _pSz.y > whl.y ? _pSz.y - whl.y : 0;
		olpz0 = _pSz.z > whl.z ? _pSz.z - whl.z : 0;

		Point3i tmp0, tmp1, whlB;

		cellSz.x = cvRound(((float) whl.x) / ((float) _numCell.x) + 10E-7);
		cellSz.y = cvRound(((float) whl.y) / ((float) _numCell.y) + 10E-7);
		cellSz.z = cvRound(((float) whl.z) / ((float) _numCell.z) + 10E-7);
		blkSz.x = cvRound(((float) cellSz.x) / ((float) _numBlock.x) + 10E-7);
		blkSz.y = cvRound(((float) cellSz.y) / ((float) _numBlock.y) + 10E-7);
		blkSz.z = cvRound(((float) cellSz.z) / ((float) _numBlock.z) + 10E-7);

		if (olpx0)
		{
			blkSz.x = 1;
			cellSz.x = _numBlock.x;
		}
		if (olpy0)
		{
			blkSz.y = 1;
			cellSz.y = _numBlock.y;
		}
		if (olpz0)
		{
			blkSz.z = 1;
			cellSz.z = _numBlock.z;
		}

		whl0.x = cellSz.x * _numCell.x;
		whl0.y = cellSz.y * _numCell.y;
		whl0.z = cellSz.z * _numCell.z;

		tmp0 = whl0 - whl;

		int tx = cvCeil(abs(tmp0.x/2.));
		int ty = cvCeil(abs(tmp0.x/2.));
		int tz = cvCeil(abs(tmp0.z/2.)); 
	
		tlp0.x = (tmp0.x < 0) ? (tlp.x + tx) : tlp.x;
		tlp0.y = (tmp0.y < 0) ? (tlp.y + ty) : tlp.y;
		tlp0.z = (tmp0.z < 0) ? (tlp.z + tz) : tlp.z;

		whlB.x = blkSz.x * _numBlock.x;
		whlB.y = blkSz.y * _numBlock.y;
		whlB.z = blkSz.z * _numBlock.z;

		tmp1 = whlB - cellSz;

		int tBx = cvCeil(abs(tmp1.x/2.));
		int tBy = cvCeil(abs(tmp1.y/2.));
		int tBz = cvCeil(abs(tmp1.z/2.)); 
	

		int step = 0;
		#ifdef HOG3D_DESC_FEATURE_DEBUG
			int testN = 0;
		#endif

		//there are one pixel overlap for some cells and blocks. olpx, olpy, olpz...
		for (int cz = tlp0.z, olpz = tmp0.z; cz < tlp0.z + whl.z - tz; olpz--) {
			for (int cy = tlp0.y, olpy = tmp0.y; cy < tlp0.y + whl.y - ty; olpy--) {
				for (int cx = tlp0.x, olpx = tmp0.x; cx < tlp0.x + whl.x - tx; olpx--, step += _dim) {
					#ifdef HOG3D_DESC_FEATURE_DEBUG
						testN = 0;
					#endif
					for (int bz = ((tmp1.z < 0) ? (cz + tBz) : cz), olpBz = tmp1.z; bz < cz + cellSz.z -tBz;  olpBz--) {
						for (int by = ((tmp1.y < 0) ? (cy + tBy) : cy), olpBy = tmp1.y; by < cy + cellSz.y - tBy;  olpBy--) {
							for (int bx = ((tmp1.x < 0) ? (cx + tBx) : cx), olpBx = tmp1.x; bx < cx + cellSz.x - tBx; olpBx--)
							{
								if (root)
								{
									desc = &((_iv->getRtDesc_u(Point3i(bx, by, bz), blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step)[i] += desc[i];

									desc = &((_iv->getRtDesc_v(Point3i(bx, by, bz), blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step+ _feaSz/2)[i] += desc[i];
					
								}
								else
								{
									desc = &((_iv->getPsDesc_u(Point3i(bx, by, bz), blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step)[i] += desc[i];

									desc = &((_iv->getPsDesc_v(Point3i(bx, by, bz), blkSz, _normBlk))[0]);
									for (int i = 0; i < _dim; i++)
										(ft + step+ _feaSz/2)[i] += desc[i];
						
								}

								#ifdef HOG3D_DESC_FEATURE_DEBUG
									testN++;
								#endif
								bx = (olpBx > 0) ? (bx + blkSz.x -1) : (bx + blkSz.x);
							}
							by = (olpBy > 0) ? (by + blkSz.y -1) : (by + blkSz.y);
						}
						bz = (olpBz > 0) ? (bz + blkSz.z -1) : (bz + blkSz.z);
					}
					#ifdef HOG3D_DESC_FEATURE_DEBUG
						if (testN  != (_numBlock.x*_numBlock.y*_numBlock.z))
						{
							std::cout<<"Wrong!"<<testN<<"\n";
							cout<<whl<<whl0<<tlp<<tlp0<<cellSz<<blkSz<<tmp0<<tmp1<<endl;
							discoverUO::wait();
						}

					#endif
					if (olpx > 2)
						cx = cx + cellSz.x -2;
					else if (olpx > 0)
						cx = cx + cellSz.x -1;
					else
						cx = cx + cellSz.x;
				}
				if (olpy > 2)
					cy = cy + cellSz.y -2;
				else if (olpy > 0)
					cy = cy + cellSz.y -1;
				else
					cy = cy + cellSz.y;
			}
			if (olpz > 2)
				cz = cz + cellSz.z -2;
			else if (olpz > 0)
				cz = cz + cellSz.z -1;
			else
				cz = cz + cellSz.z;
		}

		#ifdef HOG3D_DESC_FEATURE_DEBUG
			if (step != _numCell.x * _numCell.y * _numCell.z*_dim)
			{
				std::cout<<"wrong!\n"<<step<<" "<<cellSz<<" "<<blkSz<<" "<<whl0<<" "<<whl<<" " \
					<<tlp<<" "<<tlp0<<"\n";
				discoverUO::wait();
			} 
		#endif
	}
	//_hasFeature = true;	

}

descFeature::~descFeature()
{
	delete _iv;
}
#ifdef epislon
	#undef epislon
#endif 
#ifdef _threshold_
	#undef _threshold_
#endif 