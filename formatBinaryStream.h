// {{{ copyright notice
// }}}
// {{{ file documentation
/**
 * @file
 * @brief Implements file IO in binary
 *
 * @todo Do rigrous checks that has been mode in binary mode
 *
 * originally coded by /Dalal @ Lear/INRIA, HOG, file format "formatstream.h"
 *
 * @modified by Feng Shi@discover lab, univ of ottawa @Apr 17, 2009
 */
// }}}

#ifndef _FORMAT_BINARY_STREAM_H
#define  _FORMAT_BINARY_STREAM_H

#include <fstream>
#include <string>
#include "biostream.h"
#include "biistream.h"
#include <iostream>
#include "cxcore.h"

#include "waitKeySeconds.h"

using namespace cv;

 template<class T>
    struct IOTypeIdentifier {
        typedef T   Type;
        enum { ID = 0};
    };
#define DEF_TYPE_ID(T, Value)       \
template<>                          \
struct IOTypeIdentifier<T> {        \
    typedef T    Type;              \
    enum { ID = Value};             \
};

DEF_TYPE_ID(bool,           1);
DEF_TYPE_ID(char,           2);
DEF_TYPE_ID(int,            3);
DEF_TYPE_ID(float,          4);
DEF_TYPE_ID(double,         5);
DEF_TYPE_ID(unsigned char,  6);
DEF_TYPE_ID(short,          7);
DEF_TYPE_ID(unsigned short, 8);
DEF_TYPE_ID(unsigned int,   9);
DEF_TYPE_ID(long,          10);
DEF_TYPE_ID(unsigned long, 11);

#undef DEF_TYPE_ID


class BinInStream
{
protected:
	BiIStream _stream;
public:
	BinInStream(const std::string& filename):
		_stream(filename.c_str())
	{
		if(!_stream)
		{
			std::cerr<<"BinaryInStream error! Unable to open the input file " + filename;
			discoverUO::wait();
			exit(-10);
		}
	}

	//virtual void read(cv::Mat& mat, long selectIn) = 0;
	//virtual void read(vector<cv::Mat>& mat, int stframes=-1, int endframes=-1) = 0;
	virtual ~BinInStream(){_stream.close();}

};


class BinClusterInStream : public BinInStream
{
	int  _dimFeature;
	long _numFeature;
	//char *rawData;
	int dataTp, cvTp;
	int rawDataStep;
	BiIStream::pos_type _initpos;
	typedef BinInStream		parent;
public:

	BinClusterInStream(const std::string& filename_): parent(filename_)
	{
		int version =0;
		dataTp=1;
		if (!(parent::_stream>>version))
		{
			std::cerr<<"BinClusterInStream error! Unable to read version number ";
			discoverUO::wait();
			exit(-10);
		}
		if(!(parent::_stream>>dataTp))
		{
			std::cerr<<"BinClusterInStream error! Unable to read data type";
			discoverUO::wait();
			exit(-10);
		}
		if (!(parent::_stream>>_numFeature))
		{
			std::cerr<<"BinClusterInStream error! Unable to read number of features";
			discoverUO::wait();
			exit(-10);
		}
		if (!(parent::_stream>>_dimFeature))
		{
			std::cerr<<"BinClusterInStream error! Unable to read dimension of features";
			discoverUO::wait();
			exit(-10);
		}

		_initpos = parent::_stream.tellg();

		switch(dataTp) {
			case 1:
				std::cout<<"Will read 'bool' type target data\n";
				cvTp = DataType<bool>::type;
				rawDataStep = _dimFeature*sizeof(bool);
				break;
			case 2:
				std::cout<<"Will read 'char' type target data\n";
				cvTp = DataType<char>::type;
				rawDataStep = _dimFeature*sizeof(char);
				break;
			case 3:
				std::cout<<"Will read 'int' type target data\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			case 4:
				std::cout<<"Will read 'float' type target data\n";
				cvTp = DataType<float>::type;
				rawDataStep = _dimFeature*sizeof(float);
				break;
			case 5:
				std::cout<<"Will read 'double' type target data\n";
				cvTp = DataType<double>::type;
				rawDataStep = _dimFeature*sizeof(double);
				break;
			case 6:
				std::cout<<"Will read 'uchar' type target data\n";
				cvTp = DataType<uchar>::type;
				rawDataStep = _dimFeature*sizeof(uchar);
				break;
			case 7:
				std::cout<<"Will read 'short' type target data\n";
				cvTp = DataType<short>::type;
				rawDataStep = _dimFeature*sizeof(short);
				break;
			case 8:
				std::cout<<"Will read 'unsigned short' type target data\n";
				cvTp = DataType<unsigned short>::type;
				rawDataStep = _dimFeature*sizeof(unsigned short);
				break;
			case 9:
				std::cout<<"Warning!!! OpenCV doesn't support 'unsigned int' type for cv::Mat. Will read 'int' type target data instead!\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			case 10:
				std::cout<<"Warning!!! OpenCV doesn't support 'long' type for cv::Mat. Will read 'int' type target data instead!\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			case 11:
				std::cout<<"Warning!!! OpenCV doesn't support 'unsigned long' type for cv::Mat. Will read 'int' type target data instead!\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			default:
				std::cerr<<"BinClusterInStream  error! Wrong data type!\n";
				discoverUO::wait();
				exit(-10);
		}
	}

	virtual void read(cv::Mat& mat, long selectIn = -1) //if selectIn <= 0, input all data into mat, otherwise randomly select 'selectIn' rows from total '_numFeature' 
	{
		std::cout<<_numFeature<<" ";
		parent::_stream.seekg(_initpos);

		if (selectIn <= 0 || selectIn >= _numFeature) 
		{
			mat = Mat(_numFeature, _dimFeature, cvTp);
			for (int i = 0; i < _numFeature; i++)
				parent::_stream.read(mat.ptr(i), rawDataStep);  //no matter what type 'cvTp' is, 'mat.ptr(i)' is always *uchar
		}
		else 
		{
			Mat matTmp(1, _dimFeature, cvTp);
			srand( (unsigned int)time(NULL) );
			bool *arr = new bool[_numFeature];
			memset(arr, 1, sizeof(bool)*selectIn);
			memset(arr + selectIn, 0, sizeof(bool)*(_numFeature - selectIn));
			//suuffle the array with Fisher-Yates shuffling
			/*for (int i = 0; i < (_numFeature - 1); i++)
			{
				int r = i + (rand() % (_numFeature - i);
				int tmp = arr[i];
				arr[i] = arr[r];
				arr[r] = tmp;
			}*/
			int j = _numFeature - 1;
			while (j > 0)
			{
				int r = rand() % j;
				bool tmp = arr[j];
				arr[j] = arr[r];
				arr[r] = tmp;
				j--;
			}

			mat = Mat(selectIn, _dimFeature, cvTp);
			for (int i = 0, id = 0; i < _numFeature; i++)
			{
				if (arr[i])
				{
					//std::cout<<parent::_stream.tellg()<<" \n";
					parent::_stream.read(mat.ptr(id), rawDataStep);  //no matter what type 'cvTp' is, 'mat.ptr(i)' is always *uchar
					//test if it is wrong during the reading..., the data should be less than 1.0 because of normalization
					/*for (int j0 = 0; j0 < _dimFeature; j0++)
						if (abs(((mat.ptr<float>(id)))[j0]) >= 1.0)
						{
							std::cout<<(int64)parent::_stream.tellg()<<" ";
							std::cout<<abs(((mat.ptr<float>(id)))[j0])<<" Wrong reading data or wrong data! BinClusterInStream::read(...)"<<"\n";
							discoverUO::wait();
							exit(-10);
						}*/
					//_initpos = parent::_stream.tellg();
					//std::cout<<parent::_stream.tellg()<<" \n";
					id++;
				}
				else
				{
					//std::cout<<parent::_stream.tellg()<<" \n";

					//move reading point to next row
					parent::_stream.read(matTmp.ptr(0), rawDataStep); //can't use seekg because it will overflow for "std::ios_base::cur" if larger than 32 bit integral
					//parent::_stream.seekg((int64)rawDataStep, std::ios_base::cur);
					//std::cout<<parent::_stream.tellg()<<" ";
					//std::cout<<" \n";
				}
			}
			delete []arr;
		}

		//_numFeature = selectIn;
	}


	virtual ~BinClusterInStream()
	{
		std::cout<<"Read " <<_numFeature<<" of dimension "<<_dimFeature<<std::endl;
	}
	
};



class BinMBHInStream : public BinInStream
{
	int		_dimFeature;
	int		_numFeature;  //total rows per image
	int		_frameFeature;   //total frames per video, = _rowsFeature*frames
	//char *rawData;
	int dataTp, cvTp;
	int rawDataStep;
	BiIStream::pos_type _initpos;
	typedef BinInStream		parent;
public:

	BinMBHInStream(const std::string& filename_): parent(filename_)
	{
		int version =0;
		dataTp=1;
		if (!(parent::_stream>>version))
		{
			std::cerr<<"BinClusterInStream error! Unable to read version number ";
			discoverUO::wait();
			exit(-10);
		}
		if(!(parent::_stream>>dataTp))
		{
			std::cerr<<"BinClusterInStream error! Unable to read data type";
			discoverUO::wait();
			exit(-10);
		}
		if (!(parent::_stream>>_numFeature))
		{
			std::cerr<<"BinClusterInStream error! Unable to read number of features";
			discoverUO::wait();
			exit(-10);
		}
		if (!(parent::_stream>>_dimFeature))
		{
			std::cerr<<"BinClusterInStream error! Unable to read dimension of features";
			discoverUO::wait();
			exit(-10);
		}
		if (!(parent::_stream>>_frameFeature))
		{
			std::cerr<<"BinClusterInStream error! Unable to read number of frames";
			discoverUO::wait();
			exit(-10);
		}

		_initpos = parent::_stream.tellg();

		switch(dataTp) {
			case 1:
				std::cout<<"Will read 'bool' type target data\n";
				cvTp = DataType<bool>::type;
				rawDataStep = _dimFeature*sizeof(bool);
				break;
			case 2:
				std::cout<<"Will read 'char' type target data\n";
				cvTp = DataType<char>::type;
				rawDataStep = _dimFeature*sizeof(char);
				break;
			case 3:
				std::cout<<"Will read 'int' type target data\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			case 4:
				std::cout<<"Will read 'float' type target data\n";
				cvTp = DataType<float>::type;
				rawDataStep = _dimFeature*sizeof(float);
				break;
			case 5:
				std::cout<<"Will read 'double' type target data\n";
				cvTp = DataType<double>::type;
				rawDataStep = _dimFeature*sizeof(double);
				break;
			case 6:
				std::cout<<"Will read 'uchar' type target data\n";
				cvTp = DataType<uchar>::type;
				rawDataStep = _dimFeature*sizeof(uchar);
				break;
			case 7:
				std::cout<<"Will read 'short' type target data\n";
				cvTp = DataType<short>::type;
				rawDataStep = _dimFeature*sizeof(short);
				break;
			case 8:
				std::cout<<"Will read 'unsigned short' type target data\n";
				cvTp = DataType<unsigned short>::type;
				rawDataStep = _dimFeature*sizeof(unsigned short);
				break;
			case 9:
				std::cout<<"Warning!!! OpenCV doesn't support 'unsigned int' type for cv::Mat. Will read 'int' type target data instead!\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			case 10:
				std::cout<<"Warning!!! OpenCV doesn't support 'long' type for cv::Mat. Will read 'int' type target data instead!\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			case 11:
				std::cout<<"Warning!!! OpenCV doesn't support 'unsigned long' type for cv::Mat. Will read 'int' type target data instead!\n";
				cvTp = DataType<int>::type;
				rawDataStep = _dimFeature*sizeof(int);
				break;
			default:
				std::cerr<<"BinClusterInStream  error! Wrong data type!\n";
				discoverUO::wait();
				exit(-10);
		}
	}

	virtual void read(vector<cv::Mat>& mat, int stframes = -1, int endframes = -1) //if endframes <= 0, input all data into mat 
	{
		//std::cout<<_numFeature<<" ";
		parent::_stream.seekg(_initpos);
		Mat tmp;
		mat.clear();

		if (endframes <= 0 || stframes >= _frameFeature || stframes >= endframes) 
		{
			if(endframes > _frameFeature || endframes <= 0 )
				endframes = _frameFeature;
			for (int i0 = 0; i0 < endframes; i0++)
			{
				tmp = Mat(_numFeature, _dimFeature, cvTp);
				for (int i = 0; i < _numFeature; i++)
					parent::_stream.read(tmp.ptr(i), rawDataStep);  //no matter what type 'cvTp' is, 'mat.ptr(i)' is always *uchar
				mat.push_back(tmp);
			}
		}
		else 
		{
			if(endframes > _frameFeature)
				endframes = _frameFeature;
			
			if(stframes > 0) 
			{
				long offset = stframes*_numFeature*rawDataStep;
				parent::_stream.seekg(offset, ios_base::cur);
			}
			
			for (int i0 = stframes; i0 < endframes; i0++)
			{
				tmp = Mat(_numFeature, _dimFeature, cvTp);
				for (int i = 0; i < _numFeature; i++)
					parent::_stream.read(tmp.ptr(i), rawDataStep);  //no matter what type 'cvTp' is, 'mat.ptr(i)' is always *uchar
				mat.push_back(tmp);
			}
		}
		//_numFeature = selectIn;

	}

	cv::Point3i getVsz()
	{
		return Point3i(_dimFeature,_numFeature,_frameFeature);
	}


	virtual ~BinMBHInStream()
	{
		std::cout<<"Read " <<_frameFeature<<" frames of "<<_numFeature<<" x "<<_dimFeature<<" optical flow images."<<std::endl;

	}
	
};

template<typename T1>
class BinOutStream
{
protected:
	BiOStream _stream;
public:
	BinOutStream(const std::string& filename):
		_stream(filename.c_str())
	{
		if(!_stream)
		{
			std::cerr<<"BinaryOutStream error! Unable to open the output file " + filename;
			discoverUO::wait();
			exit(-10);
		}
	}

	//virtual void write(const T1* data, const long dimFeature_ ) = 0;

	virtual ~BinOutStream(){_stream.close();}

};

template<typename T1>
class BinClusterOutStream : public BinOutStream<T1>
{
	int  _dimFeature;
	long _numFeature;
	//BiOStream::pos_type _initpos;
	long _initpos; 
	typedef BinOutStream<T1>		parent;
public:

	BinClusterOutStream(const std::string& filename_, 
			const int dimFeature_ = 0, const long numFeature_ = 0):
		parent(filename_), _numFeature(numFeature_), _dimFeature(dimFeature_)
	{
		int version =0x10000;
		parent::_stream<<version;
		int type = IOTypeIdentifier<T1>::ID;
		parent::_stream<<type;
		_initpos = parent::_stream.tellp();
		parent::_stream<<_numFeature<<_dimFeature;
	}

	void write(const T1* data, const long dimFeature_ = 0 ) //here dimFeature_ should be number of T1 in "data" array
	{
		//std::cout<<data[100]<<"\n";
		if (dimFeature_)
			_dimFeature = dimFeature_;
		if (!(parent::_stream.write(data, _dimFeature)))  //_stream.write(T1* data, int size) -> to.write((char*)data, size*sizeof(T1));
		{
			std::cerr<<"BinaryOutStream error! Unable to write data into the output file " ;
			discoverUO::wait();
			exit(-10);
		}
		//for (int i = 0; i < _dimFeature; i++)
		//	parent::_stream<<data[i];
		++_numFeature;

	}


	virtual ~BinClusterOutStream()
	{
		//parent::_stream.seekp(0,std::ios_base::beg);
		//int version =0x10000;
		//parent::_stream<<version;
		//int type = IOTypeIdentifier<T1>::ID;
		//parent::_stream<<type;		
		parent::_stream.seekp(_initpos);
		parent::_stream<<_numFeature<<_dimFeature;
		std::cout<<"Written " <<_numFeature<<" of dimension "<<_dimFeature<<std::endl;
	}
	
};




template<typename T1, typename T2>
class BinOutStreamSVM
{
protected:
	BiOStream _stream;
public:
	BinOutStreamSVM(const std::string& filename):
		_stream(filename.c_str())
	{
		if(!_stream)
		{
			std::cerr<<"BinaryOutStream error! Unable to open the output file " + filename;
			discoverUO::wait();
			exit(-10);
		}
	}

	virtual void write(const T1* data, const T2 target, const int dimFeature_ ) = 0;

	virtual ~BinOutStreamSVM(){_stream.close();}

};

template<typename T1, typename T2>
class BinSVMLightOutStream : public BinOutStreamSVM<T1,T2>
{
	int  _dimFeature, _numFeature;
	BiOStream::pos_type _initpos;
	typedef BinOutStreamSVM<T1,T2>		parent;
public:

	BinSVMLightOutStream(const std::string& filename_, 
			const int dimFeature_ = 0, const int numFeature_ = 0):
		parent(filename_), _numFeature(numFeature_), _dimFeature(dimFeature_)
	{
		int version =0x10000;
		parent::_stream<<version;
		int type = IOTypeIdentifier<T1>::ID;
		parent::_stream<<type;
		type = IOTypeIdentifier<T2>::ID;
		parent::_stream<<type;
		_initpos = parent::_stream.tellp();
		parent::_stream<<_numFeature<<_dimFeature;
	}

	virtual void write(const T1* data, const T2 target, const int dimFeature_ = 0 )
	{
		if (dimFeature_)
			_dimFeature = dimFeature_;
		parent::_stream<<target;
		for (int i = 0; i < _dimFeature; i++)
			parent::_stream<<data[i];
		++_numFeature;

	}


	virtual ~BinSVMLightOutStream()
	{
		parent::_stream.seekp(_initpos);
		parent::_stream<<_numFeature<<_dimFeature;
		std::cout<<"Written " <<_numFeature<<" of dimension "<<_dimFeature<<std::endl;

	}
	
};


template<typename T1>
class BinMBHOutStream : public BinOutStream<T1>
{
	int		_dimFeature;
	int		_numFeature;  //total rows per image
	int		_frameFeature;   //total frames per video, = _rowsFeature*frames
	//BiOStream::pos_type _initpos;
	long _initpos; 
	typedef BinOutStream<T1>		parent;
public:

	BinMBHOutStream(const std::string& filename_, int imH, int imW, int imT):
		parent(filename_), _numFeature(imH), _dimFeature(imW), _frameFeature(imT)
	{
		int version =0x10000;
		parent::_stream<<version;
		int type = IOTypeIdentifier<T1>::ID;
		parent::_stream<<type;
		_initpos = parent::_stream.tellp();
		parent::_stream<<_numFeature<<_dimFeature<<_frameFeature;
	}

	void write(const T1* data) //here dimFeature_ should be number of T1 in "data" array
	{
		//std::cout<<data[100]<<"\n";
		
		if (!(parent::_stream.write(data, _dimFeature)))  //_stream.write(T1* data, int size) -> to.write((char*)data, size*sizeof(T1));
		{
			std::cerr<<"BinaryOutStream error! Unable to write data into the output file " ;
			discoverUO::wait();
			exit(-10);
		}
		//for (int i = 0; i < _dimFeature; i++)
		//	parent::_stream<<data[i];
		//++_numFeature;

	}


	virtual ~BinMBHOutStream()
	{
		//parent::_stream.seekp(0,std::ios_base::beg);
		//int version =0x10000;
		//parent::_stream<<version;
		//int type = IOTypeIdentifier<T1>::ID;
		//parent::_stream<<type;		
		//parent::_stream.seekp(_initpos);
		//parent::_stream<<_numFeature<<_dimFeature<<_frameFeature;
		std::cout<<"Written "<<_frameFeature<<" frames of "<<_numFeature<<" x "<<_dimFeature<<" optical flow images."<<std::endl;

	}
	
};



#endif