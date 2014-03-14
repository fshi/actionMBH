//This code parse our binary file into .txt file so it can be read in text edit.

#include "cxcore.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <strstream>
#include "formatBinaryStream.h"
#include <cstdlib>
#include <string>
#include <time.h>
#include "waitKeySeconds.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	char input_file_name[1024];

	strcpy(input_file_name, argv[1]);
	string fileTp = ".dat";
	string fNameTxt = input_file_name;
	fNameTxt.replace(fNameTxt.find(fileTp),fileTp.length(),".txt");

	BinClusterInStream	*inFile = new  BinClusterInStream(input_file_name);
	Mat tmp;
	inFile->read(tmp);
	std::fstream fOut;
	fOut.open(fNameTxt.c_str(), std::ios::out);
	for(int i = 0; i < tmp.rows; i ++)
	{
		float *data = tmp.ptr<float>(i);
		for (int j = 0; j < tmp.cols; j++)
			fOut<<data[j]<<" ";
		fOut<<endl<<endl;
	}

	fOut.close();

	delete inFile;
	
	discoverUO::wait();
	//std::cin.get();
	//cin.get();
	return 0;
}