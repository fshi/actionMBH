#include "cxcore.h"
#include "cv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <fstream>

#include <ctype.h>

#include "biostream.h" 
#include "biistream.h" 
#include "formatBinaryStream.h"
#include "omp.h"

#include "svm.h"
#include "chiSquareDist.h"
using namespace std;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//#define EVAL_CPP   //Binary-class Cross Validation with Different Criteria(precision, recall, F-score, AUC...)
#ifdef EVAL_CPP 
	#include "eval.h"
#endif


static int (*info)(const char *fmt,...) = &printf;
void print_null(const char *s) {}
void read_binary(const string * fileName, const int numClass) ;
void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"	5 -- intersection (min): min(u,v)\n"
	"	6 -- chi-squared: 2uv/(u+v)\n"
	"	7 -- Jenson-Shannon's: u/2log((u+v)/u)) + v/2log((u+v)/v))\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"-x : read binary input training file with number of words (-x 4, means the input training file is binary with 4000 number of words)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, bool& binaryFile, float& numWords);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
vector< svm_model *> model;
struct svm_node *x_space;
int *classId;
int cross_validation;
int nr_fold;

static char *line0 = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line0,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line0,'\n') == NULL)
	{
		max_line_len *= 2;
		line0 = (char *) realloc(line0,max_line_len);
		len = (int) strlen(line0);
		if(fgets(line0+len,max_line_len-len,input) == NULL)
			break;
	}
	return line0;
}

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	
	bool binaryFile = 0;  //added by Feng for read binary training file. defualt is "0" (original data format. if arg -x .., read binary file)
	float numW;
	int nCls = 0;

	parse_command_line(argc, argv, input_file_name, model_file_name, binaryFile, numW);
	int numWords = 0.5+numW;
	numWords *= 1000;
	while (nCls <= 0)
	{
		std::cout<<"Pleas input total number of classes : \n";
		cin>>nCls;
		std::cout<<std::endl;
	}
	string *fileName = new string[nCls];
	char tmpC[10], tstr[10];
	itoa(numWords, tstr, 10);

	if(binaryFile)
	{
		for (int i = 0; i < nCls; i++)
		{
			fileName[i] = (string)"pWords" + (string)itoa(i+1,tmpC,10) + (string)"_" + (string)tstr+ (string)".dat";
		}

		read_binary(fileName, nCls);
		std::cout<<"done read binary file pwords."<<std::endl;
		strcpy(model_file_name,"model.dat");
		delete []fileName;
	}
	else
	{
		//read_problem(input_file_name);
		std::cout<<"Wrong! Current version only support binary training file inputs"<<std::endl;
		cin.get();cin.get();
		exit(2);
	}

	error_msg = svm_check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	/*
	if(cross_validation)
	{
#ifdef EVAL_CPP
		double cv =  binary_class_cross_validation(&prob, &param, nr_fold);
		printf("Cross Validation = %g%%\n",100.0*cv);
#else
		do_cross_validation();
#endif
	}
	else
	*/
	double *values = new double[nCls];
	double maxVal;
	struct svm_node *x;

	model.clear();
	//learning the models (number of classes)
	for(int i = 0; i < nCls; i++)
	{
		for(int i0 = 0; i0 <prob.l; i0++)
		{
			if (classId[i0] == i)
				prob.y[i0] = 0;
			else
				prob.y[i0] = 1;
		}

		model.push_back(svm_train(&prob,&param));
		/*
		struct svm_model *model0 = svm_train(&prob,&param);
		//std::cout<<model[i]->nSV[0]<<" "<<model[i]->nSV[1]<<endl;
		
		if(svm_save_model(model_file_name,model0))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}
		
		svm_free_and_destroy_model(&model0);
		if((model0=svm_load_model(model_file_name))==0)
		{
			fprintf(stderr,"can't open model file %s\n",model_file_name);
			exit(1);
		}
		model.push_back(model0);
		*/
		std::cout<<"Done training class "<<i+1<<std::endl;
	}
	free(prob.y);
	free(prob.x);
	free(classId);
	svm_destroy_param(&param);

	
	//done the training!

	fileName = new string[nCls];
	itoa(numWords, tstr, 10);
	for (int j = 0; j < nCls; j++)
	{
		fileName[j] = (string)"bagWord" + (string)itoa(j+1,tmpC,10) + (string)"_" + (string)tstr+ (string)".dat";
	}

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	int svm_type=svm_get_svm_type(model[0]);
	int j;
	using namespace cv;
	cv::Mat pMat;
	BinClusterInStream *pFile;
	cv::Mat *pMatF = new Mat[nCls];
	for (int i = 0; i < nCls; i++)
	{
		pFile = new BinClusterInStream(fileName[i]);
		pFile->read(pMat);
		pMatF[i].create(pMat.rows, pMat.cols, CV_32FC1);
		//label[i] = Mat(pMat.rows, 1, CV_32SC1, Scalar_<int>(i));

		//cout<<label[i].at<int>(200,0)<<"\n";
		if (pMat.type() == CV_32FC1)
			pMat.copyTo(pMatF[i]);
		else
		{
			fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
			exit(1);
		}

		delete pFile;
		pMat.release();
	}
	delete []fileName;
	x = (struct svm_node *) malloc((pMatF[0].cols+1)*sizeof(struct svm_node));
	FILE *output = fopen("results.txt","w");
	fpos_t position;
	fgetpos(output, &position);
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		fprintf(output,"                                                            \n");
		fprintf(output,"                                                            \n");
		fprintf(output,"                                                            \n");
		
	}
	else
	{
		fprintf(output,"                                                             \n");
		fprintf(output,"                                                      \n");
		fprintf(output,"tests\ttarget\tpredict\n");
	}

	int target_label, label;
	float *data;
	int j0;

	for(int id = 0; id < nCls; id++)
	{
		target_label = id;
		for(int j = 0; j < pMatF[id].rows; j++)  //number of features
		{
			data = pMatF[id].ptr<float>(j);

			for(j0 = 0; j0 < pMatF[id].cols; j0++)  //feature dimensions
			{
				x[j0].index = j0+1;  //feature id beginning from "1" while j0 beginning from "0"
				x[j0].value = data[j0];
			}
			x[j0].index = -1;

			
			//svm_predict_values(model[0], x, &maxVal);
			svm_predict_values(model[0], x, &values[0]);
			maxVal = values[0];
			label = 0;
			//classify x with nCls models and save the results into values[i]

			//For binary libsvm( before version 3.17), 
			//since internally class labels are ordered by their first occurrence in the training set, 
			//we need to set the classified values to negative after the 1st class
			for(int i = 1; i < nCls; i++)
			{
				svm_predict_values(model[i], x, &values[i]);
				//std::cout<<model[i]->nSV[0]<<" "<<model[i]->nSV[1]<<endl;
				values[i] = -values[i];

				if (maxVal < values[i])
				{
					maxVal = values[i];
					label = i;
				}
			}

	//std::cout<<values[0]<<" "<<values[1]<<" "<<values[2]<<" "<<values[3]<<" "<<maxVal<<" "<<label<<endl;
			fprintf(output,"%d\t%d\t%d",j,id, label);
			if(label == target_label)
				fprintf(output,"\n");
			else
				fprintf(output,"  Wrong!\n");

			if(label == target_label)
				++correct;
			error += (label-target_label)*(label-target_label);
			sump += label;
			sumt += target_label;
			sumpp += label*label;
			sumtt += target_label*target_label;
			sumpt += label*target_label;
			++total;
		}

		std::cout<<"Done predict class "<<id+1<<"!\n";
	}

	info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
	fsetpos (output, &position);
	fprintf(output,"Accuracy = %g%% (%d/%d) (classification)",
		(double)correct/total*100,correct,total);
	delete []pMatF;

	std::cout<<"\nDone testing with kernel = "<<param.kernel_type<<" and C = "<<param.C<<std::endl;
	for (int i = 0; i < model.size(); i++)
		svm_free_and_destroy_model(&model[i]);
	std::cout<<"\nDone clear model "<<model.size()<<std::endl;

	free(x_space);
	//free(x_space);
	free(line0);
	
	//model.clear();
	delete []values;
	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, bool& binaryFile, float& numWords)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	//param.kernel_type = RBF;
	//param.kernel_type = CHISQUARED;
	param.kernel_type = INTERSECTION;
	param.degree = 3;
	param.gamma = 0.00225;	// param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 32.5;		//param.C = 1;
	param.eps = 1e-3;
	//param.eps = 1e-5;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;
	binaryFile = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'x':				// Added by Feng to read binaryFile, if arg -x ..., read binary file
				binaryFile = true; 
				numWords = atof(argv[i]);
				break;
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(!binaryFile)
	{
		if(i>=argc)
			exit_with_help();

		strcpy(input_file_name, argv[i]);

		if(i<argc-1)
			strcpy(model_file_name,argv[i+1]);
		else
		{
			char *p = strrchr(argv[i],'/');
			if(p==NULL)
				p = argv[i];
			else
				++p;
			sprintf(model_file_name,"%s.model",p);
		}
	}
}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line0 = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line0," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	classId = Malloc(int,prob.l);
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line0," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

void read_binary(const string * fileName, const int numClass) 
{
	using namespace cv;
	cv::Mat pMat;
	prob.l = 0;
	BinClusterInStream *pFile;
	cv::Mat *pMatF = new Mat[numClass];

	for (int i = 0; i < numClass; i++)
	{
		pFile = new BinClusterInStream(fileName[i]);
		pFile->read(pMat);
		pMatF[i].create(pMat.rows, pMat.cols, CV_32FC1);

		if (pMat.type() == CV_32FC1)
			pMat.copyTo(pMatF[i]);
		else
		{
			fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
			exit(1);
		}
		prob.l += pMat.rows;

		delete pFile;
		pMat.release();
	}

	int elements = prob.l * (pMatF[0].cols + 1);
	float *data;

	classId = Malloc(int,prob.l);
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	int id = 0, jd = 0;
	for (int i = 0; i < numClass; i++)  //class
	{
		for(int j = 0; j < pMatF[i].rows; j++, id++)  //number of features
		{
			prob.x[id] = &x_space[jd];
			//prob.y[id] = i;   //class label
			classId[id] = i;   //class label
			data = pMatF[i].ptr<float>(j);

			for(int j0 = 0; j0 < pMatF[i].cols; j0++, jd++)  //feature dimensions
			{
				x_space[jd].index = j0+1;   //feature id beginning from "1" while j0 beginning from "0"
				x_space[jd].value = data[j0];
			}
			x_space[jd++].index = -1;
		}
	}
	if(param.kernel_type == CHISQUARED)  //compute mean chi-square dist and save it to 1/param.gamma
	{
		for (int i = 0; i < numClass; i++)  //class
			pMat.push_back(pMatF[i]);
		double sumDist = 0;
		int count = 0;
		float *p1, *p2;
		for( int i = 0; i < pMat.rows; i++)
		{
			p1 = pMat.ptr<float>(i);
			for(int j = i+1; j < pMat.rows; j++)
			{
				p2 = pMat.ptr<float>(j);
				sumDist += chiSquareDistMP<float, float>(p1, p2, pMat.cols);
				count++;
			}
		}
		param.gamma = (double)count/(sumDist*0.5);
		std::cout<<param.gamma<<" "<<pMat.rows<<" "<<count<<" "<<sumDist<<std::endl;
	}
	else if(param.gamma == 0 && pMatF[0].cols > 0)
		param.gamma = 1.0/pMatF[0].cols;

	for (int i = 0; i < numClass; i++)
		pMatF[i].release();
	delete []pMatF;
}