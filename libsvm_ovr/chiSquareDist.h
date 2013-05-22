/********************************************************************************

Filename     : chiSquareDist.h

Description  : This file is used to compute chi-square distance between two histograms (arrays). 



Typical Use  :  


				
Author       : FengShi@Discovery lab, Jul, 2010
Version No   : 1.00 


*********************************************************************************/
#ifndef _CHISQUARE_DIST_H_
#define _CHISQUARE_DIST_H_

#ifndef EPSILON_
	#define EPSILON_ 1E-7
#endif

#include <omp.h>
//#include <iostream>

template <class T1, class T2>
double chiSquareDist(const T1 *h1, const T2 *h2, const int len)   //for computation efficiency, it calculate 2*chi-squared distance.
{
	double sum = 0;
	T1 tmp;
	for (int i = 0; i < len; i++)
		if (tmp = (h1[i] + h2[i]))
			sum += (double)((h1[i] - h2[i])*(h1[i] - h2[i])/tmp);
	return sum;


}

//Multi thread version by using openMP
template <class T1, class T2>
double chiSquareDistMP(const T1 *h1, const T2 *h2, const int len)   //for computation efficiency, it calculate 2*chi-squared distance.
{
	double sum = 0.;
	
	#pragma omp parallel for reduction(+ : sum)
	for (int i = 0; i < len; i++)
	{
		T1 tmp;
		if (tmp = (h1[i] + h2[i]))
			sum += (double)((h1[i] - h2[i])*(h1[i] - h2[i])/tmp);
	}

	return sum;
}


#ifdef EPSILON_
	#undef EPSILON_
#endif

#endif