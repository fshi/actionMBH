#ifndef _I_f_TOA_H_
#define _I_f_TOA_H_

#include <stdio.h>
#include <cstdlib>
#include <string.h>

//this function change int (1-9999) into char[5]
//if int<1000, char[5] should be 3 number. Such as, 555->"555"
//if int is x000, char[5] should be a number and a k. Such as, 4000->"4k"
//if int is xxxx, char[5] should be x.xk. Such as , 4359->"4.3"
//if int is x.0xx or x.00x, char[5] should be x.0k. Such as, 4035->"4.0k"
char * i_ftoa(char tstr[], int numWords);
char * ftoa(char numStr[], float fNum, int decimalNum = 4);

#endif //_I_f_TOA_H_