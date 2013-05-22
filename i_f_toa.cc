#include "i_f_toa.h"


using namespace std;
char * i_ftoa(char tstr[], int numWords)
{
	char *pstr = tstr+2;
	if (numWords < 1000)
	{
		itoa(numWords, tstr, 10);
	}
	else if (!(numWords%1000))
	{
		itoa(numWords/100, tstr, 10);
		tstr[1] = 'k';
	}
	else
	{
		itoa(numWords/100, tstr, 10);
		tstr[1] = '.';
		int tmp = numWords%1000;
		if (tmp >= 100)
		{
			tmp /= 10;  
			itoa(tmp, pstr, 10);
			tstr[3] = 'k';
		}
		else
			strcpy(pstr, "0k");
		
	}
	return tstr;
}

char * ftoa(char numStr[], float fNum, int decimalNum )
{
	int wholeNum = (int)fNum;//wouldn't hurt to cast it to avoid warning
	int pow10 = 1;
	for(int i = 0; i < decimalNum; i++)
	 pow10 *= 10;
	int decimals= (int) ((fNum - (float)wholeNum) *pow10 + 0.5);
	char wholeNumStr[10];
	char decNumStr[5];
	itoa(wholeNum, wholeNumStr, 10);
	itoa(decimals, decNumStr, 10);
	strcpy(numStr, wholeNumStr);
	strcat(numStr, ".");
	strcat(numStr, decNumStr);

	return numStr;

}