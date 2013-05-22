#ifndef _DISCOVER_UO_WAIT_H_
#define _DISCOVER_UO_WAIT_H_

#include <stdio.h>
#include <ctime>
#include <string>
#include <iostream>
//#include <sstream>

using namespace std;
namespace discoverUO {
/*
inline void wait() //wait a key in
{ 
	using namespace std;
	cout << "\nPress any key to continue ...\n"; 
	//string z; 
	//getline(cin,z); 
	cin.sync();
	cin.get();
}
*/
	
	
inline void wait ( int seconds = 0 ) //wait some seconds
{
	using namespace std;
	if (seconds <= 0)
	{
		cout << "\nPress Enter key to continue ...\n"; 
		//string z; 
		//getline(cin,z); 
		cin.sync();
		cin.get();
	}
	else {
		clock_t endwait;
		endwait = clock () + seconds * CLOCKS_PER_SEC ;
		while (clock() < endwait) {}
	}
}
/*
std::string timestamp() 
{
        //Notice the use of a stringstream, yet another useful stream medium!
        ostringstream stream;    
        time_t rawtime;
        tm * timeinfo;
 
        time(&rawtime);
        timeinfo = localtime( &rawtime );
 
        stream << (timeinfo->tm_year)+1900<<" "<<timeinfo->tm_mon
        <<" "<<timeinfo->tm_mday<<" "<<timeinfo->tm_hour
        <<" "<<timeinfo->tm_min<<" "<<timeinfo->tm_sec;
        // The str() function of output stringstreams return a std::string.
        return stream.str();   
}*/
}//end of namespace;
#endif