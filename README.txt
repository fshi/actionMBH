Introduction
------------

This implementation is an improvement over our CVPR'13 paper with multiple channels. Please check our notepaper on ucf101 workshop “LPM for fast action recognition with large number of classes”.

If you use our code, please cite our CVPR'13 paper.
If you use the svm in our code, you should cite libsvm paper here: http://www.csie.ntu.edu.tw/~cjlin/libsvm/. You should also read its COPYRIGHT: http://www.csie.ntu.edu.tw/~cjlin/libsvm/COPYRIGHT.


Usage
-----

Dataset subdirectory::

     To run the program, you need to have the dataset videos, and replace (const string dirName = "C:\\dataSets\\hmdb51\\";) with your dataset directory. For simplicity, the videos from the same class are stored in a subdirectory. One class, one subdirectory, marked with 1,2,3...

Visual studio project properties::

     If you have compiler error, try to change the VS project properties as: Configuration properties->General->Character Set->Not Set. If using vs2012 and having compiler error of "itoa...", you can set: property pages->Configuration Properties->C/C++->General->SDL checks->No(/sdl-).
     
     
Parameters::

     You need to include “MBH_parameters_input.txt” file inside your working directory as input parameters.


Steps to do testing/training:

1. To run the program, you need to have the dataset videos, and replace (const string dirName = "C:\\dataSets\\hmdb51\\";) with your dataset directory. For simplicity, the videos from the same class are stored in a subdirectory. One class, one subdirectory, marked with 1,2,3...
2. You need to include “MBH_parameters_input.txt” file inside your working directory as input parameters.
3. You need to generate the clusters by running "getCluster.exe". the code will ask you some inputs. To start, input "4" for 4000 words.
4. Next step is to run "getTrainTestData.ext". After running the *getTrainTestData*, you should get the files representing each video file with bag-of-features representation (one .dat file for one video clip). I assume that you are testing with HMDB51 dataset, and all the .dat files are stored inside the sub-directory, such as random1, random2,..., random51.
5. Now we need to use hmdb51 split1.txt, split2.txt and split3.txt files to group individual .dat files into 70/30 training/testing files and feed them into *libsvm_ovr* to output results. 
6. The .cpp files to split training/testing files into groups are inside the Tools directory. You can build it and run under the directory with folders *random1, random2,..., random51* inside.
7. To run the code, you may need to input "4000",  "0", "1", "0" as instructed on screen for proper data scaling.
8. The output should be 3 sub-directories, such as "split1Rs0", "split2Rs0" and "split3Rs0". Now, you can copy the compiled *libsvm_ovr.exe* into each of these sub-directories to do the learning and testing.
9. The *libsvm_ovr.exe* should be run as a command line mode as follow:

libsvm_ovr -x 4000

Then input the total number of class.

"-x 4000" means 4k codewords

10. The *libsvm_ovr.exe* could be bulit with openMP (if your compiler supports) and use multiple cores for fast training/testing. It can also be built to support 64 bits OS in case of large feature dimension or large datasets.

