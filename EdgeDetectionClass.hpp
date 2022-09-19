//
//  EdgeDetectionClass.hpp
//  OpenCv
//
//  Created by Shokhina Badrieva on 5/4/21.
//  Copyright © 2021 Shokhina Badrieva. All rights reserved.
//

#ifndef EdgeDetectionClass_hpp
#define EdgeDetectionClass_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


int xGradient(Mat image, int x, int y);
int yGradient(Mat image, int x, int y);
int northGradient(Mat image, int x, int y);
int southGradient(Mat image, int x, int y);
int eastGradient(Mat image, int x, int y);
int westGradient(Mat image, int x, int y);
int northeastGradient(Mat image, int x, int y);
int northwestGradient(Mat image, int x, int y);
int southeastGradient(Mat image, int x, int y);
int southwestGradient(Mat image, int x, int y);
int laplacianGradient(Mat image, int x, int y);
bool sobelOperator(Mat &src, Mat &dst);
bool robinsonEdgeDetector(Mat &src, Mat &dst,int thresh);
bool sobelOperatorThresh(Mat &src, Mat &dst,int thresh);
bool zeroCrossingEdgeDetection(Mat &src, Mat &dst);
void cannyDeterctor(Mat&src,Mat&dst);



#endif /* EdgeDetectionClass_hpp */
