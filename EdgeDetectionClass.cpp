//
//  EdgeDetectionClass.cpp
//  OpenCv
//
//  Created by Shokhina Badrieva on 5/4/21.
//  Copyright © 2021 Shokhina Badrieva. All rights reserved.
//

#include "EdgeDetectionClass.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;



int xGradient(Mat image, int x, int y){
    return image.at<uchar>(y-1, x-1) +
    2*image.at<uchar>(y, x-1) +
    image.at<uchar>(y+1, x-1) -
    image.at<uchar>(y-1, x+1) -
    2*image.at<uchar>(y, x+1) -
    image.at<uchar>(y+1, x+1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y){
    return image.at<uchar>(y-1, x-1) +
    2*image.at<uchar>(y-1, x) +
    image.at<uchar>(y-1, x+1) -
    image.at<uchar>(y+1, x-1) -
    2*image.at<uchar>(y+1, x) -
    image.at<uchar>(y+1, x+1);
}

int northGradient(Mat image, int x, int y){
    return -image.at<uchar>(y-1, x-1) -
    2*image.at<uchar>(y, x-1) -
    image.at<uchar>(y+1, x-1) +
    image.at<uchar>(y-1, x+1) +
    2*image.at<uchar>(y, x+1) +
    image.at<uchar>(y+1, x+1);
}

int southGradient(Mat image, int x, int y){
    return image.at<uchar>(y-1, x-1) +
    2*image.at<uchar>(y-1, x) +
    image.at<uchar>(y-1, x+1) -
    image.at<uchar>(y+1, x-1) -
    2*image.at<uchar>(y+1, x) -
    image.at<uchar>(y+1, x+1);
}

int eastGradient(Mat image, int x, int y){
    return 2*image.at<uchar>(y-1, x-1) +
    image.at<uchar>(y-1, x) +
    image.at<uchar>(y, x-1) -
    image.at<uchar>(y, x+1) -
    image.at<uchar>(y+1, x) -
    2*image.at<uchar>(y+1, x+1);
}

int westGradient(Mat image, int x, int y){
    return image.at<uchar>(y-1, x) +
    2*image.at<uchar>(y-1, x+1) -
    image.at<uchar>(y, x-1) +
    image.at<uchar>(y, x+1) -
    2*image.at<uchar>(y+1, x-1) -
    image.at<uchar>(y+1, x);
}

int northeastGradient(Mat image, int x, int y){
    return -2*image.at<uchar>(y-1, x-1) -
    image.at<uchar>(y-1, x)-
    image.at<uchar>(y, x-1) +
    image.at<uchar>(y, x+1) +
    image.at<uchar>(y+1, x) +
    2*image.at<uchar>(y+1, x+1);
}

int northwestGradient(Mat image, int x, int y){
    return image.at<uchar>(y-1, x-1) +
    2*image.at<uchar>(y, x-1) +
    image.at<uchar>(y+1, x-1) -
    image.at<uchar>(y-1, x+1) -
    2*image.at<uchar>(y, x+1) -
    image.at<uchar>(y+1, x+1);
}

int southeastGradient(Mat image, int x, int y){
    return -image.at<uchar>(y-1, x-1) -
    2*image.at<uchar>(y-1, x) -
    image.at<uchar>(y-1, x+1) +
    image.at<uchar>(y+1, x-1) +
    2*image.at<uchar>(y+1, x) +
    image.at<uchar>(y+1, x+1);
}

int southwestGradient(Mat image, int x, int y){
    return -image.at<uchar>(y-1, x) -
    2*image.at<uchar>(y-1, x+1) +
    image.at<uchar>(y, x-1) -
    image.at<uchar>(y, x+1) +
    2*image.at<uchar>(y+1, x-1) +
    image.at<uchar>(y+1, x);
}

int laplacianGradient(Mat image, int x, int y){
    return image.at<uchar>(y-1, x) +
    image.at<uchar>(y, x-1) -4*
    image.at<uchar>(y, x) +
    image.at<uchar>(y, x+1) +
    image.at<uchar>(y+1, x);
}


bool sobelOperator(Mat &src, Mat &dst){
    
    int gx, gy, sum;
    int imgRows, imgCols;
    
    imgRows = src.size().height;
    imgCols = src.size().width;
    dst = Mat::zeros(imgRows, imgCols, CV_8UC1);
    
    if( !src.data )
    { return false; }
    
    
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            dst.at<uchar>(y,x) = 0.0;
    
    for(int y = 1; y < src.rows - 1; y++){
        for(int x = 1; x < src.cols - 1; x++){
            gx = xGradient(src, x, y);
            gy = yGradient(src, x, y);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            dst.at<uchar>(y,x) = sum;
        }
    }
    
    
    return true;
}


bool robinsonEdgeDetector(Mat &src, Mat &dst,int thresh){
    
    int n,s,e,w,ne,nw,se,sw;
    int imgRows, imgCols;
    int vals[8];
    
    imgRows = src.size().height;
    imgCols = src.size().width;
    dst = Mat::zeros(imgRows, imgCols, CV_8UC1);
    
    if( !src.data )
    { return false; }
    
    
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            dst.at<uchar>(y,x) = 0.0;
    
    
    for(int y = 1; y < src.rows - 1; y++){
        for(int x = 1; x < src.cols - 1; x++){
            n=northGradient(src, x, y);
            vals[0]=n;
            s=southGradient(src,x, y);
            vals[1]=s;
            e=eastGradient(src, x, y);
            vals[2]=e;
            w=westGradient(src, x, y);
            vals[3]=w;
            ne=northeastGradient(src, x, y);
            vals[4]=ne;
            nw=northwestGradient(src, x, y);
            vals[5]=nw;
            se=southeastGradient(src, x, y);
            vals[6]=se;
            sw=southwestGradient(src, x, y);
            vals[7]=sw;
            
            int max=0;
            for(int i=0;i<8;i++){
                if(vals[i]>=max){
                    max=vals[i];
                }
            }
            if(max>thresh){
                dst.at<uchar>(y,x) = 255;
            }
            else{
                dst.at<uchar>(y,x) = 0;
            }
        }
    }
    
    
    return true;
}



bool sobelOperatorThresh(Mat &src, Mat &dst,int thresh){
    
    int gx, gy, sum;
    int imgRows, imgCols;
    
    imgRows = src.size().height;
    imgCols = src.size().width;
    dst = Mat::zeros(imgRows, imgCols, CV_8UC1);
    
    if( !src.data )
    { return false; }
    
    
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            dst.at<uchar>(y,x) = 0.0;
    
    for(int y = 1; y < src.rows - 1; y++){
        for(int x = 1; x < src.cols - 1; x++){
            gx = xGradient(src, x, y);
            gy = yGradient(src, x, y);
            sum = abs(gx) + abs(gy);
            if(sum>thresh){
                dst.at<uchar>(y,x) = 255;
            }
            else{
                dst.at<uchar>(y,x) = 0;
            }
        }
    }
    
    
    return true;
}

bool zeroCrossingEdgeDetection(Mat &src, Mat &dst){
    
    int l;
    int imgRows, imgCols;
    
    imgRows = src.size().height;
    imgCols = src.size().width;
    dst = Mat::zeros(imgRows, imgCols, CV_8UC1);
    
    if( !src.data )
    { return false; }
    
    
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            dst.at<uchar>(y,x) = 0.0;
    
    for(int y = 1; y < src.rows - 1; y++){
        for(int x = 1; x < src.cols - 1; x++){
            l = laplacianGradient(src, x, y);
            l = l > 255 ? 255:l;
            l = l < 0 ? 0 : l;
            dst.at<uchar>(y,x) = l;
        }
    }
    
    
    return true;
    
}

void cannyDeterctor(Mat&src,Mat&dst){
    int imgRows, imgCols;
    
    imgRows = src.size().height;
    imgCols = src.size().width;
    dst = Mat::zeros(imgRows, imgCols, CV_8UC1);
    
    blur(src,src,Size(3,3));
    Canny(src,dst,0,100);
    imshow("CannyImg:",dst);
    
}
