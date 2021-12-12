//
//  canny.cpp
//  CVProject
//
//  Created by 김범구 on 2021/12/11.
//

#include "canny.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>
#include <utility>

using namespace std;
using namespace cv;

extern double** xBlueConvolv;
extern double** yBlueConvolv;
extern double** BlueMagnitudes;

extern double** xGreenConvolv;
extern double** yGreenConvolv;
extern double** GreenMagnitudes;

extern double** xRedConvolv;
extern double** yRedConvolv;
extern double** RedMagnitudes;

double*** magnitudes;
double** BlueOrientation;
double** GreenOrientation;
double** RedOrientation;

Mat cannyEdgeDouble(Mat* mat, double threshold_high, double threshold_low) {
    Mat canny = (*mat).clone();
    canny = nonMaxSuppress(&canny);
    Mat thin = canny.clone();
    
    for (int z = 0; z < 3; z++) {
        stack<pair<int, int>> Stack;
        
        int width = canny.size().width;
        int height = canny.size().height;
        
        int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
        
        bool** check = new bool*[height];
        for (int i = 0; i < height; i++) {
            check[i] = new bool[width];
        }
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double data = thin.at<Vec3b>(x, y)[z];
                if (data >= threshold_high) {
                    canny.at<Vec3b>(x, y)[z] = data * 5;
                    check[y][x] = true;
                    Stack.push(make_pair(x, y));
                } else {
                    canny.at<Vec3b>(x, y)[z] = 0;
                }
            }
        }
        
        while (!Stack.empty()) {
            int x = Stack.top().first;
            int y = Stack.top().second;
            Stack.pop();
            check[y][x] = true;
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    if (!check[ny][nx]) {
                        double nMag = magnitudes[z][ny][nx];
                        if (threshold_low <= nMag && nMag <= threshold_high) {
                            //check[ny][nx] = true;
                            Stack.push(make_pair(nx, ny));
                            canny.at<Vec3b>(nx, ny) = nMag * 5;
                        }
                    }
                }
                
            }
        }
        
        for (int i = 0; i < height; i++) {
            delete[] check[i];
        }
        delete[] check;
    }
    
    return canny;
}


Mat cannyEdgeTriple(Mat* mat, double threshold_high, double threshold_mid, double threshold_low) {
    Mat canny = (*mat).clone();
    canny = nonMaxSuppress(&canny);
    Mat thin = canny.clone();

    for (int z = 0; z < 3; z++) {
        stack<pair<int, int>> Stack;
        stack<pair<int, int>> _Stack;
        
        int width = canny.size().width;
        int height = canny.size().height;
        
        bool** check = new bool*[height];
        bool** _check = new bool*[height];
        for (int i = 0; i < height; i++) {
            check[i] = new bool[width];
            _check[i] = new bool[width];
        }
        
        int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double data = thin.at<Vec3b>(x, y)[z];
                if (data >= threshold_high) {
                    canny.at<Vec3b>(x, y)[z] = data * 7;
                    check[y][x] = true;
                    _check[y][x] = true;
                    Stack.push(make_pair(x, y));
                } else {
                    canny.at<Vec3b>(x, y)[z] = 0;
                }
            }
        }
        
        while (!Stack.empty()) {
            int x = Stack.top().first;
            int y = Stack.top().second;
            Stack.pop();
            check[y][x] = true;
            _check[y][x] = true;
            
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    if (!check[ny][nx]) {
                        double nMag = magnitudes[z][ny][nx];
                        if (threshold_mid <= nMag && nMag <= threshold_high) {
                            //check[ny][nx] = true;
                            //_check[ny][nx] = true;
                            Stack.push(make_pair(nx, ny));
                            _Stack.push(make_pair(nx, ny));
                            canny.at<uchar>(nx, ny) = nMag * 7;
                        }
                    }
                }
            }
        }
        
        while (!_Stack.empty()) {
            int x = _Stack.top().first;
            int y = _Stack.top().second;
            _Stack.pop();
            _check[y][x] = true;
            
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    if (!_check[ny][nx]) {
                        double nMag = magnitudes[z][ny][nx];
                        if (threshold_low <= nMag && nMag <= threshold_mid) {
                            // _check[ny][nx] = true;
                            _Stack.push(make_pair(nx, ny));
                            canny.at<uchar>(nx, ny) = nMag * 7;
                        }
                    }
                }
            }
        }
        
        for (int i = 0; i < height; i++) {
            delete[] check[i];
            delete[] _check[i];
        }
        delete[] check;
        delete[] _check;
    }
    
    return canny;
}


Mat nonMaxSuppress(Mat* mat) {
    Mat src = *mat;
    Mat thin = src.clone();
    
    int width = src.size().width;
    int height = src.size().height;
    
    //double orientation[width][height]
    BlueOrientation = new double*[height];
    GreenOrientation = new double*[height];
    RedOrientation = new double*[height];
    for (int i = 0; i < height; i++) {
        BlueOrientation[i] = new double[height];
        GreenOrientation[i] = new double[height];
        RedOrientation[i] = new double[height];
    }
    
    double*** orientation = new double**[3];
    orientation[0] = BlueOrientation;
    orientation[1] = GreenOrientation;
    orientation[2] = RedOrientation;
    
    double*** xConvolvMat = new double**[3];
    xConvolvMat[0] = xBlueConvolv;
    xConvolvMat[1] = xGreenConvolv;
    xConvolvMat[2] = xRedConvolv;
    
    double*** yConvolvMat = new double**[3];
    yConvolvMat[0] = yBlueConvolv;
    yConvolvMat[1] = yGreenConvolv;
    yConvolvMat[2] = yRedConvolv;
    
    magnitudes = new double**[3];
    magnitudes[0] = BlueMagnitudes;
    magnitudes[1] = GreenMagnitudes;
    magnitudes[2] = RedMagnitudes;
    
    
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                orientation[z][y][x] = xConvolvMat[z][y][x] / yConvolvMat[z][y][x];
            }
        }

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double orient = orientation[z][y][x];
                double mag = magnitudes[z][y][x];
                double alpha, beta;
                
                if (orient == 0.0) {
                    // 0도 / 180도
                    alpha = magnitudes[z][y][x - 1];
                    beta = magnitudes[z][y][x + 1];
                } else if (!isfinite(orient)) {
                    // 90도 / 270도
                    alpha = magnitudes[z][y + 1][x];
                    beta = magnitudes[z][y - 1][x];
                } else if (orient == 1.0) {
                    // 45도 / 225도
                    alpha = magnitudes[z][y - 1][x - 1];
                    beta = magnitudes[z][y + 1][x + 1];
                } else if (orient == -1.0) {
                    // 135도 / 315도
                    alpha = magnitudes[z][y - 1][x + 1];
                    beta = magnitudes[z][y + 1][x - 1];
                } else {
                    // 그 외 interpolation
                    if (0 < orient && orient < 1) {
                        // 0~45
                        alpha = (magnitudes[z][y + 1][x + 1] + magnitudes[z][y][x + 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x - 1] + magnitudes[z][y][x - 1]) / 2.0;
                        
                    } else if (-1 < orient && orient < 0) {
                        // 135~180
                        alpha = (magnitudes[z][y + 1][x - 1] + magnitudes[z][y][x - 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x + 1] + magnitudes[z][y][x + 1]) / 2.0;
                        
                    } else if (1 < orient) {
                        // 45 ~ 90
                        alpha = (magnitudes[z][y + 1][x] + magnitudes[z][y + 1][x + 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x] + magnitudes[z][y - 1][x - 1]) / 2.0;
                        
                    } else {//if (-1 > orient) {
                        // 90 ~ 135
                        alpha = (magnitudes[z][y + 1][x + 1] + magnitudes[z][y][x + 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x - 1] + magnitudes[z][y][x - 1]) / 2.0;
                        
                    }
                }
                
                if (mag >= alpha && mag >= beta) {
                    thin.at<Vec3b>(x, y)[z] = mag;
                } else {
                    thin.at<Vec3b>(x, y)[z] = 0;
                }
            }
        }
    }

    return thin;
}
