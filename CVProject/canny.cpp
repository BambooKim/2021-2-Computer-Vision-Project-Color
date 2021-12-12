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

// scale 보정을 위해 곱해줄 값
#define pro 5.0

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

// Double Thresholding을 통해 Canny Edge Detection 수행.
Mat cannyEdgeDouble(Mat* mat, double threshold_high, double threshold_low) {
    Mat canny = (*mat).clone();
    canny = nonMaxSuppress(&canny);
    Mat thin = canny.clone();
    
    // z는 0: Blue, 1:Green, 2:Red
    for (int z = 0; z < 3; z++) {
        // 약한 edge들을 조사하기 위한 중간 도구 Stack
        stack<pair<int, int>> Stack;
        
        int width = canny.size().width;
        int height = canny.size().height;
        
        // 중앙 픽셀의 주변 8개 픽셀을 조사하기 위해
        int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
        
        // 약한 edge를 조사할 때 이미 조사한 픽셀은 건너뛰기 위해
        bool** check = new bool*[height];
        for (int i = 0; i < height; i++) {
            check[i] = new bool[width];
        }
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                // Non-Max Suppress된 이미지의 값들.
                double data = thin.at<Vec3b>(x, y)[z];
                
                // high threshold와 비교하여 큰 값은 살리고 작으면 날린다.
                if (data >= threshold_high) {
                    canny.at<Vec3b>(x, y)[z] = data * pro;
                    // 스케일 보정을 위해 임의의 값을 곱한다.
                    
                    check[y][x] = true;
                    // 강한 edge는 나중에 조사할 필요가 없으므로 이미 확인되었다고 표시.
                    
                    Stack.push(make_pair(x, y));
                    // 스택에 해당 픽셀의 좌표를 넣는다.
                } else {
                    canny.at<Vec3b>(x, y)[z] = 0;
                }
            }
        }
        
        // 스택이 빌 때까지 반복.
        while (!Stack.empty()) {
            int x = Stack.top().first;
            int y = Stack.top().second;
            Stack.pop();
            check[y][x] = true;
            
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                // nx, ny는 조사할 8개 픽셀의 좌표. 해당 좌표가 인덱스 범위에 해당하는지 확인.
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    if (!check[ny][nx]) {
                        double nMag = magnitudes[z][ny][nx];
                        
                        // 조사할 픽셀이 low보다 높고 high보다 작다면
                        if (threshold_low <= nMag && nMag <= threshold_high) {
                            // 스택에 또 넣는 이유는, 약한 edge와 이어져 있는 약한 edge들도 조사해야 하기 때문.
                            Stack.push(make_pair(nx, ny));
                            canny.at<Vec3b>(nx, ny) = nMag * pro;
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

// Triple Thresholding을 통해 Canny Edge Detection 수행.
Mat cannyEdgeTriple(Mat* mat, double threshold_high, double threshold_mid, double threshold_low) {
    Mat canny = (*mat).clone();
    canny = nonMaxSuppress(&canny);
    Mat thin = canny.clone();

    // z는 0: Blue, 1:Green, 2:Red
    for (int z = 0; z < 3; z++) {
        // 중간 edge들과 약한 edge들을 조사하기 위해.
        stack<pair<int, int>> Stack;
        stack<pair<int, int>> _Stack;
        
        int width = canny.size().width;
        int height = canny.size().height;
        
        // 중간 edge와 약한 edge를 조사할 때 중복을 방지하기 위해.
        bool** check = new bool*[height];
        bool** _check = new bool*[height];
        for (int i = 0; i < height; i++) {
            check[i] = new bool[width];
            _check[i] = new bool[width];
        }
        
        // 중앙 픽셀의 주변 8개 픽셀을 조사하기 위해
        int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                // Non-Max Suppress에서 살아남은 값들.
                double data = thin.at<Vec3b>(x, y)[z];
                
                if (data >= threshold_high) {
                    canny.at<Vec3b>(x, y)[z] = data * pro;
                    
                    // 중간 edge 조사시 중복 방지용
                    check[y][x] = true;
                    // 약한 edge 조사시 중복 방지용
                    _check[y][x] = true;
                    
                    // 중간 edge들을 조사하기 위해 강한 edge를 Stack에 넣는다.
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
                // nx, ny는 조사할 8개 픽셀의 좌표. 해당 좌표가 인덱스 범위에 해당하는지 확인.
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    if (!check[ny][nx]) {
                        double nMag = magnitudes[z][ny][nx];
                        
                        // 조사할 픽셀이 mid와 high 사이라면
                        if (threshold_mid <= nMag && nMag <= threshold_high) {
                            // 중간edge와 연속인 중간 edge 조사용.
                            Stack.push(make_pair(nx, ny));
                            // 중간edge와 연속인 약한 edge 조사용.
                            _Stack.push(make_pair(nx, ny));
                            canny.at<Vec3b>(nx, ny)[z] = nMag * pro;
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
                // nx, ny는 조사할 8개 픽셀의 좌표. 해당 좌표가 인덱스 범위에 해당하는지 확인.
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    if (!_check[ny][nx]) {
                        double nMag = magnitudes[z][ny][nx];
                        
                        // 조사할 픽셀이 low와 mid 사이라면
                        if (threshold_low <= nMag && nMag <= threshold_mid) {
                            // 약한 edge와 연속인 약한 edge를 조사하기 위해.
                            _Stack.push(make_pair(nx, ny));
                            canny.at<Vec3b>(nx, ny)[z] = nMag * pro;
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

// Non-Maximum을 Suppress하는 함수.
Mat nonMaxSuppress(Mat* mat) {
    Mat src = *mat;
    Mat thin = src.clone();
    
    int width = src.size().width;
    int height = src.size().height;
    
    // 각 RGB들의 orientation을 담을 배열.
    // 전역 변수이며 2차원 배열로 동적 할당.
    BlueOrientation = new double*[height];
    GreenOrientation = new double*[height];
    RedOrientation = new double*[height];
    for (int i = 0; i < height; i++) {
        BlueOrientation[i] = new double[height];
        GreenOrientation[i] = new double[height];
        RedOrientation[i] = new double[height];
    }
    
    // 바로 위에서 동적 할당한 2차원 배열들을 가리키는
    // 3중 포인터 orientation.
    double*** orientation = new double**[3];
    orientation[0] = BlueOrientation;
    orientation[1] = GreenOrientation;
    orientation[2] = RedOrientation;
    
    // 각 RGB들의 x방향 gradient를 담은 배열.
    // main과 gradient에서 이미 동적 할당하고 계산되어 있음.
    double*** xConvolvMat = new double**[3];
    xConvolvMat[0] = xBlueConvolv;
    xConvolvMat[1] = xGreenConvolv;
    xConvolvMat[2] = xRedConvolv;
    
    // 각 RGB들의 y방향 gradient를 담은 배열.
    // main과 gradient에서 이미 동적 할당하고 계산되어 있음.
    double*** yConvolvMat = new double**[3];
    yConvolvMat[0] = yBlueConvolv;
    yConvolvMat[1] = yGreenConvolv;
    yConvolvMat[2] = yRedConvolv;
    
    // 각 RGB들의 gradient magnitude를 담은 배열
    // main과 gradient에서 이미 계산되어 있으며
    // magnitudes는 이들을 가리키는 3중 포인터.
    magnitudes = new double**[3];
    magnitudes[0] = BlueMagnitudes;
    magnitudes[1] = GreenMagnitudes;
    magnitudes[2] = RedMagnitudes;
    
    // z는 0: Blue, 1:Green, 2:Red
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 각 픽셀별로, 각 RGB별로 orientation을 구한다.
                orientation[z][y][x] = xConvolvMat[z][y][x] / yConvolvMat[z][y][x];
            }
        }

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double orient = orientation[z][y][x];
                double mag = magnitudes[z][y][x];
                
                // non-max를 suppress하기 위한 비교 대상.
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
                        // 0~45 사이
                        alpha = (magnitudes[z][y + 1][x + 1] + magnitudes[z][y][x + 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x - 1] + magnitudes[z][y][x - 1]) / 2.0;
                        
                    } else if (-1 < orient && orient < 0) {
                        // 135~180 사이
                        alpha = (magnitudes[z][y + 1][x - 1] + magnitudes[z][y][x - 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x + 1] + magnitudes[z][y][x + 1]) / 2.0;
                        
                    } else if (1 < orient) {
                        // 45 ~ 90 사이
                        alpha = (magnitudes[z][y + 1][x] + magnitudes[z][y + 1][x + 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x] + magnitudes[z][y - 1][x - 1]) / 2.0;
                        
                    } else {
                        // 90 ~ 135 사이
                        alpha = (magnitudes[z][y + 1][x + 1] + magnitudes[z][y][x + 1]) / 2.0;
                        beta = (magnitudes[z][y - 1][x - 1] + magnitudes[z][y][x - 1]) / 2.0;
                    }
                }
                
                if (mag >= alpha && mag >= beta) {
                    // 만약 local maximum이면 남겨둔다.
                    thin.at<Vec3b>(x, y)[z] = mag;
                } else {
                    // local maximum이 아니면 제거한다.
                    thin.at<Vec3b>(x, y)[z] = 0;
                }
            }
        }
    }

    return thin;
}
