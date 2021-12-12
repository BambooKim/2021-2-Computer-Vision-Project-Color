//
//  main.cpp
//  CVProject
//
//  Created by 김범구 on 2021/12/09.
//

#include <iostream>
#include "gradient.hpp"
#include "gaussian.hpp"
#include "canny.hpp"

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

void initGlobalArray(int width, int height) {
    xBlueConvolv = new double*[height];
    yBlueConvolv = new double*[height];
    BlueMagnitudes = new double*[height];
    xGreenConvolv = new double*[height];
    yGreenConvolv = new double*[height];
    GreenMagnitudes = new double*[height];
    xRedConvolv = new double*[height];
    yRedConvolv = new double*[height];
    RedMagnitudes = new double*[height];
    
    for (int i = 0; i < height; i++) {
        xBlueConvolv[i] = new double[width];
        yBlueConvolv[i] = new double[width];
        BlueMagnitudes[i] = new double[width];
        xGreenConvolv[i] = new double[width];
        yGreenConvolv[i] = new double[width];
        GreenMagnitudes[i] = new double[width];
        xRedConvolv[i] = new double[width];
        yRedConvolv[i] = new double[width];
        RedMagnitudes[i] = new double[width];
    }
}

void deleteGlobalArray(int height) {
    for (int i = 0; i < height; i++) {
        delete[] xBlueConvolv[i];
        delete[] yBlueConvolv[i];
        delete[] BlueMagnitudes[i];
        delete[] xGreenConvolv[i];
        delete[] yGreenConvolv[i];
        delete[] GreenMagnitudes[i];
        delete[] xRedConvolv[i];
        delete[] yRedConvolv[i];
        delete[] RedMagnitudes[i];
    }
    delete[] xBlueConvolv;
    delete[] yBlueConvolv;
    delete[] BlueMagnitudes;
    delete[] xGreenConvolv;
    delete[] yGreenConvolv;
    delete[] GreenMagnitudes;
    delete[] xRedConvolv;
    delete[] yRedConvolv;
    delete[] RedMagnitudes;
}

int main(int argc, const char * argv[]) {
    // 이미지 파일을 회색으로 읽어들인다. 해당 정보는 OpenCV의 Mat 구조체에 저장된다.
    string fileName;
    //cin >> fileName;
    fileName = "IMG_5624.jpeg";
    String filePath = "/Users/bambookim/Desktop/CVProject2/CVProject/" + fileName;
    Mat src = imread(filePath);
    
    imshow("asdf", src);
    waitKey();
    
    // 이미지의 가로, 세로, 채널 정보를 얻는다.
    int width = src.size().width;
    int height = src.size().height;
    int channel = src.channels();
    
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;
    cout << "channel: " << channel << endl;
    
    // 전역 변수 2차원 배열 xConvolvMat와 yConvolvMat를 동적 할당한다.
    initGlobalArray(width, height);
    
    Mat gausGradNon = src.clone();
    gausGradNon = gradFilter(&gausGradNon);
    gausGradNon = gaussianFilter(&gausGradNon, 25, 10);
    //gausGradNon = cannyEdgeDouble(&gausGradNon, 8, 2);
    gausGradNon = cannyEdgeTriple(&gausGradNon, 8, 7.5, 3);
    imshow("cannyTriple", gausGradNon);
    waitKey();

    imwrite("/Users/bambookim/Desktop/CVProject2/CVProject/result/color_triple_js_s10_h8_m7.5_l3.jpg", gausGradNon);
    
    // 동적 할당을 해제한다.
    deleteGlobalArray(height);
    
    return 0;
}

