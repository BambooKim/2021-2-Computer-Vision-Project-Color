//
//  gradient.cpp
//  CVProject
//
//  Created by 김범구 on 2021/12/10.
//

#include "gradient.hpp"

#define none
#define PRODUCT 2

using namespace cv;
using namespace std;

double** xBlueConvolv;
double** yBlueConvolv;
double** BlueMagnitudes;

double** xGreenConvolv;
double** yGreenConvolv;
double** GreenMagnitudes;

double** xRedConvolv;
double** yRedConvolv;
double** RedMagnitudes;

// x방향의 gradient를 구하는 함수.
void gradXFilter(Mat* mat) {
    Mat src;
    (*mat).copyTo(src);
    
    int width = src.size().width;
    int height = src.size().height;
    
    // x convolution filter
    // gradient
    int x_kernel[3] = { 1, 0, -1 };
       
    int bmin = 255;
    int bmax = 0;
    int gmin = 255;
    int gmax = 0;
    int rmin = 255;
    int rmax = 0;
    
    for (int y = 1; y < height - 1 ; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            // x 방향의 부분 픽셀값.
            int x_img[3][3];
            for (int i = -1; i < 2; i++) {
                x_img[0][i + 1] = (int) src.at<Vec3b>(x + i, y)[0];
                x_img[1][i + 1] = (int) src.at<Vec3b>(x + i, y)[1];
                x_img[2][i + 1] = (int) src.at<Vec3b>(x + i, y)[2];
            }
            
            // x kernel과 x 방향의 부분 픽셀값 convolution.
            int xb_gra = 0;
            int xg_gra = 0;
            int xr_gra = 0;
            for (int i = 0; i < 3; i++) {
                xb_gra = xb_gra + x_img[0][i] * x_kernel[i];
                xg_gra = xg_gra + x_img[1][i] * x_kernel[i];
                xr_gra = xr_gra + x_img[2][i] * x_kernel[i];
            }
        
            // Mat 구조체에 저장하면 음수는 자동으로 값이 양수로 변하므로
            // 배열에 x방향의 gradient 값을 저장한다.
            xBlueConvolv[y][x] = xb_gra;
            xGreenConvolv[y][x] = xg_gra;
            xRedConvolv[y][x] = xr_gra;
    
            
            if (bmin > xb_gra)
                bmin = xb_gra;
            if (bmax < xb_gra)
                bmax = xb_gra;
            
            if (gmin > xg_gra)
                gmin = xg_gra;
            if (gmax < xg_gra)
                gmax = xg_gra;
            
            if (rmin > xg_gra)
                rmin = xg_gra;
            if (rmax < xg_gra)
                rmax = xg_gra;
        }
    }
      
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            // 시각화를 위해 음수를 양수로 만들어주기 위해
            // 그만큼 더해준 뒤 폭의 scaling을 위해
            // 절반으로 나눠준다.
            int absbMin = bmin < 0 ? -bmin : bmin;
            int absgMin = gmin < 0 ? -gmin : gmin;
            int absrMin = rmin < 0 ? -rmin : rmin;

#ifdef none
            int xb_gra = (xBlueConvolv[y][x] + absbMin) * PRODUCT;
            int xg_gra = (xGreenConvolv[y][x] + absgMin) * PRODUCT;
            int xr_gra = (xRedConvolv[y][x] + absrMin) * PRODUCT;
#endif
        
            // Mat 구조체에 보정된 gradient 값을 저장한다.
            src.at<Vec3b>(x, y)[0] = xb_gra;
            src.at<Vec3b>(x, y)[1] = xg_gra;
            src.at<Vec3b>(x, y)[2] = xr_gra;
        }
    }
}

// y 방향의 gradient를 구하고 화면에 보여주는 함수.
void gradYFilter(Mat* mat) {
    Mat src;
    (*mat).copyTo(src);
    
    int width = src.size().width;
    int height = src.size().height;
    
    // y convolution filter
    // gradient
    int y_kernel[3] = { 1, 0, -1 };
       
    int bmin = 255;
    int bmax = 0;
    int gmin = 255;
    int gmax = 0;
    int rmin = 255;
    int rmax = 0;
    
    for (int y = 1; y < height - 1 ; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            // y 방향의 부분 픽셀값.
            int y_img[3][3];
            for (int i = -1; i < 2; i++) {
                y_img[0][i + 1] = (int) src.at<Vec3b>(x, y + i)[0];
                y_img[1][i + 1] = (int) src.at<Vec3b>(x, y + i)[1];
                y_img[2][i + 1] = (int) src.at<Vec3b>(x, y + i)[2];
            }
            
            // y kernel과 y 방향의 부분 픽셀값 convolution.
            int yb_gra = 0;
            int yg_gra = 0;
            int yr_gra = 0;
            for (int i = 0; i < 3; i++) {
                yb_gra = yb_gra + y_img[0][i] * y_kernel[i];
                yg_gra = yg_gra + y_img[1][i] * y_kernel[i];
                yr_gra = yr_gra + y_img[2][i] * y_kernel[i];
            }
        
            // Mat 구조체에 저장하면 음수는 자동으로 값이 양수로 변하므로
            // 배열에 y방향의 gradient 값을 저장한다.
            yBlueConvolv[y][x] = yb_gra;
            yGreenConvolv[y][x] = yg_gra;
            yRedConvolv[y][x] = yr_gra;
    
            
            if (bmin > yb_gra)
                bmin = yb_gra;
            if (bmax < yb_gra)
                bmax = yb_gra;
            
            if (gmin > yg_gra)
                gmin = yg_gra;
            if (gmax < yg_gra)
                gmax = yg_gra;
            
            if (rmin > yg_gra)
                rmin = yg_gra;
            if (rmax < yg_gra)
                rmax = yg_gra;
        }
    }
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            // 시각화를 위해 음수를 양수로 만들어주기 위해
            // 그만큼 더해준 뒤 폭의 scaling을 위해
            // 절반으로 나눠준다.
            int absbMin = bmin < 0 ? -bmin : bmin;
            int absgMin = gmin < 0 ? -gmin : gmin;
            int absrMin = rmin < 0 ? -rmin : rmin;

#ifdef none
            int yb_gra = (yBlueConvolv[y][x] + absbMin) * PRODUCT;
            int yg_gra = (yGreenConvolv[y][x] + absgMin) * PRODUCT;
            int yr_gra = (yRedConvolv[y][x] + absrMin) * PRODUCT;
#endif
        
            // Mat 구조체에 보정된 gradient 값을 저장한다.
            src.at<Vec3b>(x, y)[0] = yb_gra;
            src.at<Vec3b>(x, y)[1] = yg_gra;
            src.at<Vec3b>(x, y)[2] = yr_gra;
        }
    }
}

// x방향과 y방향의 gradient를 가지고 magnitude를 구한다.
Mat gradFilter(Mat* mat) {
    gradXFilter(mat);
    gradYFilter(mat);
    // 위 두 함수가 수행되면 전역 변수에 gradient가 저장된다.
    
    Mat src = (*mat).clone();
    int width = src.size().width;
    int height = src.size().height;
    
    for (int x = 1; x < width - 1; x++) {
        for (int y = 1; y < height - 1; y++) {
            // x와 y의 gradient를 이용해 magnitude를 구한다.
            double bmagnitude = sqrt(xBlueConvolv[y][x] * xBlueConvolv[y][x] + yBlueConvolv[y][x] * yBlueConvolv[y][x]);
            double gmagnitude = sqrt(xGreenConvolv[y][x] * xGreenConvolv[y][x] + yGreenConvolv[y][x] * yGreenConvolv[y][x]);
            double rmagnitude = sqrt(xRedConvolv[y][x] * xRedConvolv[y][x] + yRedConvolv[y][x] * yRedConvolv[y][x]);
            
#ifdef none
            src.at<Vec3b>(x, y)[0] = bmagnitude;
            src.at<Vec3b>(x, y)[1] = gmagnitude;
            src.at<Vec3b>(x, y)[2] = rmagnitude;
#endif
            
            // 전역 변수 2차원 배열 magnitudes에 magnitude값을 저장한다.
            BlueMagnitudes[y][x] = bmagnitude;
            GreenMagnitudes[y][x] = gmagnitude;
            RedMagnitudes[y][x] = rmagnitude;
        }
    }
    
    return src;
}
