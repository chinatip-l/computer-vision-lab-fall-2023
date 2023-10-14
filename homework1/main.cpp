#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string.h>

#define MAX_LEN 250
#define BASE_PATH "/home/chinatip/work/computer_vision/homework1"
#define SRC_PATH "test_img"
#define OUT_PATH "output"
#define FILENAME "taipei101"
// #define FILENAME "aeroplane"
// #define FILENAME "test"
#define OUT_SUFX "Q1_BW"
#define EXT "png"




using namespace cv;

Mat canvas;


void appendImgToCanvas(Mat);
Mat applyGreyscaleFilter(Mat);
Mat applyConvolve(Mat, Mat);
Mat applyConvolve(Mat, Mat,int,int);
Mat applyMaxPooling(Mat ,int , int , int );
Mat applyBinarisation(Mat ,int);


int main(int argc, char** argv )
{

    Mat og_img;
    char og_file[MAX_LEN]="",bw_file[MAX_LEN]="";
    snprintf(og_file,MAX_LEN,"%s/%s/%s.%s",BASE_PATH,SRC_PATH,FILENAME,EXT);
    snprintf(bw_file,MAX_LEN,"%s/%s/%s_%s.%s",BASE_PATH,OUT_PATH,FILENAME,OUT_SUFX,EXT);
    og_img = imread(og_file);
    if ( !og_img.data )
    {
        printf("No image data \n");
        return -1;
    }
    else{
        printf("Img Size: w x h %d x %d\n",og_img.cols,og_img.rows);
        
        
        
    }
    Mat bw_img(og_img.size(),og_img.type());
    for(int i = 0; i < og_img.rows; i++)
    {
        for(int j = 0; j < og_img.cols; j++)
        {
            Vec3b bgrPixel = og_img.at<Vec3b>(i, j);
            int b=(bgrPixel[0]+bgrPixel[1]+bgrPixel[2])/3;
            bgrPixel[0]=b;
            bgrPixel[1]=b;
            bgrPixel[2]=b;
            bw_img.at<Vec3b>(i,j)=bgrPixel;
            // do something with BGR values...
        }
    }
    printf("BW Img Size: w x h %d x %d\n",bw_img.cols,bw_img.rows);

    // printf("Size: %d\n",og_img.dims);
    imwrite(bw_file,bw_img);
    int raw_kernel[]={  -1,-1,-1,\
                        -1,8,-1,\
                        -1,-1,-1};
    Mat kernel(3,3,CV_32S,raw_kernel);
    Mat edge;
    edge=applyConvolve(bw_img,kernel);
    printf("Edge Img Size: w x h %d x %d\n",edge.cols,edge.rows);
    
    Mat pooled;
    pooled=applyMaxPooling(edge,2,2,2);
    Mat bin;
    bin=applyBinarisation(pooled,20);
    
    namedWindow("Original", WINDOW_AUTOSIZE );
    // showMultipleImages("Original",og_img,bw_img);
    
    // appendImgToCanvas(bw_img);
    appendImgToCanvas(og_img);
    appendImgToCanvas(bw_img);
    // imwrite("OUT.png",res);
    
    appendImgToCanvas(edge);
    appendImgToCanvas(pooled);
    appendImgToCanvas(bin);
    imshow("Original", canvas);
    waitKey(0);
    return 0;
}

void appendImgToCanvas(Mat img){
    if(canvas.empty()){
        canvas=img;
    }else{
        Size s(canvas.cols+img.cols+5,canvas.rows);
        size_t old_w=canvas.cols;
        copyMakeBorder(canvas,canvas,0,0,0,img.cols+5,BORDER_CONSTANT,Scalar(0,0,0,0));
        // canvas.resize(canvas.cols+img.cols+5);
        img.copyTo(canvas(Rect(old_w+5,0,img.cols,img.rows)));
        printf("%d\n",canvas.cols);

    }
}


Mat applyConvolve(Mat img,Mat kernel){
    int padding=1,stride=1;
    return applyConvolve(img,kernel,padding,stride);
}

Mat applyConvolve(Mat img,Mat kernel,int padding, int stride){

    int ker_x_offset=kernel.cols/2;
    int ker_y_offset=kernel.rows/2;
    int ker_w=kernel.cols;
    int ker_h=kernel.rows;
    int out_w=((img.cols+(2*padding)-kernel.cols)/stride)+1;
    int out_h=((img.rows+(2*padding)-kernel.rows)/stride)+1;



    Mat padded;
    // Mat ker_scalar;
    copyMakeBorder(img,padded,padding,padding,padding,padding,BORDER_CONSTANT,Scalar(0,0,0,0));
    // kernel.convertTo(ker_scalar,Scalar);

    Mat img_res=Mat::zeros(out_h,out_w,CV_32SC3);
    Vec3i res(0,0,0);
    for(int img_j=1;img_j<=img.rows;img_j+=stride){
        for(int img_i=1;img_i<=img.cols;img_i+=stride){
            Mat windowed=padded(Rect(img_i-ker_x_offset,img_j-ker_y_offset,ker_w,ker_h));
            // printf("%d %d %d %d\n",windowed.cols,windowed.rows,kernel.cols,kernel.rows);
            // multiply(windowed,kernel,res);
            
            // int ker_x_offset=kernel.cols/2,ker_y_offset=kernel.rows/2;
            res=Vec3i(0,0,0);
            for(int ker_j=0;ker_j<kernel.rows;ker_j++){
                for(int ker_i=0;ker_i<kernel.cols;ker_i++){
                    // int loc_img_i=img_i+ker_i-ker_x_offset;
                    // int loc_img_j=img_j+ker_j-ker_y_offset;
                    // tmp=safeAccess(padded,loc_img_i,loc_img_j)*kernel.at<int>(ker_i,ker_j);
                    // res+=windowed.at<Vec3i>(ker_i,ker_j)*kernel.at<int>(ker_i,ker_j);
                    Vec3b t=windowed.at<Vec3b>(ker_i,ker_j);
                    int k=kernel.at<int>(ker_i,ker_j);
                    for(int c=0;c<3;c++){
                        res[c]+=t[c]*k;
                    }
                    
                    // res+=tmp;
                    // if(img_i<500 && img_i>480 && img_j==400){
                        // printf("img %d,%d %d   ",img_i+ker_i-ker_x_offset,img_j+ker_j-ker_y_offset,tmp[0]);

                    // }
                    
                
                }
            }
            
            
            img_res.at<Vec3i>((img_j-1)/stride,(img_i-1)/stride)=res;


        }
    }
    
    return img_res;
    
    

}

Mat applyMaxPooling(Mat img,int kernel_w, int kernel_h, int stride){
    int pad_w=0,pad_h=0;
    if (img.cols%kernel_w>0)
    {
        pad_w=kernel_w-(img.cols%kernel_w);
    }
    if (img.rows%kernel_h>0)
    {
        pad_h=kernel_h-(img.rows%kernel_h);
    }
    
    Mat padded;
    // Mat ker_scalar;
    copyMakeBorder(img,padded,0,pad_h,0,pad_w,BORDER_CONSTANT,Scalar(0,0,0,0));
    int out_w=((padded.cols-kernel_w)/stride)+1;
    int out_h=((padded.rows-kernel_h)/stride)+1;



    Mat img_res=Mat::zeros(out_h,out_w,CV_32SC3);
    
    Vec3i res(0,0,0);
    for(int img_j=0;img_j<img.rows;img_j+=stride){
        for(int img_i=0;img_i<img.cols;img_i+=stride){
            Mat windowed=padded(Rect(img_i,img_j,kernel_w,kernel_h));
            res=Vec3i(0,0,0);
            for(int ker_j=0;ker_j<kernel_h;ker_j++){
                
                for(int ker_i=0;ker_i<kernel_w;ker_i++){

                    Vec3i t=windowed.at<Vec3i>(ker_i,ker_j);
                    if(t[0]>res[0]){
                        res[0]=t[0];
                    }
                    if(t[1]>res[1]){
                        res[1]=t[1];
                    }
                    if(t[2]>res[2]){
                        res[2]=t[2];
                    }
                    // if(img_i<500 && img_i>480 && img_j==400){
                        // printf("img %d,%d %d\n",img_i+ker_i,img_j+ker_j,res[0]);

                    // }
                
                }
            }
            
            img_res.at<Vec3i>((img_j-1)/stride,(img_i-1)/stride)=res;

        }
    }
    // imshow("Original",img_res);
    // waitKey(0);
    return img_res;
    
    

}

Mat applyBinarisation(Mat img,int thres){


    Mat img_res=Mat::zeros(img.rows,img.cols,CV_32SC3);
    Vec3i res = Vec3i(0);
    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            res=Vec3i(0,0,0);
            Vec3i bgrPixel = img.at<Vec3i>(i, j);
            for(int c=0;c<3;c++){
                res[c]=bgrPixel[c]>thres?255:0;
            }
            img_res.at<Vec3i>(i,j)=res;
        }
    }
    return img_res;
}