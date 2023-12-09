#include "include/func.hpp"
#include <vector>

typedef struct node
{
    int32_t val;
    struct node* next;
    bool is_term;
}node_t;

node_t END={.val=0,.next=nullptr,.is_term=true};


Mat applyMeanFilter (Mat img, int kernel_size, int padding, int stride)
{
    // calculate kernel offset ie. number of pixel before and 
    // after since the target pixel will be the center of 
    // matrix,
    // in order to get the neighbour pixel with size of kernel
    // size, offset will be applied for both before, and after
    // the target pixel
    // example
    // width = 5 => offset = (int)5/2 = 2;
    // px   ... k   n-offset    ... n   ... n+offset    k
    int ker_x_offset = kernel_size / 2;
    int ker_y_offset = kernel_size / 2;

    // get kernel size
    int ker_w = kernel_size;
    int ker_h = kernel_size;

    // calculate output image size
    int out_w = ((img.cols + (2 * padding) - kernel_size) / stride) + 1;
    int out_h = ((img.rows + (2 * padding) - kernel_size) / stride) + 1;

    // create intermediate buffer and add padding, size is 
    // w+(2*padding),h+(2*padding) then copy the content 
    // of image to the buffer
    Mat padded;
    padded = Mat::zeros(img.rows + (2 * padding), img.cols + (2 * padding), CV_32SC3);
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols; i++)
        {
            padded.at<Vec3i>(j+padding,i+padding)=img.at<Vec3b>(j,i);
        }
    }
    
    // create output buffer with calculated output size
    // data type need to be vector of signed int 32 bit
    // in case negative data is possible, this can help
    // preserve information through the convolution process
    // and rasterise to 8 bit at the last step
    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);
    Vec3i res(0, 0, 0);
    Point2i start=Point2i(0,0);

    // iterate over the size of input image
    for (int out_j = 0; out_j < out_h; out_j++)
    {
        for (int out_i = 0; out_i < out_w; out_i++)
        {
            
            // calculate the start point to slice a windowed from 
            // the padded buffer
            start.x=(out_i*stride)+padding-ker_x_offset;
            start.y=(out_j*stride)+padding-ker_x_offset;

            // get windowed matrix from the padded buffer with 
            // the size of kernel size, and target pixel is in
            // the center of windowed matrix
            Mat windowed = padded(Rect(start.x, start.y, ker_w, ker_h));
            res = Vec3i(0, 0, 0);


            // item-wise multiply windowed matrix with kernel 
            // matrix and summarise 
            for (int ker_j = 0; ker_j < kernel_size; ker_j++)
            {
                for (int ker_i = 0; ker_i < kernel_size; ker_i++)
                {
                    Vec3b t = windowed.at<Vec3i>(ker_i, ker_j);
                    
                    // add to buffer
                    res+=t;
                }
            }

            for (int c = 0; c < 3; c++)
            {
                // divide by the number of elements
                res[c] /= (kernel_size*kernel_size);
            }

            // write to output buffer
            img_res.at<Vec3i>(out_j,out_i) = res;
            
        }
    }

    // return the output image
    return img_res;
}

Mat applyMedianFilter (Mat img, int kernel_size, int padding, int stride)
{
    // calculate kernel offset ie. number of pixel before and 
    // after since the target pixel will be the center of 
    // matrix,
    // in order to get the neighbour pixel with size of kernel
    // size, offset will be applied for both before, and after
    // the target pixel
    // example
    // width = 5 => offset = (int)5/2 = 2;
    // px   ... k   n-offset    ... n   ... n+offset    k
    int ker_x_offset = kernel_size / 2;
    int ker_y_offset = kernel_size / 2;

    // get kernel size
    int ker_w = kernel_size;
    int ker_h = kernel_size;

    // calculate output image size
    int out_w = ((img.cols + (2 * padding) - kernel_size) / stride) + 1;
    int out_h = ((img.rows + (2 * padding) - kernel_size) / stride) + 1;

    // create intermediate buffer and add padding, size is 
    // w+(2*padding),h+(2*padding) then copy the content 
    // of image to the buffer
    Mat padded;
    padded = Mat::zeros(img.rows + (2 * padding), img.cols + (2 * padding), CV_32SC3);
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols; i++)
        {
            padded.at<Vec3i>(j+padding,i+padding)=img.at<Vec3b>(j,i);
        }
    }
    
    // create output buffer with calculated output size
    // data type need to be vector of signed int 32 bit
    // in case negative data is possible, this can help
    // preserve information through the convolution process
    // and rasterise to 8 bit at the last step
    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);
    Vec3i res(0, 0, 0);
    Point2i start=Point2i(0,0);

    // iterate over the size of input image
    for (int out_j = 0; out_j < out_h; out_j++)
    {
        for (int out_i = 0; out_i < out_w; out_i++)
        {

            
            // calculate the start point to slice a windowed from 
            // the padded buffer
            start.x=(out_i*stride)+padding-ker_x_offset;
            start.y=(out_j*stride)+padding-ker_x_offset;

            // get windowed matrix from the padded buffer with 
            // the size of kernel size, and target pixel is in
            // the center of windowed matrix
            Mat windowed = padded(Rect(start.x, start.y, ker_w, ker_h));
            res = Vec3i(0, 0, 0);

            // create buffer lists for each channel
            std::vector<int32_t> b,g,r;


            // item-wise split channel data             
            for (int ker_j = 0; ker_j < kernel_size; ker_j++)
            {
                for (int ker_i = 0; ker_i < kernel_size; ker_i++)
                {
                    Vec3b t = windowed.at<Vec3i>(ker_i, ker_j);
                
                    // add each channel to its buffer list
                    b.push_back(t[0]);
                    g.push_back(t[1]);
                    r.push_back(t[2]);
                }
            }
            
            // apply sort function to each channel
            b=sort(b);
            r=sort(r);
            g=sort(g);

            // get the median ie. the middle item of the list
            // as a representative for each window
            int idx=b.size()/2;
            res[0]=b.at(idx);
            res[1]=g.at(idx);
            res[2]=r.at(idx);

            // write to output buffer
            img_res.at<Vec3i>(out_j,out_i) = res;
            
        }
    }

    // return the output image
    return img_res;
}

std::vector<int32_t> sort(std::vector<int32_t> list){
    // using insertion sort

    // initialise head of the list, and point to the end
    node_t head={.next=&END};
    node_t* cur;
    node_t* tmp;

    // for each item in list
    for(int i=0;i<list.size();i++){

        //first, set current node to head
        cur=&head;

        //create new node with current data
        tmp=(node_t*)malloc(sizeof(node_t));
        tmp->val=list.at(i);

        // traverse through linked list
        while(true){
            if(cur->next->is_term==true){
                // if reach end node, insert new node as
                // the last node of the linked list
                tmp->next=cur->next;
                cur->next=tmp;
                break;
            }else if(tmp->val<=cur->next->val){
                // if not the last node, compare value with
                // the next node, if lesser or equal to the 
                // next node, put new node at the current 
                // position and point to the next
                tmp->next=cur->next;
                cur->next=tmp;
                break;
            }
            // else, continue to next node
            cur=cur->next;
        }
    }

    //after sort, traverse through linked list and put back to vector
    cur=&head;
    cur=cur->next;
    std::vector<int32_t> res;
    while(!cur->is_term){
        res.push_back(cur->val);
        tmp=cur;
        cur=cur->next;
        free(tmp);
    }
    return res;
    
}