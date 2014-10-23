#include <tld_utils.h>
using namespace cv;
using namespace std;

void drawBox(Mat& image, CvRect box, Scalar color, int thick){
  rectangle( image, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),color, thick);
} 

/*
 *函数 cvRound, cvFloor, cvCeil 用一种舍入方法将输入浮点数转换成整数.
 *cvRound 返回和参数最接近的整数值。 cvFloor 返回不大于参数的最大整数值.
 *cvCeil 返回不小于参数的最小整数值。
 */
void drawPoints(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i )
      {
		Point center( cvRound(i->x ), cvRound(i->y));//据说是初始化x = i,y = i 
		circle(image,*i,2,color,1);
      }
}

//画了一个圈面~
Mat createMask(const Mat& image, CvRect box){
  Mat mask = Mat::zeros(image.rows,image.cols,CV_8U);
  drawBox(mask,box,Scalar::all(255),CV_FILLED);
  return mask;
}

//用在中值流算法中，寻找中值
float median(vector<float> v)
{
    int n = floor(v.size() / 2);
    nth_element(v.begin(), v.begin()+n, v.end());//寻找中值
    return v[n];
}

//index_shuffle()用于产生指定范围[begin:end]的随机数，返回随机数数组
vector<int> index_shuffle(int begin,int end){
  vector<int> indexes(end-begin);
  for (int i=begin;i<end;i++){
    indexes[i]=i;
  }
  random_shuffle(indexes.begin(),indexes.end());//STL中的random_shuffle()算法
  return indexes;
}

