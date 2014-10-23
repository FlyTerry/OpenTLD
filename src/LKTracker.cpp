#include <LKTracker.h>
using namespace cv;

/*
 * 金字塔LK光流法跟踪
 * Media Flow中值光流跟踪和跟踪错误检测
 */
LKTracker::LKTracker(){//初始化成员变量

  term_criteria = TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);//三个参数分别是：类型，迭代的最大次数，阈值
  window_size = Size(4,4);
  level = 5;
  lambda = 0.5;
}


  /*
   * 从t时刻的图像的A点，跟踪到t+1时刻的图像B点；然后倒回来，从t+1时刻的图像的B点往回跟踪
   * 假如跟踪到t时刻的图像的C点，这样就产生了前向和后向两个轨迹，比较t时刻中 A点 和 C点 的距离，
   * 如果距离小于阈值FB-error，那么就认为前向跟踪是正确的
   */
bool LKTracker::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
  //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
  //Forward-Backward tracking
  
  /*
   * calcOpticalFlowPyrLK(imgA, imgB, pyrA,pyrB,featuresA,featuresB, count, 
   *						winsize,level,status,track_error,criteria,flags)
   * 是opencv中计算一个稀疏特征集的光流(LK金字塔)的函数
   * 其中imgA,imgB是两帧相临的图像
   * pyrA,pyrB是存放两副图像的金字塔的缓存，至少(img.width+8)*img.height/3
   * featuresA是用于寻找运动的点，featuresB存放featuresA中点的新位置
   * count是featuresA中点的数目
   * 其他参数是一致的了就不一一注解了...
   */
  calcOpticalFlowPyrLK( img1,img2, points1, points2, status, similarity, window_size, level, term_criteria, lambda, 0);//forward trajectory 前向轨迹跟踪

  calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status, FB_error, window_size, level, term_criteria, lambda, 0);////backward trajectory 后向轨迹跟踪

  //Compute the real FB-error
  ///计算前向与后向轨迹的误差
  for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);
  }

  //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
  normCrossCorrelation(img1,img2,points1,points2);
  return filterPts(points1,points2);
}


/*
 * 利用NCC把跟踪预测的结果周围取10*10的小片与原始位置周围10*10的小片（使用函数getRectSubPix得到）
 * 进行模板匹配(调用opencv的matchTemplate方法)
 */
void LKTracker::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
        Mat rec0(10,10,CV_8U);
        Mat rec1(10,10,CV_8U);
        Mat res(1,1,CV_32F);

        for (int i = 0; i < points1.size(); i++) {
                if (status[i] == 1) {//假如特征点追踪成功
                        getRectSubPix( img1, Size(10,10), points1[i],rec0 );//提取10*10的像素块
                        getRectSubPix( img2, Size(10,10), points2[i],rec1);

						//CV_TM_CCOEFF_NORMED 归一化相关系数匹配法  
						//参数分别为：欲搜索的图像，搜索模板，比较结果的映射图像，指定匹配方法  
                        matchTemplate( rec0, rec1, res, CV_TM_CCOEFF_NORMED);//匹配前一帧和当前帧中提取的小块，得到匹配后的映射图像
                        similarity[i] = ((float *)(res.data))[0];//得到各个特征点的相似度大小

                } else {
                        similarity[i] = 0.0;
                }
        }
        rec0.release();
        rec1.release();
        res.release();
}

/*
 * 筛选出 FB_error[i] <= median(FB_error) 和 sim_error[i] > median(sim_error) 的特征点
 * 得到NCC和FB error结果的中值，分别去掉中值一半的跟踪结果不好的点  
 */
bool LKTracker::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
  //Get Error Medians
  simmed = median(similarity);
  size_t i, k;
  for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
          continue;
        if(similarity[i]> simmed){//筛选
          points1[k] = points1[i];
          points2[k] = points2[i];
          FB_error[k] = FB_error[i];
          k++;
        }
    }
  if (k==0)
    return false;
  points1.resize(k);
  points2.resize(k);
  FB_error.resize(k);

  fbmed = median(FB_error);
  for( i=k = 0; i<points2.size(); ++i ){
      if( !status[i])
        continue;
      if(FB_error[i] <= fbmed){//对上一步剩下的特征点进一步筛选
        points1[k] = points1[i];
        points2[k] = points2[i];
        k++;
      }
  }
  points1.resize(k);
  points2.resize(k);
  if (k>0)
    return true;
  else
    return false;
}


/*
 * old OpenCV style
void LKTracker::init(Mat img0, vector<Point2f> &points){
  //Preallocate
  //pyr1 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //pyr2 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //const int NUM_PTS = points.size();
  //status = new char[NUM_PTS];
  //track_error = new float[NUM_PTS];
  //FB_error = new float[NUM_PTS];
}


void LKTracker::trackf2f(..){
  cvCalcOpticalFlowPyrLK( &img1, &img2, pyr1, pyr1, points1, points2, points1.size(), window_size, level, status, track_error, term_criteria, CV_LKFLOW_INITIAL_GUESSES);
  cvCalcOpticalFlowPyrLK( &img2, &img1, pyr2, pyr1, points2, pointsFB, points2.size(),window_size, level, 0, 0, term_criteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY );
}
*/

