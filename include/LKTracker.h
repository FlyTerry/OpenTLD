#include<tld_utils.h>
#include <opencv2/opencv.hpp>

//金字塔LK光流法追踪，调用Opencv的calcOpticalFlowPyrLK()函数，所以蛮多是定义函数的参数
class LKTracker{
private:
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size;//每个金字塔层的搜索窗口尺寸
  int level;//金字塔的最大层数
  std::vector<uchar> status;//对应特征的光流被发现，status内均为1,否则为0
  std::vector<uchar> FB_status;
  std::vector<float> similarity;//相似度

  /*
   * Forward-Backward error方法，作者在他的文章
   * Forward-Backward Error：Automatic Detection of Tracking Failures里
   * 提到用FB+NCC（交叉验证）的方案，可以使跟踪的结果最佳
   */
  std::vector<float> FB_error;//求FB_error的结果与原始位置的距离  
							  //把距离过大的跟踪结果舍弃  
  float simmed;
  float fbmed;
  
  /* 
   * TermCriteria模板类，作为迭代算法的终止条件(以前是CvTermCriteria)
   * 该类变量需要3个参数，第一个是类型，第二个参数为迭代的最大次数，第三个是特定的阈值
   * 指定在每个金字塔层，为某点寻找光流的迭代过程的终止条件
   */
  cv::TermCriteria term_criteria;

  float lambda;//阈值？

  /*
   * NCC 归一化交叉相关，FB error与NCC结合，使跟踪更稳定
   * 交叉相关的图像的匹配算法，作用是进行云团移动的短时预测
   * 选取连续两个时次的云图，将云图区域划分为32×32像素
   * 的图像子集，采用交叉相关法计算获取两幅云图的最佳匹配区域，
   * 根据前后云图匹配区域的位置和时间间隔，确定出每个图像子集的移动矢量（速度和方向），
   * 并对图像子集的移动矢量进行客观分析，其后，基于检验后的云图移动矢量集，
   * 利用后向轨迹方法对云图作时外推预测
   */
  void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, 
		                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);

  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
  LKTracker();

  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);//追踪特征点
  float getFB(){return fbmed;}
};

