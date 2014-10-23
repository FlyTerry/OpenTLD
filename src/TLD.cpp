/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */

#include <TLD.h>
#include <stdio.h>
using namespace cv;
using namespace std;


TLD::TLD(){
}
TLD::TLD(const FileNode& file){
  read(file);
}

void TLD::read(const FileNode& file){
  ///Bounding Box Parameters
  min_win = (int)file["min_win"];
  ///Genarator Parameters
  //initial parameters for positive examples
  patch_size = (int)file["patch_size"];
  num_closest_init = (int)file["num_closest_init"];
  num_warps_init = (int)file["num_warps_init"];
  noise_init = (int)file["noise_init"];
  angle_init = (float)file["angle_init"];
  shift_init = (float)file["shift_init"];
  scale_init = (float)file["scale_init"];
  //update parameters for positive examples
  num_closest_update = (int)file["num_closest_update"];
  num_warps_update = (int)file["num_warps_update"];
  noise_update = (int)file["noise_update"];
  angle_update = (float)file["angle_update"];
  shift_update = (float)file["shift_update"];
  scale_update = (float)file["scale_update"];
  //parameters for negative examples
  bad_overlap = (float)file["overlap"];
  bad_patches = (int)file["num_patches"];
  classifier.read(file);
}

void TLD::init(const Mat& frame1,const Rect& box,FILE* bb_file){
  //bb_file = fopen("bounding_boxes.txt","w");
  //Get Bounding Boxes
    buildGrid(frame1,box); //此函数根据传入的box(目标边界框)在传入的图像frame1中构建全部的扫描窗口
						   //并计算重叠度  
    printf("Created %d bounding boxes\n",(int)grid.size());
  
  //Preparation
  //allocation
  //积分图像，用以计算2bitBP特征
  iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);
  iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
  
  // reserve是容器预留空间，但在空间内不真正创建元素对象
  dconf.reserve(100);
  dbb.reserve(100);
  bbox_step =7;

  //tmp.conf.reserve(grid.size());
  tmp.conf = vector<float>(grid.size());//grid.size()一幅图像中全部的扫描窗口个数
  tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
  //tmp.patt.reserve(grid.size());
  dt.bb.reserve(grid.size());
  good_boxes.reserve(grid.size());
  bad_boxes.reserve(grid.size());
  pEx.create(patch_size,patch_size,CV_64F);
  
  //Init Generator
  //背景最小值：0,背景最大值：0, 噪声范围：0 ~ noise_init, 是否模糊化：true
  //最小尺度：1-scale_init,  最大尺度：1+scale_init
  //最小角度：-angle_init*CV_PI/180, 最大角度：angle_init*CV_PI/180
  generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);
  
  getOverlappingBoxes(box,num_closest_init);//区分good_boxes和bad_boxes
  
  printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
  printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
  printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);
  
  //Correct Bounding Box
  lastbox=best_box;
  lastconf=1;
  lastvalid=true;
  //Print
  //新得到的BoundingBox(即便是空的的)写入文件
  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  
  //Prepare Classifier
  classifier.prepare(scales);//scales容器里是所有扫描窗口的尺度，由buildGrid()函数初始化  
  ///Generate Data
  // Generate positive data
  generatePositiveData(frame1,num_warps_init);
  // Set variance threshold
  Scalar stdev, mean;//统计best_box的标准差和均值

  meanStdDev(frame1(best_box),mean,stdev);//frame1(best_box)表示在frame1中提取best_box区域
										  //即目标区域的图像片
  integral(frame1,iisum,iisqsum);//用积分图像去计算每个待检测窗口的方差
  
  //级联分类器模块一：方差检测模块，利用积分图计算每个待检测窗口的方差，
  //方差大于var阈值(目标patch方差的50%)的则认为其含有前景目标方差；
  //var 为标准差的平方  
  var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);
  cout << "variance: " << var << endl;
  
  //check variance
  double vr =  getVar(best_box,iisum,iisqsum)*0.5;//getVar函数通过积分图像计算输入的best_box的方差
  cout << "check variance: " << vr << endl;
  
  // Generate negative data
  generateNegativeData(frame1);
  
  //Split Negative Ferns into Training and Testing sets (they are already shuffled)
  int half = (int)nX.size()*0.5f;
  nXT.assign(nX.begin()+half,nX.end());//将一半的负样本集作为测试集  
  nX.resize(half);
  
  ///Split Negative NN Examples into Training and Testing sets
  half = (int)nEx.size()*0.5f;
  nExT.assign(nEx.begin()+half,nEx.end());//然后将剩下的一半作为训练集  
  nEx.resize(half);

  //Merge Negative Data with Positive Data and shuffle it
  vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
  vector<int> idx = index_shuffle(0,ferns_data.size());
  int a=0;
  for (int i=0;i<pX.size();i++){
      ferns_data[idx[a]] = pX[i];
      a++;
  }
  for (int i=0;i<nX.size();i++){
      ferns_data[idx[a]] = nX[i];
      a++;
  }

  //Data already have been shuffled, just putting it in the same vector
  vector<cv::Mat> nn_data(nEx.size()+1);
  nn_data[0] = pEx;
  for (int i=0;i<nEx.size();i++){
      nn_data[i+1]= nEx[i];
  }
  ///Training
  //训练集合分类器和最近邻分类器
  classifier.trainF(ferns_data,2); //bootstrap = 2
  classifier.trainNN(nn_data);

  ///Threshold Evaluation on testing sets
  classifier.evaluateTh(nXT,nExT);
}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps){
  Scalar mean;//均值
  Scalar stdev;//标准差
  getPattern(frame(best_box),pEx,mean,stdev);//将frame图像best_box区域的图像片归一化为均值为0
											 //的15*15大小的patch，存在pEx正样本中
  //Get Fern features on warped patches
  Mat img;
  Mat warped;
  GaussianBlur(frame,img,Size(9,9),1.5);//高斯滤波

  warped = img(bbhull);//在img中截取bbhull(goodbox)信息的部分图像赋给warped  
  RNG& rng = theRNG();//生成随机数
  Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);//取bbhull中心坐标
  
  vector<int> fern(classifier.getNumStructs());//nstructs树木的森林（由一个特征组构建，
											   //每组特征代表图像块的不同视图表示）的个数
  pX.clear();
  Mat patch;
  //pX为处理后的RectBox最大边界处理后的像素信息
  //pEx最近邻的RectBox的Pattern，bbP0为最近邻的RectBox
  if (pX.capacity() < num_warps*good_boxes.size())
    pX.reserve(num_warps * good_boxes.size());//pX正样本个数 = 仿射变换个数 * good_box个数
  int idx;
  for (int i=0;i<num_warps;i++){
     if (i>0)
       generator(frame, pt, warped, bbhull.size(), rng);//对frame进行仿射变换，将仿射变换结果保存到warped
       for (int b=0;b<good_boxes.size();b++){
         idx=good_boxes[b];
		 patch = img(grid[idx]);//提取img的 grid[idx] 区域,也就是bounding box重叠度高的图像片
         classifier.getFeatures(patch,grid[idx].sidx,fern);
         pX.push_back(make_pair(fern,1));//positive ferns <features, labels=1>  正样本
     }
  }
  printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}
 
/*
 * 先对最接近box的RectBox区域得到其patch ,然后归一化RectBox对应的patch的size
 * (放缩至patch_size = 15*15)，将2维的矩阵变成一维的向量，
 * 然后将向量信息均值设为0，调整为zero mean and unit variance（ZMUV）    
 */
void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
  //Output: resized Zero-Mean patch
  resize(img,pattern,Size(patch_size,patch_size));//将img放缩至patch_size = 15*15，存到pattern中 
  
  //Computes a mean value and a standard deviation of matrix elements. 
  meanStdDev(pattern,mean,stdev);
  pattern.convertTo(pattern,CV_32F);
  pattern = pattern-mean.val[0];//矩阵所有元素减去其均值，把patch的均值设为零
}

/* Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 * Describe
 * -之前重叠度小于0.2的都归入 bad_boxes，数量略多，函数用于打乱顺序，是为了  
 *  后面随机选择bad_boxes
 */
void TLD::generateNegativeData(const Mat& frame){
  random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
  int idx;
  //Get Fern Features of the boxes with big variance (calculated using integral images)
  int a=0;
  //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
  printf("negative data generation started.\n");
  vector<int> fern(classifier.getNumStructs());
  nX.reserve(bad_boxes.size());
  Mat patch;
  for (int j=0;j<bad_boxes.size();j++){//方差较大的bad_boxes加入负样本  
      idx = bad_boxes[j];
          if (getVar(grid[idx],iisum,iisqsum)<var*0.5f)
            continue;
      patch =  frame(grid[idx]);
	  classifier.getFeatures(patch,grid[idx].sidx,fern);
      nX.push_back(make_pair(fern,0));//得到负样本
      a++;
  }
  printf("Negative examples generated: ferns: %d ",a);
  //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);
  //Randomly selects 'bad_patches' and get the patterns for NN;
  Scalar dum1, dum2;
  nEx=vector<Mat>(bad_patches);//bad_patches 初始化文件中读取
  for (int i=0;i<bad_patches;i++){
      idx=bad_boxes[i];
	  patch = frame(grid[idx]);
      getPattern(patch,nEx[i],dum1,dum2);//归一化负样本对应的patch的size
  }
  printf("NN: %d\n",(int)nEx.size());
}

//通过积分图像计算输入的box的方差 
double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  //方差 = E(X^2)-(EX)^2，其中EX为均值 
  return sqmean-mean*mean;
}

void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound, bool tl, FILE* bb_file){
  vector<BoundingBox> cbb;
  vector<float> cconf;
  int confident_detections=0;
  int didx; //detection index

  ///Track
  if(lastboxfound && tl){//tl:train and learn
      track(img1,img2,points1,points2);
  }
  else{
      tracked = false;
  }
  ///Detect
  detect(img2);

  ///Integration综合模块
  //TLD只跟踪单目标，所以综合模块综合跟踪器跟踪到的单个目标和
  //检测器检测到的多个目标，然后只输出有效相似度最大的一个目标
  if (tracked){
      bbnext=tbb;
      lastconf=tconf;//表示相关相似度的阈值  
      lastvalid=tvalid; //表示有效相似度的阈值  
      printf("Tracked\n");
      if(detected){

          clusterConf(dbb,dconf,cbb,cconf); //cluster detections
											//通过重叠度对检测器检测到的目标bounding box进行聚类
											//每个类其重叠度小于0.5
          printf("Found %d clusters\n",(int)cbb.size());
          
		  for (int i=0;i<cbb.size();i++){
              if (bbOverlap(tbb,cbb[i])<0.5 && cconf[i]>tconf){
				  //Get index of a clusters that is far from tracker and are more confident than the tracker
                  confident_detections++;//记录可信度比较高的目标box的个数
                  didx=i; //detection index
              }
          }
          if (confident_detections==1){                                
			  //if there is ONE such a cluster, re-initialize the tracker
              printf("Found a better match..reinitializing tracking\n");
              bbnext=cbb[didx];
              lastconf=cconf[didx];
              lastvalid=false;
          }
		  //找到检测器检测到的box与跟踪器预测到的box距离很近(重叠度大于0.7)的box
		  //对其坐标和大小进行累加
          else {
              printf("%d confident cluster was found\n",confident_detections);
              int cx=0,cy=0,cw=0,ch=0;
              int close_detections=0;
              for (int i=0;i<dbb.size();i++){
                  if(bbOverlap(tbb,dbb[i])>0.7){                     
					  // Get mean of close detections
                      cx += dbb[i].x;
                      cy +=dbb[i].y;
                      cw += dbb[i].width;
                      ch += dbb[i].height;
                      close_detections++;//记录最近邻box的个数  
                      printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                  }
              }

			 //对 与跟踪器预测到的box距离很近的box 和 跟踪器本身预测到的box 进行坐标与大小的平均作为最终的
			 //目标bounding box，其中跟踪器的权值较大
              if (close_detections>0){
				  // weighted average trackers trajectory with the close detections
                  bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));  
                  bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                  bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                  bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                  printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                  printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                  printf("Weighting %d close detection(s) with tracker..\n",close_detections);
              }
              else{
                printf("%d close detections were found\n",close_detections);

              }
          }
      }
  }

  //If NOT tracking
  else{       
      printf("Not tracking..\n");
      lastboxfound = false;
      lastvalid = false;
    
	  //如果跟踪器没有跟踪到目标，但是检测器检测到了一些可能的目标box，
	  //那么同样对其进行聚类，但只是简单的将聚类的cbb[0]作为新的跟踪目标box
	  //重新初始化跟踪器  
	  if(detected){                           //  and detector is defined
          clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          if (cconf.size()==1){
              bbnext=cbb[0];
              lastconf=cconf[0];
              printf("Confident detection..reinitializing tracker\n");
              lastboxfound = true;
          }
      }
  }
  lastbox=bbnext;
  if (lastboxfound)
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  else
    fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
  
  ///learn
  if (lastvalid && tl)
    learn(img2);
}

  /*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2){
  //Generate points
  //网格均匀撒点（均匀采样），在lastbox中共产生最多10*10=100个特征点，存于points1  
  bbPoints(points1,lastbox);
  if (points1.size()<1){
      printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
      tvalid=false;
      tracked=false;
      return;
  }
  vector<Point2f> points = points1;
  
  //Frame-to-frame tracking with forward-backward error cheking
  tracked = tracker.trackf2f(img1,img2,points,points2);//trackef2f函数定义在LKTracker.h，功能是
													   //跟踪、计算FB error和匹配相似度sim，然后筛选
  if (tracked){
      //Bounding box prediction
      bbPredict(points,points2,lastbox,tbb);
	  //跟踪失败检测：如果FB error的中值大于10个像素，或者预测到的当前box的位置移出图像时
	  //则认为跟踪错误，此时不返回bounding box
      if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){
          tvalid =false; //too unstable prediction or bounding box out of image
          tracked = false;
          printf("Too unstable predictions FB error=%f\n",tracker.getFB());
          return;
      }

      //Estimate Confidence and Validity
      Mat pattern;
      Scalar mean, stdev;
      BoundingBox bb;
      bb.x = max(tbb.x,0);
      bb.y = max(tbb.y,0);
      bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
      bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
      
	  getPattern(img2(bb),pattern,mean,stdev);//归一化img2(bb)对应的patch的size，存入pattern
      vector<int> isin;
      float dummy;
      classifier.NNConf(pattern,isin,dummy,tconf); //Conservative Similarity
      tvalid = lastvalid;

      if (tconf>classifier.thr_nn_valid){//保守相似度大于阈值，则评估跟踪有效  
          tvalid =true;
      }
  }
  else
    printf("No points tracked\n");

}

//网格均匀撒点，box共10*10=100个特征点 
void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
  int max_pts=10;
  int margin_h=0;//采样边界
  int margin_v=0;
  //网格均匀撒点
  int stepx = ceil((bb.width-2*margin_h)/max_pts);//ceil()取整
  int stepy = ceil((bb.height-2*margin_v)/max_pts);
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
          points.push_back(Point2f(x,y));
      }
  }
}

//利用剩下的不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小
void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);//位移  
  vector<float> yoff(npoints);
  printf("tracked points : %d\n",npoints);
  for (int i=0;i<npoints;i++){//计算每个特征点在两帧之间的位移 
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }

  float dx = median(xoff);//计算位移的中值 
  float dy = median(yoff);
  float s;

  //计算bounding box尺度scale的变化
  //通过计算当前特征点相互间的距离与上一帧特征点相互间的距离的
  //比值，以比值的中值作为尺度的变化因子
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++){
              d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
          }
      }
      s = median(d);
  }
  else {
      s = 1.0;
  }

  float s1 = 0.5*(s-1)*bb1.width;
  float s2 = 0.5*(s-1)*bb1.height;
  printf("s= %f s1= %f s2= %f \n",s,s1,s2);
  
  //得到当前bounding box的位置与大小信息  
  //当前box的x坐标 = 前一帧box的x坐标 + 全部特征点位移的中值 - 当前box宽的一半  
  bb2.x = round( bb1.x + dx -s1);
  bb2.y = round( bb1.y + dy -s2);
  bb2.width = round(bb1.width*s);
  bb2.height = round(bb1.height*s);
  printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}

void TLD::detect(const cv::Mat& frame){
  //cleaning
  dbb.clear();
  dconf.clear();
  dt.bb.clear();
  double t = (double)getTickCount();//GetTickCount返回从操作系统启动到现在所经过的时间
  Mat img(frame.rows,frame.cols,CV_8U);
  integral(frame,iisum,iisqsum);//计算frame的积分图
  GaussianBlur(frame,img,Size(9,9),1.5);//高斯去噪
  int numtrees = classifier.getNumStructs();
  float fern_th = classifier.getFernTh();//getFernTh()返回thr_fern, 集合分类器的分类阈值
  vector <int> ferns(10);
  float conf;
  int a=0;
  Mat patch;
  
  //级联分类器模块一：方差检测模块，利用积分图计算每个待检测窗口的方差，
  //方差大于var阈值（目标patch方差的50%）的,则认为其含有前景目标
  for (int i=0;i<grid.size();i++){//FIXME: BottleNeck瓶颈
      if (getVar(grid[i],iisum,iisqsum)>=var){//计算每一个扫描窗口的方差  
          a++;
		   //级联分类器模块二：集合分类器检测模块  
		  patch = img(grid[i]);
          classifier.getFeatures(patch,grid[i].sidx,ferns);//得到该patch特征（13位的二进制代码）  
          conf = classifier.measure_forest(ferns);//计算该特征值对应的后验概率累加值  
          tmp.conf[i]=conf;
          tmp.patt[i]=ferns;
		  //如果集合分类器的后验概率的平均值大于阈值fern_th（由训练得到），就认为含有前景目标
          if (conf>numtrees*fern_th){
              dt.bb.push_back(i);
          }
      }
      else
        tmp.conf[i]=0.0;
  }
  int detections = dt.bb.size();
  printf("%d Bounding boxes passed the variance filter\n",a);
  printf("%d Initial detection from Fern Classifier\n",detections);
  
 //如果通过以上两个检测模块的扫描窗口数大于100个，则只取后验概率大的前100个  
  if (detections>100){
      nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      dt.bb.resize(100);
      detections=100;
  }
//  for (int i=0;i<detections;i++){
//        drawBox(img,grid[dt.bb[i]]);
//    }
//  imshow("detections",img);
  if (detections==0){
        detected=false;
        return;
      }
  printf("Fern detector made %d detections ",detections);
  t=(double)getTickCount()-t;
  printf("in %gms\n", t*1000/getTickFrequency());
  //  Initialize detection structure
  dt.patt = vector<vector<int> >(detections,vector<int>(10,0));//Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);//  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);//  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));//  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
  
  int idx;
  Scalar mean, stdev;
  float nn_th = classifier.getNNTh();
  //级联分类器模块三：最近邻分类器检测模块  
  for (int i=0;i<detections;i++){ //  for every remaining detection
      idx=dt.bb[i]; //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPattern(patch,dt.patch[i],mean,stdev); //  Get pattern within bounding box
      //计算图像片pattern到在线模型M的相关相似度和保守相似度 
	  classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      ////相关相似度大于阈值，则认为含有前景目标 
	  if (dt.conf1[i]>nn_th){                                               
		  //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
          dbb.push_back(grid[idx]); //  BB    = dt.bb(:,idx); % bounding boxes
          dconf.push_back(dt.conf2[i]);//  Conf  = dt.conf2(:,idx); % conservative confidences
      }
  }  //  end
  
  ////打印检测到的可能存在目标的扫描窗口数（可以通过三个级联检测器的）
  if (dbb.size()>0){
      printf("Found %d NN matches\n",(int)dbb.size());
      detected=true;
  }
  else{
      printf("No NN matches found.\n");
      detected=false;
  }
}

//作者已经用python脚本../datasets/evaluate_vis.py来完成算法评估功能，具体见README
void TLD::evaluate(){
}


void TLD::learn(const Mat& img){
  printf("[Learning] ");
  ///Check consistency
  BoundingBox bb;
  bb.x = max(lastbox.x,0);
  bb.y = max(lastbox.y,0);
  bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
  bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
  Scalar mean, stdev;
  Mat pattern;

  getPattern(img(bb),pattern,mean,stdev);//归一化...
  vector<int> isin;
  float dummy, conf;
  classifier.NNConf(pattern,isin,conf,dummy);//计算输入图像片（跟踪器的目标box）与在线模型之间的相关相似度conf  
  if (conf<0.5) {///相似度太小的就不训练 
      printf("Fast change..not training\n");
      lastvalid =false;
      return;
  }
  if (pow(stdev.val[0],2)<var){//方差太小也不训练  
      printf("Low variance..not training\n");
      lastvalid=false;
      return;
  }
  if(isin[2]==1){//被识别为负样本也不训练 
      printf("Patch in negative data..not traing");
      lastvalid=false;
      return;
  }

/// Data generation
//样本生成
  for (int i=0;i<grid.size();i++){////计算所有的扫描窗口与目标box的重叠度  
      grid[i].overlap = bbOverlap(lastbox,grid[i]);
  }
  //集合分类器 
  vector<pair<vector<int>,int> > fern_examples;
  good_boxes.clear();
  bad_boxes.clear();
  getOverlappingBoxes(lastbox,num_closest_update);//
  if (good_boxes.size()>0)
    generatePositiveData(img,num_warps_update);//仿射模型产生正样本
  else{
    lastvalid = false;
    printf("No good boxes..Not training");
    return;
  }
  fern_examples.reserve(pX.size()+bad_boxes.size());
  fern_examples.assign(pX.begin(),pX.end());
  int idx;
  for (int i=0;i<bad_boxes.size();i++){
      idx=bad_boxes[i];
      if (tmp.conf[idx]>=1){//相似度大于1加入负样本
          fern_examples.push_back(make_pair(tmp.patt[idx],0));
      }
  }

  //最近邻分类器
  vector<Mat> nn_examples;
  nn_examples.reserve(dt.bb.size()+1);
  nn_examples.push_back(pEx);
  for (int i=0;i<dt.bb.size();i++){
      idx = dt.bb[i];
      if (bbOverlap(lastbox,grid[idx]) < bad_overlap)
        nn_examples.push_back(dt.patch[i]);
  }
  /// Classifiers update
  //分类器训练
  classifier.trainF(fern_examples,2);
  classifier.trainNN(nn_examples);
  classifier.show();
}

//检测器采用扫描窗口的策略 
//根据传入的box（目标边界框）在传入的图像中构建全部的扫描窗口，并计算每个窗口与box的重叠度
void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box){
  const float SHIFT = 0.1;//扫描窗口步长为 宽高的 10% 
  const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                          0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                          2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
  int width, height, min_bb_side;
  //Rect bbox;
  BoundingBox bbox;
  Size scale;
  int sc=0;

  for (int s=0;s<21;s++){
    width = round(box.width*SCALES[s]);
    height = round(box.height*SCALES[s]);
    min_bb_side = min(height,width);
    if (min_bb_side < min_win || width > img.cols || height > img.rows)
      continue;
    scale.width = width;
    scale.height = height;
    scales.push_back(scale);//把该尺度的窗口存入scales容器，避免在扫描时计算，加快检测速度
    for (int y=1;y<img.rows-height;y+=round(SHIFT*min_bb_side)){//按步长移动窗口
      for (int x=1;x<img.cols-width;x+=round(SHIFT*min_bb_side)){
        bbox.x = x;
        bbox.y = y;
        bbox.width = width;
        bbox.height = height;
		//判断传入的bounding box（目标边界框）与 传入图像中的此时窗口的 重叠度，  
		//以此来确定该图像窗口是否含有目标  
        bbox.overlap = bbOverlap(bbox,BoundingBox(box));
        bbox.sidx = sc;
        grid.push_back(bbox);
      }
    }
    sc++;
  }
}

//计算两个bounding box 的重叠度(两个box的交集与它们的并集 的比)
float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
	//先判断坐标，假如它们都没有重叠的地方，就直接返回0  
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}


  /*
   * 此函数根据传入的box,在整帧图像中的全部窗口中寻找与该box距离最小(重叠度最大)
   * 的num_closest_init个窗口,然后把这些窗口 归入good_boxes容器,同时把重叠度小于0.2的
   * 归入 bad_boxes 容器.
   * 首先根据overlap的比例信息选出重复区域比例大于60%并且前num_closet_init= 10个的最接近box的RectBox
   * 相当于对RectBox进行筛选并通过BBhull函数得到这些RectBox的最大边界
   */
void TLD::getOverlappingBoxes(const cv::Rect& box1,int num_closest){
  float max_overlap = 0;
  for (int i=0;i<grid.size();i++){
      if (grid[i].overlap > max_overlap) { //找出重叠度最大的box 
          max_overlap = grid[i].overlap;
          best_box = grid[i];
      }
      if (grid[i].overlap > 0.6){//重叠度大于0.6的，归入 good_boxes 
          good_boxes.push_back(i);
      }
      else if (grid[i].overlap < bad_overlap){//重叠度小于0.2的，归入 bad_boxes  
          bad_boxes.push_back(i);
      }
  }
  //Get the best num_closest (10) boxes and puts them in good_boxes
  if (good_boxes.size()>num_closest){
	  //找出一个数列中排第num_closest的数
	  //也就是找到最好的num_closest个box了
    std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
    good_boxes.resize(num_closest);
  }
  getBBHull();//获取good_boxes的窗口边框
}


//获取good_boxes的窗口边框
void TLD::getBBHull(){
  int x1=INT_MAX, x2=0;
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<good_boxes.size();i++){
      idx= good_boxes[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  bbhull.x = x1;
  bbhull.y = y1;
  bbhull.width = x2-x1;
  bbhull.height = y2 -y1;
}

//如果两个box的重叠度小于0.5，返回false，否则返回true
bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
  TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
      return false;
    else
      return true;
}

int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
  //FIXME: Conditional jump or move depends on uninitialised value(s)
  const int c = dbb.size();
  //1. Build proximity matrix
  Mat D(c,c,CV_32F); 
  float d;
  for (int i=0;i<c;i++){
      for (int j=i+1;j<c;j++){
        d = 1-bbOverlap(dbb[i],dbb[j]);
        D.at<float>(i,j) = d;
        D.at<float>(j,i) = d;
      }
  }
  //2. Initialize disjoint clustering
 float L[c-1]; //Level
 int nodes[c-1][2];
 int belongs[c];
 int m=c;
 for (int i=0;i<c;i++){
    belongs[i]=i;
 }
 for (int it=0;it<c-1;it++){
 //3. Find nearest neighbor
     float min_d = 1;
     int node_a, node_b;
     for (int i=0;i<D.rows;i++){
         for (int j=i+1;j<D.cols;j++){
             if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                 min_d = D.at<float>(i,j);
                 node_a = i;
                 node_b = j;
             }
         }
     }
     if (min_d>0.5){
         int max_idx =0;
         bool visited;
         for (int j=0;j<c;j++){
             visited = false;
             for(int i=0;i<2*c-1;i++){
                 if (belongs[j]==i){
                     indexes[j]=max_idx;
                     visited = true;
                 }
             }
             if (visited)
               max_idx++;
         }
         return max_idx;
     }

 //4. Merge clusters and assign level
     L[m]=min_d;
     nodes[it][0] = belongs[node_a];
     nodes[it][1] = belongs[node_b];
     for (int k=0;k<c;k++){
         if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
           belongs[k]=m;
     }
     m++;
 }
 return 1;

}

//对检测器检测到的目标bounding box进行聚类
//聚类（Cluster）分析是由若干模式（Pattern）组成的，通常模式是一个度量（Measurement）
//的向量，或者是多维空间中的一个点。聚类分析以相似性为基础，在一个聚类中的模式之间
//比不在同一聚类中的模式之间具有更多的相似性。  
void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
  int numbb =dbb.size();
  vector<int> T;
  float space_thr = 0.5;
  int c=1;//记录 聚类的类个数  
  switch (numbb){//检测到的含有目标的bounding box个数 
  case 1:
    cbb=vector<BoundingBox>(1,dbb[0]);//如果只检测到一个，那么这个就是检测器检测到的目标
    cconf=vector<float>(1,dconf[0]);
    return;
    break;
  case 2:
    T =vector<int>(2,0);
	//计算两个bounding box 的重叠度  
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){//如果只检测到两个box，但他们的重叠度小于0.5  
      T[1]=1;
      c=2; //重叠度小于0.5的box，属于不同的类
    }
    break;
  default://检测到的box数目大于2个，则筛选出重叠度大于0.5的  
    T = vector<int>(numbb,0);
	//stable_partition()重新排列元素，使得满足指定条件的元素排在不满足条件的元素前面。
	//它维持着两组元素的顺序关系。  
    c = partition(dbb,T,(*bbcomp));//重叠度小于0.5的box，属于不同的类，所以c是不同的类别个数  
    //c = clusterBB(dbb,T);
    break;
  }

  cconf=vector<float>(c);
  cbb=vector<BoundingBox>(c);
  printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){
      float cnf=0;
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){
          if (T[j]==i){//将聚类为同一个类别的box的坐标和大小进行累加
              printf("%d ",i);
              cnf=cnf+dconf[j];
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){//然后求该类的box的坐标和大小的平均值，将平均值作为该类的box的代表  
          cconf[i]=cnf/N;
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;//返回的是聚类，每一个类都有一个代表的bounding box  
      }
  }
  printf("\n");
}

