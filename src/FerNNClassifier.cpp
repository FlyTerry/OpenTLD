/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include <FerNNClassifier.h>

using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];//决策树数目
  structSize = (int)file["num_features"];//树的特征个数，即每棵树的节点个数
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}

void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  //二维向量包含全部尺度（scales）的扫描窗口，每个尺度包含totalFeatures个特征  
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();//产生随机数
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;

  ///用随机数去填充每一个尺度扫描窗口的特征  
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2);//第s种尺度的第i个特征  两个随机分配的像素点坐标  
      }

  }
  //Thresholds
  thrN = 0.5*nstructs;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  } 
}

//获取输入的image的特征(13位2进制代码),树的形式
void FerNNClassifier::get Features(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;//叶子
  for (int t=0;t<nstructs; t++){
      leaf=0;
      for (int f=0; f<struc tSize; f++){
          leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);//leaf<<1 左移一位
      }
      fern[t]=leaf;
  }
}

float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];///每棵树的每个特征值对应的后验概率累加值
   }
  return votes;
} 

////更新正负样本数，同时更新后验概率  
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int  i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCoun ter[i][idx]==0) {
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
} 

//训练集合分类器（n个基本分类器集合
void FerNNClassi fier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i  = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].seco nd==1){                           //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
  //}
}

////训练最近邻分类器 
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;
  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){                         
	  //  For each example
      NNConf(nn_examples[i],isin,conf,dummy);//  Measure Relative similarity
      
	  if (y[i]==1 && conf<=thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){                                          //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                         //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);
      }                                                             //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                                  //  end
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                   //  end


  /*Inputs:  
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
void FerNNClassifier::NNConf(const Mat& examp le, vector<int>& isin,float& rsconf,float& csconf){
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);
  float nccN, maxN=0;
  bool anyN=false;
  
  //比较图像片p到在线模型M的距离（相似度），计算正样本最近邻相似度，也就是将输入的图像片与 
  //在线模型中所有的图像片进行匹配，找出最相似的那个图像片，也就是相似度的最大值  
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5; //计算匹配相似度  
      if (nccP>ncc_thesame)//ncc_thesame: 0.95  
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;//记录最大的相似度以及对应的图像片index索引值 
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }

  ////计算负样本最近邻相似度  
  for (int i=0;i<nEx.size();i++){ 
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  
  //Measure Relative Similarity
  //相关相似度 = 正样本最近邻相似度 / （正样本最近邻相似度 + 负样本最近邻相似度）  
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}

void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
  
  float fconf;
   //所有基本分类器的后验概率的平均值如果大于thr_fern，则认为含有前景目标
  for (int i=0;i<nXT.size();i++){
	fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)
      thr_fern=fconf;//取这个平均值作为 该集合分类器的 新的阈值
}
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf;
  }
  if (thr_nn>thr_nn_valid)//thr_nn_valid: 0.7  
    thr_nn_valid = thr_nn;
}

//把正样本库（在线模型）包含的所有正样本输出  
void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  //minMaxLoc寻找矩阵（一维数组当作向量，用Mat定义）中最小值和最大值的位置.   
  for (int i=0;i<pEx.size();i++){ 
    minMaxLoc(pEx[i],&minval);//寻找pEx[i]的最小值  
    pEx[i].copyTo(ex);
    ex = ex-minval;//把像素亮度最小的像素重设为0，其他像素按此重设  
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
  }
  imshow("Examples",examples);
}
