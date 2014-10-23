#include <opencv2/opencv.hpp>
#include <tld_utils.h>
#include <iostream>
#include <sstream>
#include <TLD.h>
#include <stdio.h>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;

//读取bounding box文件，获得四个参数：左上角坐标x,y和寬高
void readBB(char* file){
  ifstream bb_file (file);//打开文件

  string line;
  getline(bb_file, line);//从流bb_file 中读取的字符存到line中

  istringstream linestream(line);//istringstream 对象可以绑定一行字符串，并且以空格为分隔符将该行分隔
  string x1,y1,x2,y2;

  //从istringstream 中读取字符串存在x,y中,','为分隔符
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');

  //字符串转为整形数
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}

//bounding box mouse callback
//用鼠标选中bounding box,即目标区域
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:// 滑動
     if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN://左键点击
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP://左键放开
    drawing_box = false;
     if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
     if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;//表示已经获取到bounding box
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

//分析运行时的命令参数
void read_options(int argc, char** argv, VideoCapture& capture, FileStorage &fs){
  for (int i=0;i<argc;i++){
      if (strcmp(argv[ i],"-b")==0){
          if (argc>i){ 
              readBB(argv[i+1]);//是否指定初始bounding box
              gotBB = true;
          }
          else
            print_help(argv);
      }

	//从视频中读取,"-s"表示后面跟一个视频
      if (strcmp(argv[i],"-s")==0){
          if (argc>i){
              video = string(argv[i+1]);
              capture.open(video);
              fromfile = true;
          }
          else
            print_help(argv);

      }

	  //读取参数文件parameters.yml
      if (strcmp(argv[i],"-p")==0){ 
          if (argc>i){ 
              fs.open(argv[i+1], FileStorage::READ);
          }
          else
            print_help(argv);
      }

	  //to train only in the first frame(no tracking ,no learning)
      if (strcmp(argv[i],"-no_tl")==0){
          tl = false;
      
	  }

	  //to test the final detector(reapeat the video)
       if (strcmp(argv[i],"-r")==0){
          rep = true;
      }
  }
}
/*
%To run from camera
./run_tld -p ../parameters.yml
%To run from file
./run_tld -p ../parameters.yml -s ../datasets/06_car/car.mpg
%To init bounding box from file
./run_tld -p ../parameters.yml -s ../datasets/06_car/car.mpg -b ../datasets/06_car/init.txt
%To train only in the firs frame (no tracking, no learning)
./run_tld -p ../parameters.yml -s ../datasets/06_car/car.mpg -b ../datasets/06_car/init.txt -no_tl 
%To test the final detector (Repeat the video, first time learns, second time detects)
./run_tld -p ../parameters.yml -s ../datasets/06_car/car.mpg -b ../datasets/06_car/init.txt -r
*/

//对起始帧初始化，然后逐帧处理
int main(int argc, char * argv[]){
  VideoCapture capture;
  capture.open(0);

 //OpenCV的C++接口中，用于保存图像的imwrite只能保存整数数据，且需作为图像格式。
 //而浮点数据或XML\YAML文件在OpenCV中的数据结构为FileStorage  
  FileStorage fs;
  //Read options
  read_options(argc,argv,capture,fs);//分析命令行参数  
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);

  /*
   * void cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse, void* param=NULL );
   * window_name  窗口的名字。 
   * on_mouse 指定窗口里每次鼠标事件发生的时候，被调用的函数指针。这个函数的原型应该为 
   * void Foo(int event, int x, int y, int flags, void* param);
  */
  cvSetMouseCallback( "TLD", mouseHandler, NULL );//鼠标选中初始目标的bounding box
  //TLD framework
  TLD tld; 
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  if (fromfile){
	  //若是读取文件
      capture >> frame;//读取当前帧
      cvtColor(frame, last_gray, CV_RGB2GRAY);//转换为灰度图像
      frame.copyTo(first);//拷贝做第一帧
  }else{
	  //若为读取摄像头，则设置获取的图像大小为340*240
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  }

  ///Initialization

  //got语句块开始了。。。
GETBOUNDINGBOX:
  while(!gotBB)
  {
    if (!fromfile){
      capture >> frame;
    }
    else
      first.copyTo(frame);
    cvtColor(frame, last_gray, CV_RGB2GRAY);
    drawBox(frame,box);//画出BoundingBox,这函数在tld_utils.h中
    imshow("TLD", frame);
    if (cvWaitKey(33) == 'q')//按键q退出
	    return 0;
  }
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
	  //由于图像片min_win(15x15)是在bounding box中采样得到的,所以box必须比min_win要大 
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }

  //Remove callback
  //如果已经获得第一帧用户框定的box了，就取消鼠标响应
  cvSetMouseCallback( "TLD", NULL, NULL );
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");
  //TLD initialization
  tld.init(last_gray, box, bb_file);//用bounding_boxe和第一帧图片去初始化TLD系统!!这个是重点

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1;
  vector<Point2f> pts2;
  bool status=true;//记录跟踪成功与否的状态 lastbox been found  
  int frames = 1;//记录已过去的帧数
  int detections = 1;//记录成功检测到的目标box数目

REPEAT:
  while(capture.read(frame)){ 
    //get frame
    cvtColor(frame, current_gray, CV_RGB2GRAY);
    //Process Frame
    tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);///!!!!这个重点
    //Draw Points
    if (status){
		//如果追踪成功
      drawPoints(frame,pts1);
      drawPoints(frame,pts2,Scalar(0,255,0));//当前特帧点用蓝色点标示
      drawBox(frame,pbox);
      detections++;
    } 
    //Display
    imshow("TLD", frame);
    //swap points and images
    swap(last_gray, current_gray);//交换二者的值
    pts1.clear();
    pts2.clear();
    frames++;
    printf("Detection rate: %d/%d\n", detections, frames);
    if (cvWaitKey(33) == 'q')
      break;
  }
  //把检测器村起来~
  if (rep){ 
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  fclose(bb_file);
  return 0;
}
