#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#define PAI 3.141592653589793

using namespace cv;
using namespace std;
using namespace dnn;

typedef struct BoxInfo
{
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	float pitch;
	float yaw;
	float roll;
} BoxInfo;

void drawPred(Mat& frame, vector<BoxInfo> faceboxes);

class Head_Pose
{
public:
	Head_Pose(string modelpath, float confThreshold, float iouThreshold);
	vector<BoxInfo> detect(Mat srcimg);
private:
	int inpWidth;
	int inpHeight;
	float confThreshold;
	float nmsThreshold;

	Net net;
};

Head_Pose::Head_Pose(string model_path, float confThreshold, float iouThreshold)
{
	this->net = readNet(model_path);
	this->confThreshold = confThreshold;
	this->nmsThreshold = iouThreshold;

	size_t pos = model_path.rfind("_");
	size_t pos_ = model_path.rfind(".");
	int len = pos_ - pos - 1;
	string hxw = model_path.substr(pos + 1, len);

	pos = hxw.rfind("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);
}

vector<BoxInfo> Head_Pose::detect(Mat srcimg)
{
	Mat blob;
	blobFromImage(srcimg, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);

	this->net.setInput(blob);
	vector<Mat> outs;
	net.enableWinograd(false);  ///����һ�У������forward�Ͳ��ᱨ��ԭ�����https://github.com/opencv/opencv/issues/23080
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // ��ʼ����
	////�ն˴�ӡһ����־����ô�ر�������μ����� https://blog.csdn.net/benobug/article/details/117653268
	float* pred = (float*)outs[0].data;
	const int num_face = outs[0].size[1];
	const int len = outs[0].size[2];
	const float scale_h = (float)srcimg.rows / this->inpHeight;
	const float scale_w = (float)srcimg.cols / this->inpWidth;
	/////generate proposals
	vector<Rect> bboxes_wh;
	vector<float> confidences;
	vector<float> pitchs;
	vector<float> yaws;
	vector<float> rolls;
	for (int i = 0; i < num_face; i++)
	{
		const float conf = pred[4] * pred[5];
		if (conf > this->confThreshold)
		{
			float xmin = max((float)(pred[0] - 0.5*pred[2]) * scale_w, 0.f);
			float ymin = max((float)(pred[1] - 0.5*pred[3]) * scale_h, 0.f);
			float xmax = min((float)(pred[0] + 0.5*pred[2]) * scale_w, float(srcimg.cols));
			float ymax = min((float)(pred[1] + 0.5*pred[3]) * scale_h, float(srcimg.rows));
			bboxes_wh.push_back(Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)));
			confidences.push_back(conf);
			pitchs.push_back(pred[6]);
			yaws.push_back(pred[7]);
			rolls.push_back(pred[8]);
		}
		pred += len;  ///ָ�뻬�������Ҳ�ɸĳ�pred[i*len+j]�ķ�ʽ����
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(bboxes_wh, confidences, this->confThreshold, this->nmsThreshold, indices);
	vector<BoxInfo> faceboxes;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		faceboxes.push_back({ (float)bboxes_wh[idx].x,(float)bboxes_wh[idx].y,float(bboxes_wh[idx].x + bboxes_wh[idx].width),float(bboxes_wh[idx].y + bboxes_wh[idx].height), confidences[idx], float((pitchs[idx]-0.5)*180),float((yaws[idx]-0.5)*360),float((rolls[idx]-0.5)*180) });
	}
	return faceboxes;
}

void drawPred(Mat& frame, vector<BoxInfo> faceboxes)
{
	for (int i = 0; i < faceboxes.size(); i++)
	{
		rectangle(frame, Point(int(faceboxes[i].xmin), int(faceboxes[i].ymin)), Point(int(faceboxes[i].xmax), int(faceboxes[i].ymax)), Scalar(0, 0, 255), 2);
		float pitch = faceboxes[i].pitch*PAI / 180;
		float yaw = -faceboxes[i].yaw*PAI / 180;
		float roll = faceboxes[i].roll*PAI / 180;
		float tdx = (faceboxes[i].xmin + faceboxes[i].xmax)*0.5;
		float tdy = (faceboxes[i].ymin + faceboxes[i].ymax)*0.5;
		int size_ = floor((faceboxes[i].xmax - faceboxes[i].xmin) / 3);
		////X - Axis pointing to right.drawn in red
		float x1 = size_ * (cos(yaw) * cos(roll)) + tdx;
		float y1 = size_ * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy;
		////Y-Axis | drawn in green
		float x2 = size_ * (-cos(yaw) * sin(roll)) + tdx;
		float y2 = size_ * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy;
		////Z-Axis (out of the screen) drawn in blue
		float x3 = size_ * (sin(yaw)) + tdx;
		float y3 = size_ * (-cos(yaw) * sin(pitch)) + tdy;
		line(frame, Point(int(tdx), int(tdy)), Point(int(x1), int(y1)), Scalar(0, 0, 255), 2);
		line(frame, Point(int(tdx), int(tdy)), Point(int(x2), int(y2)), Scalar(0, 255, 0), 2);
		line(frame, Point(int(tdx), int(tdy)), Point(int(x3), int(y3)), Scalar(255, 0, 0), 2);
	}
}

int main()
{
	Head_Pose mynet("./directmhp_cmu_m_512x640.onnx", 0.5, 0.5);
	string imgpath = "./1.jpg";
	Mat srcimg = imread(imgpath);
	vector<BoxInfo> faceboxes = mynet.detect(srcimg);
	drawPred(srcimg, faceboxes);
	////imwrite("opencv_result.jpg", srcimg);
	string winname = "Deep learning Head Pose Estimation in OpenCV";
	namedWindow(winname, WINDOW_NORMAL);
	imshow(winname, srcimg);
	waitKey(0);
	destroyAllWindows();
}