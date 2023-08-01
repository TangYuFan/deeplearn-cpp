#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo
{
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	float score;
	string name;
} BoxInfo;

class PP_YOLOE_Plus
{
public:
	PP_YOLOE_Plus(string model_path, float confThreshold);
	void detect(Mat& cv_image);
private:
	float confThreshold;
	const string classesFile = "coco.names";
	vector<string> class_names;
	int num_class;

	void normalize_(Mat img);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;
	vector<float> scale_factor = { 1,1 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "pp-yoloe-plus");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

PP_YOLOE_Plus::PP_YOLOE_Plus(string model_path, float confThreshold)
{
	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

//	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->confThreshold = confThreshold;
}

void PP_YOLOE_Plus::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

void PP_YOLOE_Plus::detect(Mat& srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	array<int64_t, 2> scale_shape_{ 1, 2 };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	vector<Value> ort_inputs;
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size()));
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, scale_factor.data(), scale_factor.size(), scale_shape_.data(), scale_shape_.size()));
	// ��ʼ����
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), ort_inputs.data(), 2, output_names.data(), output_names.size());
	const float* outs = ort_outputs[0].GetTensorMutableData<float>();
	const int* box_num = ort_outputs[1].GetTensorMutableData<int>();
	const int nout = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);

	const float ratioh = float(srcimg.rows) / this->inpHeight;
	const float ratiow = float(srcimg.cols) / this->inpWidth;
	vector<BoxInfo> boxs;
	for (int i = 0; i < box_num[0]; i++)
	{
		if (outs[0] > -1 && outs[1] > this->confThreshold)
		{
			boxs.push_back({ int(outs[2] * ratiow), int(outs[3] * ratioh), int(outs[4] * ratiow), int(outs[5] * ratioh), outs[1], this->class_names[int(outs[0])] });
		}
		outs += nout;
	}

	for (size_t i = 0; i < boxs.size(); ++i)
	{
		rectangle(srcimg, Point(boxs[i].xmin, boxs[i].ymin), Point(boxs[i].xmax, boxs[i].ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", boxs[i].score);
		label = boxs[i].name + ":" + label;
		putText(srcimg, label, Point(boxs[i].xmin, boxs[i].ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main()
{

	PP_YOLOE_Plus mynet("weights/ppyoloe_plus_crn_s_80e_coco_640x640.onnx", 0.5);
	string imgpath = "images/bus.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}