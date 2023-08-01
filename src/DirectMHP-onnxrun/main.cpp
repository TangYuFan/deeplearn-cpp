#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///nvidia-cuda加速
#include <onnxruntime_cxx_api.h>

#define PAI 3.141592653589793

using namespace cv;
using namespace std;
using namespace Ort;

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
    Head_Pose(string modelpath, float confThreshold);
    vector<BoxInfo> detect(Mat srcimg);
private:
    vector<float> input_image_;
    int inpWidth;
    int inpHeight;
    float confThreshold;

    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Head Pose Estimation");
    Ort::Session *ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

Head_Pose::Head_Pose(string model_path, float confThreshold)
{
    //OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);  ///nvidia-cuda加速
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
//    std::wstring widestr = std::wstring(model_path.begin(), model_path.end());   ///如果在windows系统就这么写
//    ort_session = new Session(env, widestr.c_str(), sessionOptions);   ///如果在windows系统就这么写
    ort_session = new Session(env, model_path.c_str(), sessionOptions);  ///如果在linux系统，就这么写

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

vector<BoxInfo> Head_Pose::detect(Mat srcimg)
{
    Mat dstimg;
    resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight));
    this->input_image_.resize(this->inpWidth * this->inpHeight * dstimg.channels());
    int k = 0;
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < this->inpHeight; i++)
        {
            for (int j = 0; j < this->inpWidth; j++)
            {
                float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
                this->input_image_[k] = pix / 255.0;
                k++;
            }
        }
    }
    array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    // 开始推理
    vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
    float* pred = ort_outputs[0].GetTensorMutableData<float>();
    auto pred_dims = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_face = pred_dims.at(0);
    const int len = pred_dims.at(1);
    vector<BoxInfo> faceboxes;
    const float scale_h = (float)srcimg.rows / this->inpHeight;
    const float scale_w = (float)srcimg.cols / this->inpWidth;
    for (int i = 0; i < num_face; i++)
    {
        const float score = pred[6];
        if (score > this->confThreshold)
        {
            float xmin = max(pred[2] * scale_w, 0.f);
            float ymin = max(pred[3] * scale_h, 0.f);
            float xmax = min(pred[4] * scale_w, float(srcimg.cols));
            float ymax = min(pred[5] * scale_h, float(srcimg.rows));
            faceboxes.push_back({ xmin,ymin,xmax,ymax, score, pred[7],pred[8],pred[9] });
        }
        pred += len;  ///指针滑动
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

    Head_Pose mynet("./directmhp_cmu_m_post_512x640.onnx", 0.5);
    string imgpath = "./1.jpg";
    Mat srcimg = imread(imgpath);
    vector<BoxInfo> faceboxes = mynet.detect(srcimg);
    drawPred(srcimg, faceboxes);

    string winname = "Deep learning Head Pose Estimation in ONNXRuntime";
    namedWindow(winname, WINDOW_NORMAL);
    imshow(winname, srcimg);
    waitKey(0);
    destroyAllWindows();
}