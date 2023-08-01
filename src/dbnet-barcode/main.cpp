#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;
class DBNet
{
public:
    DBNet(const float binaryThreshold = 0.5, const float polygonThreshold = 0.7, const float unclipRatio = 1.5, const int maxCandidates = 1000)
    {
        cout<<"run dbnet"<<endl;
        this->binaryThreshold = binaryThreshold;
        this->polygonThreshold = polygonThreshold;
        this->unclipRatio = unclipRatio;
        this->maxCandidates = maxCandidates;
        //OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);  ////gpu
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

        // 因为可执行文件在build下生成,这里是 ../onnx
        net = new Session(env, "./model_0.88_depoly.onnx", sessionOptions);
        size_t numInputNodes = net->GetInputCount();
        size_t numOutputNodes = net->GetOutputCount();
        AllocatorWithDefaultOptions allocator;
        for(int i=0;i<numInputNodes;i++)
        {
            input_names.push_back(net->GetInputName(i, allocator));
        }
        for(int i=0;i<numOutputNodes;i++)
        {
            output_names.push_back(net->GetOutputName(i, allocator));
            TypeInfo outputTypeInfo = net->GetOutputTypeInfo(i);
            auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            vector<int64_t> outputDims = outputTensorInfo.GetShape();
            int count = 1;
            for(int j=0;j<outputDims.size();j++)
            {
                count *= outputDims[j];
            }
            outputCount.push_back(count);
        }
    }
    void detect(Mat& srcimg);
private:
    float binaryThreshold;
    float polygonThreshold;
    float unclipRatio;
    int maxCandidates;
    const int inpWidth = 736;
    const int inpHeight = 736;
    const float meanValues[3] = {0.485, 0.456, 0.406};
    const float normValues[3] = {0.229, 0.224, 0.225};
    float contourScore(const Mat& binary, const vector<Point>& contour);
    void unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly);
    vector<float> normalize_(Mat img);

    Session *net;
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "dbnet");
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
    vector<int> outputCount;
};

vector<float> DBNet::normalize_(Mat img)
{
//    img.convertTo(img, CV_32F);
    int row = img.rows;
    int col = img.cols;
    vector<float> output(row * col * img.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + c];
                output[c * row * col + i * col + j] = (pix / 255.0 - this->meanValues[c]) / this->normValues[c];
            }
        }
    }
    return output;
}

void DBNet::detect(Mat& srcimg)
{
    int h = srcimg.rows;
    int w = srcimg.cols;
    Mat dst;
    cvtColor(srcimg, dst, COLOR_BGR2RGB);
    resize(dst, dst, Size(this->inpWidth, this->inpHeight));
    vector<float> input_image_ = this->normalize_(dst);
    array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    vector<Value> ort_outputs = net->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
    const float* floatArray = ort_outputs[0].GetTensorMutableData<float>();
    Mat binary(dst.rows, dst.cols, CV_32FC1);
    memcpy(binary.data, floatArray, outputCount[0] * sizeof(float));

    // Threshold
    Mat bitmap;
    threshold(binary, bitmap, binaryThreshold, 255, THRESH_BINARY);
    // Scale ratio
    float scaleHeight = (float)(h) / (float)(binary.size[0]);
    float scaleWidth = (float)(w) / (float)(binary.size[1]);
    // Find contours
    vector< vector<Point> > contours;
    bitmap.convertTo(bitmap, CV_8UC1);
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Candidate number limitation
    size_t numCandidate = min(contours.size(), (size_t)(maxCandidates > 0 ? maxCandidates : INT_MAX));
    vector<float> confidences;
    vector< vector<Point2f> > results;
    for (size_t i = 0; i < numCandidate; i++)
    {
        vector<Point>& contour = contours[i];

        // Calculate text contour score
        if (contourScore(binary, contour) < polygonThreshold)
            continue;

        // Rescale
        vector<Point> contourScaled; contourScaled.reserve(contour.size());
        for (size_t j = 0; j < contour.size(); j++)
        {
            contourScaled.push_back(Point(int(contour[j].x * scaleWidth),
                                          int(contour[j].y * scaleHeight)));
        }

        // Unclip
        RotatedRect box = minAreaRect(contourScaled);

        // minArea() rect is not normalized, it may return rectangles with angle=-90 or height < width
        const float angle_threshold = 60;  // do not expect vertical text, TODO detection algo property
        bool swap_size = false;
        if (box.size.width < box.size.height)  // horizontal-wide text area is expected
            swap_size = true;
        else if (fabs(box.angle) >= angle_threshold)  // don't work with vertical rectangles
            swap_size = true;
        if (swap_size)
        {
            swap(box.size.width, box.size.height);
            if (box.angle < 0)
                box.angle += 90;
            else if (box.angle > 0)
                box.angle -= 90;
        }

        Point2f vertex[4];
        box.points(vertex);  // order: bl, tl, tr, br
        vector<Point2f> approx;
        for (int j = 0; j < 4; j++)
            approx.emplace_back(vertex[j]);
        vector<Point2f> polygon;
        unclip(approx, polygon);
        results.push_back(polygon);
    }
    confidences = vector<float>(contours.size(), 1.0f);
    for (int i = 0; i < results.size(); i++)
    {
        for (int j = 0; j < 4; j++)
        {
            circle(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), 2, Scalar(0, 0, 255), -1);
            if(j<3)
            {
                line(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), Point((int)results[i][j+1].x, (int)results[i][j+1].y), Scalar(0, 255, 0));
            }
            else
            {
                line(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), Point((int)results[i][0].x, (int)results[i][0].y), Scalar(0, 255, 0));
            }
        }
    }
}

float DBNet::contourScore(const Mat& binary, const vector<Point>& contour)
{
    Rect rect = boundingRect(contour);
    int xmin = max(rect.x, 0);
    int xmax = min(rect.x + rect.width, binary.cols - 1);
    int ymin = max(rect.y, 0);
    int ymax = min(rect.y + rect.height, binary.rows - 1);

    Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

    Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    vector<Point> roiContour;
    for (size_t i = 0; i < contour.size(); i++) {
        Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
        roiContour.push_back(pt);
    }
    vector<vector<Point>> roiContours = {roiContour};
    fillPoly(mask, roiContours, Scalar(1));
    float score = mean(binROI, mask).val[0];
    return score;
}

void DBNet::unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly)
{
    float area = contourArea(inPoly);
    float length = arcLength(inPoly, true);
    float distance = area * unclipRatio / length;

    size_t numPoints = inPoly.size();
    vector<vector<Point2f>> newLines;
    for (size_t i = 0; i < numPoints; i++)
    {
        vector<Point2f> newLine;
        Point pt1 = inPoly[i];
        Point pt2 = inPoly[(i - 1) % numPoints];
        Point vec = pt1 - pt2;
        float unclipDis = (float)(distance / norm(vec));
        Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
        newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
        newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
        newLines.push_back(newLine);
    }

    size_t numLines = newLines.size();
    for (size_t i = 0; i < numLines; i++)
    {
        Point2f a = newLines[i][0];
        Point2f b = newLines[i][1];
        Point2f c = newLines[(i + 1) % numLines][0];
        Point2f d = newLines[(i + 1) % numLines][1];
        Point2f pt;
        Point2f v1 = b - a;
        Point2f v2 = d - c;
        float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

        if( fabs(cosAngle) > 0.7 )
        {
            pt.x = (b.x + c.x) * 0.5;
            pt.y = (b.y + c.y) * 0.5;
        }
        else
        {
            float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
                          d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
            float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
            float s = num / denom;

            pt.x = a.x + s*(b.x - a.x);
            pt.y = a.y + s*(b.y - a.y);
        }
        outPoly.push_back(pt);
    }
}


int main(){

    DBNet mynet(0.5, 0.7, 1.5, 1000);
    string imgpath = "./img.png";
    Mat srcimg = imread(imgpath);
    mynet.detect(srcimg);

    static const string kWinName = "Deep learning object detection in onnxrun";
    namedWindow(kWinName, WINDOW_NORMAL);

    // 保存图片
//    imwrite("out.png",srcimg);

    imshow(kWinName, srcimg);
    waitKey(0);
    destroyAllWindows();
}
