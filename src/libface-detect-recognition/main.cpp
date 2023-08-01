#include"libfacedet.h"

int main()
{


	const int execute = 0;  ///// 0��ʾִ��������⣬1��ʾִ����ȡ������������������2��ʾִ������ʶ��������ʾ�������+PFLD�ؼ�����
	const bool align = false;   ////�Ƿ�����������
	const Rec_thresh rec_threh = { 0.5,"cos" };  ////"cos" ���� "euclid"

	const string imgpath = "���ͼƬ·��, ���޸�";   /// ·���ָ�����/
	const string fileroot = "����ļ���·��, ���޸�";  ////����ļ��������������ļ���, ÿ�����ļ��е�����������, ���ļ������������˵�����ͼƬ, ͼƬ��Ф����, ����ֻ��һ������
	const char* bin_name = "���뱣���bin�ļ���, ���޸�";

	libface face_detect(align);
	arcface face_feature;
	//openface face_feature;
	pfld pfld_detect;

	if (execute == 0)   ////ֻ���������
	{
		Mat srcimg = imread(imgpath);
		vector<Face> dets = face_detect.detect(srcimg);
		face_detect.draw(srcimg, dets, true);

		static const string kWinName = "Deep learning face detection in OpenCV";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, srcimg);
		waitKey(0);
		destroyAllWindows();
	}
	else if (execute == 1)    ////��ȡ����������������, ����Ϊbin�ļ�
	{
		if (_access(fileroot.c_str(), 0) == -1)
		{
			cout << fileroot << " Dir don't exist!!!" << endl;
			return -1;
		}
		vector<string> filepaths;
		getAllFiles(fileroot, filepaths);

		cout << "get " << filepaths.size() << " images" << endl;

		vector<vector<float> > face_features;
		vector<string> facenames;
		for (int n = 0; n < filepaths.size(); n++)
		{
			cout << filepaths[n] << endl;
			Mat srcimg = imread(filepaths[n]);
			vector<Face> dets = face_detect.detect(srcimg);

			if (dets.size() == 1)   ///ͼƬ��Ф����, ����ֻ��һ������
			{
				string name = fromPath_Getname(filepaths[n]);
				facenames.push_back(name);

				Mat face_roi = face_detect.crop_face(dets[0], srcimg);
				vector<float> out_feature = face_feature.get_feature(face_roi);
				face_features.push_back(out_feature);
			}
		}
		int num_face = facenames.size(), len_feature = face_feature.get_feature_length();
		float* output = new float[num_face * len_feature];
		for (int i = 0; i < num_face; i++)
		{
			for (int j = 0; j < len_feature; j++)
			{
				output[i * len_feature + j] = face_features[i][j];
			}
		}
		write_face_feature_name2bin(num_face, len_feature, output, facenames, bin_name); ///Ҳ���԰�vector<float>д��bin�ļ�
		cout << "write finish!!!" << endl;

		/////�������Ϊ�˲��Ա����bin�ļ��Ƿ���ȷ, �������
		int len_face = 0, num_feature = 0;
		vector<string> read_names;
		float* input = read_face_feature_name2bin(&len_face, &num_feature, read_names, bin_name);
		cout << "read " << len_face << " data" << endl;

		if (num_face == len_face)
		{
			cout << "��ȡ����������ȷ!!!" << endl;
		}
		if (len_feature == num_feature)
		{
			cout << "��ȡ������������������ȷ!!!" << endl;
		}
		for (int i = 0; i < len_face; i++)
		{
			if (read_names[i] != facenames[i])
			{
				cout << "read names[" << i << "] error, write is " << facenames[i] << " , but read is " << read_names[i] << endl;
			}
		}
		float sum = 0;
		for (int i = 0; i < (len_face * num_feature); i++)
		{
			sum += (output[i] - input[i]);
		}
		cout << "mean error = " << sum / (len_face * num_feature) << endl;
		delete[] output;
		delete[] input;
	}
	else if (execute == 2)   ////��ȡ���������֪����������������������bin�ļ�, ������ʶ��
	{
		if (_access(bin_name, 0) == -1)
		{
			cout << bin_name << " File don't exist!!!" << endl;
			return -1;
		}
		int num_face = 0, len_feature = 0;
		vector<string> face_names;
		float* face_features = read_face_feature_name2bin(&num_face, &len_feature, face_names, bin_name);
		
		string rec_name = "unknown";
		Mat srcimg = imread(imgpath);
		vector<Face> dets = face_detect.detect(srcimg);
		float* dist_features = new float[num_face];
		int x_start = 0, y_start = 0, x_end = 0, y_end = 0;
		for (int i = 0; i < dets.size(); i++)
		{
			Mat face_roi = face_detect.crop_face(dets[i], srcimg);
			vector<float> out_feature = face_feature.get_feature(face_roi);
			
			if (rec_threh.type == "cos")
			{
				int max_ind = Get_Max_Cos_Dist(face_features, out_feature, num_face, len_feature, dist_features);
				cout << "max cos_dist = " << dist_features[max_ind] << endl;
				if (dist_features[max_ind] >= rec_threh.thresh)
				{
					rec_name = face_names[max_ind];
				}
			}
			else
			{
				int min_ind = Get_Min_Euclid_Dist(face_features, out_feature, num_face, len_feature, dist_features);
				cout << "min euclid_dist = " << dist_features[min_ind] << endl;
				if (dist_features[min_ind] <= rec_threh.thresh)
				{
					rec_name = face_names[min_ind];
				}
			}
			rectangle(srcimg, Point(dets[i].bbox.top_left.x, dets[i].bbox.top_left.y), Point(dets[i].bbox.bottom_right.x, dets[i].bbox.bottom_right.y), Scalar(0, 0, 255));
			putText(srcimg, rec_name, Point(dets[i].bbox.top_left.x, dets[i].bbox.top_left.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0));
		}
		delete[] face_features;
		delete[] dist_features;
		static const string kWinName = "Deep learning face recognition in OpenCV";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, srcimg);
		waitKey(0);
		destroyAllWindows();
	}
	else
	{
		Mat srcimg = imread(imgpath);
		vector<Face> dets = face_detect.detect(srcimg);
		for (int i = 0; i < dets.size(); i++)
		{
			Mat face_roi = face_detect.crop_face(dets[i], srcimg);
			vector<Point> landmarks = pfld_detect.detect(face_roi);
			pfld_detect.face_detect_draw_landmarks(landmarks, srcimg, (int)dets[i].bbox.top_left.x, (int)dets[i].bbox.top_left.y);
		}
		face_detect.draw(srcimg, dets, false);
		static const string kWinName = "Deep learning face detection in OpenCV";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, srcimg);
		waitKey(0);
		destroyAllWindows();
	}

	system("pause");
	return 0;
}