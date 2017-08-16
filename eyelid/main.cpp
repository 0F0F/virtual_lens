#include "test.h"

#if defined(_LSH_WRITE_PROCEDURE_LOG) || defined(_LSH_TIME_CONSUMPTION_CHECK)
#include <ctime>
char buffer[128];
#endif

#ifdef _LSH_WRITE_PROCEDURE_LOG
#include <fstream>

std::ofstream procedure_log{ "log.txt", std::ios::out | std::ios::app };
std::string procedure_name;
#endif //_LSH_WRITE_PROCEDURE_LOG


#ifdef _LSH_TIME_CONSUMPTION_CHECK
#include <chrono>

std::chrono::time_point<std::chrono::system_clock> proc_start, proc_end;
std::chrono::duration<double> elapsed_seconds;
std::time_t start_time, end_time;

inline void measure_start();
inline void measure_end(const char* proc_name);

#endif //_LSH_TIME_CONSUMPTION_CHECK

int main(int argc, char* argv[]) {
	const char* const arg_details[] 
	{
		"",
		"path_to_face",		// argv[1]
		"path_to_lens",		// argv[2]
		"path_to_result"	// argv[3]
	};
	if (argc < 4)
	{
		cout << "please input: ";
		for (const auto& arg_detail : arg_details)
		{
			printf(arg_detail);
		}
		procedure_log.close();
		return RETURN_VALUES::UNEXPECTED_ARGUMENTS;
	}

	for (int i = 1; i < argc; i++)
	{
		printf("%s : %s\n", arg_details[i], argv[i]);
	}
#ifdef _LSH_WRITE_PROCEDURE_LOG
	procedure_name = (argc == 5 ? argv[4] : argv[3]);
	auto started_time =
		std::chrono::system_clock::to_time_t(
			std::chrono::system_clock::now()
		);
	auto started_time_as_str = std::ctime(&started_time);

	sprintf(buffer, "%s[started procedure : %s]\n", started_time_as_str, procedure_name.c_str());
	procedure_log.write(buffer, strlen(buffer));
#endif //_LSH_WRITE_PROCEDURE_LOG


	//opencv cascade classifier
	//using haar classifier
	cv::CascadeClassifier face_cascade;
	cv::CascadeClassifier eyes_cascade;

	//frame: face image (3 or 4 channel)
	//lens_image: 4channel png image
	Mat frame;
	Mat lens_image;

	try {
		if (!face_cascade.load(face_cascade_name)) { throw std::runtime_error("face cascade load"); }
		if (!eyes_cascade.load(eyes_cascade_name)) { throw std::runtime_error("eyes cascade load"); }

#ifdef _LSH_TIME_CONSUMPTION_CHECK
		measure_start();
#endif //_LSH_TIME_CONSUMPTION_CHECK

		//loading images
		frame = cv::imread(argv[1]);
		lens_image = cv::imread(argv[2], cv::IMREAD_UNCHANGED);

#ifdef _LSH_TIME_CONSUMPTION_CHECK
		measure_end("imread");
#endif //_LSH_TIME_CONSUMPTION_CHECK

		//렌즈 합성시 에러 해결
		//8UC3 to 8UC4
		cv::cvtColor(frame, frame, CV_BGR2BGRA);


		if (!frame.empty() && !lens_image.empty()) {

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_start();
#endif //_LSH_TIME_CONSUMPTION_CHECK

			//얼굴에서 눈 영역 ROI 갖고오기
			cv::Mat eyesROI[2];
			extract_eye_roi(frame, eyesROI[0], eyesROI[1],
				&face_cascade, &eyes_cascade);

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_end("extract_eye_roi");
#endif //_LSH_TIME_CONSUMPTION_CHECK

			//각 눈 영역 ROI에서 눈동자 영역 검출
			//center point와 radius를 각각 분리
			cv::Point center[2];
			double radius[2];

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_start();
#endif //_LSH_TIME_CONSUMPTION_CHECK

			detect_pupil(eyesROI[0], &center[0], &radius[0]);
			detect_pupil(eyesROI[1], &center[1], &radius[1]);

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_end("detect_pupil");
#endif //_LSH_TIME_CONSUMPTION_CHECK

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_start();
#endif //_LSH_TIME_CONSUMPTION_CHECK

			auto left_eyelids = detect_eyelids(eyesROI[0], center[0], radius[0]);
			auto right_eyelids = detect_eyelids(eyesROI[1], center[1], radius[1]);

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_end("detect_eyelids");
#endif //_LSH_TIME_CONSUMPTION_CHECK

#ifdef _CSH_DEBUG
			//ROI offset 디버그
			cv::Point offset[2];
			Size whole_sz;

			eyesROI[0].locateROI(whole_sz, offset[0]);
			eyesROI[1].locateROI(whole_sz, offset[1]);

			circle(frame, Point(center[0].x + offset[0].x, center[0].y + offset[0].y),
				radius[0], Scalar(255, 255, 0));
			circle(frame, Point(center[1].x + offset[1].x, center[1].y + offset[1].y),
				radius[1], Scalar(255, 255, 0));
			imshow("debug2", frame);
#endif // _CSH_DEBUG

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_start();
#endif //_LSH_TIME_CONSUMPTION_CHECK

			//렌즈 이미지를 이용해 각 눈 부분에 합성
			overlay_lens(lens_image, eyesROI[0], &center[0], &radius[0], left_eyelids);
			overlay_lens(lens_image, eyesROI[1], &center[1], &radius[1], right_eyelids);

#ifdef _LSH_TIME_CONSUMPTION_CHECK
			measure_end("overlay_lens");
#endif //_LSH_TIME_CONSUMPTION_CHECK

			//합성된 이미지 출력
			cv::imshow("Result", frame);
			cv::imwrite(argv[3], frame);
		}
		else {
			//이미지가 없는 경우
			throw std::runtime_error("frame is empty");
		}

		//입력 대기
		//waitKey 없으면 창이 바로 종료됨

#if defined(_LSH_DEBUG) || defined(_CSH_DEBUG)
		cv::waitKey(0);
#endif

#ifdef _LSH_WRITE_PROCEDURE_LOG
		auto end_time =
			std::chrono::system_clock::to_time_t(
				std::chrono::system_clock::now()
			);
		auto end_time_as_str = std::ctime(&end_time);

		sprintf(buffer, "%s[ended procedure : %s]\n\n", end_time_as_str, procedure_name.c_str());
		procedure_log.write(buffer, strlen(buffer));
		procedure_log.close();
#endif //_LSH_WRITE_PROCEDURE_LOG

		return RETURN_VALUES::COMPLETE;
	}

	catch (std::runtime_error& e) {
		std::cerr << "error: " << e.what() << std::endl;
#ifdef _LSH_WRITE_PROCEDURE_LOG
		auto end_time =
			std::chrono::system_clock::to_time_t(
				std::chrono::system_clock::now()
			);
		auto end_time_as_str = std::ctime(&end_time);
		auto error = e.what();

		sprintf(buffer, "%s[unhandled_exception : %s]\n\n", end_time_as_str, error);
		procedure_log.write(buffer, strlen(buffer));
		procedure_log.close();
#endif
		return RETURN_VALUES::UNHANDLED_EXCEPTIONS;
	}
}

#ifdef _LSH_TIME_CONSUMPTION_CHECK
inline void measure_start()
{
	proc_start = std::chrono::system_clock::now();
}
inline void measure_end(const char* proc_name)
{
	proc_end = std::chrono::system_clock::now();
	elapsed_seconds = proc_end - proc_start;
	start_time = std::chrono::system_clock::to_time_t(proc_start);
	end_time = std::chrono::system_clock::to_time_t(proc_end);

	sprintf(buffer,
		"[%s] "
		"elapsed time : %f\n",
		proc_name, elapsed_seconds.count()
	);
	printf(buffer);

#ifdef _LSH_WRITE_PROCEDURE_LOG
	auto buffer_length = strlen(buffer);
	procedure_log.write(buffer, buffer_length);
#endif //_LSH_WRITE_PROCEDURE_LOG
}
#endif //_LSH_TIME_CONSUMPTION_CHECK

