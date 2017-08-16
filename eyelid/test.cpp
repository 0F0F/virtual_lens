#include "test.h"

/**
*	Function Definitions
*/

#ifdef _LSH_DEBUG
auto display_in_border(const cv::Mat& eye, std::function<bool(const Point&)> border_checker)
{
	list<Point> in_border;
	for (auto x = 0; x < eye.cols; x++)
	{
		for (auto y = 0; y < eye.rows; y++)
		{
			auto pos = Point{ x, y };
			if (border_checker(pos))
				in_border.push_back(pos);
		}
	}
	for (auto& pos : in_border)
		cv::circle(eye, pos, 2, Scalar{ 255 });
	cv::imshow("border", eye);
	return;
}
#endif
std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

#ifdef _CSH_DEBUG
void matdebug(const cv::Mat& m) {
	std::string ty = type2str(m.type());

	printf("mat debug: %s %dx%d\n", ty.c_str(), m.cols, m.rows);
}

std::string random_string(std::size_t length) {
	auto randchar = []() -> char
	{
		const char charset[] =
			"0123456789"
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"abcdefghijklmnopqrstuvwxyz";
		const std::size_t max_index = (sizeof(charset) - 1);
		return charset[rand() % max_index];
	};
	std::string str(length, 0);
	std::generate_n(str.begin(), length, randchar);
	return str;
}
#endif //_CSH_DEBUG

void extract_eye_roi(cv::Mat& img,
	cv::Mat& eye1ROI, cv::Mat& eye2ROI,
	cv::CascadeClassifier *pFace_cascade,
	cv::CascadeClassifier *pEyes_cascade) {
	//얼굴에서 눈 영역 ROI 설정
	std::vector<Rect> faces_vect;


	cv::Mat frame_gray;

	cvtColor(img, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces 
	pFace_cascade->detectMultiScale(frame_gray, faces_vect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (faces_vect.size() == 0) {
		throw std::runtime_error("no face detected");
	}
	else if (faces_vect.size() > 1) {
		//throw std::runtime_error("there is so many faces");
	}


	cv::Rect eyes_rect[2];

	point face_center(faces_vect[0].x + faces_vect[0].width*0.5, faces_vect[0].y + faces_vect[0].height*0.5);
	std::vector<cv::Rect> eyes_vect;

	pEyes_cascade->detectMultiScale(img, eyes_vect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
	if (eyes_vect.size() < 2) {
		throw std::runtime_error("too few eyes detected");
	}

	for (int i = 0; i < 2; ++i) {
		eyes_rect[i] = cv::Rect(eyes_vect[i].x, eyes_vect[i].y,
			eyes_vect[i].width, eyes_vect[i].height);
	}


	//ROI 설정
	eye1ROI = img(eyes_rect[0]);
	eye2ROI = img(eyes_rect[1]);
}

cv::Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh)
{
	cv::Mat oriMap = cv::Mat::zeros(ori.size(), CV_8UC3);
	cv::Vec3b red(0, 0, 255);
	cv::Vec3b cyan(255, 255, 0);
	cv::Vec3b green(0, 255, 0);
	cv::Vec3b yellow(0, 255, 255);
	for (int i = 0; i < mag.rows*mag.cols; i++)
	{
		float* magPixel = reinterpret_cast<float*>(mag.data + i * sizeof(float));
		if (*magPixel > thresh)
		{
			float* oriPixel = reinterpret_cast<float*>(ori.data + i * sizeof(float));
			cv::Vec3b* mapPixel = reinterpret_cast<cv::Vec3b*>(oriMap.data + i * 3 * sizeof(char));
			if (*oriPixel < 90.0)
				*mapPixel = red;
			else if (*oriPixel >= 90.0 && *oriPixel < 180.0)
				*mapPixel = cyan;
			else if (*oriPixel >= 180.0 && *oriPixel < 270.0)
				*mapPixel = green;
			else if (*oriPixel >= 270.0 && *oriPixel < 360.0)
				*mapPixel = yellow;
		}
	}

	return oriMap;
}

std::array<point, 4> detect_intersection_points(cv::Mat& eye, point center, double radius)
{
	cv::Mat hsv;
	cvtColor(eye, hsv, CV_BGR2HSV);

	const auto thresh = 30;
	const auto kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0;
	cv::Mat hsv_mean, over;
	cv::filter2D(hsv, hsv_mean, -1, kernel);
	cv::threshold(hsv_mean, over, thresh, 0xff, THRESH_BINARY_INV);

	const double tolerance = 0.1;
	std::vector<point> close_to_zero;
	std::mutex mtx_push;
	over.forEach<cv::Point3_<uint8_t>>(
		[center, radius, tolerance, &close_to_zero, &mtx_push]
	(cv::Point3_<uint8_t>& p, const int position[]) -> void
	{
		if (p.y == 0)
			return;

		auto pixel_pos = point{ position[1], position[0] };
		auto dist = (distance(pixel_pos, center));
		auto diff = abs(dist - radius);

		if (diff < tolerance)
		{
			mtx_push.lock();
			close_to_zero.emplace_back(std::move(pixel_pos));
			mtx_push.unlock();
		}
	});
	
	std::array<point, 4> intersect_points{
		point{eye.cols, eye.rows},	// for left top, start from right bottom
		point{eye.cols, 0},			// for left bottom, start from right top
		point{0, eye.rows},			// for right top, start from left bottom
		point{0,0} };				// for right bottom, start from left top

	for (auto& point : close_to_zero)
	{
		auto place = (point.x < center.x ? 0 : 2);
		if (point.y < intersect_points[place].y)
			intersect_points[place] = point;
		if (point.y > intersect_points[place + 1].y)
			intersect_points[place + 1] = point;
	}

#ifdef _LSH_DEBUG
	auto temp = eye;
	for (auto& point : intersect_points)
		cv::circle(temp, point, 5, Scalar{ 255 });
	cv::imshow("4 points", temp);
	cv::imshow("over", over);
#endif // _LSH_DEBUG
	
	return intersect_points;
}
void detect_pupil(cv::Mat& eyeROI, point *pCenter, double *pRadius) {

	//hough circle은 gray scale 이미지에 대해 동작
	cv::Mat frame_gray;
	cv::cvtColor(eyeROI, frame_gray, CV_BGR2GRAY);


	//Hough circle 전처리
	//histogram equalization + 노이즈 감소
	//gaussian blur 또는 median blur 사용
	cv::equalizeHist(frame_gray, frame_gray);
	//GaussianBlur(frame_gray, frame_gray, cv::Size(3, 3), 1.2, 1.2);
	cv::medianBlur(frame_gray, frame_gray, 3);

	//추출된 눈동자가 담길 벡터
	std::vector<cv::Vec3f> circles;

	//hough circle을 이용해 눈동자 추출
	cv::HoughCircles(
		frame_gray,
		circles,
		cv::HOUGH_GRADIENT,
		2,
		(double)frame_gray.rows / 8,
		200,
		std::min((double)frame_gray.rows / 2, (double)100),
		(double)frame_gray.rows / 20,
		(double)frame_gray.rows / 2
	);

#ifdef _CSH_DEBUG
	printf("# of Circles: %d\n", circles.size());
#endif // _CSH_DEBUG

	//눈동자 영역 output으로 넘기기
	const Vec3f& detected = circles[0];
	*pCenter = point(detected[0], detected[1]);
	*pRadius = detected[2];

#ifdef _CSH_DEBUG
	for (auto& c : circles) {
		cv::circle(frame_gray, point(c[0], c[1]), c[2], Scalar(255));
	}

	cv::imshow("debug_" + random_string(5), frame_gray);
#endif // _CSH_DEBUG
}

auto flood_fill(const point& seed, std::function<void(const point&)> fill, std::function<bool(const Point&)> in_area)
{
	function<bool(const Point&, const Point&)> comp =
		[]
	(const Point& a, const Point& b) -> bool
	{
		const auto point2value = [](const Point& point)
		{
			uint64_t value = point.x;
			value = value << (sizeof(point.x) * 8);
			value |= point.y;
			return value;
		};
		return
			point2value(a) < point2value(b);
	};

	stack<Point> targets;
	set<Point, decltype(comp)> processed{ comp };

	auto should_process =
		[&processed, &in_area]
	(const Point& point) -> bool
	{
		bool already_processed = (processed.find(point) != processed.end());
		processed.insert(point);

		return !already_processed && in_area(point);
	};

	targets.push(seed);


	while (!targets.empty())
	{
		auto target = targets.top();
		targets.pop();

		if (should_process(target))
		{
			fill(target);
			auto up = Point{ target.x, target.y - 1 };
			auto down = Point{ target.x, target.y + 1 };
			auto left = Point{ target.x - 1, target.y };
			auto right = Point{ target.x + 1, target.y };
			targets.push(up);
			targets.push(down);
			targets.push(left);
			targets.push(right);
		}
	}
}

cv::Mat adjust_lens(cv::Mat lens, point center, double radius, const array<point_set, 2>& eyelid, int rectX, int rectY)
{
	auto fill =
		[&lens]
	(const Point& point) -> void
	{
		try {
			auto& lens_pixel = lens.at<Vec4b>(point.y - 1, point.x - 1);
			lens_pixel[3] = 0;
#ifdef _LSH_DEBUG
			//std::cout << "cutting: " << point << endl;
#endif //_LSH_DEBUG
		}

		catch (...)
		{
			cout << point;
		}
	};

	auto in_eyeROI =
		std::bind(
			in_circle,
			std::placeholders::_1,
			center,
			radius);

	auto in_area = 
		[&in_eyeROI, rectX, rectY]
	(const point_set& eyelid)
	{
		return 
			[&eyelid, &in_eyeROI, rectX, rectY]
		(const Point& point)
		{
			// lens[j][i] == eye[j + rectY][i + rectX]
			auto as_eye = Point{ point.x + rectX, point.y + rectY };

			auto inside_eyeROI = in_eyeROI(as_eye);
			auto it = eyelid.find(as_eye);
			auto inside_eyelid = (eyelid.find(as_eye) == eyelid.cend());
			return
				inside_eyeROI &&
				inside_eyelid;
		};
	};

	auto in_area_upper = in_area(eyelid[0]);
	auto in_area_lower = in_area(eyelid[1]);
	auto upper_seed = Point{ center.x - rectX, int(center.y - rectY - radius + 1) };
	auto lower_seed = Point{ center.x - rectX, int(center.y - rectY + radius - 1) };
#ifdef _LSH_DEBUG
	cout << "starting lens cut : " << lens.rows << " , " << lens.cols << endl;
#endif
	flood_fill(upper_seed, fill, in_area_upper);
	flood_fill(lower_seed, fill, in_area_lower);

	return lens;
}
void overlay_lens(cv::Mat lens, cv::Mat& eyeROI,
	point *pCenter, double *pRadius,const array<point_set, 2>& eyelid) {

	//눈동자 영역 Rect 생성
	cv::Rect rect(point(pCenter->x - *pRadius, pCenter->y - *pRadius),
		point(pCenter->x + *pRadius, pCenter->y + *pRadius));

#ifdef _CSH_DEBUG
	//rectangle(eyeROI, rect, Scalar(255, 0, 0));
#endif // _CSH_DEBUG


	//렌즈 이미지를 눈동자 이미지에 맞춰서 resize 및 투명도 조절
	double diameter = *pRadius * 2;
	cv::resize(lens, lens, Size(diameter, diameter),
		0, 0,
		CV_INTER_CUBIC);

	//auto& adjusted_lens = lens;
	auto adjusted_lens = adjust_lens(lens, *pCenter, *pRadius, eyelid, rect.x, rect.y);

#ifdef _CSH_DEBUG
	printf("rect x:%d, y:%d, width:%d, height:%d\n", rect.x, rect.y,
		rect.width, rect.height);
	printf("resz lens rows:%d, cols:%d\n", lens.rows, lens.cols);
	//printf("lens ch:%d, eye_roi ch:%d\n", lens.channels(), eyeROI.channels());
#endif // _CSH_DEBUG


	//ROI 내 매 픽셀마다 합성 수행
	register int i, j;
	for (i = 0; i < adjusted_lens.cols; ++i) {
		for (j = 0; j < adjusted_lens.rows; ++j) {
			auto& adjusted_lens_pixel = adjusted_lens.at<Vec4b>(j, i);
			auto& eye_pixel = eyeROI.at<Vec4b>(j + rect.y, i + rect.x);
			if (adjusted_lens_pixel[3] == 0) continue;	//렌즈 알파값이 0일경우

												//BGR 각 채널마다 합성 수행
												//channel == 0 --> B channel
												//channel == 1 --> G channel
												//channel == 2 --> R channel
												//channel == 3 --> alpha channel
			for (int channel = 0; channel < 3; ++channel) {
				float target = eye_pixel[channel] / 255.0;
				float blend = adjusted_lens_pixel[channel] / 255.0;
				float result;

				//blend 모드 따라 formula 변경
				//각 모드는 포토샵에 있는 모드들 활용
				switch (blend_mode) {
				case BLEND_MODE::BLEND_DARKEN:
					result = std::min(target, blend);
					break;

				case BLEND_MODE::BLEND_MULTIPLY:
					result = target * blend;
					break;

				case BLEND_MODE::BLEND_COLOR_BURN:
					result = 1 - (1 - target) / blend;
					break;

				case BLEND_MODE::BLEND_LINEAR_BURN:
					result = target + blend - 1;
					break;

				case BLEND_MODE::BLEND_LIGHTEN:
					result = std::max(target, blend);
					break;

				case BLEND_MODE::BLEND_SCREEN:
					result = 1 - (1 - target)*(1 - blend);
					break;

				case BLEND_MODE::BLEND_COLOR_DODGE:
					result = target + blend;
					break;

				case BLEND_MODE::BLEND_OVERLAY:
					if (target > 0.5) {
						result = (1 - (1 - 2 * (target - 0.5)) * (1 - blend));
					}
					else {
						result = (2 * target*blend);
					}
					break;
				}

				eye_pixel[channel] = result * 255.0;
			}
		}
	}
}
double distance(Point p1, Point p2)
{
	return pow(
		pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2), 0.5);
}

array<point_set, 2> detect_eyelids(cv::Mat& eye, point center, double radius)
{
	auto cost = get_cost_func(eye, center, radius, center);

	auto intersect_points = detect_intersection_points(eye, center, radius);

	auto upper_left = intersect_points[0], upper_right = intersect_points[2];
	auto upper_border_checker = get_upper_border_checker(upper_left, upper_right, center, radius);
#ifdef _LSH_DEBUG
	//display_in_border(eye, upper_border_checker);
#endif
	auto upper_eyelid = detect_eyelid(eye, center, radius, upper_left, upper_right, upper_border_checker, cost);

	auto lower_left = intersect_points[1], lower_right = intersect_points[3];
	auto lower_border_checker = get_lower_border_checker(lower_left, lower_right, center, radius);
	auto lower_eyelid = detect_eyelid(eye, center, radius, lower_left, lower_right, lower_border_checker, cost);

#ifdef _LSH_DEBUG
	cout << "[upper eyelid] : " << upper_left << " ~ " << upper_right << endl;
	cout << "[lower eyelid] : " << lower_left << " ~ " << lower_right << endl;

	for (auto& pos : upper_eyelid)
	{
		cv::circle(eye, pos, 1, Scalar{ 255 });
	}
	for (auto& pos : lower_eyelid)
	{
		cv::circle(eye, pos, 1, Scalar{ 255 });
	}
	cv::imshow("eyelid?", eye);
#endif // _LSH_DEBUG

	return { upper_eyelid, lower_eyelid };

	// old ver
	/*
	static const Point directions[]
	{
	//{0, -1},	// 위
	{ 1, -1 },	// 위 오른쪽
	{ 1, 0 },		// 오른쪽
	{ 1, 1 }		// 오른쪽 아래
	//{0, 1}		// 아래
	};

	list<Point> ret{ left };
	auto tolerance = 10;
	while(distance(ret.back(), right) > tolerance && ret.size() < 80)
	//while (ret.back() != right && ret.size() < 80)	// 우상까지
	{
	auto curr = ret.back();
	unsigned int smallest = -1;
	auto dir = -1;

	for (auto i = 0; i < sizeof(directions) / sizeof(Point); i++)
	{
	auto next = curr + directions[i];
	unsigned int  weight = cost(curr, next);
	if (weight < smallest)
	{
	smallest = weight;
	dir = i;
	}
	}

	if (dir != -1)
	ret.push_back(curr + directions[dir]);
	else
	throw std::runtime_error("can't find eyelid path");
	}
	return ret;
	*/
}

point_set detect_eyelid(cv::Mat& eye, point center, double radius,
	Point left, Point right, 
	std::function<bool(const Point&)>& border_checker,
	std::function<double(const Point&)>& cost)
{
	auto cost_map = get_cost_map(left, border_checker, cost);
	point_set ret{ []
	(const Point& a, const Point& b) -> bool
	{
		const auto point2value = [](const Point& point)
		{
			uint64_t value = point.x;
			value = value << (sizeof(point.x) * 8);
			value |= point.y;
			return value;
		};
		return
			point2value(a) < point2value(b);
	} };
	auto curr = right;
	while (curr != left)
	{
		ret.insert(curr);
		curr = cost_map[curr].prev;
	}
#ifdef _LSH_DEBUG
	//for (auto& pix : cost_map)
	//{
	//	cv::circle(eye, pix.first, 2, Scalar{ 255 });
	//}
	//cv::imshow("area", eye);

#endif
	return ret;
}

auto get_neighbors(const Point& point, std::function<bool(const Point& point)>& on_mat) noexcept
{
	std::forward_list<weighted_point> ret;
	static const float ROUTE2 = pow(2, 0.5);
	const weighted_point candidates[]
	{
		{ { -1,-1 } , ROUTE2 },{ { 0,-1 } , 1 },{ { 1,-1 } , ROUTE2 },
		{ { -1,0 } , 1 }, /* { { 0,0 } , 0 },*/{ { 1,0 } , 1 },
		{ { -1,1 } , ROUTE2 },{ { 0,1 } , 1 },{ { 1,1 } , ROUTE2 }
	};

	for (auto& candidate : candidates)
	{
		auto candidate_pos = point + candidate.point;
		auto on_matrix = on_mat(candidate_pos);
		if (on_matrix)
		{
			ret.push_front({ candidate_pos, candidate.info.weight });
		}
	}
	return ret;
}


void add_weighted_point(list<weighted_point>& list, weighted_point&& point)
{
	auto it = list.begin();
	for (; it != list.end(); ++it)
	{
		if (it->info.weight > point.info.weight)
		{
			list.insert(it, std::move(point));
			return;
		}
	}
	list.emplace_back(std::move(point));
}

cost_map get_cost_map(
	Point seed,
	std::function<bool(const Point&)> in_map,
	std::function<double(const Point&)> cost)
{
	std::list<weighted_point> active_points{ weighted_point(seed, 0) };

	cost_map total_cost{ 
		[]
	(const Point& a, const Point& b) -> bool
	{
		const auto point2value = [](const Point& point)
		{
			uint64_t value = point.x;
			value = value << (sizeof(point.x) * 8);
			value |= point.y;
			return value;
		};
		return
			point2value(a) < point2value(b);
	}};
	total_cost[seed] = cost_info{};

	while (!active_points.empty())
	{
		auto min_cost = active_points.front();
		active_points.pop_front();

		auto neighbors = get_neighbors(min_cost.point, in_map);

		for (const auto& neighbor : neighbors)
		{
			auto x = min_cost.point;
			auto x_info = total_cost[x];
			auto Tx = x_info.weight;

			auto y = neighbor.point;
			auto current_Ty = Tx + cost(y) * neighbor.info.weight;

			auto y_processed = total_cost.find(y) != total_cost.end();
			if (y_processed)
			{
				auto y_info = total_cost[y];
				auto old_Ty = y_info.weight;
				if (current_Ty < old_Ty)
				{
					active_points.remove_if(
						[y]
					(const weighted_point& wp)
					{
						return wp.point == y;
					});
				}
				else continue;
			}

			add_weighted_point(active_points, { y, current_Ty });
			total_cost[y] = { current_Ty, x };
		}
	}

	return total_cost;
}
bool in_circle(const Point& point, Point center, double radius)
{
	auto dist = distance(point, center);
	return dist <= radius;
}
std::function<bool(const Point&)> 
get_upper_border_checker(Point left, Point right, Point eye_center, double eye_radius)
{
	auto border_line = 
		[left, right]
	(const int x) -> int
	{
		auto slope = (right.y - left.y) / double(right.x - left.x);
		auto constant = left.y - slope * left.x;
		return slope * x + constant;
	};
	auto upper =
		[border_line]
	(const Point& pos) -> bool
	{
		return border_line(pos.x) >= pos.y;
	};
	auto adjusted_radius = eye_radius * 1.1;

	return 
		[=]
	(const Point& point) -> bool
	{
		return
			point == left || point == right ||
			(
				in_circle(point, eye_center, adjusted_radius) &&
				upper(point)
			);
	};
}
std::function<bool(const Point&)>
get_lower_border_checker(Point left, Point right, Point eye_center, double eye_radius)
{
	auto border_line =
		[left, right]
	(const int x) -> int
	{
		auto slope = ((right.y - left.y) / double(right.x - left.x));
		auto constant = left.y - slope * left.x;
		return slope * x + constant;
	};
	auto lower =
		[border_line]
	(const Point& pos) -> bool
	{
		return border_line(pos.x) <= pos.y;
	};

	auto adjusted_radius = eye_radius * 1.1;

	return
		[=]
	(const Point& point) -> bool
	{
		return
			point == left || point == right ||
			(
				in_circle(point, eye_center, adjusted_radius) &&
				lower(point)
			);
	};
}

std::function<double(const Point&)> 
get_cost_func(const cv::Mat& eye, point center, double radius, point seed)
{
	// local cost fucntion l(p,q) := p->q = 
	//			0.4 canny(q) + 0.1orientation(p,q) + 0.1 magnitude(q) + 0.4 laplacian(q)
	Mat gray;
	cv::cvtColor(eye, gray, CV_BGR2GRAY);

	Mat canny;
	cv::Canny(gray, canny, 0, 255);

	auto& gradient_dst = eye;
	Mat orientation, magnitude; // gradient -
	Mat Sx, Sy;
	cv::Sobel(gradient_dst, Sx, CV_32F, 1, 0);
	cv::Sobel(gradient_dst, Sy, CV_32F, 0, 1);
	cv::phase(Sx, Sy, orientation, false);
	cv::magnitude(Sx, Sy, magnitude);

	Mat temp, laplacian;
	cv::Laplacian(gray, temp, CV_16S, 3, 1, 0);
	cv::convertScaleAbs(temp, laplacian);

#ifdef _LSH_DEBUG
	//cv::imshow("canny?", canny);
	//cv::imshow("Sx?", Sx);
	//cv::imshow("Sy?", Sy);
	//cv::imshow("ori?", orientation);
	//cv::imshow("mag?", magnitude);
	//cv::imshow("laplacian?", laplacian);
#endif // _LSH_DEBUG

	return
	[canny, orientation, magnitude, laplacian, seed]
	(const Point& point) -> double {
		return
			//canny.at<uchar>(q);
			0.4 * canny.at<uchar>(point) +
			//0.1 * (orientation.at<float>(point) - orientation.at<float>(seed)) +	// ??? 잘모르겟다
			0.2 * magnitude.at<float>(point) +
			0.4 * laplacian.at<uchar>(point);
	};
}
