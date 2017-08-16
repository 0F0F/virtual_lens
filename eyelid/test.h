#pragma once
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>
#include <array>
#include <forward_list>
#include <list>
#include <mutex>
#include <map>
#include <set>
#include <stack>
using namespace std;
using namespace cv;

// optional compile for debugging
//#define	_CSH_DEBUG
//#define _LSH_DEBUG

// optional compile for utilities
//#define _LSH_TIME_CONSUMPTION_CHECK
#define _LSH_WRITE_PROCEDURE_LOG
/**
*	enums
*/
enum RETURN_VALUES {
	UNHANDLED_EXCEPTIONS = -1,
	COMPLETE = 0,
	UNEXPECTED_ARGUMENTS
};
enum BLEND_MODE {
	BLEND_DARKEN,		//min(target, blend)
	BLEND_MULTIPLY,		//target * blend
	BLEND_COLOR_BURN,	//1-(1-target) / blend
	BLEND_LINEAR_BURN,	//target + blend - 1
	BLEND_LIGHTEN,		//max(target, blend)
	BLEND_SCREEN,		//1-(1-target)*(1-blend)
	BLEND_COLOR_DODGE,	//target/(1-blend)
	BLEND_OVERLAY		//...
};
/**
*	classes
*/
struct cost_info
{
	double weight;
	cv::Point prev;

	cost_info() : prev(-1, -1), weight(0) {}
	cost_info(double weight, cv::Point prev = { -1,-1 }) : prev(prev), weight(weight) {}
};

struct weighted_point
{
	cv::Point point;
	cost_info info;
	weighted_point(cv::Point point, double weight, cv::Point prev) : point(point), info(weight, prev) {}
	weighted_point(cv::Point point, double weight) : point(point), info(weight) {}
	weighted_point(cv::Point point, cost_info info) : point(point), info(info) {}
	weighted_point(cv::Point point) : point(point), info() {}
	weighted_point() : point(-1, -1), info() {}
};

/**
*	constants
*/
const cv::String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
const cv::String eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
const std::string window_name = "csh lens";
const std::string img_path = "data/face/002.jpg";
const std::string lens_path = "data/lens/007.png";
const BLEND_MODE blend_mode = BLEND_MODE::BLEND_OVERLAY;

using point = cv::Point;
using point_comparator = std::function<bool(const point&, const point&)>;
using point_set = std::set<point, point_comparator>;
using cost_map = std::map<point, cost_info, point_comparator>;

/**
*	function declaration
*/
//디버그용 함수들
std::string type2str(int type);
void matdebug(const cv::Mat& m);
std::string random_string(std::size_t length);

//영상처리용 함수들
void extract_eye_roi(cv::Mat& img,
	cv::Mat& eye1ROI, cv::Mat& eye2ROI,
	cv::CascadeClassifier *pFace_cascade,
	cv::CascadeClassifier *pEyes_cascade);

void detect_pupil(cv::Mat& eyeROI, point *pCenter, double *pRadius);

void overlay_lens(cv::Mat lens, cv::Mat& eyeROI,
	point *pCenter, double *pRadius, const std::array<point_set, 2>& eyelid);

cv::Mat adjust_lens(cv::Mat lens, point center, double radius, const std::array<point_set, 2>& eyelid, int rectX, int rectY);

std::array<point, 4> detect_intersection_points(cv::Mat& eye, point center, double radius);

Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0);

double distance(Point p1, Point p2);

std::array<point_set, 2>
detect_eyelids(cv::Mat& eye, point center, double radius);

point_set
detect_eyelid(cv::Mat& eye, point center, double radius,
	Point left, Point right,
	std::function<bool(const Point&)>& border_checker,
	std::function<double(const Point&)>& cost);

std::function<double(const Point&)> 
get_cost_func(const cv::Mat& eye, point center, double radius, point seed);


cost_map
get_cost_map(Point seed,
	std::function<bool(const Point&)> in_map,
	std::function<double(const Point&)> cost);

std::function<bool(const Point&)>
get_upper_border_checker(Point left, Point right, Point eye_center, double eye_radius);

std::function<bool(const Point&)>
get_lower_border_checker(Point left, Point right, Point eye_center, double eye_radius);

bool in_circle(const Point& point, Point center, double radius);
auto flood_fill(const Point& seed, std::function<void(const Point&)> fill, std::function<bool(const Point&)> in_area);
#ifdef _LSH_DEBUG
auto display_in_border(const cv::Mat& eye, std::function<bool(const Point&)> border_checker);
#endif