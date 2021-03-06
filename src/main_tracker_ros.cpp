#include "libfreenect/libfreenect.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <ros/ros.h>
#include <geometry_msgs/Point.h>

using namespace cv;
using namespace std;


class myMutex {
	public:
		myMutex() {
			pthread_mutex_init( &m_mutex, NULL );
		}
		void lock() {
			pthread_mutex_lock( &m_mutex );
		}
		void unlock() {
			pthread_mutex_unlock( &m_mutex );
		}
	private:
		pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice {
	public:
		MyFreenectDevice(freenect_context *_ctx, int _index)
	 		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),
			m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false),
			m_new_depth_frame(false), depthMat(Size(640,480),CV_16UC1),
			rgbMat(Size(640,480), CV_8UC3, Scalar(0)),
			ownMat(Size(640,480),CV_8UC3,Scalar(0)) {
			
			for( unsigned int i = 0 ; i < 2048 ; i++) {
				float v = i/2048.0;
				v = std::pow(v, 3)* 6;
				m_gamma[i] = v*6*256;
			}
		}
		
		// Do not call directly even in child
		void VideoCallback(void* _rgb, uint32_t timestamp) {
			// std::cout << "RGB callback" << std::endl;
			m_rgb_mutex.lock();
			uint8_t* rgb = static_cast<uint8_t*>(_rgb);
			rgbMat.data = rgb;
			m_new_rgb_frame = true;
			m_rgb_mutex.unlock();
		};
		
		// Do not call directly even in child
		void DepthCallback(void* _depth, uint32_t timestamp) {
			// std::cout << "Depth callback" << std::endl;
			m_depth_mutex.lock();
			uint16_t* depth = static_cast<uint16_t*>(_depth);
			depthMat.data = (uchar*) depth;
			m_new_depth_frame = true;
			m_depth_mutex.unlock();
		}
		
		bool getVideo(Mat& output) {
			m_rgb_mutex.lock();
			if(m_new_rgb_frame) {
				cv::cvtColor(rgbMat, output, CV_RGB2BGR);
				m_new_rgb_frame = false;
				m_rgb_mutex.unlock();
				return true;
			} else {
				m_rgb_mutex.unlock();
				return false;
			}
		}
		
		bool getDepth(Mat& output) {
				m_depth_mutex.lock();
				if(m_new_depth_frame) {
					depthMat.copyTo(output);
					m_new_depth_frame = false;
					m_depth_mutex.unlock();
					return true;
				} else {
					m_depth_mutex.unlock();
					return false;
				}
			}
	private:
		std::vector<uint8_t> m_buffer_depth;
		std::vector<uint8_t> m_buffer_rgb;
		std::vector<uint16_t> m_gamma;
		Mat depthMat;
		Mat rgbMat;
		Mat ownMat;
		myMutex m_rgb_mutex;
		myMutex m_depth_mutex;
		bool m_new_rgb_frame;
		bool m_new_depth_frame;
};


int main(int argc, char **argv) {
	const int MIN_OBJECT_AREA = 120;//20*20;
	bool die(false);
	
	Mat depthMat(Size(640,480),CV_16UC1);
	Mat depthf (Size(640,480),CV_8UC1);
	Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
	Mat hsvMat(Size(640,480), CV_8UC3,Scalar(0));
	Mat threshMat(Size(640,480), CV_8UC3,Scalar(0));
	Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));
	
	// The next two lines must be changed as Freenect::Freenect
	// isn't a template but the method createDevice:
	// Freenect::Freenect<MyFreenectDevice> freenect;
	// MyFreenectDevice& device = freenect.createDevice(0);
	// by these two lines:

	Freenect::Freenect freenect;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

	namedWindow("rgb",CV_WINDOW_AUTOSIZE);
	// namedWindow("drawing",CV_WINDOW_AUTOSIZE);
	// namedWindow("filteredImg", CV_WINDOW_AUTOSIZE);
	
	int Blvalue = 0;
	int Glvalue = 64;
	int Rlvalue = 0;
	int Bhvalue = 38;
	int Ghvalue = 241;
	int Rhvalue = 255;

	// createTrackbar("lowB", "filteredImg", &Blvalue, 255);
	// createTrackbar("lowG", "filteredImg", &Glvalue, 255);
	// createTrackbar("lowR", "filteredImg", &Rlvalue, 255);
	// createTrackbar("highB", "filteredImg", &Bhvalue, 255);
	// createTrackbar("highG", "filteredImg", &Ghvalue, 255);
	// createTrackbar("highR", "filteredImg", &Rhvalue, 255);

	Moments momentArr;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	// namedWindow("depth",CV_WINDOW_AUTOSIZE);
	device.startVideo();

	device.startDepth();
	RNG rng(12345);
	Mat drawing;
	Point2f center;
	Point2f endEffector;
	Point2f pendulum;
	Scalar color;
	float radius;
	std::vector<Point2f> tracked_object_centers(2);
	float dx, dy, dxerr, dyerr, dxprev = 0, dyyprev = 0;
	int point_idx = 0;


	// Ros stuff -----------------------------------------------------------

	ros::init(argc, argv, "ball_tracker");
	ros::NodeHandle nh;
	ros::Rate loop_rate(100); // is this going that fast? Ans: yes it is
	
	ros::Publisher state_publisher = nh.advertise<geometry_msgs::Point>("ball_tracker", 10);

	geometry_msgs::Point relative_position;

	// End of Ros Stuff ----------------------------------------------------

	while ( ros::ok() && !die) {

		device.getVideo(rgbMat);
		// device.getDepth(depthMat);

		cv::cvtColor(rgbMat, threshMat, COLOR_BGR2HSV);
		cv::GaussianBlur( rgbMat, rgbMat, Size(9,9), 0,0);

		cv::inRange(rgbMat, cv::Scalar(Blvalue, Glvalue, Rlvalue), cv::Scalar(Bhvalue,Ghvalue,Rhvalue), threshMat);
		// cv::imshow("hsv", hsvMat);
		// cv::imshow("filteredImg", threshMat);
		// cv::blur(threshMat, threshMat, Size( 4, 4 ));

		cv::findContours(threshMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		//drawing = Mat::zeros( threshMat.size(), CV_8UC3 );
		point_idx = 0;
  		for( size_t i = 0; i < contours.size(); i++ )
		{
			momentArr = cv::moments(contours[(int)i]);
			if (momentArr.m00 > MIN_OBJECT_AREA) {
				// std::cout << momentArr.m00 << std::endl;
				cv::minEnclosingCircle(contours[(int)i], center, radius);
				// if (center.x > 180 && center.x < 540){
				color = Scalar( 255, 255, 255 );
				//cv::circle(drawing, center, radius, color);
				// cv::circle(rgbMat, center, radius, color, 1);
				cv::circle(rgbMat, center, 20, color, 1);
				std::string center_string = std::to_string((int)center.x) + " " + std::to_string((int)center.y);
				cv::putText(rgbMat, center_string, center, FONT_HERSHEY_SIMPLEX, 0.5, color);
				
				tracked_object_centers[point_idx] = center;
				point_idx++;
				// }
			}
			//drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy);
		}
		if (point_idx == 2) {
			for (int i = 0; i < 2; i++) {
				dx = tracked_object_centers[i].x - 290;
				dy = tracked_object_centers[i].y - 180;
				if (sqrt(dx*dx + dy*dy) < 50) {
					endEffector = tracked_object_centers[i];
				} else {
					pendulum = tracked_object_centers[i];
				}
			}
			// dx = tracked_object_centers[0].x - tracked_object_centers[1].x;
			// dy = tracked_object_centers[0].y - tracked_object_centers[1].y;
			dx = pendulum.x - endEffector.x;
			dy = pendulum.y - endEffector.y;
			dy = -dy;
			relative_position.x = dx;
			relative_position.y = dy;
			state_publisher.publish(relative_position);
			// std::cout << "there were two" << std::endl;
			// double l = sqrt( dx * dx + dy * dy);
			// // std::cout << dx << " " << dy << " " << asin(dx/l) << std::endl;
			// if (dx < 0 && dy > 0) {
			// 	// std::cout << asin(dx/l) - M_PI/2 << std::endl;
			// 	std::cout << dx << " " << dy << " " << atan2(dx,dy)<< std::endl;
			// } else if ( dx > 0  && dy > 0) {
			// 	// std::cout << asin(dx/l) + M_PI/2 << std::endl;
			// 	std::cout << dx << " " << dy << " " << atan2(dx,dy) << std::endl;
			// } else {
			// 	std::cout << dx << " " << dy << " " << atan2(dx,dy) << std::endl;
			// }
		} else {
			std::cout << "There were not two " << std::endl;
		}
		// std::cout << dx << " " << dy << std::endl;
		cv::imshow("rgb", rgbMat);

		// cv::imshow("drawing", drawing);
		// depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
		// cv::imshow("depth",depthf);
		char k = cvWaitKey(5);
		if( k == 27 ){
			cvDestroyWindow("rgb");
			cvDestroyWindow("depth");
			break;
		}

		ros::spinOnce();
		loop_rate.sleep();
	}
	
	device.stopVideo();
	device.stopDepth();
	return 0;
}
