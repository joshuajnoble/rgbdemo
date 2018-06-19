#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include <opencv2/rgbd.hpp>

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
			int num_modalities, cv::Point offset, cv::Size size,
			cv::Mat& dst);

		std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
			int num_modalities, cv::Point offset, cv::Size size,
			cv::Mat& mask, cv::Mat& dst);

		void subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f);
		void filterPlane(IplImage * ap_depth, std::vector<IplImage *> & a_masks, std::vector<CvPoint> & a_chain, double f);
		void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename);
		void reprojectPoints(const std::vector<cv::Point3d>& proj, std::vector<cv::Point3d>& real, double f);
		
		int learning_lower_bound, learning_upper_bound;

		// Initialize LINEMOD data structures
		cv::Ptr<cv::linemod::Detector> detector;
		int num_modalities;

		cv::Mat depth, color;
		float matching_threshold;
		int num_classes;
		CvPoint templateRegion;
		bool learn_online;

		
};
