#include "ofApp.h"

double focal_length = 0; 

int previewWidth = 640;
int previewHeight = 480;

//--------------------------------------------------------------
void ofApp::setup(){

	//detector = cv::linemod::getDefaultLINEMOD();
	static const int T_LVLS[] = { 4, 15 };
	std::vector< cv::Ptr<cv::linemod::Modality> > modalities;
	modalities.push_back(new cv::linemod::ColorGradient());
	modalities.push_back(new cv::linemod::DepthNormal());
	detector = new cv::linemod::Detector(modalities, std::vector<int>(T_LVLS, T_LVLS + 2));

	// ??
	focal_length = cv::CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH;
	num_modalities = (int)detector->getModalities().size();

	learning_lower_bound = 90;
	learning_upper_bound = 95;

	num_classes = 0;
	learn_online = false;

	kinect.open();
	kinect.initDepthSource();
	kinect.initColorSource();

	updateDepthLookupTable();

	ofSetWindowShape(previewWidth * 2, previewWidth * 2);
}

void ofApp::updateDepthLookupTable() {
	unsigned char nearColor = 255;
	unsigned char farColor = 0;
	unsigned int maxDepthLevels = 10001;
	depthLookupTable.resize(maxDepthLevels);
	depthLookupTable[0] = 0;
	for (unsigned int i = 1; i < maxDepthLevels; i++) {
		depthLookupTable[i] = ofMap(i, 500, 4000, nearColor, farColor, true);
		//depthLookupTable[i] = ofMap(i, nearClipping, farClipping, nearColor, farColor, true);
	}
	depthPixels.allocate(512, 424, OF_IMAGE_GRAYSCALE);
}

//--------------------------------------------------------------
void ofApp::update(){

	// update the kinect
	kinect.update();

	if (!kinect.getColorSource()->isFrameNew() || !kinect.getDepthSource()->isFrameNew())
	{
		return;
	}

	color = ofxCv::toCv(kinect.getColorSource()->getPixels());

	auto dTemp = kinect.getDepthSource();

	////this next part only needs to happen once
	//{
	//	//load the depth to world table
	//	dTemp->getDepthToWorldTable(depthToWorldTable);

	//	//load it into our preview
	//	depthToWorldPreview.loadData(depthToWorldTable);
	//}

	int n = dTemp->getHeight() * dTemp->getWidth();
	for (int i = 0; i < n; i++) {
		depthPixels[i] = depthLookupTable[dTemp->getPixels()[i]];
	}

	depthToWorldPreview.loadData(depthPixels);
	depth = ofxCv::toCv(depthPixels);

	if (color.rows < 1 || depth.rows < 1) {
		return;
	}


	// get the sources:

	std::vector<cv::Mat> sources;

	cv::Mat resizedColor;
	cv::Size sdSize(960, 540);
	cv::resize(color, resizedColor, sdSize, CV_INTER_AREA);

	cv::Mat resizedDepth;
	cv::resize(depth, resizedDepth, sdSize, CV_INTER_AREA);

	sources.push_back(resizedColor);
	sources.push_back(resizedDepth);

	// Perform matching
	std::vector<cv::linemod::Match> matches;
	std::vector<cv::Mat> quantized_images;
	//match_timer.start();

	if (num_classes > 0) {

		detector->match(sources, (float) matching_threshold, matches, class_ids, quantized_images);

		int classes_visited = 0;
		std::set<std::string> visited;

		for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
		{
			cv::linemod::Match m = matches[i];

			if (visited.insert(m.class_id).second)
			{
				++classes_visited;
				//printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n", m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
				ofLog() << " similarity " << m.similarity << " x " << m.x << " y " << m.y << " id " << m.class_id << " template " << m.template_id << endl;

				// Draw matching template
				const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
				//drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));

				if (learn_online == true)
				{
					/// @todo Online learning possibly broken by new gradient feature extraction,
					/// which assumes an accurate object outline.

					// make a display - this might need to be elsewhere
					cv::Mat display = color.clone();

					// Compute masks based on convex hull of matched template
					cv::Mat color_mask, depth_mask;
					std::vector<CvPoint> chain = maskFromTemplate(templates, num_modalities, cv::Point(m.x, m.y), color.size(), color_mask, display);
					subtractPlane(depth, depth_mask, chain, focal_length);

					// If pretty sure (but not TOO sure), add new template
					if (learning_lower_bound < m.similarity && m.similarity < learning_upper_bound)
					{
						int template_id = detector->addTemplate(sources, m.class_id, depth_mask);
						if (template_id != -1)
						{
							//printf("***  (id %d) for existing object class %s***\n", template_id, m.class_id.c_str());
							ofLog() << " Added template " << endl;
						}
					}
				}
			}
		}
	}
}


//--------------------------------------------------------------
void ofApp::draw() {

	// Color is at 1920x1080 instead of 512x424 so we should fix aspect ratio
	float colorHeight = previewWidth * (kinect.getColorSource()->getHeight() / kinect.getColorSource()->getWidth());
	
	kinect.getColorSource()->draw(0, 0, previewWidth, colorHeight);
	//kinect.getDepthSource()->draw(previewWidth, 0, previewWidth, previewHeight);
	depthToWorldPreview.draw(previewWidth, 0);

	if (maskedImage.isAllocated()) {
		maskedImage.draw(0, previewHeight);
	}

	if (ofGetMousePressed()) {
		ofSetColor(0, 255, 0);
		ofNoFill();
		ofDrawRectangle(templateRegion.x, templateRegion.y, mouseX - templateRegion.x, mouseY - templateRegion.y);
		ofSetColor(255, 255, 255);
	}

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	string filename = "linemod_templates.yml";
	
	switch (key)
	{
	case 'h':
		//help();
		break;
	case 'm':
		// toggle printing match result
		//show_match_result = !show_match_result;
		//printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
		break;
	case 't':
		// toggle printing timings
		//show_timings = !show_timings;
		//printf("Show timings %s\n", show_timings ? "ON" : "OFF");
		break;
	case 'l':
		// toggle online learning
		learn_online = !learn_online;
		//printf("Online learning %s\n", learn_online ? "ON" : "OFF");
		break;
	case '[':
		// decrement threshold
		matching_threshold = max(matching_threshold - 1, -100.f);
		//printf("New threshold: %d\n", matching_threshold);
		break;
	case ']':
		// increment threshold
		matching_threshold = min(matching_threshold + 1, +100.f);
		//printf("New threshold: %d\n", matching_threshold);
		break;
	case 'w':
		// write model to disk
		
		writeLinemod(detector, filename);
		//printf("Wrote detector and templates to %s\n", filename.c_str());
		break;
	default:
		;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {
	templateRegion.x = x;
	templateRegion.y = y;
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {
	// Compute object mask by subtracting the plane within the ROI

	ofLog() << x << " " << y << endl;

	std::vector<CvPoint> chain(4);
	chain[0] = templateRegion;
	chain[1] = cv::Point(x, templateRegion.y);
	chain[2] = cv::Point(x, y);
	chain[3] = cv::Point(templateRegion.x, y);
	cv::Mat mask;
	subtractPlane(depth, mask, chain, focal_length);

	ofxCv::toOf(mask, maskedImage);

	cv::Size dSize(640, 480);
	cv::Mat resizedMask;
	cv::resize(mask, resizedMask, dSize, CV_INTER_LINEAR);

	cv::Mat resizedColor;
	cv::resize(color, resizedColor, dSize, CV_INTER_AREA);

	cv::Mat resizedDepth;
	cv::resize(depth, resizedDepth, dSize, CV_INTER_LINEAR);

	std::vector<cv::Mat> sources;
	sources.push_back(resizedColor);
	sources.push_back(resizedDepth);

	// Extract template
	std::string class_id = cv::format("class%d", num_classes);
	cv::Rect bb;
	int template_id = detector->addTemplate(sources, class_id, resizedMask, &bb);
	if (template_id != -1)
	{
		ofLog() << " *** Added template (id " << template_id << " for new object class  " << num_classes << " *** " << endl;
		//printf("Extracted at (%d, %d) size %dx%d\n", bb.x, bb.y, bb.width, bb.height);
	}
	else
	{
		ofLog() << "adding template failed " << endl;
	}
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}


// Adapted from cv_line_template::convex_hull
void ofApp::templateConvexHull(const std::vector<cv::linemod::Template>& templates,
	int num_modalities, cv::Point offset, cv::Size size,
	cv::Mat& dst)
{
	std::vector<cv::Point> points;
	for (int m = 0; m < num_modalities; ++m)
	{
		for (int i = 0; i < (int)templates[m].features.size(); ++i)
		{
			cv::linemod::Feature f = templates[m].features[i];
			points.push_back(cv::Point(f.x, f.y) + offset);
		}
	}

	std::vector<cv::Point> hull;
	cv::convexHull(points, hull);

	dst = cv::Mat::zeros(size, CV_8U);
	const int hull_count = (int)hull.size();
	const cv::Point* hull_pts = &hull[0];
	cv::fillPoly(dst, &hull_pts, &hull_count, 1, cv::Scalar(255));
}

std::vector<CvPoint> ofApp::maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
	int num_modalities, cv::Point offset, cv::Size size,
	cv::Mat& mask, cv::Mat& dst)
{
	templateConvexHull(templates, num_modalities, offset, size, mask);

	const int OFFSET = 30;
	cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), OFFSET);

	CvMemStorage * lp_storage = cvCreateMemStorage(0);
	CvTreeNodeIterator l_iterator;
	CvSeqReader l_reader;
	CvSeq * lp_contour = 0;

	cv::Mat mask_copy = mask.clone();
	IplImage mask_copy_ipl = mask_copy;
	cvFindContours(&mask_copy_ipl, lp_storage, &lp_contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	std::vector<CvPoint> l_pts1; // to use as input to cv_primesensor::filter_plane

	cvInitTreeNodeIterator(&l_iterator, lp_contour, 1);
	while ((lp_contour = (CvSeq *)cvNextTreeNode(&l_iterator)) != 0)
	{
		CvPoint l_pt0;
		cvStartReadSeq(lp_contour, &l_reader, 0);
		CV_READ_SEQ_ELEM(l_pt0, l_reader);
		l_pts1.push_back(l_pt0);

		for (int i = 0; i < lp_contour->total; ++i)
		{
			CvPoint l_pt1;
			CV_READ_SEQ_ELEM(l_pt1, l_reader);
			/// @todo Really need dst at all? Can just as well do this outside
			cv::line(dst, l_pt0, l_pt1, CV_RGB(0, 255, 0), 2);

			l_pt0 = l_pt1;
			l_pts1.push_back(l_pt0);
		}
	}
	cvReleaseMemStorage(&lp_storage);

	return l_pts1;
}


void ofApp::reprojectPoints(const std::vector<cv::Point3d>& proj, std::vector<cv::Point3d>& real, double f)
{
	real.resize(proj.size());
	double f_inv = 1.0 / f;

	for (int i = 0; i < (int)proj.size(); ++i)
	{
		double Z = proj[i].z;
		real[i].x = (proj[i].x - 320.) * (f_inv * Z);
		real[i].y = (proj[i].y - 240.) * (f_inv * Z);
		real[i].z = Z;
	}
}

void ofApp::filterPlane(IplImage * ap_depth, std::vector<IplImage *> & a_masks, std::vector<CvPoint> & a_chain, double f)
{
	const int l_num_cost_pts = 200;

	float l_thres = 4;

	IplImage * lp_mask = cvCreateImage(cvGetSize(ap_depth), IPL_DEPTH_8U, 1);
	cvSet(lp_mask, cvRealScalar(0));

	std::vector<CvPoint> l_chain_vector;

	float l_chain_length = 0;
	float * lp_seg_length = new float[a_chain.size()];

	for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
	{
		float x_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x);
		float y_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y);
		lp_seg_length[l_i] = sqrt(x_diff*x_diff + y_diff * y_diff);
		l_chain_length += lp_seg_length[l_i];
	}
	for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
	{
		if (lp_seg_length[l_i] > 0)
		{
			int l_cur_num = cvRound(l_num_cost_pts * lp_seg_length[l_i] / l_chain_length);
			float l_cur_len = lp_seg_length[l_i] / l_cur_num;

			for (int l_j = 0; l_j < l_cur_num; ++l_j)
			{
				float l_ratio = (l_cur_len * l_j / lp_seg_length[l_i]);

				CvPoint l_pts;

				l_pts.x = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x) + a_chain[l_i].x);
				l_pts.y = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y) + a_chain[l_i].y);

				l_chain_vector.push_back(l_pts);
			}
		}
	}
	std::vector<cv::Point3d> lp_src_3Dpts(l_chain_vector.size());

	for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
	{
		lp_src_3Dpts[l_i].x = l_chain_vector[l_i].x;
		lp_src_3Dpts[l_i].y = l_chain_vector[l_i].y;
		lp_src_3Dpts[l_i].z = CV_IMAGE_ELEM(ap_depth, unsigned short, cvRound(lp_src_3Dpts[l_i].y), cvRound(lp_src_3Dpts[l_i].x));
		//CV_IMAGE_ELEM(lp_mask,unsigned char,(int)lp_src_3Dpts[l_i].Y,(int)lp_src_3Dpts[l_i].X)=255;
	}
	//cv_show_image(lp_mask,"hallo2");

	reprojectPoints(lp_src_3Dpts, lp_src_3Dpts, f);

	CvMat * lp_pts = cvCreateMat((int)l_chain_vector.size(), 4, CV_32F);
	CvMat * lp_v = cvCreateMat(4, 4, CV_32F);
	CvMat * lp_w = cvCreateMat(4, 1, CV_32F);

	for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
	{
		CV_MAT_ELEM(*lp_pts, float, l_i, 0) = (float)lp_src_3Dpts[l_i].x;
		CV_MAT_ELEM(*lp_pts, float, l_i, 1) = (float)lp_src_3Dpts[l_i].y;
		CV_MAT_ELEM(*lp_pts, float, l_i, 2) = (float)lp_src_3Dpts[l_i].z;
		CV_MAT_ELEM(*lp_pts, float, l_i, 3) = 1.0f;
	}
	cvSVD(lp_pts, lp_w, 0, lp_v);

	float l_n[4] = { CV_MAT_ELEM(*lp_v, float, 0, 3),
		CV_MAT_ELEM(*lp_v, float, 1, 3),
		CV_MAT_ELEM(*lp_v, float, 2, 3),
		CV_MAT_ELEM(*lp_v, float, 3, 3) };

	float l_norm = sqrt(l_n[0] * l_n[0] + l_n[1] * l_n[1] + l_n[2] * l_n[2]);

	l_n[0] /= l_norm;
	l_n[1] /= l_norm;
	l_n[2] /= l_norm;
	l_n[3] /= l_norm;

	float l_max_dist = 0;

	for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
	{
		float l_dist = l_n[0] * CV_MAT_ELEM(*lp_pts, float, l_i, 0) +
			l_n[1] * CV_MAT_ELEM(*lp_pts, float, l_i, 1) +
			l_n[2] * CV_MAT_ELEM(*lp_pts, float, l_i, 2) +
			l_n[3] * CV_MAT_ELEM(*lp_pts, float, l_i, 3);

		if (fabs(l_dist) > l_max_dist)
			l_max_dist = l_dist;
	}
	ofLog() << "plane: " << l_n[0] << ";" << l_n[1] << ";" << l_n[2] << ";" << l_n[3] << " maxdist: " << l_max_dist << " end" << std::endl;
	int l_minx = ap_depth->width;
	int l_miny = ap_depth->height;
	int l_maxx = 0;
	int l_maxy = 0;

	for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
	{
		l_minx = std::min(l_minx, a_chain[l_i].x);
		l_miny = std::min(l_miny, a_chain[l_i].y);
		l_maxx = std::max(l_maxx, a_chain[l_i].x);
		l_maxy = std::max(l_maxy, a_chain[l_i].y);
	}
	int l_w = l_maxx - l_minx + 1;
	int l_h = l_maxy - l_miny + 1;
	int l_nn = (int)a_chain.size();

	CvPoint * lp_chain = new CvPoint[l_nn];

	for (int l_i = 0; l_i < l_nn; ++l_i)
		lp_chain[l_i] = a_chain[l_i];

	cvFillPoly(lp_mask, &lp_chain, &l_nn, 1, cvScalar(255, 255, 255));

	delete[] lp_chain;

	//cv_show_image(lp_mask,"hallo1");

	std::vector<cv::Point3d> lp_dst_3Dpts(l_h * l_w);

	int l_ind = 0;

	for (int l_r = 0; l_r < l_h; ++l_r)
	{
		for (int l_c = 0; l_c < l_w; ++l_c)
		{
			lp_dst_3Dpts[l_ind].x = l_c + l_minx;
			lp_dst_3Dpts[l_ind].y = l_r + l_miny;
			lp_dst_3Dpts[l_ind].z = CV_IMAGE_ELEM(ap_depth, unsigned short, l_r + l_miny, l_c + l_minx);
			++l_ind;
		}
	}
	reprojectPoints(lp_dst_3Dpts, lp_dst_3Dpts, f);

	l_ind = 0;

	for (int l_r = 0; l_r < l_h; ++l_r)
	{
		for (int l_c = 0; l_c < l_w; ++l_c)
		{
			float l_dist = (float)(l_n[0] * lp_dst_3Dpts[l_ind].x + l_n[1] * lp_dst_3Dpts[l_ind].y + lp_dst_3Dpts[l_ind].z * l_n[2] + l_n[3]);

			++l_ind;

			if (CV_IMAGE_ELEM(lp_mask, unsigned char, l_r + l_miny, l_c + l_minx) != 0)
			{
				if (fabs(l_dist) < std::max(l_thres, (l_max_dist * 2.0f)))
				{
					for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
					{
						int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
						int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

						CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 0;
					}
				}
				else
				{
					for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
					{
						int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
						int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

						CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 255;
					}
				}
			}
		}
	}
	cvReleaseImage(&lp_mask);
	cvReleaseMat(&lp_pts);
	cvReleaseMat(&lp_w);
	cvReleaseMat(&lp_v);
}

void ofApp::subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f)
{
	mask = cv::Mat::zeros(depth.size(), CV_8U);
	std::vector<IplImage*> tmp;
	IplImage mask_ipl = mask;
	tmp.push_back(&mask_ipl);
	IplImage depth_ipl = depth;
	filterPlane(&depth_ipl, tmp, chain, f);
}


void ofApp::writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	detector->write(fs);

	std::vector<cv::String> ids = detector->classIds();
	fs << "classes" << "[";
	for (int i = 0; i < (int)ids.size(); ++i)
	{
		fs << "{";
		detector->writeClass(ids[i], fs);
		fs << "}"; // current class
	}
	fs << "]"; // classes
}