#include "people_counter.h"
#include <ctime>

PeopleCounter::PeopleCounter(cv::VideoCapture& cap,
                             std::string cnf_path, std::string wts_path, std::string nms_path,
                             float ct, float st,
                             int iw, int ih,
                             int cw, int ch,
                             float zsf, float fct, float hst, float rmg, int dt) :
_capture(cap),
_frameRegionToShow({ 0, 0, 0, 0 }),
_zoomSpeedFactor(zsf),
_peopleQty(0),
_modelConfigurationFile(cnf_path),
_modelWeightsFile(wts_path),
_classesFile(nms_path),
_confThreshold(ct),
_nmsThreshold(st),
_inpWidth(iw),
_inpHeight(ih),
_threadsEnabled(true),
_frameRegionUpdateThreshold(fct),
_frameRegionUpdateThresholdHyst(hst),
_frameRegionMarginRatio(rmg),
_timeCountForRegion(dt)

{
    _capture.set(cv::CAP_PROP_FRAME_WIDTH, cw);
    _capture.set(cv::CAP_PROP_FRAME_HEIGHT, ch);
    
    _captureFrameWidth = static_cast<int>(_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    _captureFrameHeight = static_cast<int>(_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    _frameRegionToShow = { 0, 0, _captureFrameWidth, _captureFrameHeight };
    _frameRegionToShowZoomed = { 0, 0, _captureFrameWidth, _captureFrameHeight };
    
    // Setup the model
    _net = cv::dnn::readNetFromDarknet(_modelConfigurationFile, _modelWeightsFile);
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    std::ifstream ifs(_classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) {
        _classes.push_back(line);
    }
}

void PeopleCounter::producer() {
    std::cout << "\nStarting Producer Thread\n";
    cv::Mat frame;
    
    while (_threadsEnabled) {
        _capture.read(frame);
        
        {
            std::lock_guard<std::mutex> lck(_mutexFrameCapture);
            _lastCapturedFrame = frame.clone();
        }
        
        // Stop the program if no video stream
        if (frame.empty()) {
            _threadsEnabled = false;
            break;
        }
    }
    if (_capture.isOpened()) {
        _capture.release();
    }
    std::cout << "\nStopping Producer Thread\n";
}

void PeopleCounter::processor() {
    std::cout << "\nStarting Processor Thread\n";
    cv::Mat frame;
    
    while (_threadsEnabled) {
        {
            std::lock_guard<std::mutex> lck(_mutexFrameCapture);
            frame = _lastCapturedFrame.clone();
        }
        
        if (!frame.empty()) {
            processFrame(frame);
            std::cout << "There are [ " << _peopleQty << " ] peoples\n";
        }
    }
    std::cout << "\nStopping Processor Thread\n";
}

void PeopleCounter::runThreads() {
    std::thread producer_t(&PeopleCounter::producer, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::thread processor_t(&PeopleCounter::processor, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Create a window
    static const std::string kWinName = "people counter";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    while (_threadsEnabled) {
        if (cv::waitKey(1) >= 0) {
            _threadsEnabled = false;
            break;
        }
        
        {
            std::lock(_mutexFrameRegion, _mutexFrameOverlay, _mutexFrameCapture);
            std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
            std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
            std::lock_guard<std::mutex> lckCapture(_mutexFrameCapture, std::adopt_lock);
            
            updateFrameRegionToShow();
            
            if (!_lastCapturedFrame.empty() && !_lastOverlayFrame.empty()) {
                // Blur the background
                cv::Mat blurred = cv::Mat::zeros(_lastCapturedFrame.size(), _lastCapturedFrame.type());
                cv::GaussianBlur(_lastCapturedFrame, blurred, cv::Size(15, 15), 0.0);
               
                blurred.copyTo(_lastCapturedFrame, _blurMask);
                
                cv::bitwise_or(_lastCapturedFrame, _lastOverlayFrame, _lastOverlayedFrame);
            }
            
            if (!_lastOverlayedFrame.empty()) {
                cv::Mat frame = _lastOverlayedFrame(_frameRegionToShowZoomed);
                cv::resize(frame, frame, cv::Size(_captureFrameWidth, _captureFrameHeight));
                
                if (!frame.empty()) {
                    cv::imshow(kWinName, frame);
                }
            }
        }
    }
    
    cv::destroyAllWindows();
    
    producer_t.join();
    processor_t.join();
}

int PeopleCounter::getPeopleQty() {
    return _peopleQty;
}

void PeopleCounter::processFrame(cv::Mat& frame) {
    // Create a 4D blob from a frame.
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(_inpWidth, _inpHeight), cv::Scalar(0, 0, 0), true, false);
    
    // Nets forward pass
    std::vector<cv::Mat> outs;
    _net.setInput(blob);
    _net.forward(outs, getOutputsNames(_net));
    
    // Filter out low confidence objects
    _peopleQty = countPeople(frame, outs);
}

int PeopleCounter::countPeople(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
	static int countFrame = _timeCountForRegion;
	static cv::Rect prevFrameRect = { 0, 0, _captureFrameWidth, _captureFrameHeight };
	static cv::Rect prevFrameRegionToShow = { 0, 0, _captureFrameWidth, _captureFrameHeight };
	static std::clock_t start = std::clock();
	static std::clock_t update = std::clock();
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
	static int count_test = 0;
	int i_num = 0;
	int j_num = 0;
	double mat_data[85];
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
			for (int index = 0; index < 85; index++) {
				mat_data[index] = data[index];
			}
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			count_test++;
			i_num = i;
			j_num = j;
            if (confidence > _confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression
    std::vector<int> indices;
	count_test = 0;
    int peopleQty = 0;
    cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _nmsThreshold, indices);
    
    {
        std::lock(_mutexFrameRegion, _mutexFrameOverlay);
        std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
        std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
        
        _lastOverlayFrame = cv::Mat::zeros(frame.size(), frame.type());
        _blurMask = cv::Mat::ones(frame.size(), CV_8UC1);
        cv::Rect frameRegion = { _captureFrameWidth, _captureFrameHeight, (-1)*_captureFrameWidth, (-1)*_captureFrameHeight };
        
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            
            //if (_classes[classIds[idx]] == "person") {
                peopleQty++;
                drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, _lastOverlayFrame);
                
                adjustBlurMask(box);
                
                // Expand the frame region to show to contain all objects
                adjustFrameRegion(frameRegion, box, true);				
            //}
        }

		if (std::clock() - start > countFrame){
			float prevArea = getRegionArea(prevFrameRect);
			float curArea = getRegionArea(frameRegion);
			if ((prevArea / curArea <= 1 && prevArea / curArea > 0.95) || (curArea / prevArea <= 1 && curArea / prevArea > 0.95)) {
				bool isOver = isOverRect(prevFrameRegionToShow, frameRegion);
				if (isOver || curArea / getRegionArea(prevFrameRegionToShow) < _frameRegionUpdateThreshold - _frameRegionUpdateThresholdHyst) {
					if ((peopleQty > 0) && (frameRegion.height > 0 && frameRegion.width > 0)) {
						prevFrameRegionToShow = frameRegion;
					} else {
						prevFrameRegionToShow = cv::Rect(0, 0, _captureFrameWidth, _captureFrameHeight);
					}
				} 
			}
			prevFrameRect = frameRegion;
			start = std::clock();			
		}

		_frameRegionToShow = prevFrameRegionToShow;

        
        // Put efficiency information.
        // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = _net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        cv::putText(_lastOverlayFrame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    }
    
    return peopleQty;
}

void PeopleCounter::padAspectRatio(cv::Mat& img, float ratio) {
    float width = img.cols;
    float height = img.rows;

    int widthPadding = std::round((height * ratio - width) / 2);
    int heightPadding = std::round((width / ratio - height) / 2);
    
    widthPadding = clip(widthPadding, 0, int(width));
    heightPadding = clip(heightPadding, 0, int(height));
    
    cv::copyMakeBorder(img, img, heightPadding, heightPadding, 0, 0, cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(img, img, 0, 0, widthPadding, widthPadding, cv::BORDER_ISOLATED, 0);
}

void PeopleCounter::adjustBlurMask(cv::Rect& region) {
    boundRegionToCaptureFrame(region);
    _blurMask(region).setTo(cv::Scalar(0));
}

float PeopleCounter::getRegionArea(cv::Rect& region) {
    return region.width * region.height;
}

bool PeopleCounter::isOverRect(cv::Rect prevRect, cv::Rect curRect) {
	int preBottomX = prevRect.x + prevRect.width;
	int preBottomY = prevRect.y + prevRect.height;
	int curBottomX = curRect.x + curRect.width;
	int curBottomY = curRect.y + curRect.height;
	toMarginPoints(prevRect.x, prevRect.y, preBottomX, preBottomY);
	if (prevRect.x > curRect.x || prevRect.y > curRect.y || preBottomX < curBottomX || preBottomY < curBottomY) {
		return true;
	}
	else {
		return false;
	}
}

void PeopleCounter::adjustFrameRegion(cv::Rect& region, cv::Rect& box, bool keepAspectRatio) {
    int leftTopX = std::min(box.x, region.x);
    int leftTopY = std::min(box.y, region.y);
    int rightBottomX = std::max(region.x + region.width, box.x + box.width);
    int rightBottomY = std::max(region.y + region.height, box.y + box.height);
    region.x = leftTopX;
    region.y = leftTopY;
    region.width = rightBottomX - leftTopX;
    region.height = rightBottomY - leftTopY;
    
    // Keep aspect ratio
    if (keepAspectRatio) {
        double width = region.width;
        double height = region.height;
        double ratio = (double)_captureFrameWidth / (double)_captureFrameHeight;
        cv::Point regionCenter(region.x + region.width / 2, region.y + region.height / 2);
        
        // Calculate the new hight and width to retain the aspect ratio
        if (height > width * 2.5 / ratio) {
            width = width * 2.5;
            width = clip(int(width), 0, _captureFrameWidth);            
            height = width / ratio;
        } else {
			double tmpWidth = height * ratio;
			if (tmpWidth > width) {
				tmpWidth = clip(int(tmpWidth), 0, _captureFrameWidth);
				width = tmpWidth;
				height = width / ratio;
			} else {
				height = width / ratio;
				height = clip(int(height), 0, _captureFrameHeight);
				width = height * ratio;
			}
        }
        
        /* PROCESS THE HORIZONTAL DIMENSION */
        
        leftTopX = regionCenter.x - width / 2;
        
        if (leftTopX < 0) {
            // shift the region by border overshoot
            rightBottomX = regionCenter.x + width / 2 - leftTopX;
            leftTopX = 0;
        } else {
            rightBottomX = regionCenter.x + width / 2;
        }
        
        if (rightBottomX > _captureFrameWidth) {
            leftTopX -= (rightBottomX - _captureFrameWidth);
            rightBottomX = _captureFrameWidth;
        }
        
        /* PROCESS THE VERTICAL DIMENSION */
        
        rightBottomY = leftTopY + height;
        
        if (rightBottomY > _captureFrameHeight) {
            leftTopY -= (rightBottomY - _captureFrameHeight);
            rightBottomY = _captureFrameHeight;
        }
        
        region.x = leftTopX;
        region.y = leftTopY;
        region.width = rightBottomX - leftTopX;
        region.height = rightBottomY - leftTopY;
    }
}

void PeopleCounter::boundRegionToCaptureFrame(cv::Rect& region) {
    int leftTopX, leftTopY, rightBottomX, rightBottomY;
    
    boxToPoints(region, leftTopX, leftTopY, rightBottomX, rightBottomY);
    
    leftTopX = clip(leftTopX, 0, _captureFrameWidth);
    leftTopY = clip(leftTopY, 0, _captureFrameHeight);
    rightBottomX = clip(rightBottomX, 0, _captureFrameWidth);
    rightBottomY = clip(rightBottomY, 0, _captureFrameHeight);
    
    pointsToBox(region, leftTopX, leftTopY, rightBottomX, rightBottomY);
}

void PeopleCounter::boxToPoints(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY) {
    leftTopX = box.x;
    leftTopY = box.y;
    rightBottomX = box.x + box.width;
    rightBottomY = box.y + box.height;
}

int PeopleCounter::toMarginPoints(int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY){
	int width = rightBottomX - leftTopX;
	int height = rightBottomY - leftTopY;
	int newWidth = width * (1 + 2 * _frameRegionMarginRatio);
	int newHeight = height * (1 + 2 * _frameRegionMarginRatio);
	int newLeftTopX = leftTopX - width * _frameRegionMarginRatio;
	int newLeftTopY = leftTopY - height * _frameRegionMarginRatio;
	int newRightBottomX = rightBottomX + width * _frameRegionMarginRatio;
	int newRightBottomY = rightBottomY + height * _frameRegionMarginRatio;
	if (newWidth > _captureFrameWidth || newHeight > _captureFrameHeight){
		return 0;
	}
	else {
		if (newLeftTopX < 0) {
			newLeftTopX = 0;
			newRightBottomX = newWidth;
		}
		if (newLeftTopY < 0) {
			newLeftTopY = 0;
			newRightBottomY = newHeight;
		}
		if (newRightBottomX > _captureFrameWidth) {
			newRightBottomX = _captureFrameWidth;
			newLeftTopX = _captureFrameWidth - newWidth;
		}
		if (newRightBottomY > _captureFrameHeight) {
			newRightBottomY = _captureFrameHeight;
			newLeftTopY = _captureFrameHeight - newHeight;
		}
	}
	leftTopX = newLeftTopX;
	leftTopY = newLeftTopY;
	rightBottomX = newRightBottomX;
	rightBottomY = newRightBottomY;
	return 1;
}

void PeopleCounter::pointsToBox(cv::Rect& box, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY) {
    box.x = leftTopX;
    box.y = leftTopY;
    box.width = rightBottomX - leftTopX;
    box.height = rightBottomY - leftTopY;
}

double PeopleCounter::expAvg(double prev, double curr, double factor) {
    return ((1 - factor) * prev + factor * curr);
}

void PeopleCounter::updateFrameRegionToShow() {
    static int leftTopX, leftTopY, rightBottomX, rightBottomY;
    static double ltx(0), lty(0), lbx(_captureFrameWidth), lby(_captureFrameHeight);
    static float hyst(0.0);
    
    float roiArea = getRegionArea(_frameRegionToShow);
    
    if ((roiArea / (_captureFrameWidth * _captureFrameHeight)) > _frameRegionUpdateThreshold - hyst) {
        _frameRegionToShow = cv::Rect(0, 0, _captureFrameWidth, _captureFrameHeight);
        hyst = _frameRegionUpdateThresholdHyst;
    } else {
        hyst = 0.0;
    }
    
    boxToPoints(_frameRegionToShow, leftTopX, leftTopY, rightBottomX, rightBottomY);
	toMarginPoints(leftTopX, leftTopY, rightBottomX, rightBottomY);
	if (abs(ltx - leftTopX) > 10 && abs(ltx - leftTopX) > 10 && abs(ltx - leftTopX) > 10 && abs(ltx - leftTopX) > 10) {
		ltx = expAvg(ltx, leftTopX, _zoomSpeedFactor);
		lty = expAvg(lty, leftTopY, _zoomSpeedFactor);
		lbx = expAvg(lbx, rightBottomX, _zoomSpeedFactor);
		lby = expAvg(lby, rightBottomY, _zoomSpeedFactor);
	}        
    pointsToBox(_frameRegionToShowZoomed, (int)ltx, (int)lty, (int)lbx, (int)lby);
    
    boundRegionToCaptureFrame(_frameRegionToShowZoomed);
}

// Draw the predicted bounding box
void PeopleCounter::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!_classes.empty()) {
        CV_Assert(classId < (int)_classes.size());
        label = _classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

std::vector<std::string> PeopleCounter::getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}
