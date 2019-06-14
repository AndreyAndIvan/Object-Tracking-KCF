#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class PeopleCounter
{
public:
    PeopleCounter(cv::VideoCapture& capture,
                  std::string cnf_path, std::string wts_path, std::string nms_path,
                  float ct, float st,
                  int iw, int ih,
                  int cw, int ch,
                  float zsf, float fct, float hst, float rmg, int frm);
    
    void runThreads();
    int getPeopleQty();
    
private:
    // Get the names of the output layers
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
    // Filter out low confidence objects with non-maxima suppression
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    int countPeople(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    void processFrame(cv::Mat& frame);
    void updateFrameRegionToShow();
    void boundRegionToCaptureFrame(cv::Rect& region);
    void adjustFrameRegion(cv::Rect& region, cv::Rect& box, bool keepAspectRatio = false);
    void adjustBlurMask(cv::Rect& region);
    void padAspectRatio(cv::Mat& img, float ratio);
    void cutAspectRatio(cv::Mat& img, float ratio);
    void boxToPoints(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY);
	int toMarginPoints(int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY);
    void pointsToBox(cv::Rect& box, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);
    float getRegionArea(cv::Rect& region);
	bool isOverRect(cv::Rect prevRect, cv::Rect curRect);
    
    template <typename T>
    T clip(const T& val, const T& lower, const T& upper) {
        return std::max(lower, std::min(val, upper));
    }

    double expAvg(double prev, double curr, double factor);
    
    void producer();
    void processor();
    
    cv::VideoCapture _capture;
    cv::Mat _lastCapturedFrame;
    cv::Mat _lastOverlayFrame;
    cv::Mat _lastOverlayedFrame;
    cv::Mat _lastProcessedFrame;
    cv::Mat _blurMask;
    cv::Rect _frameRegionToShow;
    cv::Rect _frameRegionToShowZoomed;
    float _zoomSpeedFactor;
    int _captureFrameWidth;
    int _captureFrameHeight;
    int _peopleQty;
    
    cv::dnn::Net _net;                      // object detection neural network
    
    std::string _modelConfigurationFile;    // network configuration file
    std::string _modelWeightsFile;          // network weights file
    std::string _classesFile;               // network classes file
    
    float _confThreshold;                   // Confidence threshold
    float _nmsThreshold;                    // Non-maximum suppression threshold
    int _inpWidth;                          // Width of network's input image
    int _inpHeight;                         // Height of network's input image
    float _frameRegionUpdateThreshold;      // ROI dimensions update threshold
    float _frameRegionUpdateThresholdHyst;  // ROI dimensions update threshold hysteresis
	float _frameRegionMarginRatio;			//frame region margin
	int _timeCountForRegion;				//count of frame for decide to region
    
    std::vector<std::string> _classes;
    
    bool _threadsEnabled;
    std::mutex _mutexFrameCapture;
    std::mutex _mutexFrameRegion;
    std::mutex _mutexFrameOverlay;
};

