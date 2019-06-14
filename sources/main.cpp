#include "people_counter.h"
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

const char* keys =
"{help h ?|| usage examples: peoplecounter.exe --dev=0 }"
"{mov     |<none>| video file name                    }"
"{dev     |0| input device id                          }"
"{ct      |0.6| confidence threshold                   }"
"{st      |0.5| non-maximum suppression threshold      }"
"{iw      |320| width of network's input image         }"
"{ih      |320| height of network's input image        }"
"{cw      |1920| width of the camera input image       }"
"{ch      |1080| height of the camera input image      }"
"{cfg     |net.cfg| network configuration              }"
"{wts     |net.wts| network weights                    }"
"{nms     |net.nms| network object classes             }"
"{zsf     |0.005| zooming speed factor                  }"
"{fct     |0.9| ROI update threshold in percents      }"
"{hst     |0.1| ROI update threshold hysteresis        }"
"{rmg     |0.075| region margin						   }"
"{dt     |0|  delay time for region				   }"
;

int main(int argc, char** argv)
{
	char szEXEPath[2048];
	GetModuleFileName(NULL, szEXEPath, 2048);
	PathRemoveFileSpec(szEXEPath);

	std::string strExePath = szEXEPath;
	strExePath += "\\";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this application to count the number of people in a video stream.");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    
    cv::VideoCapture cap;
    
    try {
        // Open the video file
        if (parser.has("mov")) {
			std::string file_name = strExePath + parser.get<std::string>("mov");
            cap.open(strExePath + parser.get<std::string>("mov"));
        } else {
            // Open the cam
            cap.open(parser.get<int>("dev"));
        }
    }
    catch(...) {
        std::cout << "Could not open the input video stream" << std::endl;
        return 0;
    }
    
    if (cap.isOpened()) {        
        PeopleCounter peopleCounter(cap,
			strExePath + parser.get<std::string>("cfg"), strExePath + parser.get<std::string>("wts"), strExePath + parser.get<std::string>("nms"),
            parser.get<double>("ct"), parser.get<double>("st"),
            parser.get<int>("iw"), parser.get<int>("ih"),
            parser.get<int>("cw"), parser.get<int>("ch"),
            parser.get<double>("zsf"), parser.get<double>("fct"), parser.get<double>("hst"), parser.get<double>("rmg"), parser.get<double>("dt"));
        
        peopleCounter.runThreads();
    }
    
    cv::waitKey(1000);
    
    return 0;
}
