#include <ros/ros.h>
#include <jetson-inference/detectNet.h>
#include <jetson-inference/loadImage.h>
#include <jetson-inference/cudaFont.h>
#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/cudaNormalize.h>
#include <jetson-inference/cudaFont.h>

#include <jetson-inference/glDisplay.h>
#include <jetson-inference/glTexture.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <vr_msgs/BoundingBox.h>
#include <vr_msgs/BoundingBoxes.h>

bool signal_recieved;


ros::Publisher detection_publisher;
image_transport::Publisher image_publisher;
bool publish_image = true;


cv::Mat cv_image;
sensor_msgs::ImagePtr pub_detectnet_image;

detectNet *net;

float4 *gpu_data;

uint32_t imgWidth;
uint32_t imgHeight;
size_t imgSize;

float *bbCPU;
float *bbCUDA;
float *confCPU;
float *confCUDA;

uint32_t maxBoxes;
uint32_t classes;

void callback(const sensor_msgs::ImageConstPtr &input) {
    cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image;
    cv_image = cv_im;

    // convert bit depth
    cv_im.convertTo(cv_im, CV_32FC3);

    // convert color
    cv::cvtColor(cv_im, cv_im, CV_BGR2RGBA);

    // allocate GPU data if necessary
    if (!gpu_data) {
        ROS_INFO("first allocation");
        CUDA(cudaMalloc(&gpu_data, cv_im.rows * cv_im.cols * sizeof(float4)));
    } else if (imgHeight != cv_im.rows || imgWidth != cv_im.cols) {
        ROS_INFO("re allocation");

        // reallocate for a new image size if necessary
        CUDA(cudaFree(gpu_data));
        CUDA(cudaMalloc(&gpu_data, cv_im.rows * cv_im.cols * sizeof(float4)));
    }

    imgHeight = (uint32_t)cv_im.rows;
    imgWidth = (uint32_t)cv_im.cols;
    imgSize = cv_im.rows * cv_im.cols * sizeof(float4);
    float4 *cpu_data = (float4 *) (cv_im.data);

    // copy to device
    CUDA(cudaMemcpy(gpu_data, cpu_data, imgSize, cudaMemcpyHostToDevice));

    float confidence = 0.0f;

    // detect image with detectNet
    int numBoundingBoxes = maxBoxes;
    // number of bounding boxes
    vr_msgs::BoundingBoxes detections;

    if (net->Detect((float *) gpu_data, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU)) {
        int lastClass = 0;
        int lastStart = 0;

        //printf("%i bounding boxes detected\n", numBoundingBoxes);

        for (int n = 0; n < numBoundingBoxes; n++) {
            vr_msgs::BoundingBox bbox;
            float conf = confCPU[n * 2 + 0];

            const int nc = confCPU[n * 2 + 1];
            float *bb = bbCPU + (n * 4);

			//printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0],
            //       bb[3] - bb[1]);
            if (publish_image)
            {
				int clr = 255 * conf;
				
				// draw a green line(CW) on the overlay copy
				cv::line(cv_image, cv::Point(bb[0],bb[1]), cv::Point(bb[2],bb[1]),cv::Scalar(0, clr, 0),2);
				cv::line(cv_image, cv::Point(bb[2],bb[1]), cv::Point(bb[2],bb[3]),cv::Scalar(0, clr, 0),2);
				cv::line(cv_image, cv::Point(bb[2],bb[3]), cv::Point(bb[0],bb[3]),cv::Scalar(0, clr, 0),2);
				cv::line(cv_image, cv::Point(bb[0],bb[3]), cv::Point(bb[0],bb[1]),cv::Scalar(0, clr, 0),2);				
			}     
            

            bbox.Class = nc;
            bbox.confidence = conf; //TODO: check DetectNet.cpp docs to extract confidence range
            bbox.xmin = (int)bb[0];
            bbox.ymin = (int)bb[1];
            bbox.xmax = (int)bb[2];
            bbox.ymax = (int)bb[3];

            detections.boxes.push_back(bbox);
            if( nc != lastClass || n == (numBoundingBoxes - 1) ){
                CUDA(cudaDeviceSynchronize());
            }
        }
    }

    detection_publisher.publish(detections);
    if (publish_image)
    {
		pub_detectnet_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image).toImageMsg(); 
		image_publisher.publish(pub_detectnet_image);
	}
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_detectnet_publisher");
    ros::NodeHandle nh("~");
    detectNet::NetworkType networkType;
    std::string input_topic, output_topic;
    std::string prototxt_path, model_path, mean_binary_path;

    net = NULL;

    gpu_data = NULL;

    imgWidth = 0;
    imgHeight = 0;
    imgSize = 0;

    bbCPU = NULL;
    bbCUDA = NULL;
    confCPU = NULL;
    confCUDA = NULL;

    maxBoxes = 0;
    classes = 0;

    nh.getParam("prototxt_path", prototxt_path);
    nh.getParam("model_path", model_path);
    if ( !nh.getParam("input_topic", input_topic)) { ROS_ERROR("failed to get input topic"); }
    if ( !nh.getParam("output_topic", output_topic)) { ROS_ERROR("failed to get output topic"); }

    // make sure files exist (and we can read them)
    if (access(prototxt_path.c_str(), R_OK))
        ROS_ERROR("unable to read file \"%s\", check filename and permissions", prototxt_path.c_str());
    if (access(model_path.c_str(), R_OK))
        ROS_ERROR("unable to read file \"%s\", check filename and permissions", model_path.c_str());



    net = detectNet::Create( prototxt_path.c_str(), model_path.c_str() );


    if (!net) {
        ROS_ERROR("ros-detectnet-camera: failed to create detectNet\n");
    }

    // allocate memory for output bounding boxes and class confidence
    maxBoxes = net->GetMaxBoundingBoxes();
    printf("maximum bounding boxes:  %u\n", maxBoxes);

    classes = net->GetNumClasses();

    if (!cudaAllocMapped((void **) &bbCPU, (void **) &bbCUDA, maxBoxes * sizeof(float4)) ||
        !cudaAllocMapped((void **) &confCPU, (void **) &confCUDA, maxBoxes * classes * sizeof(float))) {
        ROS_ERROR("detectnet-console:  failed to alloc output memory\n");
    }

    // setup image transport
    image_transport::ImageTransport it(nh);

    // publisher for number of detected bounding boxes output
    detection_publisher = nh.advertise<vr_msgs::BoundingBoxes>(output_topic, 1);
    
    if (publish_image)
    {
		image_publisher = it.advertise("detector/debug", 1);
	}
	
    // subscriber for passing in images
    image_transport::Subscriber sub = it.subscribe(input_topic, 1, callback);




    ros::Rate loop_rate(25); //Hz


    ros::spin();
}
