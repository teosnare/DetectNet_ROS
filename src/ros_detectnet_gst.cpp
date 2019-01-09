#include <ros/ros.h>
//#include <jetson-inference/gstCamera.h>
#include <jetson-inference/gstPipeline.h>
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


cv::Mat cv_image;
sensor_msgs::ImagePtr pub_detectnet_image;

detectNet *net;

float4 *gpu_data;

bool show_debug = false;
int imgWidth;
int imgHeight;
int imgBpp;
size_t imgSize;

float *bbCPU;
float *bbCUDA;
float *confCPU;
float *confCUDA;

uint32_t maxBoxes;
uint32_t classes;

#define CAM_WIDTH 1920
#define CAM_HEIGHT 1080
#define DEFAULT_CAMERA 1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)


int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_detectnet_publisher");
    ros::NodeHandle nh("~");
    detectNet::NetworkType networkType;
    std::string output_topic;
    std::string launch_str;
    std::string prototxt_path, model_path, mean_binary_path;

    net = NULL;

    gpu_data = NULL;

    imgWidth = CAM_WIDTH;
    imgHeight = CAM_HEIGHT;
    imgBpp = 12;
    imgSize = 0;

    bbCPU = NULL;
    bbCUDA = NULL;
    confCPU = NULL;
    confCUDA = NULL;

    maxBoxes = 0;
    classes = 0;

    nh.getParam("prototxt_path", prototxt_path);
    nh.getParam("model_path", model_path);
    if ( !nh.getParam("launch_string", launch_str)) { ROS_ERROR("failed to get launch string"); }
    if ( !nh.getParam("output_topic", output_topic)) { ROS_ERROR("failed to get output topic"); }
    imgWidth = nh.param("image_width", imgWidth);
    imgHeight = nh.param("image_height", imgHeight);
    imgBpp = nh.param("image_bpp", imgBpp);
    show_debug = nh.param("show_debug", show_debug);

    // make sure files exist (and we can read them)
    if (access(prototxt_path.c_str(), R_OK))
        ROS_ERROR("unable to read file \"%s\", check filename and permissions", prototxt_path.c_str());
    if (access(model_path.c_str(), R_OK))
        ROS_ERROR("unable to read file \"%s\", check filename and permissions", model_path.c_str());

    gstPipeline * camera = gstPipeline::Create(launch_str, imgWidth, imgHeight, imgBpp);

    if( !camera )
    {
        printf("\nros-detectnet-gst:  failed to initialize video device\n");
        return 0;
    }
    printf("\nros-detectnet-gst:  successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("    height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());

    net = detectNet::Create(prototxt_path.c_str(), model_path.c_str() );


    if (!net) {
        ROS_ERROR("ros-detectnet-gst: failed to create detectNet\n");
    }

    // allocate memory for output bounding boxes and class confidence
    maxBoxes = net->GetMaxBoundingBoxes();
    printf("maximum bounding boxes:  %u\n", maxBoxes);

    classes = net->GetNumClasses();

    if (!cudaAllocMapped((void **) &bbCPU, (void **) &bbCUDA, maxBoxes * sizeof(float4)) ||
        !cudaAllocMapped((void **) &confCPU, (void **) &confCUDA, maxBoxes * classes * sizeof(float))) {
        ROS_ERROR("detectnet-console:  failed to alloc output memory\n");
    }


    /*
     * create openGL window
     */

    glDisplay* display = NULL;
    glTexture* texture = NULL;

    if (show_debug)
    {
        display = glDisplay::Create();
        if( !display ) {
            printf("\ndetectnet-camera:  failed to create openGL display\n");
        }
        else
        {
            texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

            if( !texture )
                printf("detectnet-camera:  failed to create openGL texture\n");
        }
    }

    // setup image transport
//    image_transport::ImageTransport it(nh);

    // publisher for number of detected bounding boxes output
    detection_publisher = nh.advertise<vr_msgs::BoundingBoxes>(output_topic, 1);


    /*
     * start streaming
     */
    if( !camera->Open() )
    {
        printf("\nros-detectnet-gst:  failed to open camera for streaming\n");
        return 0;
    }

    printf("\nros-detectnet-gst:  camera open for streaming\n");

    /*
     * processing loop
     */
    float confidence = 0.0f;

    while (ros::ok())
    {
        void* imgCPU  = NULL;
        void* imgCUDA = NULL;

        // get the latest frame
        if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
            printf("\ndetectnet-camera:  failed to capture frame\n");

        // convert from YUV to RGBA
        void* imgRGBA = NULL;

        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
            printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");

        // classify image with detectNet
        int numBoundingBoxes = maxBoxes;
        vr_msgs::BoundingBoxes detections;

        if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
        {
            printf("%i bounding boxes detected\n", numBoundingBoxes);

            int lastClass = 0;
            int lastStart = 0;

            for( int n=0; n < numBoundingBoxes; n++ )
            {
                vr_msgs::BoundingBox bbox;
                const int nc = confCPU[n*2+1];

                std::ostringstream stringStream;
                stringStream << nc;
                std::string classId = stringStream.str();

                float conf = confCPU[n*2];
                float* bb = bbCPU + (n * 4);

                printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);

                bbox.Class = classId;
                bbox.confidence = conf; //1.0; //TODO: check DetectNet.cpp docs to extract confidence range
                bbox.xmin = (int)bb[0];
                bbox.ymin = (int)bb[1];
                bbox.xmax = (int)bb[2];
                bbox.ymax = (int)bb[3];

                detections.boxes.push_back(bbox);
                if( nc != lastClass || n == (numBoundingBoxes - 1) ){

                    if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
                                                bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
                        printf("detectnet-console:  failed to draw boxes\n");

                    lastClass = nc;
                    lastStart = n;

                    CUDA(cudaDeviceSynchronize());
                }
            }

            if( display != NULL )
            {
                char str[256];
                sprintf(str, "TensorRT build %i.%i.%i | %s | %04.1f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
                //sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
                display->SetTitle(str);
            }
        }

        // update display
        if( display != NULL )
        {
            display->UserEvents();
            display->BeginRender();

            if( texture != NULL )
            {
                // rescale image pixel intensities for display
                CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
                                   (float4*)imgRGBA, make_float2(0.0f, 1.0f),
                                   camera->GetWidth(), camera->GetHeight()));

                // map from CUDA to openGL using GL interop
                void* tex_map = texture->MapCUDA();

                if( tex_map != NULL )
                {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
                    texture->Unmap();
                }

                // draw the texture
                texture->Render(100,100);
            }

            display->EndRender();
        }

        detection_publisher.publish(detections);
    }

    printf("\nros-detectnet-gst:  un-initializing video device\n");


    /*
     * shutdown the camera device
     */
    if( camera != NULL )
    {
        delete camera;
        camera = NULL;
    }

    if( display != NULL )
    {
        delete display;
        display = NULL;
    }

    printf("ros-detectnet-gst:  video device has been un-initialized.\n");
    printf("ros-detectnet-gst:  this concludes the test of the video device.\n");
}
