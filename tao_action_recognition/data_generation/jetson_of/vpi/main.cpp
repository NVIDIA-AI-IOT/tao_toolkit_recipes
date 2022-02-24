/*
* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#    include <opencv2/videoio.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <regex>

#define TAG_STRING "PIEH"    // use this when WRITING the file

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

static std::string generateRegexPattern(const std::string& imageNamePattern)
{
    std::string regex_pat;
    std::string image;
    std::string temp;

    for (auto it = imageNamePattern.cbegin(); it != imageNamePattern.cend(); ++it)
    {
        if (*it == '*')
        {
            image.append(".*");
        }
        else if (*it == '?')
        {
            image.append(".");
        }
        else
        {
            image.append(1, *it);
        }
    }

    size_t pos = image.find_first_of("%");
    if (pos != std::string::npos)
    {
        if (pos > 0)
        {
            regex_pat.append(image.substr(0, pos));
        }
        temp = image.substr(pos + 1);
        pos = temp.find_first_of("d");
        if (pos != std::string::npos)
        {
            if (pos > 0)
            {
                auto nd = atoi(temp.substr(0, pos).c_str());
                std::ostringstream ss;
                ss << "([0-9]){" << nd << ",}";
                regex_pat.append(ss.str());
            }
            else
            {
                regex_pat.append("([0 - 9]){1,}");
            }
            regex_pat.append(temp.substr(pos + 1));
        }
    }
    else
    {
        regex_pat.append(image);
    }
    return regex_pat;
}

static std::vector<std::pair<std::string, std::string>> ReadDirectory(const std::string& path)
{
    std::vector<std::pair<std::string, std::string>> files;
    DIR* d;
    struct dirent* dir;
    d = opendir(path.c_str());
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            const char* name = dir->d_name;
            if ((name[0] == 0) ||
                (name[0] == '.' && name[1] == 0) ||
                (name[0] == '.' && name[1] == '.' && name[2] == 0))
                continue;

            struct stat buf;
            if ((stat(name, &buf) == 0) &&
                S_ISDIR(buf.st_mode))
                continue;

            files.push_back(std::make_pair(path + "/" + std::string(name), std::string(name)));
        }

        closedir(d);
    }

    return files;
}

static void glob(const std::string& image, std::vector<std::string>& result)
{
    const char dir_separators[] = "/\\";
    std::string wildchart;
    std::string path;
    size_t pos = image.find_last_of(dir_separators);
    if (pos == std::string::npos)
    {
        wildchart = image;
        path = ".";
    }
    else
    {
        path = image.substr(0, pos);
        wildchart = image.substr(pos + 1);
    }
    std::string regex_str = generateRegexPattern(wildchart);
    std::regex regex_pat{ regex_str };
#ifndef NDEBUG
    std::cout << "Input file directory path : " << path << std::endl;
    std::cout << "Input file pattern : " << wildchart << std::endl;
#endif
    std::vector<std::pair<std::string, std::string>> fileNames = ReadDirectory(path);
    for (const auto & p : fileNames)
    {
        if (!p.first.empty() && !p.second.empty())
        {
            auto fileName = p.second;
            if (!wildchart.empty())
            {
                if (regex_match(fileName, regex_pat))
                {
                    result.push_back(p.first);
                }
            }
        }
    }

    if (!result.empty())
    {
        std::sort(result.begin(), result.end());
    }
}

static void ProcessMotionVector(VPIImage mvImg, cv::Mat &outputImage)
{
    // Lock the input image to access it from CPU
    VPIImageData mvData;
    CHECK_STATUS(vpiImageLock(mvImg, VPI_LOCK_READ, &mvData));

    // Create a cv::Mat that points to the input image data
    cv::Mat mvImage;
    CHECK_STATUS(vpiImageDataExportOpenCVMat(mvData, &mvImage));

    // Convert S10.5 format to float
    cv::Mat flow(mvImage.size(), CV_32FC2);
    mvImage.convertTo(flow, CV_32F, 1.0f / (1 << 5));

    // Image not needed anymore, we can unlock it.
    CHECK_STATUS(vpiImageUnlock(mvImg));

    outputImage = flow;

}

static void WriteFlowVectors(const std::string& outputFilePattern,
                             const int frameIdx,
                             const cv::Mat& outputImage,
                             const int mvWidth,
                             const int mvHeight)
{
    std::ostringstream fileName;
    fileName << outputFilePattern << "_";
    fileName << std::setw(5) << std::setfill('0') << frameIdx << std::string("_middlebury.flo") ;

    std::ofstream fpOut(fileName.str(), std::ios::out | std::ios::binary);

    fpOut << TAG_STRING;

    fpOut.write((char*)(&mvWidth), sizeof(uint32_t));
    fpOut.write((char*)(&mvHeight), sizeof(uint32_t));
    fpOut.write((char*)outputImage.data, sizeof(float) * mvWidth * mvHeight * 2);
    fpOut.close();
}

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvPrevFrame, cvCurFrame;

    // VPI objects that will be used
    VPIStream stream         = NULL;
    VPIImage imgPrevFramePL  = NULL;
    VPIImage imgPrevFrameTmp = NULL;
    VPIImage imgPrevFrameBL  = NULL;
    VPIImage imgCurFramePL   = NULL;
    VPIImage imgCurFrameTmp  = NULL;
    VPIImage imgCurFrameBL   = NULL;
    VPIImage imgMotionVecBL  = NULL;
    VPIPayload payload       = NULL;

    int retval = 0;

    try
    {
        if (argc != 4)
        {
            std::cout<<argc;
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <input_files_pattern> <output_files> <low|medium|high>");
        }

        // Parse input parameters
        std::string strInputFilesPattern = argv[1];
        std::string strOuputFilesPattern = argv[2];
        std::string strQuality    = argv[3];

        VPIOpticalFlowQuality quality;
        if (strQuality == "low")
        {
            quality = VPI_OPTICAL_FLOW_QUALITY_LOW;
        }
        else if (strQuality == "medium")
        {
            quality = VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
        }
        else if (strQuality == "high")
        {
            quality = VPI_OPTICAL_FLOW_QUALITY_HIGH;
        }
        else
        {
            throw std::runtime_error("Unknown quality provided");
        }

        VPIBackend backend;
            backend = VPI_BACKEND_NVENC;
        // Load the files list
        std::vector<std::string> inputFilesList;
        glob(strInputFilesPattern, inputFilesList);

        // Create the stream where processing will happen. We'll use user-provided backend
        // for Optical Flow, and CUDA/VIC for image format conversions.
        CHECK_STATUS(vpiStreamCreate(backend | VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &stream));

        cvPrevFrame = cv::imread(inputFilesList[0]);

        // Create the previous and current frame wrapper using the first frame. This wrapper will
        // be set to point to every new frame in the main loop.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvPrevFrame, 0, &imgPrevFramePL));
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvPrevFrame, 0, &imgCurFramePL));

        // Define the image formats we'll use throughout this sample.
        VPIImageFormat imgFmt   = VPI_IMAGE_FORMAT_NV12_ER;
        VPIImageFormat imgFmtBL = VPI_IMAGE_FORMAT_NV12_ER_BL;

        int32_t width  = cvPrevFrame.cols;
        int32_t height = cvPrevFrame.rows;

        // Create Dense Optical Flow payload to be executed on the given backend
        CHECK_STATUS(vpiCreateOpticalFlowDense(backend, width, height, imgFmtBL, quality, &payload));

        // The Dense Optical Flow on NVENC backend expects input to be in block-linear format.
        // Since Convert Image Format algorithm doesn't currently support direct BGR
        // pitch-linear (from OpenCV) to NV12 block-linear conversion, it must be done in two
        // passes, first from BGR/PL to NV12/PL using CUDA, then from NV12/PL to NV12/BL using VIC.
        // The temporary image buffer below will store the intermediate NV12/PL representation.
        CHECK_STATUS(vpiImageCreate(width, height, imgFmt, 0, &imgPrevFrameTmp));
        CHECK_STATUS(vpiImageCreate(width, height, imgFmt, 0, &imgCurFrameTmp));

        // Now create the final block-linear buffer that'll be used as input to the
        // algorithm.
        CHECK_STATUS(vpiImageCreate(width, height, imgFmtBL, 0, &imgPrevFrameBL));
        CHECK_STATUS(vpiImageCreate(width, height, imgFmtBL, 0, &imgCurFrameBL));

        // Motion vector image width and height, align to be multiple of 4
        int32_t mvWidth  = (width + 3) / 4;
        int32_t mvHeight = (height + 3) / 4;


        // Create the output motion vector buffer
        CHECK_STATUS(vpiImageCreate(mvWidth, mvHeight, VPI_IMAGE_FORMAT_2S16_BL, 0, &imgMotionVecBL));

        // First convert the first frame to NV12_BL. It'll be used as previous frame when the algorithm is called.
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgPrevFramePL, imgPrevFrameTmp, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, imgPrevFrameTmp, imgPrevFrameBL, nullptr));

        // Create a output image which holds the rendered motion vector image.
        cv::Mat mvOutputImage;

        // Fetch a new frame until video ends
        int idxFrame = 1;
        int outIdxFrame = 0;
        for(idxFrame = 1; idxFrame < inputFilesList.size(); idxFrame++)
        {
            printf("Processing frame %d\n", idxFrame);
            cvCurFrame = cv::imread(inputFilesList[idxFrame]);
            // Wrap frame into a VPIImage, reusing the existing imgCurFramePL.
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFramePL, cvCurFrame));

            // Convert current frame to NV12_BL format
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgCurFramePL, imgCurFrameTmp, nullptr));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, imgCurFrameTmp, imgCurFrameBL, nullptr));

            CHECK_STATUS(
                vpiSubmitOpticalFlowDense(stream, backend, payload, imgPrevFrameBL, imgCurFrameBL, imgMotionVecBL));

            // Wait for processing to finish.
            CHECK_STATUS(vpiStreamSync(stream));

            // Render the resulting motion vector in the output image
            ProcessMotionVector(imgMotionVecBL, mvOutputImage);

            // Save to output files:
            WriteFlowVectors(strOuputFilesPattern, outIdxFrame++, mvOutputImage, mvWidth, mvHeight);

            // Swap previous frame and next frame
            std::swap(cvPrevFrame, cvCurFrame);
            std::swap(imgPrevFramePL, imgCurFramePL);
            std::swap(imgPrevFrameBL, imgCurFrameBL);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // Destroy all resources used
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);

    vpiImageDestroy(imgPrevFramePL);
    vpiImageDestroy(imgPrevFrameTmp);
    vpiImageDestroy(imgPrevFrameBL);
    vpiImageDestroy(imgCurFramePL);
    vpiImageDestroy(imgCurFrameTmp);
    vpiImageDestroy(imgCurFrameBL);
    vpiImageDestroy(imgMotionVecBL);

    return retval;
}

// vim: ts=8:sw=4:sts=4:et:ai
