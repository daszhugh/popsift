/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <popsift/common/device_prop.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_config.h>
#include <popsift/version.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <mutex>


#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#ifdef USE_DEVIL
#include <devil_cpp_wrapper.hpp>
#endif
#include "pgmread.h"

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif


#include "timer.h"
#include "threading.h"

static bool print_dev_info  = false;
static bool print_time_info = false;
static bool write_as_uchar  = false;
static bool dont_write      = false;
static bool pgmread_loading = false;
static bool float_mode      = false;

static void parseargs(int argc, char** argv, popsift::Config& config, std::string& inputFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "")
            ("log,l", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files")

            ("input-file,i", value<std::string>(&inputFile)->required(), "Input file");
    
    }
    options_description parameters("Parameters");
    {
        parameters.add_options()
            ("octaves", value<int>(&config.octaves), "Number of octaves")
            ("levels", value<int>(&config.levels), "Number of levels per octave")
            ("sigma", value<float>()->notifier([&](float f) { config.setSigma(f); }), "Initial sigma value")

            ("threshold", value<float>()->notifier([&](float f) { config.setThreshold(f); }), "Contrast threshold")
            ("edge-threshold", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("edge-limit", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("downsampling", value<float>()->notifier([&](float f) { config.setDownsampling(f); }), "Downscale width and height of input by 2^N")
            ("initial-blur", value<float>()->notifier([&](float f) {config.setInitialBlur(f); }), "Assume initial blur, subtract when blurring first time");
    }
    options_description modes("Modes");
    {
    modes.add_options()
        ( "gauss-mode", value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
          popsift::Config::getGaussModeUsage() )
        // "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending on sigma. "
        // "Options are: vlfeat, relative, relative-all, opencv, fixed9, fixed15"
        ("desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
        "Choice of descriptor extraction modes:\n"
        "loop, iloop, grid, igrid, notile\n"
	"Default is loop\n"
        "loop is OpenCV-like horizontal scanning, computing only valid points, grid extracts only useful points but rounds them, iloop uses linear texture and rotated gradiant fetching. igrid is grid with linear interpolation. notile is like igrid but avoids redundant gradiant fetching.")
        ("popsift-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::PopSift); }),
        "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ("vlfeat-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::VLFeat); }),
        "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
        "In extrema refinement, steps up to 0.6, levels remain unchanged, "
        "do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh.")
        ("opencv-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::OpenCV); }),
        "During the initial upscale, shift pixels by 0.5. "
        "In extrema refinement, steps up to 0.5, "
        "reject points when reaching max iterations, "
        "first contrast threshold is floor(.5 * peak thresh). "
        "Computed filter width are lower than VLFeat/PopSift")
        ("direct-scaling", bool_switch()->notifier([&](bool b) { if(b) config.setScalingMode(popsift::Config::ScaleDirect); }),
         "Direct each octave from upscaled orig instead of blurred level.")
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ( "norm-mode", value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
          popsift::Config::getNormModeUsage() )
        ( "root-sift", bool_switch()->notifier([&](bool b) { if(b) config.setNormMode(popsift::Config::RootSift); }),
          popsift::Config::getNormModeUsage() )
        ("filter-max-extrema", value<int>()->notifier([&](int f) {config.setFilterMaxExtrema(f); }), "Approximate max number of extrema.")
        ("filter-grid", value<int>()->notifier([&](int f) {config.setFilterGridSize(f); }), "Grid edge length for extrema filtering (ie. value 4 leads to a 4x4 grid)")
        ("filter-sort", value<std::string>()->notifier([&](const std::string& s) {config.setFilterSorting(s); }), "Sort extrema in each cell by scale, either random (default), up or down");

    }
    options_description informational("Informational");
    {
        informational.add_options()
        ("print-gauss-tables", bool_switch()->notifier([&](bool b) { if(b) config.setPrintGaussTables(); }), "A debug output printing Gauss filter size and tables")
        ("print-dev-info", bool_switch(&print_dev_info)->default_value(false), "A debug output printing CUDA device information")
        ("print-time-info", bool_switch(&print_time_info)->default_value(false), "A debug output printing image processing time after load()")
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int.\n"
         "Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar")
        ("dont-write", bool_switch(&dont_write)->default_value(false), "Suppress descriptor output")
        ("pgmread-loading", bool_switch(&pgmread_loading)->default_value(false), "Use the old image loader instead of LibDevIL")
        ("float-mode", bool_switch(&float_mode)->default_value(false), "Upload image to GPU as float instead of byte")
        ;
        
        //("test-direct-scaling")
    }

    options_description all("Allowed options");
    all.add(options).add(parameters).add(modes).add(informational);
    variables_map vm;
    
    try
    {    
       store(parse_command_line(argc, argv, all), vm);

       if (vm.count("help")) {
           std::cout << all << '\n';
           exit(EXIT_SUCCESS);
       }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        std::cerr << "Usage:\n\n" << all << std::endl;
        exit(EXIT_FAILURE);
    }
}


static void collectFilenames(std::list<std::string>& inputFiles, const boost::filesystem::path& inputFile)
{
    std::vector<boost::filesystem::path> vec;
    std::copy( boost::filesystem::directory_iterator( inputFile ),
               boost::filesystem::directory_iterator(),
               std::back_inserter(vec) );
    for (const auto& currPath : vec)
    {
        if( boost::filesystem::is_regular_file(currPath) )
        {
            inputFiles.push_back( currPath.string() );
        }
        else if( boost::filesystem::is_directory(currPath) )
        {
            collectFilenames( inputFiles, currPath);
        }
    }
}

SiftJob* process_image(const std::string& inputFile, PopSift& popSift)
{
    SiftJob* job;
    unsigned char* image_data;

#ifdef USE_OPENCV
    if(1)
    {
        nvtxRangePushA("load and convert image - pgmread");
        int w{};
        int h{};

        cv::Mat image = cv::imread(inputFile, cv::IMREAD_GRAYSCALE);
        w = image.cols;
        h = image.rows;
        image_data = new unsigned char[w * h];
        memcpy(image_data, image.data, w * h* sizeof(unsigned char));

        nvtxRangePop(); // "load and convert image - pgmread"

        if(!float_mode)
        {
            // popSift.init( block_w, block_h );
            job = popSift.enqueue(w, h, image_data);

            delete[] image_data;
        }
        else
        {
            auto f_image_data = new float[w * h];
            for(int i = 0; i < w * h; i++)
            {
                f_image_data[i] = float(image_data[i]) / 256.0f;
            }
            job = popSift.enqueue(w, h, f_image_data);

            delete[] image_data;
            delete[] f_image_data;
        }
    }
    else 
#else defined USE_DEVIL
    if( ! pgmread_loading )
    {
        if( float_mode )
        {
            std::cerr << "Cannot combine float-mode test with DevIL image reader" << std::endl;
            exit( -1 );
        }

        nvtxRangePushA( "load and convert image - devil" );

        ilImage img;
        if( img.Load( inputFile.c_str() ) == false ) {
            cerr << "Could not load image " << inputFile << endl;
            return 0;
        }
        if( img.Convert( IL_LUMINANCE ) == false ) {
            cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << std::endl;
            exit( -1 );
        }
        const auto block_w = img.Width();
        const auto block_h = img.Height();
        std::cout << "Loading " << block_w << " x " << block_h << " image " << inputFile << std::endl;

        image_data = img.GetData();

        nvtxRangePop( ); // "load and convert image - devil"

        job = popSift.enqueue( block_w, block_h, image_data );

        img.Clear();
    }
    else
#endif
    {
        nvtxRangePushA( "load and convert image - pgmread" );
        int w{};
        int h{};
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == nullptr ) {
            exit( EXIT_FAILURE );
        }

        nvtxRangePop( ); // "load and convert image - pgmread"

        if( ! float_mode )
        {
            // popSift.init( block_w, block_h );
            job = popSift.enqueue( w, h, image_data );

            delete [] image_data;
        }
        else
        {
            auto f_image_data = new float [w * h];
            for( int i=0; i<w*h; i++ )
            {
                f_image_data[i] = float( image_data[i] ) / 256.0f;
            }
            job = popSift.enqueue( w, h, f_image_data );

            delete [] image_data;
            delete [] f_image_data;
        }
    }

    return job;
}

bool process_image2(const std::string& inputFile,
                    PopSift& popSift,
                    std::mutex& jobs_mutex,
                    std::queue<SiftJob*>& jobs,
                    int max_rows = 4000,
                    int max_cols = 6000)
{  
    if(inputFile.empty())
    {
        return false;
    }

    cv::Mat image = cv::imread(inputFile, cv::IMREAD_GRAYSCALE);
    if(image.empty())
    {
        return false;
    }

    std::vector<cv::Rect> rects;
    {
        int rows = image.rows;
        int cols = image.cols;

        while(rows > max_rows)
        {
            rows /= 2;
        }

        while(cols > max_cols)
        {
            cols /= 2;
        }

        int row_blocks = (image.rows + rows - 1) / rows;
        int col_blocks = (image.cols + cols - 1) / cols;

        for(int r = 0; r < row_blocks; ++r)
        {
            int y_beg = r * rows;
            int y_end = std::min(y_beg + rows, image.rows);
            int y_size = y_end - y_beg;

            for(int c = 0; c < col_blocks; ++c)
            {
                int x_beg = c * cols;
                int x_end = std::min(x_beg + cols, image.cols);
                int x_size = x_end - x_beg;

                rects.emplace_back(x_beg, y_beg, x_size, y_size);
            }
        }
    }
  
    {
        std::unique_lock<std::mutex> jobs_lock(jobs_mutex);
        for(const auto& rect : rects)
        {
            cv::Mat block_image_u8 = image(rect).clone();
            int block_w = rect.width;
            int block_h = rect.height;

            SiftJob* job = nullptr;

            if(!float_mode)
            {
                job = popSift.enqueue(block_w, block_h, block_image_u8.ptr<uchar>(0));
            }
            else
            {
                cv::Mat block_image_f32;
                block_image_u8.convertTo(block_image_f32, CV_32F, 1 / 256.0);
                job = popSift.enqueue(block_w, block_h, block_image_f32.ptr<float>(0));
            }

            jobs.push(job);
        }
    }

    return true;
}

void read_job( SiftJob* job, bool really_write )
{
    popsift::Features* feature_list = job->getHost();
    std::cerr << "Number of feature points: " << feature_list->getFeatureCount()
         << " number of feature descriptors: " << feature_list->getDescriptorCount() << std::endl;

    if( really_write ) {
        nvtxRangePushA( "Writing features to disk" );
        colmap::Timer timer;
        timer.Start();
        std::ofstream of( "output-features.txt" );
        feature_list->print( of, write_as_uchar );

        std::cout << "Writing time: " << timer.ElapsedSeconds() << std::endl;
    }
    delete feature_list;

    if( really_write ) {
        nvtxRangePop( ); // Writing features to disk
    }
}

struct PopSIFTFeatures
{
    std::vector<popsift::Feature> features;
    std::vector<popsift::Descriptor> descriptors;
};

PopSIFTFeatures CreateFeatures(popsift::Features* in_features)
{
    PopSIFTFeatures out_features;
    out_features.features.resize(in_features->getFeatureCount());
    out_features.descriptors.resize(in_features->getDescriptorCount());

    std::memcpy(out_features.features.data(),
                in_features->getFeatures(),
                sizeof(popsift::Feature) * in_features->getFeatureCount());

    std::memcpy(out_features.descriptors.data(),
                in_features->getDescriptors(),
                sizeof(popsift::Descriptor) * in_features->getDescriptorCount());

    return out_features;
}

void read_job2(SiftJob* job, std::vector<PopSIFTFeatures>& features)
{
    popsift::Features* feature_list = job->getHost();
    std::cerr << "Number of feature points: " << feature_list->getFeatureCount()
              << " number of feature descriptors: " << feature_list->getDescriptorCount() << std::endl;

    features.emplace_back(CreateFeatures(feature_list));
    delete feature_list;
}

int main(int argc, char **argv)
{
    cudaDeviceReset();

    popsift::Config config;
    std::list<std::string> inputFiles;
    std::string inputFile{};

    std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;

    try {
        parseargs( argc, argv, config, inputFile ); // Parse command line
        std::cout << inputFile << std::endl;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if( boost::filesystem::exists( inputFile ) ) {
        if( boost::filesystem::is_directory( inputFile ) ) {
            std::cout << "BOOST " << inputFile << " is directory" << std::endl;
            collectFilenames( inputFiles, inputFile );
            if( inputFiles.empty() ) {
                std::cerr << "No files in directory, nothing to do" << std::endl;
                return EXIT_SUCCESS;
            }
        } else if( boost::filesystem::is_regular_file( inputFile ) ) {
            inputFiles.push_back( inputFile );
        } else {
            std::cout << "Input file is neither regular file nor directory, nothing to do" << std::endl;
            return EXIT_FAILURE;
        }
    }

    colmap::Timer init_timer;
    init_timer.Start();

    int device_id = 0;

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(device_id, print_dev_info);
    if(print_dev_info)
    {
        deviceInfo.print();
    }

    std::vector<std::string> input_files;
    input_files.reserve(inputFiles.size());
    for(const auto& currFile : inputFiles)
    {
        input_files.emplace_back(currFile);
    }

    int max_rows = 2000;
    int max_cols = 6000;

    PopSift popSift(
      config, popsift::Config::ExtractingMode, float_mode ? PopSift::FloatImages : PopSift::ByteImages, device_id);

    std::cout << "Init time: " << init_timer.ElapsedSeconds() << std::endl;

    colmap::Timer all_timer;
    all_timer.Start();

    std::unordered_map<size_t, std::queue<SiftJob*>> image_jobs;

    std::mutex jobs_mutex;
    std::mutex temp_mutex;

    colmap::ThreadPool thread_pool(6);
    for(size_t i = 0; i < input_files.size(); ++i)
    {
        auto& jobs = image_jobs[i];

        thread_pool.AddTask(
          [&](size_t i) {
              const auto& currFile = input_files[i];
              colmap::Timer reading_timer;
              reading_timer.Start();

              process_image2(currFile, popSift, jobs_mutex, jobs, max_rows, max_cols);

              double readingimg_time = reading_timer.ElapsedSeconds();

              std::lock_guard<std::mutex> temp_lock(temp_mutex);
              std::cout << "Reading image time: " << readingimg_time << std::endl;
          },
          i);
    }
    thread_pool.Wait();

    for(auto& kv : image_jobs)
    {
        auto& jobs = kv.second;

        std::vector<PopSIFTFeatures> features;
        while(!jobs.empty())
        {
            SiftJob* job = jobs.front();
            jobs.pop();
            if(job)
            {
                read_job2(job, features);
                delete job;
            }
        }
    }

    popSift.uninit();

    std::cout << "All time: " << all_timer.ElapsedSeconds() << std::endl;
  

    return EXIT_SUCCESS;
}

