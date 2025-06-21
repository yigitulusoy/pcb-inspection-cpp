// component_extraction.cpp
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem> // For C++17 filesystem operations
#include <algorithm>  // For std::max, std::min
#include <cmath>      // For std::round (implicitly used in int conversions)

#include <opencv2/opencv.hpp> // OpenCV core and image processing
#include <opencv2/imgcodecs.hpp> // For imread, imwrite
#include <opencv2/imgproc.hpp>   // For rectangle, putText

#include "augmentation.h"


int main() {

    augmentation aug;

    fs::path output_base_path = aug.comp_extract();

    std::cout << "Component extraction complete. Images saved to " << output_base_path.string() << std::endl; //

    aug.rotatedComponentAugmenter();

    aug.shiftedComponentAugmentation();

    aug.missingComponentAugmenter();


    return 0;
}
