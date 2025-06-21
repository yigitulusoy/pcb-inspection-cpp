#ifndef AUGMENTATION_H
#define AUGMENTATION_H

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm> // For std::max, std::min
#include <cmath>     // For std::sqrt, std::round
#include <chrono>    // For timestamp
#include <iomanip>   // For std::put_time
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "config.h"

namespace fs = std::filesystem; // Alias for convenience

// Re-using structures from previous files.
// In a real project, these would be in a common header.
struct PixelDimensions
{
    int width;
    int height;
};

struct PixelPoint
{
    int x;
    int y;
};

struct BoundingBoxSizes
{
    int clean_box_size;
    int comp_box_size;
};

// Define a structure for frame boundaries (same as in component_extraction.cpp)
struct FrameBoundaries
{
    int left;
    int right;
    int top;
    int bottom;
};

// Random number generator setup
extern std::default_random_engine generator;
extern std::uniform_real_distribution<double> distribution;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class augmentation
{
public:
    augmentation();

    PixelPoint calculate_component_position(
        const config::Component &comp,
        const std::pair<double, double> &pattern_pos_mm,
        const config::Offset &frame_center_mm,
        int img_center_x_px,
        int img_center_y_px,
        int frame_id,
        int pattern_index);

    PixelPoint calculate_component_position_px(
        const config::Component &comp, const std::pair<double, double> &pattern_pos_mm,
        const config::Offset &frame_center_mm, int img_center_x_px, int img_center_y_px,
        int frame_id, int pattern_index_in_frame);

    FrameBoundaries calculate_frame_boundaries(const config::Offset &frame_center_mm,
                                               int img_center_x_px,
                                               int img_center_y_px);
    bool is_component_fully_visible(
        int x_px, int y_px, // component center
        int width_px, int height_px,
        const FrameBoundaries &frame_boundaries);
    std::vector<double> get_shift_percentages_for_comp(const std::string &component_ref);

    fs::path comp_extract();

    PixelDimensions get_component_dimensions_px(const config::Component &comp_template);
    // Get the index of a pattern in its frame (same as in missing_component_augmentation.cpp)
    int get_pattern_index_in_frame(int pattern_id, int frame_id);
    // Calculate component position in image coordinates (same as in missing_component_augmentation.cpp)

    // Calculate two bounding boxes: one for cleaning and one for component handling.
    BoundingBoxSizes get_bounding_boxes(int width_px, int height_px, const std::string &comp_ref);
    // Get appropriate rotation angles based on component type.
    std::vector<int> get_rotation_angles_for_comp(const std::string &comp_ref);
    // Returns pair of paths: {augmented_frame_path, component_region_path} or empty strings on failure
    std::pair<std::string, std::string> create_rotated_component_augmentation(
        int frame_id, int pattern_id, const config::Component &comp_template,
        int angle_deg, int direction,
        const fs::path &base_frames_dir, const fs::path &base_components_dir);
    int rotatedComponentAugmenter();

    FrameBoundaries calculate_frame_boundaries_px(
        const config::Offset &frame_center_mm,
        int img_center_x_px,
        int img_center_y_px);

    int missingComponentAugmenter();

    std::pair<std::string, std::string> create_missing_component_augmentation(
    int frame_id,
    int pattern_id,
    const config::Component& comp_template,
    const fs::path& base_frames_dir,
    const fs::path& base_components_dir);


    // Returns pair of paths: {augmented_frame_path, component_region_path} or empty strings on failure
    std::pair<std::string, std::string> create_shifted_component_augmentation(
        int frame_id, int pattern_id, const config::Component &comp_template, int shift_idx,
        const fs::path &base_frames_dir, const fs::path &base_components_dir, int num_total_shifts_for_angle_calc);
    int shiftedComponentAugmentation();
};

#endif // AUGMENTATION_H
