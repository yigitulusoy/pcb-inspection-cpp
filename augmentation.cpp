#include "augmentation.h"



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

#include "config.h"
#include "augmentation.h"

std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
std::uniform_real_distribution<double> distribution(-M_PI / 12.0, M_PI / 12.0);

augmentation::augmentation() {}


PixelPoint augmentation::calculate_component_position(
    const config::Component& comp,
    const std::pair<double, double>& pattern_pos_mm,
    const config::Offset& frame_center_mm,
    int img_center_x_px,
    int img_center_y_px,
    int frame_id,
    int pattern_index) {

    // Get frame-specific offsets
    config::Offset frame_offset_mm = {0.0, 0.0};
    auto it_frame_offset = config::FRAME_OFFSETS.find(frame_id);
    if (it_frame_offset != config::FRAME_OFFSETS.end()) {
        frame_offset_mm = it_frame_offset->second;
    } else {
        // std::cerr << "Warning: Frame offset not found for frame_id: " << frame_id << std::endl;
        // Defaulting to 0,0 or handle as an error
    }

    // Get pattern-specific offset based on position in frame
    config::Offset pattern_specific_offset_mm = {0.0, 0.0};
    if (pattern_index == 0) { // Bottom pattern
        pattern_specific_offset_mm = config::PATTERN_OFFSETS.at("bottom");
    } else if (pattern_index == 1) { // Middle pattern
        pattern_specific_offset_mm = config::PATTERN_OFFSETS.at("middle");
    } else { // Top pattern
        pattern_specific_offset_mm = config::PATTERN_OFFSETS.at("top");
    }

    // Calculate true origin (bottom-left of frame) in mm
    double true_origin_x_mm = frame_center_mm.x - config::FRAME_SIZE_X / 2.0;
    double true_origin_y_mm = frame_center_mm.y - config::FRAME_SIZE_Y / 2.0; // In python, Y is often top-left, but here it's bottom-left for consistency before pixel conversion

    // Calculate true origin in pixels (image coordinates: Y positive downwards)
    int true_origin_x_px = img_center_x_px - static_cast<int>((config::FRAME_SIZE_X / 2.0) * config::SCALING_X);
    int true_origin_y_px = img_center_y_px + static_cast<int>((config::FRAME_SIZE_Y / 2.0) * config::SCALING_Y); // Y positive downwards

    // Calculate component position relative to pattern (in mm)
    double comp_x_rel_pattern_mm = comp.x_mm - config::PCB_BORDER + pattern_pos_mm.first;
    double comp_y_rel_pattern_mm = comp.y_mm - config::PCB_BORDER + pattern_pos_mm.second;

    // Convert to position relative to true origin (in mm)
    double rel_x_mm = comp_x_rel_pattern_mm - true_origin_x_mm;
    // For y_mm, the Python script subtracts PCB_BORDER then adds pattern_pos[1].
    // Then rel_y_mm = comp_y_mm - true_origin_y_mm.
    // True origin_y_mm is frame_center.y - FRAME_SIZE_Y/2 (bottom of frame).
    // comp.y_mm is from bottom of pattern. pattern_pos[1] is pattern bottom from PCB origin.
    // So comp_y_mm (relative to pattern's bottom-left) + pattern_pos[1] (pattern's bottom-left y from some global origin).
    // The python script seems to handle y consistently with its coordinate system.
    // Let's stick to the formula:
    double rel_y_mm = comp_y_rel_pattern_mm - true_origin_y_mm;


    // Convert to pixel coordinates and apply frame-specific and pattern-specific offsets
    // Note: In image coordinates, Y is positive downwards.
    // The Python script has:
    // y_px = true_origin_y_px - int(rel_y_mm * SCALING_Y) + int(y_offset * SCALING_Y) + int(pattern_offset['y'] * SCALING_Y)
    // true_origin_y_px is already bottom-left in pixel space (large y).
    // rel_y_mm is distance from bottom of frame. So true_origin_y_px - (rel_y_mm * SCALING_Y) moves up.
    // Offsets: y_offset positive implies moving down in image.
    // pattern_offset['y'] positive implies moving down in image.

    int x_px = true_origin_x_px + static_cast<int>(rel_x_mm * config::SCALING_X) +
               static_cast<int>(frame_offset_mm.x * config::SCALING_X) +
               static_cast<int>(pattern_specific_offset_mm.x * config::SCALING_X);
    int y_px = true_origin_y_px - static_cast<int>(rel_y_mm * config::SCALING_Y) + // Subtract because rel_y_mm is from bottom, and pixel Y increases downwards
               static_cast<int>(frame_offset_mm.y * config::SCALING_Y) +         // Positive y_offset in mm shifts component down in image (increases y_px)
               static_cast<int>(pattern_specific_offset_mm.y * config::SCALING_Y); // Positive pattern y_offset shifts component down in image

    return {x_px, y_px};
}

FrameBoundaries augmentation::calculate_frame_boundaries(
    const config::Offset& frame_center_mm,
    int img_center_x_px,
    int img_center_y_px) {

    // Calculate frame boundaries in pixels
    // Image origin (0,0) is top-left. Y increases downwards.
    int frame_left_px = img_center_x_px - static_cast<int>((config::FRAME_SIZE_X / 2.0) * config::SCALING_X);
    int frame_right_px = img_center_x_px + static_cast<int>((config::FRAME_SIZE_X / 2.0) * config::SCALING_X);
    int frame_top_px = img_center_y_px - static_cast<int>((config::FRAME_SIZE_Y / 2.0) * config::SCALING_Y);
    int frame_bottom_px = img_center_y_px + static_cast<int>((config::FRAME_SIZE_Y / 2.0) * config::SCALING_Y);

    return {frame_left_px, frame_right_px, frame_top_px, frame_bottom_px};
}

bool augmentation::is_component_fully_visible(
    int x_px, int y_px, // component center
    int width_px, int height_px,
    const FrameBoundaries& frame_boundaries) {

    int x1 = x_px - width_px / 2 - config::PADDING_PX;
    int y1 = y_px - height_px / 2 - config::PADDING_PX;
    int x2 = x_px + width_px / 2 + config::PADDING_PX;
    int y2 = y_px + height_px / 2 + config::PADDING_PX;

    return (x1 >= frame_boundaries.left &&
            y1 >= frame_boundaries.top &&
            x2 <= frame_boundaries.right &&
            y2 <= frame_boundaries.bottom);
}




fs::path  augmentation::comp_extract() {




    fs::path output_base_path = config::DATA_DIR; //
    fs::path components_dir = config::COMPONENTS_DIR_EXTRACTION; //
    fs::path frames_dir = config::FRAMES_DIR_EXTRACTION; //

    try {
        fs::create_directories(components_dir); //
        fs::create_directories(frames_dir); //
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directories: " << e.what() << std::endl;
        return output_base_path;
    }

    // Process each frame
    for (const auto& frame_entry : config::FRAME_PATTERNS) {
        int frame_id = frame_entry.first;
        const std::vector<int>& pattern_ids = frame_entry.second;

        config::Offset frame_center_mm;
        auto it_fc = config::FRAME_CENTERS.find(frame_id);
        if (it_fc != config::FRAME_CENTERS.end()) {
            frame_center_mm = it_fc->second;
        } else {
            std::cerr << "Error: Frame center not found for frame_id: " << frame_id << std::endl;
            continue;
        }

        fs::path image_path = fs::path(config::FILLED_DIR) / ("F" + std::to_string(frame_id) + ".jpg"); //

        if (!fs::exists(image_path)) { //
            std::cout << "Skipping missing frame image: F" << frame_id << std::endl; //
            continue;
        }

        cv::Mat img = cv::imread(image_path.string()); //
        if (img.empty()) { //
            std::cout << "Failed to read image: " << image_path.string() << std::endl; //
            continue;
        }

        int h = img.rows; //
        int w = img.cols; //
        int img_center_x_px = w / 2; //
        int img_center_y_px = h / 2; //

        FrameBoundaries frame_boundaries = calculate_frame_boundaries(frame_center_mm, img_center_x_px, img_center_y_px); //

        cv::Mat vis_img = img.clone(); //

        int total_components_in_frame = 0; //
        int visible_components_in_frame = 0; //

        // Process each pattern in the frame
        for (size_t pattern_index = 0; pattern_index < pattern_ids.size(); ++pattern_index) {
            int pid = pattern_ids[pattern_index];

            std::pair<double, double> pattern_pos_mm;
            auto it_pp = config::PATTERN_POSITIONS.find(pid);
            if (it_pp != config::PATTERN_POSITIONS.end()) {
                pattern_pos_mm = it_pp->second;
            } else {
                std::cerr << "Error: Pattern position not found for pattern_id: " << pid << std::endl;
                continue;
            }


            // Process each component in the pattern
            for (const auto& comp_template : config::BASE_COMPONENTS) {
                std::string pkg_name;
                auto it_pkg_map = config::COMPONENT_PACKAGES.find(comp_template.part_no);
                if (it_pkg_map != config::COMPONENT_PACKAGES.end()) {
                    pkg_name = it_pkg_map->second;
                } else {
                    // Default or error for unknown part_no
                    // std::cerr << "Warning: Package name not found for part_no: " << comp_template.part_no << std::endl;
                    // Using a default, as Python's .get with default
                }

                config::PackageDimensions dims_mm = {1.6, 0.8}; // Default if not found
                auto it_pkg_db = config::PACKAGE_DB.find(pkg_name);
                if (it_pkg_db != config::PACKAGE_DB.end()) {
                    dims_mm = it_pkg_db->second;
                } else if (!pkg_name.empty()) { // Only warn if pkg_name was found but details are missing
                    // std::cerr << "Warning: Package dimensions not found for package: " << pkg_name << std::endl;
                }

                double width_mm = dims_mm.width; //
                double height_mm = dims_mm.height; //

                // Swap dimensions for rotated components
                if ((comp_template.angle == 90 || comp_template.angle == 270) &&
                    comp_template.ref != "R4" && comp_template.ref != "D1") { //
                    std::swap(width_mm, height_mm); //
                }

                PixelPoint comp_center_px = this->calculate_component_position(
                    comp_template, pattern_pos_mm, frame_center_mm,
                    img_center_x_px, img_center_y_px, frame_id, static_cast<int>(pattern_index)); //

                int width_px = static_cast<int>(width_mm * config::SCALING_X); //
                int height_px = static_cast<int>(height_mm * config::SCALING_Y); //

                total_components_in_frame++; //

                if (!is_component_fully_visible(comp_center_px.x, comp_center_px.y, width_px, height_px, frame_boundaries)) { //
                    std::cout << "⚠️ Component " << comp_template.ref << "_p" << pid << " in F" << frame_id
                              << " is not fully visible, skipping" << std::endl; //
                    continue;
                }
                visible_components_in_frame++; //

                // Calculate bounding box with padding
                int x1 = std::max(comp_center_px.x - width_px / 2 - config::PADDING_PX, 0); //
                int y1 = std::max(comp_center_px.y - height_px / 2 - config::PADDING_PX, 0); //
                int x2 = std::min(comp_center_px.x + width_px / 2 + config::PADDING_PX, w); //
                int y2 = std::min(comp_center_px.y + height_px / 2 + config::PADDING_PX, h); //

                // Ensure x1 < x2 and y1 < y2 for cv::Rect
                if (x1 >= x2 || y1 >= y2) {
                    std::cout << "⚠️ Invalid bounding box for " << comp_template.ref << "_p" << pid << "_F" << frame_id
                              << ": box (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ") in image " << w << "x" << h << std::endl;
                    continue;
                }


                // Draw rectangle for visualization
                cv::rectangle(vis_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2); //
                std::string text = comp_template.ref + "_p" + std::to_string(pid);
                cv::putText(vis_img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1); //

                // Extract and save the component image
                cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
                cv::Mat crop = img(roi); //

                if (crop.empty() || crop.total() == 0) { // Python checks crop.size == 0
                    std::cout << "⚠️ Empty crop for " << comp_template.ref << "_p" << pid << "_F" << frame_id
                              << ": box (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ") in image " << w << "x" << h << std::endl; //
                    continue;
                }

                std::string filename_str = comp_template.ref + "_p" + std::to_string(pid) + "_F" + std::to_string(frame_id) + ".jpg"; //
                fs::path component_output_path = components_dir / filename_str;
                if (!cv::imwrite(component_output_path.string(), crop)) { //
                    std::cerr << "Failed to write component image: " << component_output_path.string() << std::endl;
                }
            }
        }
        std::cout << "Frame " << frame_id << ": " << visible_components_in_frame << "/" << total_components_in_frame
                  << " components fully visible" << std::endl; //

        // Save visualization image
        fs::path vis_path = frames_dir / ("visualization_F" + std::to_string(frame_id) + ".jpg"); //
        if (!cv::imwrite(vis_path.string(), vis_img)) { //
            std::cerr << "Failed to write visualization image: " << vis_path.string() << std::endl;
        } else {
            std::cout << "Saved visualization for frame " << frame_id << " to " << vis_path.string() << std::endl; //
        }
    }

    return output_base_path;
}


/*
 *
 */




// Helper function to get component dimensions in pixels (same as in missing_component_augmentation.cpp)
PixelDimensions augmentation::get_component_dimensions_px(const config::Component& comp_template) {
    std::string pkg_name;
    auto it_pkg_map = config::COMPONENT_PACKAGES.find(comp_template.part_no);
    if (it_pkg_map != config::COMPONENT_PACKAGES.end()) {
        pkg_name = it_pkg_map->second;
    }

    config::PackageDimensions dims_mm = {1.6, 0.8}; // Default
    auto it_pkg_db = config::PACKAGE_DB.find(pkg_name);
    if (it_pkg_db != config::PACKAGE_DB.end()) {
        dims_mm = it_pkg_db->second;
    }

    double width_mm = dims_mm.width;
    double height_mm = dims_mm.height;

    if ((comp_template.angle == 90 || comp_template.angle == 270) &&
        comp_template.ref != "R4" && comp_template.ref != "D1") {
        std::swap(width_mm, height_mm);
    }
    return {static_cast<int>(width_mm * config::SCALING_X), static_cast<int>(height_mm * config::SCALING_Y)};
}

// Get the index of a pattern in its frame (same as in missing_component_augmentation.cpp)
int augmentation::get_pattern_index_in_frame(int pattern_id, int frame_id) {
    auto it_frame = config::FRAME_PATTERNS.find(frame_id);
    if (it_frame != config::FRAME_PATTERNS.end()) {
        const auto& patterns_in_frame = it_frame->second;
        for (size_t i = 0; i < patterns_in_frame.size(); ++i) {
            if (patterns_in_frame[i] == pattern_id) {
                return static_cast<int>(i);
            }
        }
    }
    return 0;
}

// Calculate component position in image coordinates (same as in missing_component_augmentation.cpp)
PixelPoint augmentation::calculate_component_position_px(
    const config::Component& comp, const std::pair<double, double>& pattern_pos_mm,
    const config::Offset& frame_center_mm, int img_center_x_px, int img_center_y_px,
    int frame_id, int pattern_index_in_frame) {
    config::Offset frame_offset_mm = {0.0, 0.0};
    auto it_frame_offset = config::FRAME_OFFSETS.find(frame_id);
    if (it_frame_offset != config::FRAME_OFFSETS.end()) {
        frame_offset_mm = it_frame_offset->second;
    }
    config::Offset pattern_specific_offset_mm = {0.0, 0.0};
    if (pattern_index_in_frame == 0) pattern_specific_offset_mm = config::PATTERN_OFFSETS.at("bottom");
    else if (pattern_index_in_frame == 1) pattern_specific_offset_mm = config::PATTERN_OFFSETS.at("middle");
    else pattern_specific_offset_mm = config::PATTERN_OFFSETS.at("top");

    double true_origin_x_mm = frame_center_mm.x - config::FRAME_SIZE_X / 2.0;
    double true_origin_y_mm = frame_center_mm.y - config::FRAME_SIZE_Y / 2.0;
    int true_origin_x_px = img_center_x_px - static_cast<int>((config::FRAME_SIZE_X / 2.0) * config::SCALING_X);
    int true_origin_y_px = img_center_y_px + static_cast<int>((config::FRAME_SIZE_Y / 2.0) * config::SCALING_Y);
    double comp_x_rel_pattern_mm = comp.x_mm - config::PCB_BORDER + pattern_pos_mm.first;
    double comp_y_rel_pattern_mm = comp.y_mm - config::PCB_BORDER + pattern_pos_mm.second;
    double rel_x_mm = comp_x_rel_pattern_mm - true_origin_x_mm;
    double rel_y_mm = comp_y_rel_pattern_mm - true_origin_y_mm;
    int x_px = true_origin_x_px + static_cast<int>(rel_x_mm * config::SCALING_X) +
               static_cast<int>(frame_offset_mm.x * config::SCALING_X) +
               static_cast<int>(pattern_specific_offset_mm.x * config::SCALING_X);
    int y_px = true_origin_y_px - static_cast<int>(rel_y_mm * config::SCALING_Y) +
               static_cast<int>(frame_offset_mm.y * config::SCALING_Y) +
               static_cast<int>(pattern_specific_offset_mm.y * config::SCALING_Y);
    return {x_px, y_px};
}

// Calculate two bounding boxes: one for cleaning and one for component handling.
BoundingBoxSizes augmentation::get_bounding_boxes(int width_px, int height_px, const std::string& comp_ref) {
    int comp_box_size = std::max(width_px, height_px) + 2 * config::PADDING_PX;
    int clean_box_size;

    if (comp_ref == "U1") {
        clean_box_size = std::max(width_px, height_px) + 2 * (config::PADDING_PX + 1); // Minimal padding
    } else {
        int diagonal = static_cast<int>(std::sqrt(width_px * width_px + height_px * height_px));
        clean_box_size = diagonal + 2 * (config::PADDING_PX + config::CLEAN_PADDING_PX);
    }
    return {clean_box_size, comp_box_size};
}

// Get appropriate rotation angles based on component type.
std::vector<int> augmentation::get_rotation_angles_for_comp(const std::string& comp_ref) {
    char comp_type_char = ' ';
    if (!comp_ref.empty()) {
        comp_type_char = comp_ref[0];
    }
    auto it = config::ROTATION_ANGLES.find(comp_type_char);
    if (it != config::ROTATION_ANGLES.end()) {
        return it->second;
    }
    return {15, 20, 25}; // Default angles
}

// Returns pair of paths: {augmented_frame_path, component_region_path} or empty strings on failure
std::pair<std::string, std::string> augmentation::create_rotated_component_augmentation(
    int frame_id, int pattern_id, const config::Component& comp_template,
    int angle_deg, int direction,
    const fs::path& base_frames_dir, const fs::path& base_components_dir) {
    namespace fs = std::filesystem;

    fs::path filled_path = fs::path(config::FILLED_DIR) / ("F" + std::to_string(frame_id) + ".jpg");
    cv::Mat filled_img = cv::imread(filled_path.string());
    if (filled_img.empty()) {
        std::cerr << "Failed to read filled image: " << filled_path.string() << std::endl;
        return {};
    }

    fs::path empty_path = fs::path(config::EMPTY_DIR) / ("F" + std::to_string(frame_id) + ".jpg");
    cv::Mat empty_img = cv::imread(empty_path.string());
    if (empty_img.empty()) {
        std::cerr << "Failed to read empty image: " << empty_path.string() << std::endl;
        return {};
    }

    int h_img = filled_img.rows;
    int w_img = filled_img.cols;
    int img_center_x_px = w_img / 2;
    int img_center_y_px = h_img / 2;

    config::Offset frame_center_mm = config::FRAME_CENTERS.at(frame_id);
    std::pair<double, double> pattern_pos_mm = config::PATTERN_POSITIONS.at(pattern_id);
    int pattern_index = this->get_pattern_index_in_frame(pattern_id, frame_id);

    PixelPoint comp_center_px =  this->calculate_component_position_px(
        comp_template, pattern_pos_mm, frame_center_mm,
        img_center_x_px, img_center_y_px, frame_id, pattern_index);

    PixelDimensions comp_dims_px = this->get_component_dimensions_px(comp_template);

    BoundingBoxSizes P_SIZES = get_bounding_boxes(comp_dims_px.width, comp_dims_px.height, comp_template.ref);


    // Calculate cleaning box (for empty PCB)
    int clean_x1 = std::max(comp_center_px.x - P_SIZES.clean_box_size / 2, 0);
    int clean_y1 = std::max(comp_center_px.y - P_SIZES.clean_box_size / 2, 0);
    int clean_x2 = std::min(comp_center_px.x + P_SIZES.clean_box_size / 2, w_img);
    int clean_y2 = std::min(comp_center_px.y + P_SIZES.clean_box_size / 2, h_img);

    // Calculate component box (for rotation)
    int comp_box_x1 = std::max(comp_center_px.x - P_SIZES.comp_box_size / 2, 0);
    int comp_box_y1 = std::max(comp_center_px.y - P_SIZES.comp_box_size / 2, 0);
    int comp_box_x2 = std::min(comp_center_px.x + P_SIZES.comp_box_size / 2, w_img);
    int comp_box_y2 = std::min(comp_center_px.y + P_SIZES.comp_box_size / 2, h_img);

    if (comp_box_x1 >= comp_box_x2 || comp_box_y1 >= comp_box_y2) {
        std::cerr << "⚠️ Invalid component box for " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id << std::endl;
        return {};
    }
    if (clean_x1 >= clean_x2 || clean_y1 >= clean_y2) {
        std::cerr << "⚠️ Invalid clean box for " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id << std::endl;
        return {};
    }


    cv::Rect comp_roi(comp_box_x1, comp_box_y1, comp_box_x2 - comp_box_x1, comp_box_y2 - comp_box_y1);
    cv::Mat component_region_original = filled_img(comp_roi);
    cv::Mat empty_comp_region_for_blend = empty_img(comp_roi);

    if (component_region_original.empty() || empty_comp_region_for_blend.empty()) {
        std::cerr << "⚠️ Empty component or blend region for " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id << std::endl;
        return {};
    }

    cv::Point2f rot_center(static_cast<float>(component_region_original.cols) / 2.0f, static_cast<float>(component_region_original.rows) / 2.0f);
    double actual_angle = static_cast<double>(angle_deg) * direction;
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(rot_center, actual_angle, 1.0);

    cv::Mat rotated_component_on_black_bg;
    cv::warpAffine(component_region_original, rotated_component_on_black_bg, rotation_matrix, component_region_original.size());

    cv::Mat mask;
    cv::cvtColor(rotated_component_on_black_bg, mask, cv::COLOR_BGR2GRAY);
    cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

    // Blend: Start with empty PCB region, then copy rotated component parts using the mask
    cv::Mat blended_rotated_component = empty_comp_region_for_blend.clone();
    rotated_component_on_black_bg.copyTo(blended_rotated_component, mask);


    // Create augmented frame image
    cv::Mat augmented_frame = filled_img.clone();

    // First, replace the area with empty PCB (cleaning wider area)
    cv::Rect clean_roi(clean_x1, clean_y1, clean_x2 - clean_x1, clean_y2 - clean_y1);
    cv::Mat empty_clean_region = empty_img(clean_roi);
    if(empty_clean_region.empty()){
        std::cerr << "⚠️ Empty clean region for " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id << std::endl;
        return {};
    }
    empty_clean_region.copyTo(augmented_frame(clean_roi));

    // Paste the blended component back into the component box area
    blended_rotated_component.copyTo(augmented_frame(comp_roi));


    // Create component-specific directories
    fs::path comp_specific_frames_dir = base_frames_dir / comp_template.ref;
    fs::path comp_specific_components_dir = base_components_dir / comp_template.ref;
    try {
        fs::create_directories(comp_specific_frames_dir);
        fs::create_directories(comp_specific_components_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating component specific directories: " << e.what() << std::endl;
        return {};
    }

    std::string dir_str = (direction == 1) ? "cw" : "ccw";
    std::string base_filename = "rotated_" + comp_template.ref + "_p" + std::to_string(pattern_id) +
                                "_F" + std::to_string(frame_id) + "_" + std::to_string(angle_deg) + dir_str + ".jpg";

    fs::path augmented_frame_path = comp_specific_frames_dir / base_filename;
    if (!cv::imwrite(augmented_frame_path.string(), augmented_frame)) {
        std::cerr << "Failed to write augmented frame: " << augmented_frame_path.string() << std::endl;
    }

    fs::path component_image_path = comp_specific_components_dir / base_filename;
    if (!cv::imwrite(component_image_path.string(), blended_rotated_component)) {
        std::cerr << "Failed to write component image: " << component_image_path.string() << std::endl;
    }

    return {augmented_frame_path.string(), component_image_path.string()};
}




int augmentation::rotatedComponentAugmenter() {
    namespace fs = std::filesystem;

    // Get current time for timestamp
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss_timestamp;
    ss_timestamp << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss_timestamp.str();

    fs::path output_base_aug = fs::path(config::AUGMENTED_DIR);
    fs::path frames_output_dir = output_base_aug / ("rotated_components_" + timestamp) / "frames";
    fs::path components_output_dir = output_base_aug / ("rotated_components_" + timestamp) / "components";

    try {
        fs::create_directories(frames_output_dir);
        fs::create_directories(components_output_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating base augmentation directories: " << e.what() << std::endl;
        return 1;
    }

    size_t total_base_components = config::BASE_COMPONENTS.size();
    std::cout << "Total base components to process: " << total_base_components << std::endl;

    for (size_t i = 0; i < total_base_components; ++i) {
        const auto& comp_template = config::BASE_COMPONENTS[i];
        std::cout << "\nProcessing component " << (i + 1) << "/" << total_base_components << ": " << comp_template.ref << std::endl;

        int instance_count = 0;
        for (const auto& frame_pattern_entry : config::FRAME_PATTERNS) {
            const std::vector<int>& pattern_ids_in_frame = frame_pattern_entry.second;
            instance_count += pattern_ids_in_frame.size();
        }
        std::cout << "Found " << instance_count << " potential instances of " << comp_template.ref << std::endl;

        std::vector<int> angles_for_comp = get_rotation_angles_for_comp(comp_template.ref);

        for (const auto& frame_pattern_entry : config::FRAME_PATTERNS) {
            int frame_id = frame_pattern_entry.first;
            const std::vector<int>& pattern_ids_in_frame = frame_pattern_entry.second;
            for (int pattern_id : pattern_ids_in_frame) {
                std::cout << "Processing " << comp_template.ref << " in F" << frame_id << ", pattern " << pattern_id << std::endl;
                for (int angle_val : angles_for_comp) {
                    for (int direction_val : config::ROTATION_DIRECTIONS) {
                        auto result_paths = create_rotated_component_augmentation(
                            frame_id, pattern_id, comp_template, angle_val, direction_val,
                            frames_output_dir, components_output_dir);

                        if (!result_paths.first.empty())
                            std::cout << "Created frame augmentation: " << result_paths.first << std::endl;
                        if (!result_paths.second.empty())
                            std::cout << "Created component augmentation: " << result_paths.second << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "\nRotated component augmentation complete." << std::endl;
    return 0;
}



// Calculate frame boundaries in image coordinates (same as in component_extraction.cpp)
FrameBoundaries augmentation::calculate_frame_boundaries_px(
    const config::Offset& frame_center_mm,
    int img_center_x_px,
    int img_center_y_px) {
    int frame_left_px = img_center_x_px - static_cast<int>((config::FRAME_SIZE_X / 2.0) * config::SCALING_X);
    int frame_right_px = img_center_x_px + static_cast<int>((config::FRAME_SIZE_X / 2.0) * config::SCALING_X);
    int frame_top_px = img_center_y_px - static_cast<int>((config::FRAME_SIZE_Y / 2.0) * config::SCALING_Y);
    int frame_bottom_px = img_center_y_px + static_cast<int>((config::FRAME_SIZE_Y / 2.0) * config::SCALING_Y);
    return {frame_left_px, frame_right_px, frame_top_px, frame_bottom_px};
}


// The Python script defines its own is_component_fully_visible for augmentation,
// which is slightly different from the one in component_extraction.
// Here, it's integrated into the create_missing_component_augmentation logic.


// Returns pair of paths: {augmented_frame_path, component_region_path} or empty strings on failure
std::pair<std::string, std::string> augmentation::create_missing_component_augmentation(
    int frame_id,
    int pattern_id,
    const config::Component& comp_template,
    const fs::path& base_frames_dir, // Base for augmented frames output (e.g., .../missing_components/frames)
    const fs::path& base_components_dir // Base for "missing" component regions (e.g., .../missing_components/components)
    ) {
    namespace fs = std::filesystem;

    fs::path filled_path = fs::path(config::FILLED_DIR) / ("F" + std::to_string(frame_id) + ".jpg");
    cv::Mat filled_img = cv::imread(filled_path.string());
    if (filled_img.empty()) {
        std::cerr << "Failed to read filled image: " << filled_path.string() << std::endl;
        return {};
    }

    fs::path empty_path = fs::path(config::EMPTY_DIR) / ("F" + std::to_string(frame_id) + ".jpg");
    cv::Mat empty_img = cv::imread(empty_path.string());
    if (empty_img.empty()) {
        std::cerr << "Failed to read empty image: " << empty_path.string() << std::endl;
        return {};
    }

    int h = filled_img.rows;
    int w = filled_img.cols;
    int img_center_x_px = w / 2;
    int img_center_y_px = h / 2;

    config::Offset frame_center_mm = config::FRAME_CENTERS.at(frame_id);
    std::pair<double, double> pattern_pos_mm = config::PATTERN_POSITIONS.at(pattern_id);
    int pattern_index = this->get_pattern_index_in_frame(pattern_id, frame_id);

    PixelPoint comp_center_px =  this->calculate_component_position_px(
        comp_template, pattern_pos_mm, frame_center_mm,
        img_center_x_px, img_center_y_px, frame_id, pattern_index);

    PixelDimensions comp_dims_px =  this->get_component_dimensions_px(comp_template);

    FrameBoundaries frame_boundaries = this->calculate_frame_boundaries_px(frame_center_mm, img_center_x_px, img_center_y_px);

    // Visibility check (from the python script's augmentation logic)
    int check_x1 = comp_center_px.x - comp_dims_px.width / 2 - config::PADDING_PX;
    int check_y1 = comp_center_px.y - comp_dims_px.height / 2 - config::PADDING_PX;
    int check_x2 = comp_center_px.x + comp_dims_px.width / 2 + config::PADDING_PX;
    int check_y2 = comp_center_px.y + comp_dims_px.height / 2 + config::PADDING_PX;

    if (!(check_x1 >= frame_boundaries.left &&
          check_y1 >= frame_boundaries.top &&
          check_x2 <= frame_boundaries.right &&
          check_y2 <= frame_boundaries.bottom)) {
        std::cout << "⚠️ Component " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id
                  << " is not fully visible for augmentation, skipping" << std::endl;
        return {};
    }

    // Calculate bounding box with padding, clamped to image dimensions
    int roi_x1 = std::max(comp_center_px.x - comp_dims_px.width / 2 - config::PADDING_PX, 0);
    int roi_y1 = std::max(comp_center_px.y - comp_dims_px.height / 2 - config::PADDING_PX, 0);
    int roi_x2 = std::min(comp_center_px.x + comp_dims_px.width / 2 + config::PADDING_PX, w);
    int roi_y2 = std::min(comp_center_px.y + comp_dims_px.height / 2 + config::PADDING_PX, h);

    if (roi_x1 >= roi_x2 || roi_y1 >= roi_y2) {
        std::cerr << "⚠️ Invalid ROI for " << comp_template.ref << "_p" << pattern_id << "_F" << frame_id << std::endl;
        return {};
    }

    // Create component-specific directories
    fs::path comp_specific_frames_dir = base_frames_dir / comp_template.ref;
    fs::path comp_specific_components_dir = base_components_dir / comp_template.ref;
    try {
        fs::create_directories(comp_specific_frames_dir);
        fs::create_directories(comp_specific_components_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating component specific directories: " << e.what() << std::endl;
        return {};
    }

    cv::Mat augmented_frame = filled_img.clone();
    cv::Rect roi(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1);

    cv::Mat empty_region = empty_img(roi);
    if (empty_region.empty()) {
        std::cerr << "⚠️ Empty region extracted from empty_img for " << comp_template.ref << "_p" << pattern_id << "_F" << frame_id << std::endl;
        return {};
    }
    empty_region.copyTo(augmented_frame(roi));

    std::string base_filename = "missing_" + comp_template.ref + "_p" + std::to_string(pattern_id) + "_F" + std::to_string(frame_id) + ".jpg";

    fs::path augmented_frame_path = comp_specific_frames_dir / base_filename;
    if (!cv::imwrite(augmented_frame_path.string(), augmented_frame)) {
        std::cerr << "Failed to write augmented frame: " << augmented_frame_path.string() << std::endl;
        // Continue to try saving component image
    }

    fs::path component_region_path = comp_specific_components_dir / base_filename;
    if (!cv::imwrite(component_region_path.string(), empty_region)) { // Save the empty part
        std::cerr << "Failed to write component region: " << component_region_path.string() << std::endl;
    }

    return {augmented_frame_path.string(), component_region_path.string()};
}


int augmentation::missingComponentAugmenter() {
    namespace fs = std::filesystem;

    fs::path output_base_aug = fs::path(config::AUGMENTED_DIR);
    fs::path frames_output_dir = output_base_aug / "missing_components" / "frames";
    fs::path components_output_dir = output_base_aug / "missing_components" / "components";

    try {
        fs::create_directories(frames_output_dir);
        fs::create_directories(components_output_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating base augmentation directories: " << e.what() << std::endl;
        return 1;
    }

    size_t total_base_components = config::BASE_COMPONENTS.size();
    std::cout << "Total base components to process: " << total_base_components << std::endl;

    for (size_t i = 0; i < total_base_components; ++i) {
        const auto& comp_template = config::BASE_COMPONENTS[i];
        std::cout << "\nProcessing component " << (i + 1) << "/" << total_base_components << ": " << comp_template.ref << std::endl;

        int instance_count = 0;
        for (const auto& frame_pattern_entry : config::FRAME_PATTERNS) {
            int frame_id = frame_pattern_entry.first;
            const std::vector<int>& pattern_ids_in_frame = frame_pattern_entry.second;
            for (int pattern_id : pattern_ids_in_frame) {
                instance_count++;
                // The Python script has a separate loop to count instances first, then processes.
                // Here, we can process directly. The "Found X instances" log can be adapted.
            }
        }
        std::cout << "Found " << instance_count << " potential instances of " << comp_template.ref << std::endl;


        for (const auto& frame_pattern_entry : config::FRAME_PATTERNS) {
            int frame_id = frame_pattern_entry.first;
            const std::vector<int>& pattern_ids_in_frame = frame_pattern_entry.second;
            for (int pattern_id : pattern_ids_in_frame) {
                std::cout << "Processing " << comp_template.ref << " in F" << frame_id << ", pattern " << pattern_id << std::endl;
                auto result_paths = create_missing_component_augmentation(
                    frame_id, pattern_id, comp_template, frames_output_dir, components_output_dir);

                if (!result_paths.first.empty() || !result_paths.second.empty()) {
                    if (!result_paths.first.empty())
                        std::cout << "Created frame augmentation: " << result_paths.first << std::endl;
                    if (!result_paths.second.empty())
                        std::cout << "Created component augmentation: " << result_paths.second << std::endl;
                }
            }
        }
    }
    std::cout << "\nMissing component augmentation complete." << std::endl;
    return 0;
}


// Check if a component is fully visible within frame boundaries (consistent with other scripts)
bool is_component_fully_visible(
    int center_x_px, int center_y_px, int width_px, int height_px,
    const FrameBoundaries& frame_boundaries) {
    int x1 = center_x_px - width_px / 2 - config::PADDING_PX;
    int y1 = center_y_px - height_px / 2 - config::PADDING_PX;
    int x2 = center_x_px + width_px / 2 + config::PADDING_PX;
    int y2 = center_y_px + height_px / 2 + config::PADDING_PX;
    return (x1 >= frame_boundaries.left &&
            y1 >= frame_boundaries.top &&
            x2 <= frame_boundaries.right &&
            y2 <= frame_boundaries.bottom);
}



// Get appropriate shift percentages based on component type.
std::vector<double> augmentation::get_shift_percentages_for_comp(const std::string& component_ref) {
    if (component_ref.rfind("U1", 0) == 0) { // starts_with U1
        return config::SHIFT_PERCENTAGES.at("U1");
    } else if (component_ref.rfind("U2", 0) == 0) { // starts_with U2
        return config::SHIFT_PERCENTAGES.at("U2");
    } else {
        return config::SHIFT_PERCENTAGES.at("default");
    }
}





// Returns pair of paths: {augmented_frame_path, component_region_path} or empty strings on failure
std::pair<std::string, std::string> augmentation::create_shifted_component_augmentation(
    int frame_id, int pattern_id, const config::Component& comp_template, int shift_idx,
    const fs::path& base_frames_dir, const fs::path& base_components_dir, int num_total_shifts_for_angle_calc) {
    namespace fs = std::filesystem;

    fs::path filled_path = fs::path(config::FILLED_DIR) / ("F" + std::to_string(frame_id) + ".jpg");
    cv::Mat filled_img = cv::imread(filled_path.string());
    if (filled_img.empty()) {
        std::cerr << "Failed to read filled image: " << filled_path.string() << std::endl;
        return {};
    }

    fs::path empty_path = fs::path(config::EMPTY_DIR) / ("F" + std::to_string(frame_id) + ".jpg");
    cv::Mat empty_img = cv::imread(empty_path.string());
    if (empty_img.empty()) {
        std::cerr << "Failed to read empty image: " << empty_path.string() << std::endl;
        return {};
    }

    int h_img = filled_img.rows;
    int w_img = filled_img.cols;
    int img_center_x_px = w_img / 2;
    int img_center_y_px = h_img / 2;

    config::Offset frame_center_mm = config::FRAME_CENTERS.at(frame_id);
    std::pair<double, double> pattern_pos_mm = config::PATTERN_POSITIONS.at(pattern_id);
    int pattern_index = this->get_pattern_index_in_frame(pattern_id, frame_id);

    PixelPoint original_comp_center_px =  this->calculate_component_position_px(
        comp_template, pattern_pos_mm, frame_center_mm,
        img_center_x_px, img_center_y_px, frame_id, pattern_index);

    PixelDimensions comp_dims_px =  this->get_component_dimensions_px(comp_template);
    FrameBoundaries frame_boundaries = calculate_frame_boundaries_px(frame_center_mm, img_center_x_px, img_center_y_px);

    if (!is_component_fully_visible(original_comp_center_px.x, original_comp_center_px.y,
                                    comp_dims_px.width, comp_dims_px.height, frame_boundaries)) {
        std::cout << "⚠️ Component " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id
                  << " is not fully visible (original position), skipping shift." << std::endl;
        return {};
    }

    std::vector<double> shift_percentages = get_shift_percentages_for_comp(comp_template.ref);
    double shift_percentage = shift_percentages[shift_idx % shift_percentages.size()];

    int shift_amount_x_max = static_cast<int>(shift_percentage * comp_dims_px.width);
    int shift_amount_y_max = static_cast<int>(shift_percentage * comp_dims_px.height);

    double base_angle = (static_cast<double>(shift_idx) * 2.0 * M_PI / static_cast<double>(num_total_shifts_for_angle_calc));
    double random_offset_angle = distribution(generator); // Generate random offset
    double angle_rad = base_angle + random_offset_angle;

    int shift_delta_x = static_cast<int>(shift_amount_x_max * std::cos(angle_rad));
    int shift_delta_y = static_cast<int>(shift_amount_y_max * std::sin(angle_rad));

    PixelPoint new_comp_center_px = {original_comp_center_px.x + shift_delta_x, original_comp_center_px.y + shift_delta_y};

    if (!is_component_fully_visible(new_comp_center_px.x, new_comp_center_px.y,
                                    comp_dims_px.width, comp_dims_px.height, frame_boundaries)) {
        std::cout << "⚠️ Shifted component " << comp_template.ref << "_p" << pattern_id << " in F" << frame_id
                  << " would be out of bounds, skipping." << std::endl;
        return {};
    }

    // Bounding box for original component (with padding)
    int orig_roi_x1 = std::max(original_comp_center_px.x - comp_dims_px.width / 2 - config::PADDING_PX, 0);
    int orig_roi_y1 = std::max(original_comp_center_px.y - comp_dims_px.height / 2 - config::PADDING_PX, 0);
    int orig_roi_x2 = std::min(original_comp_center_px.x + comp_dims_px.width / 2 + config::PADDING_PX, w_img);
    int orig_roi_y2 = std::min(original_comp_center_px.y + comp_dims_px.height / 2 + config::PADDING_PX, h_img);

    // Bounding box for new shifted component (with padding)
    int new_roi_x1 = std::max(new_comp_center_px.x - comp_dims_px.width / 2 - config::PADDING_PX, 0);
    int new_roi_y1 = std::max(new_comp_center_px.y - comp_dims_px.height / 2 - config::PADDING_PX, 0);
    int new_roi_x2 = std::min(new_comp_center_px.x + comp_dims_px.width / 2 + config::PADDING_PX, w_img);
    int new_roi_y2 = std::min(new_comp_center_px.y + comp_dims_px.height / 2 + config::PADDING_PX, h_img);

    if (orig_roi_x1 >= orig_roi_x2 || orig_roi_y1 >= orig_roi_y2 || new_roi_x1 >= new_roi_x2 || new_roi_y1 >= new_roi_y2) {
        std::cerr << "⚠️ Invalid ROI for shift augmentation " << comp_template.ref << "_p" << pattern_id << "_F" << frame_id << std::endl;
        return {};
    }

    cv::Mat augmented_frame = filled_img.clone();

    // 1. Clear original position with empty PCB
    cv::Rect original_location_roi(orig_roi_x1, orig_roi_y1, orig_roi_x2 - orig_roi_x1, orig_roi_y2 - orig_roi_y1);
    cv::Mat empty_patch_for_original_loc = empty_img(original_location_roi);
    if(empty_patch_for_original_loc.empty()){
        std::cerr << "⚠️ Empty patch from empty_img for " << comp_template.ref << "_p" << pattern_id << "_F" << frame_id << std::endl;
        return {};
    }
    empty_patch_for_original_loc.copyTo(augmented_frame(original_location_roi));

    // 2. Get component from original filled image
    cv::Mat component_to_move = filled_img(original_location_roi); // Use original_location_roi to get the component
    if(component_to_move.empty()){
        std::cerr << "⚠️ Empty component_to_move from filled_img for " << comp_template.ref << "_p" << pattern_id << "_F" << frame_id << std::endl;
        return {};
    }

    // 3. Paste component to new position
    // Important: The ROI for pasting must match the size of component_to_move.
    // new_roi_x1, new_roi_y1 define the top-left of the new component's bounding box.
    // The width and height of this destination ROI must be component_to_move.cols and component_to_move.rows.
    cv::Rect new_component_dest_roi(new_roi_x1, new_roi_y1, component_to_move.cols, component_to_move.rows);
    // Ensure new_component_dest_roi is within augmented_frame bounds
    new_component_dest_roi &= cv::Rect(0, 0, augmented_frame.cols, augmented_frame.rows);
    if (new_component_dest_roi.width != component_to_move.cols || new_component_dest_roi.height != component_to_move.rows) {
        // This can happen if the shifted component is partially out of bounds.
        // We need to crop component_to_move to fit the valid part of new_component_dest_roi.
        cv::Mat cropped_component_to_move = component_to_move(cv::Rect(0,0, new_component_dest_roi.width, new_component_dest_roi.height));
        cropped_component_to_move.copyTo(augmented_frame(new_component_dest_roi));
    } else {
        component_to_move.copyTo(augmented_frame(new_component_dest_roi));
    }


    // 4. Get the "overlapping region" from the original position in the NOW MODIFIED augmented_frame
    cv::Mat component_level_img = augmented_frame(original_location_roi);


    // Create component-specific directories
    fs::path comp_specific_frames_dir = base_frames_dir / comp_template.ref;
    fs::path comp_specific_components_dir = base_components_dir / comp_template.ref;
    try {
        fs::create_directories(comp_specific_frames_dir);
        fs::create_directories(comp_specific_components_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating component specific directories: " << e.what() << std::endl;
        return {};
    }

    std::string base_filename = "shifted_" + comp_template.ref + "_p" + std::to_string(pattern_id) +
                                "_F" + std::to_string(frame_id) + "_s" + std::to_string(shift_idx) +
                                "_p" + std::to_string(static_cast<int>(shift_percentage * 100)) + ".jpg";

    fs::path augmented_frame_path = comp_specific_frames_dir / base_filename;
    if (!cv::imwrite(augmented_frame_path.string(), augmented_frame)) {
        std::cerr << "Failed to write augmented frame: " << augmented_frame_path.string() << std::endl;
    }

    fs::path component_image_path = comp_specific_components_dir / base_filename;
    if (!cv::imwrite(component_image_path.string(), component_level_img)) {
        std::cerr << "Failed to write component image (original area): " << component_image_path.string() << std::endl;
    }

    return {augmented_frame_path.string(), component_image_path.string()};
}




int augmentation::shiftedComponentAugmentation() {
    namespace fs = std::filesystem;

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss_timestamp;
    ss_timestamp << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss_timestamp.str();

    fs::path output_base_aug = fs::path(config::AUGMENTED_DIR);
    fs::path frames_output_dir = output_base_aug / ("shifted_components_" + timestamp) / "frames";
    fs::path components_output_dir = output_base_aug / ("shifted_components_" + timestamp) / "components";

    try {
        fs::create_directories(frames_output_dir);
        fs::create_directories(components_output_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating base augmentation directories: " << e.what() << std::endl;
        return 1;
    }

    const int num_shifts_per_instance = 5; // As per Python script's loop range

    for (const auto& comp_template : config::BASE_COMPONENTS) {
        std::cout << "\nProcessing component: " << comp_template.ref << std::endl;

        int instance_count = 0;
        for (const auto& frame_pattern_entry : config::FRAME_PATTERNS) {
            const std::vector<int>& pattern_ids_in_frame = frame_pattern_entry.second;
            instance_count += pattern_ids_in_frame.size();
        }
        std::cout << "Found " << instance_count << " potential instances of " << comp_template.ref << std::endl;

        for (const auto& frame_pattern_entry : config::FRAME_PATTERNS) {
            int frame_id = frame_pattern_entry.first;
            const std::vector<int>& pattern_ids_in_frame = frame_pattern_entry.second;
            for (int pattern_id : pattern_ids_in_frame) {
                std::cout << "Processing " << comp_template.ref << " in F" << frame_id << ", pattern " << pattern_id << std::endl;
                for (int shift_i = 0; shift_i < num_shifts_per_instance; ++shift_i) {
                    auto result_paths = create_shifted_component_augmentation(
                        frame_id, pattern_id, comp_template, shift_i,
                        frames_output_dir, components_output_dir, num_shifts_per_instance);

                    if (!result_paths.first.empty())
                        std::cout << "Created frame augmentation: " << result_paths.first << std::endl;
                    if (!result_paths.second.empty())
                        std::cout << "Created component augmentation: " << result_paths.second << std::endl;
                }
            }
        }
    }
    std::cout << "\nShifted component augmentation complete." << std::endl;
    return 0;
}
