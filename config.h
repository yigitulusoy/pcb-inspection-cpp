// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <utility> // For std::pair

namespace config {

    // Image scaling factors (pixels/mm)
    const double SCALING_X = 1000.0 / 21.964; // Approx 45.53
    const double SCALING_Y = 1000.0 / 21.596; // Approx 46.30

    // Frame dimensions (mm)
    const double FRAME_SIZE_X = 44.228608;
    const double FRAME_SIZE_Y = 53.767872;

    // Padding for component extraction (pixels)
    const int PADDING_PX = 8;
    const double PCB_BORDER = 0.0; // mm
    const int CLEAN_PADDING_PX = 2;

    // Struct for 2D points/offsets
    struct Offset {
        double x;
        double y;
    };

    // Pattern-specific offsets within frames (in mm)
    extern const std::map<std::string, Offset> PATTERN_OFFSETS;

    // Frame-specific offset adjustments (in mm)
    extern const std::map<int, Offset> FRAME_OFFSETS;

    // Frame centers from PRG (in mm)
    extern const std::map<int, Offset> FRAME_CENTERS;

    // Pattern positions (in mm) - pair: {x, y}
    extern const std::map<int, std::pair<double, double>> PATTERN_POSITIONS;

    // Frame to pattern mapping
    extern const std::map<int, std::vector<int>> FRAME_PATTERNS;

    // Component structure
    struct Component {
        std::string ref;
        std::string part_no;
        double x_mm;
        double y_mm;
        int angle;
    };

    // Components from pattern 1
    extern const std::vector<Component> BASE_COMPONENTS;

    // Package dimensions structure
    struct PackageDimensions {
        double width;
        double height;
    };

    // Package dimensions (mm)
    extern const std::map<std::string, PackageDimensions> PACKAGE_DB;

    // Component to package mapping
    extern const std::map<std::string, std::string> COMPONENT_PACKAGES;

    // Rotation Augmentation Parameters
    extern const std::map<char, std::vector<int>> ROTATION_ANGLES;
    extern const std::vector<int> ROTATION_DIRECTIONS;

    // Shift Augmentation Parameters
    extern const std::map<std::string, std::vector<double>> SHIFT_PERCENTAGES;

    // File paths
    const std::string DATA_DIR = "./data"; 
    const std::string RAW_DIR = DATA_DIR + "/raw";
    const std::string FILLED_DIR = RAW_DIR + "/filled";
    const std::string EMPTY_DIR = RAW_DIR + "/empty";
    const std::string COMPONENTS_DIR_EXTRACTION = DATA_DIR + "/components_v8";
    const std::string FRAMES_DIR_EXTRACTION = DATA_DIR + "/frames_v8";
    const std::string AUGMENTED_DIR = DATA_DIR + "/augmented_v8";

    // Note: Other paths like BASE_DATASET_PATH, CHECKPOINT_PATH etc. are not directly used
    // in the provided Python scripts for conversion but are kept here for completeness if needed later.
    const std::string BASE_DATASET_PATH = DATA_DIR + "/yolo_dataset";
    const std::string CHECKPOINT_PATH = "pcb_defect_detection/yolov8_defect_classifier/weights/last.pt";
    const std::string PRETRAINED_MODEL = "yolov8n-cls.pt";

    // Training parameters (not used in these scripts but listed in config)
    const int EPOCHS = 100;
    const int IMAGE_SIZE = 640;
    const int BATCH_SIZE = 16;
    const int PATIENCE = 20;
    const std::string DEVICE = "cpu";

    const std::string PROJECT_NAME = "pcb_defect_detection";
    const std::string MODEL_NAME = "yolov8_defect_classifier";

    extern const std::vector<std::string> SPLITS;
    extern const std::vector<std::string> CLASSES;

} // namespace config

#endif // CONFIG_H