/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <Simulator.h>
#include <raymath.h>
#include <rcamera.h>
#include <string_view>
#include <nanoflann.h>
#include <unordered_map>

/**************************************************************************************/
// Abstract base class for model information including bounding box
/**************************************************************************************/
struct ModelInfo {
    Model model;                    // The 3D model
    BoundingBox boundingBox;        // Minimal axis-aligned bounding box
    Vector3 dimensions;             // Width (x), Height (y), Depth (z) of bounding box
    double orientation_offset;      // Default offset of model in radians
    Color color;                    // Model color
    
    ModelInfo() = default;
    ModelInfo(Model m, BoundingBox bb, Vector3 dims, double of, Color c = WHITE)
        : model(m), boundingBox(bb), dimensions(dims), orientation_offset(of), color(c) {}
    virtual ~ModelInfo() = default;
    
    // Virtual method for nearest point queries - returns k nearest points as (point, squared_dist)
    virtual std::vector<std::pair<Eigen::Vector2d, double>> getNearestPoints(int k, const Eigen::Vector2d &query_pt) const {
        // Default implementation for robots - returns empty vector
        return std::vector<std::pair<Eigen::Vector2d, double>>();
    }
};

/**************************************************************************************/
// Concrete implementation for robot models (no KDTree needed)
/**************************************************************************************/
struct RobotModelInfo : public ModelInfo {
    RobotModelInfo() = default;
    RobotModelInfo(Model m, BoundingBox bb, Vector3 dims, double of)
        : ModelInfo(m, bb, dims, of) {}
};

/**************************************************************************************/
// Concrete implementation for obstacle models with KDTree support
/**************************************************************************************/
struct ObstacleModelInfo : public ModelInfo {
    Eigen::Matrix<double, 2, Eigen::Dynamic> mat_;  // Point cloud matrix
    using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, 2, Eigen::Dynamic>, 2, nanoflann::metric_L2, false>;
    std::unique_ptr<KDTree> kdtree_;
    
    ObstacleModelInfo() = default;
    ObstacleModelInfo(Model m, BoundingBox bb, Vector3 dims, double of, Color c = GRAY)
        : ModelInfo(m, bb, dims, of, c) {}
    
    // Initialize KDTree with point cloud
    void initializeKDTree(const std::vector<Eigen::Vector2d> &points);
    
    // Override nearest point query with actual implementation
    std::vector<std::pair<Eigen::Vector2d, double>> getNearestPoints(int k, const Eigen::Vector2d &query_pt) const override;
};


/**************************************************************************************/
// Graphics class that deals with the nitty-gritty of display.
// Camera is also included here. You can set different camera positions/trajectories
// and then during simulation cycle through them using the SPACEBAR

// Please note: Raylib camera defines the world with positive X = right, positive Z = down, and positive Y = out-of-plane
// But in our work we use the standard convention of positive X = right, positive Y = down, and positive Z = into-plane
/**************************************************************************************/
class Graphics
{
public:
    // Constructor
    Graphics(Image obstacleImg);
    ~Graphics();

    Image obstacleImg_;      // Image representing obstacles in the environment
    Texture2D texture_img_;  // Raylib Texture created from obstacleImg
    Model groundModel_;      // Model representing the ground plane
    Vector3 groundModelpos_; // Ground plane position
    Shader lightShader_;     // Light shader
    
    // Map storing model information for each robot type
    std::unordered_map<RobotType, std::shared_ptr<RobotModelInfo>> robotModels_;
    std::unordered_map<ObstacleType, std::shared_ptr<ObstacleModelInfo>> obstacleModels_;

    Camera3D camera3d = {0}; // Define the camera to look into our 3d world
    // These represent a set of camera transition frames.
    std::vector<Vector3> camera_positions_{};
    std::vector<Vector3> camera_ups_{};
    std::vector<Vector3> camera_targets_{};
    int camera_idx_ = 0;
    uint32_t camera_clock_ = 0;
    bool camera_transition_ = false;

    // Function to update camera based on mouse and key input
    void update_camera();
    
    // Helper functions for model management
    void loadRobotModels();     // Load all robot models and compute their bounding boxes
    void loadObstacleModels();  // Load all obstacle models and compute their bounding boxes
    void loadHeadlessModelInfo(); // Load minimal model info for headless mode
    
    // Helper functions for creating obstacle models
    BoundingBox computeMeshBoundingBox(const Mesh& mesh);  // Compute minimal bounding box for a mesh
    std::shared_ptr<ObstacleModelInfo> createBoxObstacleModel(float width, float height, float depth, double angle_offset = 0.0, Color color = GRAY);
    std::shared_ptr<ObstacleModelInfo> createCustomObstacleModel(const std::string_view mesh_file, double angle_offset = 0.0, Color color = GRAY, bool use_bbox = true);
};