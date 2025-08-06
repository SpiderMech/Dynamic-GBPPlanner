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

/**************************************************************************************/
// Geometry interface which stores model for drawing and KDTree of obstacle point clouds
// to support closest-points queries.
/**************************************************************************************/
struct IGeometry
{
    std::shared_ptr<Model> model_;
    Color color_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> mat_;
    // Define KDTree type with row_major = false (so that each column vec is a point rather than each row)
    using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, 2, Eigen::Dynamic>, 2, nanoflann::metric_L2, false>;
    std::unique_ptr<KDTree> kdtree_;

    IGeometry(std::shared_ptr<Model> model, Color color)
        : model_(std::move(model)), color_(color) {}
    virtual ~IGeometry() = default;

    // Returns the k nearest neighbours to query_pt as vector of (point, squared_dist)
    std::vector<std::pair<Eigen::Vector2d, double>> getNearestPoints(int k, const Eigen::Vector2d &query_pt) const
    {
        if (!kdtree_) {
            throw std::runtime_error("Geometry::getNearestPoints called before KDTree was built.");
        }

        if (mat_.cols() == 0) {
            throw std::runtime_error("Geometry::getNearestPoints called with empty point cloud.");
        }

        if (k > mat_.cols()) {
            k = mat_.cols();  // clamp to max available points
        }

        std::vector<size_t> ret_indexes(k);
        std::vector<double> out_dists_sqr(k);
        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(ret_indexes.data(), out_dists_sqr.data());
        nanoflann::SearchParameters params;
        kdtree_->index_->findNeighbors(resultSet, query_pt.data(), params);

        std::vector<std::pair<Eigen::Vector2d, double>> results;
        for (int i = 0; i < k; ++i)
        {
            results.emplace_back(mat_.col(ret_indexes[i]), out_dists_sqr[i]);
        }
        return results;
    }
};

// Add Geometry types here
struct BoxGeometry : IGeometry
{
    Eigen::Vector3d min_pt_;
    Eigen::Vector3d max_pt_;
    int grid_size_ = 8;
    
    BoxGeometry(std::shared_ptr<Model> model_ptr, Eigen::Vector3d min_pt, Eigen::Vector3d max_pt, Color c)
        : IGeometry(model_ptr, c), min_pt_(min_pt), max_pt_(max_pt) 
    {
        std::vector<Eigen::Vector2d> points;
        auto lerp = [](double a, double b, double t){ return a + (b-a)*t; };
        // Generate grid outlines only on the X-Z plane
        for (int i = 0; i <= grid_size_; ++i) {
            for (int j = 0; j <= grid_size_; ++j) {
                double u = double(i)/grid_size_, v = double(j)/grid_size_;
                Eigen::Vector2d pt;
                pt.x() = lerp(min_pt.x(), max_pt.x(), u);
                pt.y() = lerp(min_pt.z(), max_pt.z(), v);
                points.push_back(pt);
            }
        }
        // Build KDTree from obstacle point cloud
        mat_.resize(2, points.size());
        for (size_t i = 0; i < points.size(); ++i)
            mat_.col(i) = points[i];
        kdtree_ = std::make_unique<KDTree>(2, std::cref(mat_));
        kdtree_->index_->buildIndex();
    }
};

struct MeshGeometry : IGeometry
{
    MeshGeometry(std::shared_ptr<Model> model_ptr, const std::vector<Eigen::Vector3d> &points, Color c)
        : IGeometry(model_ptr, c)
    {
        std::vector<Eigen::Vector2d> points2d;
        // Extract 2D boundary points
        for (const auto &pt : points) {
            points2d.emplace_back(pt.x(), pt.z());
        }
        // Build KDTree from obstacle point cloud
        mat_.resize(2, points2d.size());
        for (size_t i = 0; i < points2d.size(); ++i)
            mat_.col(i) = points2d[i];
        kdtree_ = std::make_unique<KDTree>(2, std::cref(mat_));
        kdtree_->index_->buildIndex();
    }
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
    Model robotModel_;       // Raylib Model representing a robot. This can be changed.
    Model groundModel_;      // Model representing the ground plane
    Vector3 groundModelpos_; // Ground plane position
    Shader lightShader_;     // Light shader

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
    // Function to create a cubic Geometry pointer based on specified dimensions
    std::shared_ptr<IGeometry> GenCubeGeom(float width, float height, float depth, Color color=GRAY);
    // Function to create a Geometry using imported mesh file
    std::shared_ptr<IGeometry> GenCustomGeom(const std::string_view mesh_file, Color color=GRAY);
    // Function to Create obstacle model directly from self-defined mesh object
    std::shared_ptr<IGeometry> GenPolyGeom(Mesh &mesh, Color color=GRAY);

    // Functions for generating specific meshes
    Mesh genMeshPyramid(float base, float height);
};