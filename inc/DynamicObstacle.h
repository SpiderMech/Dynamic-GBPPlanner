#pragma once
#include "Simulator.h"
#include "Graphics.h"
#include <gbp/GBPCore.h> // Included for Eigen and Globals headers
#include <tuple>

/**************************************************************************************/
// Wrapper struct to support Geometry creation and obstalce motion control
/**************************************************************************************/
struct MotionOptions
{
    std::shared_ptr<IGeometry> geom_;
    std::deque<Eigen::Vector4d> waypoints_;
    uint32_t last_spawn_time_ = -1000;
    int spawn_interval_;
    float elevation_;

    // Constructor for cuboid obstacles
    MotionOptions(float width, float height, float depth, float elevation, int spawn_interval, std::deque<Eigen::Vector4d> waypoints, Graphics *graphics_ptr, Color color=GRAY)
        : elevation_(elevation), spawn_interval_(spawn_interval), waypoints_(waypoints)
    {
        geom_ = graphics_ptr->GenCubeGeom(width, height, depth, color);
    }

    // Constructor for imported obstacles
    MotionOptions(std::string_view mesh_file, float elevation, int spawn_interval, std::deque<Eigen::Vector4d> waypoints, Graphics *graphics_ptr, Color color=GRAY)
        : elevation_(elevation), spawn_interval_(spawn_interval), waypoints_(waypoints)
    {
        geom_ = graphics_ptr->GenCustomGeom(mesh_file, color);
    }

    // Constructor for pre-defined meshes
    MotionOptions(Mesh &mesh, float elevation, int spawn_interval, std::deque<Eigen::Vector4d> waypoints, Graphics *graphics_ptr, Color color=GRAY)
        : spawn_interval_(spawn_interval), elevation_(elevation), waypoints_(waypoints)
    {
        geom_ = graphics_ptr->GenPolyGeom(mesh, color);
    }
};

/**************************************************************************************/
// DynamicObstacle class that tracks the current position and handles future state projection
// of moving obstacles.
/**************************************************************************************/
class DynamicObstacle
{
public:
    DynamicObstacle(int oid,
                    std::deque<Eigen::Vector4d> waypoints,
                    std::shared_ptr<IGeometry> geom,
                    float elevation);
    ~DynamicObstacle();

    std::shared_ptr<IGeometry> geom_;           // Pointer to the obstacle's geometry, defined in Graphics.
    int oid_ = 0;                               // ID of the dynamic obstacle.
    Eigen::Vector4d state_;                     // Stores the current most-updated position and velocity (i.e., [x, y, xdot, ydot]) of the dynamic obstacle in world frame.
    std::map<int, Eigen::Vector4d> states_;     // Map of (robot) variable timestep to future obstacle state positions.
    std::vector<int> variable_timesteps_;       // List of (robot) variable timesteps.
    Eigen::Vector3d centre_;                    // The centre of the obstacle in world frame.
    float elevation_ = 0.0;                     // Elevation (Y-axis) of the obstacle, controls how much it floats.
    bool completed_ = true;                     // Whether the obstacle has completed it's path.

    std::deque<Eigen::Vector4d> waypoints_{};   // List of waypoints which determins the obstacle's path.
    float tau_ = 0.5;                           // Seconds to reach ~63% target velocity of the next waypoint. Also used to predict future state.
    float thresh_ = 0.5;                        // Distance threshold for considering waypoint as reached. Also used to predict future state.

    // Update current position
    void updateObstacleState();

    // Compute the current local to world transformation matrix, with additional (optional) offset by delta_t
    std::pair<Eigen::Matrix4f, Eigen::Vector4d> getLocalToWorldTransform(const float delta_t) const;

    // Returns a vector of k nearest neighbours to query_pt as a vector of (point, squared_dist)
    // Computation handled primarily by Geometry.getClosestPoints, this function mainly handles conversion between frames
    std::vector<std::pair<Eigen::Vector3d, double>> getNearestPoints(const Eigen::Vector3d &query_pt, const int k = 5, const float delta_t = 0) const;

    // Compute the state vector of the obstalce delta_t seconds later
    Eigen::Vector4d getStateAfterT(const float delta_t) const;

    // Compute the next state vector based on current waypoint and state
    Eigen::Vector4d getNextState(Eigen::Vector4d state, float delta_t, std::deque<Eigen::Vector4d> &waypoints);

    // Drawing function
    void draw();
};