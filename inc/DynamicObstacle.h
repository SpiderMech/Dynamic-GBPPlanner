#pragma once
#include "Simulator.h"
#include "Graphics.h"
#include <gbp/GBPCore.h> // Included for Eigen and Globals headers
#include <tuple>

// Forward declarations
struct ObstacleModelInfo;

/**************************************************************************************/
// Wrapper struct to support Geometry creation and obstalce motion control
/**************************************************************************************/
struct MotionOptions
{
    std::shared_ptr<ObstacleModelInfo> geom_;  // Using ObstacleModelInfo for model storage
    std::deque<Eigen::VectorXd> waypoints_;
    uint32_t last_spawn_time_ = -1000;
    int spawn_interval_;
    float elevation_;

    // Constructor for cuboid obstacles
    MotionOptions(float width, float height, float depth, float elevation, int spawn_interval, Graphics *graphics_ptr, std::deque<Eigen::VectorXd> waypoints={}, Color color=GRAY, double default_angle_offset=0.0)
        : elevation_(elevation), spawn_interval_(spawn_interval), waypoints_(waypoints)
    {
        geom_ = graphics_ptr->createBoxObstacleModel(width, height, depth, default_angle_offset, color);
    }

    // Constructor for imported obstacles
    MotionOptions(std::string_view mesh_file, float elevation, int spawn_interval, Graphics *graphics_ptr, std::deque<Eigen::VectorXd> waypoints={}, Color color=GRAY, bool use_bbox=true, double default_angle_offset=0.0)
        : elevation_(elevation), spawn_interval_(spawn_interval), waypoints_(waypoints)
    {
        geom_ = graphics_ptr->createCustomObstacleModel(mesh_file, default_angle_offset, color, use_bbox);
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
                    std::deque<Eigen::VectorXd> waypoints,
                    std::shared_ptr<ObstacleModelInfo> geom);
    ~DynamicObstacle();

    std::shared_ptr<ObstacleModelInfo> geom_;   // Pointer to the obstacle's model info with KDTree support
    int oid_ = 0;                               // ID of the dynamic obstacle.
    Eigen::VectorXd state_;                     // Stores the current position, velocity and orientation (i.e., [x, y, xdot, ydot, theta]) of the dynamic obstacle in world frame.
    std::map<int, Eigen::VectorXd> states_;     // Map of (robot) variable timestep to future obstacle state positions with orientation.
    std::vector<int> variable_timesteps_;       // List of (robot) variable timesteps.
    Eigen::Vector3d centre_;                    // The centre of the obstacle in world frame.
    float orientation_ = 0.0;                   // Current orientation of the obstacle in radians
    bool completed_ = false;                     // Whether the obstacle has completed it's path.
    float spawn_time_ = 0.0;                    // Time when the obstacle was spawned (in seconds)

    std::deque<Eigen::VectorXd> waypoints_{};   // List of waypoints which determines the obstacle's path.
    float acc_tau_ = 2.5f;                      // Seconds to accelerate to ~63% target velocity of the next waypoint. Also used to predict future state.
    float dec_tau_ = 12.0f;                     // Seconds to decelerate to ~63% target velocity of the next waypoint. Also used to predict future state.
    float thresh_ = 0.5;                        // Distance threshold for considering waypoint as reached. Also used to predict future state.
    float pause_timer_ = 0.f;                   // A countdown timer for pause motion [seconds]. To suspend obstacle motion, set (xdot, ydot) as (-1000, pause_time [seconds]).

    // Update current position
    void updateObstacleState();

    // Compute the current local to world transformation matrix, with additional (optional) offset by delta_t
    std::pair<Eigen::Matrix3f, Eigen::VectorXd> getLocalToWorldTransform(const float delta_t) const;

    // Returns a vector of k nearest neighbours to query_pt as vector of (point, squared_dist)
    // Computation handled primarily by Geometry.getClosestPoints, this function mainly handles conversion between frames
    std::vector<std::pair<Eigen::Vector2d, double>> getNearestPoints(const Eigen::Vector2d &query_pt, const int k = 5, const float delta_t = 0) const;

    // Compute the state vector of the obstalce delta_t seconds later
    Eigen::VectorXd getStateAfterT(const float delta_t) const;

    // Compute the next state vector based on given waypoints and starting state
    Eigen::VectorXd getNextState(Eigen::VectorXd state, float delta_t, std::deque<Eigen::VectorXd> &waypoints, float& pause_timer);

    // Drawing function
    void draw();

    // Functions for generating waypoints for specific scenarios
    static std::vector<std::deque<Eigen::VectorXd>> GenPedWaypoints(int n);
};