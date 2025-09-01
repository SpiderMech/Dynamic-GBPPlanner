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
    Color color_ = GRAY;  // Color for this specific obstacle instance

    // Constructor for cuboid obstacles
    MotionOptions(float width, float height, float depth, float elevation, int spawn_interval, Graphics *graphics_ptr, std::deque<Eigen::VectorXd> waypoints={}, Color color=GRAY, double default_angle_offset=0.0)
        : elevation_(elevation), spawn_interval_(spawn_interval), waypoints_(waypoints), color_(color)
    {
        geom_ = graphics_ptr->createBoxObstacleModel(width, height, depth, default_angle_offset, color);
    }

    // Constructor for imported obstacles
    MotionOptions(std::string_view mesh_file, float elevation, int spawn_interval, Graphics *graphics_ptr, std::deque<Eigen::VectorXd> waypoints={}, Color color=GRAY, bool use_bbox=true, double default_angle_offset=0.0)
        : elevation_(elevation), spawn_interval_(spawn_interval), waypoints_(waypoints), color_(color)
    {
        geom_ = graphics_ptr->createCustomObstacleModel(mesh_file, default_angle_offset, color, use_bbox);
    }
};
/**************************************************************************************/
// Result structure for closest point queries
/**************************************************************************************/
struct NeighbourHit {
    Eigen::Vector2d pt_world;
    double dist_squared;
    const Eigen::Matrix2d* Sigma_pos;
};

/**************************************************************************************/
// DynamicObstacle class that tracks the current position and handles future state projection
// of moving obstacles.
/**************************************************************************************/
class DynamicObstacle
{
public:
    ObstacleType obstacle_type_ = ObstacleType::CUBE; // Type of this obstacle
    
private:
    Eigen::Matrix4d P_curr_ = Eigen::Matrix4d::Zero(); // State covariance at current tick (x, y, vx, vy)
    std::map<int, Eigen::Matrix2d> pos_covariances_;   // Lookahead timestep position covariance (same index as states_)
    double sigma_acc_ = 0.001;                           // White-noise acceleration [m/s^2]

    // Discrete constant-velocity model matrices
    static Eigen::Matrix4d Fcv(double dt) {             
        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F(0, 2) = dt; F(1, 3) = dt; return F;
    }
    static Eigen::Matrix4d Qcv(double dt, double sigma_a) {
        const double dt2 = dt*dt, dt3 = dt2 * dt, q = sigma_a*sigma_a;
        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        // x block
        Q(0,0) = dt3/3.0*q; Q(0,2) = dt2/2.0*q; Q(2,0) = dt2/2.0*q; Q(2,2) = dt*q;
        // y block
        Q(1,1) = dt3/3.0*q; Q(1,3) = dt2/2.0*q; Q(3,1) = dt2/2.0*q; Q(3,3) = dt*q;
        return Q;
    }

    // Updates P_curr_ by dt
    void propagateCurrentCovariance(double dt);
    // Roll Sigma_k for all lookahead timesteps to match `states_`
    void rollCovariancesFromCurrent();

public:
    DynamicObstacle(int oid,
                    std::deque<Eigen::VectorXd> waypoints,
                    std::shared_ptr<ObstacleModelInfo> geom,
                    Color color = GRAY,
                    ObstacleType type = ObstacleType::CUBE);
    ~DynamicObstacle();

    std::shared_ptr<ObstacleModelInfo> geom_;   // Pointer to the obstacle's model info with KDTree support
    int oid_ = 0;                               // ID of the dynamic obstacle.
    Eigen::VectorXd state_;                     // Stores the current position, velocity and orientation (i.e., [x, y, xdot, ydot, theta]) of the dynamic obstacle in world frame.
    std::map<int, Eigen::VectorXd> states_;     // Map of (robot) variable timestep to future obstacle state positions with orientation.
    std::vector<int> variable_timesteps_;       // List of (robot) variable timesteps.
    float orientation_ = 0.0;                   // Current orientation of the obstacle in radians
    bool completed_ = false;                     // Whether the obstacle has completed it's path.
    float spawn_time_ = 0.0;                    // Time when the obstacle was spawned (in seconds)

    std::deque<Eigen::VectorXd> waypoints_{};   // List of waypoints which determines the obstacle's path.
    float acc_tau_ = 2.5f;                      // Seconds to accelerate to ~63% target velocity of the next waypoint. Also used to predict future state.
    float dec_tau_ = 12.0f;                     // Seconds to decelerate to ~63% target velocity of the next waypoint. Also used to predict future state.
    float thresh_ = 0.5;                        // Distance threshold for considering waypoint as reached. Also used to predict future state.
    float pause_timer_ = 0.f;                   // A countdown timer for pause motion [seconds]. To suspend obstacle motion, set (xdot, ydot) as (-1000, pause_time [seconds]).
    Color color_ = GRAY;                        // Color for this specific obstacle instance

    // Update current position
    void updateObstacleState();

    // Compute the current local to world transformation matrix, with additional (optional) offset by delta_t
    std::pair<Eigen::Matrix3f, Eigen::VectorXd> getLocalToWorldTransform(const float delta_t) const;

    // Returns a vector of k nearest neighbour hits to query_pt as vector of NeighbourHits
    std::vector<NeighbourHit> getNearestPointsFromKDTree(const Eigen::Vector2d &query_pt, const int k = 5, const float delta_t = 0) const;
    
    // A simplified version of getNearestPointsFromKDTree, uses OBB directly
    std::vector<NeighbourHit> getNearestPoints2D(const Eigen::Vector2d& query_pt, const float delta_t = 0) const;
    
    // Compute the next state vector based on given waypoints and starting state
    Eigen::VectorXd getNextState(Eigen::VectorXd state, float delta_t, std::deque<Eigen::VectorXd> &waypoints, float& pause_timer);
    
    // Sigma_k accessor, returns nullptr if unavailable
    const Eigen::Matrix2d* getPosCovPtrAtDt(float delta_t) const;

    // Drawing function
    void draw();

    // Junction waypoint generation functions
    static std::deque<Eigen::VectorXd> generateBusWaypoints(int road, int turn, int lane, 
                                                             double world_sz, double max_speed,
                                                             double lane_width);
    static std::deque<Eigen::VectorXd> generateVanWaypoints(int lane,
                                                             double world_sz, double max_speed, 
                                                             double lane_width);
    static std::deque<Eigen::VectorXd> generatePedestrianWaypoints(double world_sz, double speed,
                                                                   double lane_width);
};
