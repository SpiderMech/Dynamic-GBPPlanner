#include <tuple>
#include <DynamicObstacle.h>
#include <nanoflann.h>
#include <Utils.h>
#include <GeometryUtils.h>
#include <Graphics.h> // For ObstacleModelInfo

DynamicObstacle::DynamicObstacle(int oid,
                                 std::deque<Eigen::VectorXd> waypoints,
                                 std::shared_ptr<ObstacleModelInfo> geom,
                                 Color color,
                                 ObstacleType type)
    : oid_(oid), waypoints_(waypoints), geom_(std::move(geom)), color_(color), obstacle_type_(type)
{
    auto start = waypoints_.front();
    // Check if this waypoint has a pause time (5th dimension > 0)
    if (start.size() >= 5 && start(4) > 0.0)
    {
        pause_timer_ += start(4); // Use 5th dimension for pause time
    }

    // Initialize state with position, velocity, and orientation
    state_.resize(5);
    state_ << start(0), start(1), start(2), start(3), 0.0; // Initial orientation is 0

    // Calculate initial orientation from velocity if moving
    Eigen::Vector2d velocity(start(2), start(3));
    if (velocity.norm() > 1e-6)
    {
        orientation_ = wrapAngle(-std::atan2(start(3), start(2))); // Store in Y-down convention
        state_(4) = orientation_;
    }

    states_[0] = state_;
    waypoints_.pop_front();

    auto wp_copy = waypoints_;
    float pt_copy = pause_timer_;
    variable_timesteps_ = getVariableTimesteps(globals.T_HORIZON / globals.T0, globals.LOOKAHEAD_MULTIPLE);
    for (int i = 1; i < variable_timesteps_.size(); ++i)
    {
        int ts = variable_timesteps_[i], prev_ts = variable_timesteps_[i - 1];
        // Note that wp_copy and pt_copy takes initial values from waypoints_ and pause_timer_, but are modified by getNextState
        states_[ts] = getNextState(states_[prev_ts], (ts - prev_ts) * globals.T0, wp_copy, pt_copy);
    }

    // Initialise current covariance and roll forward
    P_curr_.setZero();
    const double s_pos0 = 0.002; // [m]
    const double s_vel0 = 0.005; // [m/s]
    const double sp2 = s_pos0 * s_pos0, sv2 = s_vel0*s_vel0;
    P_curr_(0, 0) = sp2; P_curr_(1, 1) = sp2;
    P_curr_(2, 2) = sv2; P_curr_(3, 3) = sv2;
    
    rollCovariancesFromCurrent();
}

/***************************************************************************************************/
/* Destructor */
/***************************************************************************************************/
DynamicObstacle::~DynamicObstacle()
{
}

/***************************************************************************************************/
// Moves the obstacle forward towards the next waypoint and updates its state
/***************************************************************************************************/
void DynamicObstacle::updateObstacleState()
{
    if (waypoints_.empty())
    {
        completed_ = true;
        return;
    }

    state_ = states_[0] = getNextState(state_, globals.TIMESTEP, waypoints_, pause_timer_);
    orientation_ = state_[4];
    propagateCurrentCovariance(globals.TIMESTEP); // Propage P by one tick

    auto wp_copy = waypoints_;
    auto pt_copy = pause_timer_;
    for (int i = 1; i < variable_timesteps_.size(); ++i)
    {
        int ts = variable_timesteps_[i], prev_ts = variable_timesteps_[i - 1];
        states_[ts] = getNextState(states_[prev_ts], (ts - prev_ts) * globals.T0, wp_copy, pt_copy);
    }
    rollCovariancesFromCurrent();
};

/***************************************************************************************************/
// Updates P_curr_ by dt
/***************************************************************************************************/
void DynamicObstacle::propagateCurrentCovariance(double dt) 
{
    const Eigen::Matrix4d F = Fcv(dt);
    const Eigen::Matrix4d Q = Qcv(dt, sigma_acc_);
    P_curr_ = F * P_curr_ * F.transpose() + Q;
}

/***************************************************************************************************/
// Roll Sigma_k for all lookahead timesteps to match `states_`
/***************************************************************************************************/
void DynamicObstacle::rollCovariancesFromCurrent()
{
    pos_covariances_.clear();

    Eigen::Matrix4d P = P_curr_;
    pos_covariances_[0] = P.topLeftCorner<2,2>();

    for (int i = 1; i < variable_timesteps_.size(); ++i) {
        const int ts = variable_timesteps_[i];
        const int prev_ts = variable_timesteps_[i-1];
        const double dt = (ts - prev_ts) * globals.T0;

        const Eigen::Matrix4d F = Fcv(dt);
        const Eigen::Matrix4d Q = Qcv(dt, sigma_acc_);
        P = F * P * F.transpose() + Q;

        pos_covariances_[ts] = P.topLeftCorner<2,2>();
    }
}

/***************************************************************************************************/
// Sigma_k accessor, returns nullptr if unavailable
/***************************************************************************************************/
const Eigen::Matrix2d* DynamicObstacle::getPosCovPtrAtDt(float delta_t) const
{
    int ts = static_cast<int>(std::lround(delta_t / globals.T0));
    auto it = pos_covariances_.find(ts);
    if (it == pos_covariances_.end()) return nullptr;
    return &it->second;
}
/***************************************************************************************************/
// Compute the current local to world transformation matrix, with additional (optional) offset by delta_t
/***************************************************************************************************/
std::pair<Eigen::Matrix3f, Eigen::VectorXd> DynamicObstacle::getLocalToWorldTransform(const float delta_t) const
{
    int ts = static_cast<int>(std::lround(delta_t / globals.T0));
    Eigen::VectorXd s = states_.at(ts);

    // Create transformation matrix with rotation and translation
    Eigen::Matrix3f tf = Eigen::Matrix3f::Identity();
    float cos_theta = std::cos(s(4) + geom_->orientation_offset);
    float sin_theta = std::sin(s(4) + geom_->orientation_offset);

    // Set rotation part (2D rotation in X-Y plane)
    // For Y-down coordinate system, positive rotation is clockwise
    tf(0, 0) = cos_theta;
    tf(0, 1) = sin_theta;  // Flipped for Y-down
    tf(1, 0) = -sin_theta; // Flipped for Y-down
    tf(1, 1) = cos_theta;

    // Set translation part
    tf(0, 2) = s[0]; // X translation
    tf(1, 2) = s[1]; // Y translation

    return {tf, s};
}

/***************************************************************************************************/
// Returns a vector of k nearest neighbour hits to query_pt as vector of NeighbourHits
/***************************************************************************************************/
std::vector<NeighbourHit> DynamicObstacle::getNearestPointsFromKDTree(const Eigen::Vector2d &query_pt, const int k, const float delta_t) const
{
    auto [tf, s] = getLocalToWorldTransform(delta_t);
    Eigen::Matrix3f inv_tf = tf.inverse();
    Eigen::Vector3f world_pt;
    world_pt << query_pt.x(), query_pt.y(), 1.0f;

    Eigen::Vector3f local_h = inv_tf * world_pt;
    std::vector<std::pair<Eigen::Vector2d, double>> local_nearests = geom_->getNearestPoints(k, Eigen::Vector2d{local_h.x(), local_h.y()});

    std::vector<NeighbourHit> world_results;
    world_results.reserve(local_nearests.size());

    for (const auto &[pt_local, dist_sq_local] : local_nearests)
    {
        // Transform local point back to world coordinates
        Eigen::Vector3f pt_local_h;
        pt_local_h << pt_local.x(), pt_local.y(), 1.0f;
        Eigen::Vector3f pt_world_h = tf * pt_local_h;

        // Planar (x,z) squared distance in world frame
        Eigen::Vector2d pt_world(pt_world_h.x(), pt_world_h.y());
        double dist_sq = (pt_world - query_pt).squaredNorm();

        // Get Sigma_k
        const Eigen::Matrix2d* Sig = getPosCovPtrAtDt(delta_t);
        world_results.emplace_back( NeighbourHit{ pt_world, dist_sq, Sig } );
    }
    return world_results;
}

/***************************************************************************************************/
// Returns a vector of k nearest neighbour hits to query_pt as vector of NeighbourHits
/***************************************************************************************************/
std::vector<NeighbourHit> DynamicObstacle::getNearestPoints2D(const Eigen::Vector2d& query_pt, const float delta_t) const 
{
    auto [tf, s] = getLocalToWorldTransform(delta_t);
    const Eigen::Vector2d c(s[0], s[1]);
    const float theta = s[4] + geom_->orientation_offset;
    const Eigen::Vector2d obs_half_ext(0.5 * geom_->dimensions.x, 0.5 * geom_->dimensions.z);
    const OBB2D obb(c, obs_half_ext, theta);

    auto [q, d2] = GeometryUtils::closestPointOnOBB(query_pt, obb);
    const Eigen::Matrix2d* Sig = getPosCovPtrAtDt(delta_t);

    std::vector<NeighbourHit> hits;
    hits.push_back({q, d2, Sig});

    // (Optional) add 2-4 samples along the closest edge for smoother gradients:
    // Determine which axis clamped (|lx|>ax or |ly|>ay) to pick edge direction,
    // then push_back edge-midpoint(s) within ~safety distance.

    return hits;
}

/***************************************************************************************************/
// Compute the next state vector based on given waypoints and starting state
/***************************************************************************************************/
Eigen::VectorXd DynamicObstacle::getNextState(Eigen::VectorXd state, float delta_t, std::deque<Eigen::VectorXd> &waypoints, float &pause_timer)
{
    Eigen::VectorXd s = state;
    float time_remaining = delta_t;
    const float dt = globals.TIMESTEP;
    while (time_remaining > 0.f && !waypoints.empty())
    {
        float step = time_remaining < dt ? time_remaining : dt;

        if (pause_timer > 0.f)
        {
            pause_timer = std::max(0.f, pause_timer - step);
            time_remaining -= step;
            continue;
        }

        // Check if this waypoint has a pause time (5th dimension > 0)
        bool next_wp_is_pause = waypoints.front().size() >= 5 && waypoints.front()(4) > 0.0;

        Eigen::Vector2d current_vel = s.segment<2>(2);
        // Eigen::Vector2d target_vel = next_wp_is_pause ? current_vel :
        //                                                 waypoints.front().segment<2>(2);
        Eigen::Vector2d target_vel = waypoints.front().segment<2>(2);
        float alpha = acc_tau_;
        if (target_vel.norm() <= current_vel.norm())
        {
            // use dec_tau if decelerating
            alpha = dt / dec_tau_;
        }
        else
        {
            alpha = dt / acc_tau_;
        }
        Eigen::Vector2d new_vel = current_vel + alpha * (target_vel - current_vel);
        Eigen::Vector2d dir = waypoints.front()({0, 1}) - s({0, 1});
        float dist_to_wp = dir.norm();
        dir.normalize();
        s.segment<2>(0) += new_vel.norm() * dir * step;
        s.segment<2>(2) = new_vel;

        // Update orientation based on velocity (only if moving)
        if (new_vel.norm() > 1e-6)
        {
            s(4) = wrapAngle(-std::atan2(new_vel(1), new_vel(0)));
        }
        // If velocity is zero, keep current orientation (s(4) unchanged)

        if (dist_to_wp < thresh_)
        {
            if (next_wp_is_pause)
                pause_timer += waypoints.front()(4); // Use 5th dimension for pause time
            waypoints.pop_front();
        }
        time_remaining -= step;
    }

    return s;
}

/***************************************************************************************************/
// Drawing function
/***************************************************************************************************/
void DynamicObstacle::draw()
{
    if (globals.DRAW_OBSTACLES)
    {
        // Convert orientation from radians to degrees for Raylib
        float rotation = wrapAngle(orientation_ + geom_->orientation_offset);
        // Add the model's default offset (similar to robots)
        float rotation_degrees = rotation * (180.0f / M_PI);

        // Use DrawModelEx to include rotation with instance-specific color
        DrawModelEx(geom_->model,
                    Vector3{(float)state_[0], -geom_->boundingBox.min.y, (float)state_[1]},
                    Vector3{0.0f, 1.0f, 0.0f}, // Rotate around Y axis
                    rotation_degrees,          // Rotation angle in degrees
                    Vector3{1.0f, 1.0f, 1.0f}, // Scale
                    color_);                   // Use instance-specific color tint
    }
}

/***************************************************************************************************/
// Generate waypoints for bus obstacles in junction_twoway formation
/***************************************************************************************************/
std::deque<Eigen::VectorXd> DynamicObstacle::generateBusWaypoints(int road, int turn, int lane,
                                                                  double world_sz, double max_speed,
                                                                  double lane_width)
{
    // Constants
    const int n_lanes = 2;

    // Create rotation matrix for the road
    Eigen::Matrix<double, 5, 5> rot = Eigen::Matrix<double, 5, 5>::Identity();
    double angle = PI / 2.0 * road;
    double c = std::cos(angle);
    double s = std::sin(angle);
    rot.block<2, 2>(0, 0) << c, -s, s, c;
    rot.block<2, 2>(2, 2) << c, -s, s, c;

    // Calculate lane offsets
    double lane_v_offset = (0.5 * (1 - 2.0 * n_lanes) + lane) * lane_width;
    double lane_h_offset = (1 - turn) * (0.5 + lane - n_lanes) * lane_width;

    // Define key waypoints
    Eigen::VectorXd starting(5), turning(5), ending(5);
    starting = rot * Eigen::VectorXd{{-world_sz / 2.0 - 10.0, lane_v_offset, 1.0 * max_speed, 0.0, 0.0}};
    turning = rot * Eigen::VectorXd{{lane_h_offset, lane_v_offset, 1.0 * max_speed, 0.0, 0.0}};
    ending = rot * Eigen::VectorXd{{lane_h_offset + (turn % 2) * world_sz / 2.0,
                                    lane_v_offset + (turn - 1) * world_sz / 2.0,
                                    (turn % 2) * max_speed * 1.0,
                                    (turn - 1) * max_speed * 1.0,
                                    0.0}};

    // Build waypoint list
    std::deque<Eigen::VectorXd> waypoints = {starting};

    // Calculate segment boundaries
    double junction_edge = 2.0 * lane_width;                 // Distance from center to edge of junction box
    double segment1_length = world_sz / 2.0 - junction_edge; // Length before junction
    double segment2_length = world_sz / 2.0 - junction_edge; // Length after junction

    // Add stops in segment 1 (before junction)
    int seg1_stops = random_int(1, 2);
    for (int i = 0; i < seg1_stops; ++i)
    {
        double stop_x = -world_sz / 2.0 + (i + 1) * segment1_length / (seg1_stops + 1);
        double pause_time = random_float(2.0, 4.0);
        waypoints.push_back(rot * Eigen::VectorXd{{stop_x, lane_v_offset, 1.0 * max_speed, 0.0, pause_time}});
    }

    // Add turning waypoint at junction
    waypoints.push_back(turning);

    // Add stops in segment 2 (after junction)
    int seg2_stops = random_int(0, 2);
    for (int i = 0; i < seg2_stops; ++i)
    {
        double pause_time = random_float(2.0, 4.0);
        // Calculate stop position based on turn direction
        // turn=0 (left): move in -Y direction, turn=1 (straight): move in +X, turn=2 (right): move in +Y
        double stop_pos = junction_edge + (i + 1) * segment2_length / (seg2_stops + 1);
        Eigen::VectorXd stop_wp{{lane_h_offset + (turn % 2) * stop_pos,
                                 lane_v_offset + (turn - 1) * stop_pos,
                                 (turn % 2) * globals.MAX_SPEED * 1.,
                                 (turn - 1) * globals.MAX_SPEED * 1.,
                                 (double)pause_time}};
        waypoints.push_back(rot * stop_wp);
    }

    // Add ending waypoint
    waypoints.push_back(ending);

    return waypoints;
}

/***************************************************************************************************/
// Generate waypoints for van obstacles (delivery vehicles) in junction_twoway formation
/***************************************************************************************************/
std::deque<Eigen::VectorXd> DynamicObstacle::generateVanWaypoints(int lane,
                                                                  double world_sz, double max_speed,
                                                                  double lane_width)
{
    double junction_edge = 2.0 * lane_width;
    double segment_length = world_sz / 2.0 - junction_edge;
    const int n_roads = 4, n_lanes = 2;
    double lane_v_offset = (0.5 * (1 - 2.0 * n_lanes) + lane) * lane_width;
    double lane_h_offset = (0.5 - n_lanes) * lane_width;
    
    std::deque<Eigen::VectorXd> delivery_points;
    
    std::deque<Eigen::VectorXd> temp_entries;
    std::deque<Eigen::VectorXd> temp_exits;
    std::deque<Eigen::VectorXd> left_turns;
    std::deque<Eigen::VectorXd> gaps;

    for (int road = 0; road < n_roads; ++road) {
        // Define rotation matrix
        Eigen::Matrix<double, 5, 5> rot = Eigen::Matrix<double, 5, 5>::Identity();
        double angle = PI / 2.0 * road;
        double c = std::cos(angle);
        double s = std::sin(angle);
        rot.block<2, 2>(0, 0) << c, -s, s, c;
        rot.block<2, 2>(2, 2) << c, -s, s, c;
        
        int seg_stops = 2;
        for (int i = 0; i < seg_stops; ++i) {
            double pause_time = random_float(2.f, 4.f);
            double stop_x = -world_sz / 2.0 + (i + 1) * segment_length / (seg_stops + 1);
            temp_entries.push_back(rot * Eigen::VectorXd{{stop_x, lane_v_offset, 1.0 * max_speed, 0.0, pause_time}});
            temp_exits.push_back(rot * Eigen::VectorXd{{stop_x, -lane_v_offset, -1.0 * max_speed, 0.0, pause_time}});
            left_turns.push_back(rot * Eigen::VectorXd{{lane_h_offset, lane_v_offset, (i==0) * 1.0 * max_speed, (i==1) * -1.0 * max_speed, 0.0}});
            gaps.push_back(rot * Eigen::VectorXd{{stop_x, 0.0, 0.0, -1.0 * max_speed, 0.0}});
        }
    }
    // Shift first two exits to the back
    for (int i = 0; i < 2; ++i) {
        auto shift = temp_exits.front();
        temp_exits.pop_front();
        temp_exits.push_back(shift);
    }
    int skip_first = false, skip_second = false;
    for (int i = 0; i < temp_entries.size(); i+=2) {
        if (i > 0 && !(skip_first && skip_second)) delivery_points.push_back(gaps[skip_second ? i+1 : i]);
        if (!skip_second) delivery_points.push_back(temp_entries[i]);
        if (!skip_first) delivery_points.push_back(temp_entries[i+1]);
        delivery_points.push_back(left_turns[(skip_first && skip_second && i > 0) ? i+1 : i]);
        skip_first = static_cast<bool>(random_int(0, 1));
        skip_second = static_cast<bool>(random_int(0, 1));
        if (!skip_first) delivery_points.push_back(temp_exits[i+1]);
        if (!skip_second) delivery_points.push_back(temp_exits[i]);
    }

    Eigen::VectorXd starting = Eigen::VectorXd{{-world_sz / 2.0 - 10.0, lane_v_offset, 1.0 * max_speed, 0.0, 0.0}};
    delivery_points.push_front(starting);
    return delivery_points;
}

/***************************************************************************************************/
// Generate waypoints for a single pedestrian crossing in junction_twoway formation
/***************************************************************************************************/
std::deque<Eigen::VectorXd> DynamicObstacle::generatePedestrianWaypoints(
    double world_sz, double speed, double lane_width)
{
    // Simplified single pedestrian generation (not a horde)
    const int n_lanes = 2;
    const double road_half_width = n_lanes * lane_width;
    const double spawn_offset = globals.ROBOT_RADIUS; // offset just outside the road edge
    const double border = road_half_width + spawn_offset;
    const double world_half = world_sz / 2.0;
    
    // Randomly choose crossing direction (horizontal vs vertical)
    bool horizontal = random_int(0, 1) == 1;
    // Randomly choose crossing side (1.0 or -1.0)
    double side = (random_int(0, 1) == 1) ? 1.0 : -1.0;
    // Random spawn position along the edge
    double pos = (random_int(0, 1) == 0) ? random_float(-world_half, -road_half_width)
                                         : random_float(road_half_width, world_half);
    
    std::deque<Eigen::VectorXd> waypoints;
    Eigen::VectorXd start(5), end(5);
    
    if (horizontal) {
        // Crossing a horizontal road: vary x position, spawn above or below
        double z0 = side * border;
        start << pos, z0, 0.0, -side * speed, 0.0; // 5th dimension (pause time) = 0
        end << pos, -z0, 0.0, -side * speed, 0.0;  // 5th dimension (pause time) = 0
    } else {
        // Crossing a vertical road: vary z position, spawn left or right
        double x0 = side * border;
        start << x0, pos, -side * speed, 0.0, 0.0; // 5th dimension (pause time) = 0
        end << -x0, pos, -side * speed, 0.0, 0.0;  // 5th dimension (pause time) = 0
    }
    
    waypoints.push_back(start);
    waypoints.push_back(end);
    
    return waypoints;
}
