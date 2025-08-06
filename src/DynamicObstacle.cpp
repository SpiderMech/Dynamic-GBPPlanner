#include <tuple>
#include <DynamicObstacle.h>
#include <nanoflann.h>
#include <Utils.h>

DynamicObstacle::DynamicObstacle(int oid,
                                 std::deque<Eigen::VectorXd> waypoints,
                                 std::shared_ptr<IGeometry> geom,
                                 float elevation)
    : oid_(oid), waypoints_(waypoints),
      geom_(std::move(geom)), elevation_(elevation)
{
    auto start = waypoints_.front();
    // Check if this waypoint has a pause time (5th dimension > 0)
    if (start.size() >= 5 && start(4) > 0.0) {
        pause_timer_ += start(4);  // Use 5th dimension for pause time
    }
    // If starting waypoint a pause timer, its velocity is still used as the initial velocity
    state_ = states_[0] = Eigen::Vector4d{start(0), start(1), start(2), start(3)};
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
    auto wp_copy = waypoints_;
    auto pt_copy = pause_timer_;
    for (int i = 1; i < variable_timesteps_.size(); ++i)
    {
        int ts = variable_timesteps_[i], prev_ts = variable_timesteps_[i - 1];
        states_[ts] = getNextState(states_[prev_ts], (ts - prev_ts) * globals.T0, wp_copy, pt_copy);
    }
};

/***************************************************************************************************/
// Compute the current local to world transformation matrix, with additional (optional) offset by delta_t
/***************************************************************************************************/
std::pair<Eigen::Matrix3f, Eigen::Vector4d> DynamicObstacle::getLocalToWorldTransform(const float delta_t) const
{
    // Eigen::Vector4d s = (delta_t == 0) ? state_ : getStateAfterT(delta_t);
    int ts = static_cast<int>(delta_t / globals.T0);
    Eigen::Vector4d s = states_.at(ts);
    Eigen::Matrix3f tf = Eigen::Matrix3f::Identity();
    tf(0, 2) = s[0];  // X translation
    tf(1, 2) = s[1];  // Z translation (what was previously Z in world coordinates)
    return {tf, s};
}

/***************************************************************************************************/
// Returns a vector of k nearest neighbours to query_pt as a vector of (point, squared_dist)
// Computation handled primarily by IGeometry.kNearesNeighbours in Graphics.h, this function
// handles conversion between frames.
/***************************************************************************************************/
std::vector<std::pair<Eigen::Vector2d, double>> DynamicObstacle::getNearestPoints(const Eigen::Vector2d &query_pt, const int k, const float delta_t) const
{
    auto [tf, s] = getLocalToWorldTransform(delta_t);
    Eigen::Matrix3f inv_tf = tf.inverse();
    Eigen::Vector3f world_pt;
    world_pt << query_pt.x(), query_pt.y(), 1.0f;

    Eigen::Vector3f local_h = inv_tf * world_pt;
    std::vector<std::pair<Eigen::Vector2d, double>> local_nearests = geom_->getNearestPoints(k, Eigen::Vector2d{local_h.x(), local_h.y()});

    std::vector<std::pair<Eigen::Vector2d, double>> world_results;
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

        world_results.emplace_back(pt_world, dist_sq);
    }
    return world_results;
}

/***************************************************************************************************/
// Compute the state vector of the obstalce delta_t seconds later
/***************************************************************************************************/
Eigen::Vector4d DynamicObstacle::getStateAfterT(const float delta_t) const
{
    Eigen::Vector4d s = state_;
    auto wpts = waypoints_;
    float time_remaining = delta_t;
    const float dt = globals.TIMESTEP;
    // Simulate obstacle path for delta_t s
    while (time_remaining > 0.f && !wpts.empty())
    {
        float step = time_remaining < dt ? time_remaining : dt;
        Eigen::Vector2d current_vel = s.segment<2>(2);
        Eigen::Vector2d target_vel = wpts.front().segment<2>(2);
        float alpha = dt / tau_;
        Eigen::Vector2d new_vel = current_vel + alpha * (target_vel - current_vel);
        Eigen::Vector2d dir = wpts.front()({0, 1}) - s({0, 1});
        float dist_to_wp = dir.norm();
        dir.normalize();
        s.segment<2>(0) += new_vel.norm() * dir * step;
        s.segment<2>(2) = new_vel;
        if (dist_to_wp < thresh_)
            wpts.pop_front();
        time_remaining -= step;
    }
    return s;
}

/***************************************************************************************************/
// Compute the next state vector based on given waypoints and starting state
/***************************************************************************************************/
Eigen::Vector4d DynamicObstacle::getNextState(Eigen::Vector4d state, float delta_t, std::deque<Eigen::VectorXd> &waypoints, float& pause_timer)
{
    Eigen::Vector4d s = state;
    float time_remaining = delta_t;
    const float dt = globals.TIMESTEP;
    while (time_remaining > 0.f && !waypoints.empty())
    {
        float step = time_remaining < dt ? time_remaining : dt;
        
        if (pause_timer > 0.f) {
            pause_timer = std::max(0.f, pause_timer - step);
            time_remaining -= step;
            continue;
        }

        // Check if this waypoint has a pause time (5th dimension > 0)
        bool next_wp_is_pause = waypoints.front().size() >= 5 && waypoints.front()(4) > 0.0;
        
        Eigen::Vector2d current_vel = s.segment<2>(2);
        Eigen::Vector2d target_vel = next_wp_is_pause ? current_vel : 
                                                        waypoints.front().segment<2>(2);
        float alpha = dt / tau_;
        Eigen::Vector2d new_vel = current_vel + alpha * (target_vel - current_vel);
        Eigen::Vector2d dir = waypoints.front()({0, 1}) - s({0, 1});
        float dist_to_wp = dir.norm();
        dir.normalize();
        s.segment<2>(0) += new_vel.norm() * dir * step;
        s.segment<2>(2) = new_vel;
        if (dist_to_wp < thresh_) {
            if (next_wp_is_pause) pause_timer += waypoints.front()(4);  // Use 5th dimension for pause time
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
        DrawModel(*geom_->model_, Vector3{(float)state_[0], elevation_, (float)state_[1]}, 1.0, geom_->color_);
}

/***************************************************************************************************/
// Functions for generating waypoints for specific scenarios
/***************************************************************************************************/
std::vector<std::deque<Eigen::VectorXd>> DynamicObstacle::GenPedWaypoints(int n) {
    std::vector<std::deque<Eigen::VectorXd>> waypoints_vec;
    if (globals.FORMATION == "junction_twoway_dynamic") {
        int n_lanes = 2;
        double lane_width = 4. * globals.ROBOT_RADIUS;
        double road_half_width = n_lanes * lane_width;
        double spawn_offset = globals.ROBOT_RADIUS;  // offset just outside the road edge
        double border = road_half_width + spawn_offset;
        double world_half = globals.WORLD_SZ / 2.;

        for (int i = 0; i < n; ++i) {
            bool horizontal = random_int(0, 1) == 1;
            double speed = double(random_float(float(globals.DEFAULT_OBS_SPEED), 2.f));
            double side = (random_int(0, 1) == 1) ? 1.0 : -1.0;
            double pos  = (random_int(0, 1) == 0) ? random_float(-world_half, -road_half_width)
                                                  : random_float(road_half_width, world_half);
            Eigen::VectorXd start(5), end(5);
            if (horizontal) {
                // Crossing a horizontal road: vary x, spawn above or below
                double z0 = side * border;
                start << pos, z0, 0.0, -side * speed, 0.0;  // 5th dimension (pause time) = 0
                end   << pos, -z0, 0.0, -side * speed, 0.0; // 5th dimension (pause time) = 0
            } else {
                // Crossing a vertical road: vary z, spawn left or right
                double x0 = side * border;
                start <<  x0, pos, -side * speed, 0.0, 0.0; // 5th dimension (pause time) = 0
                end   << -x0, pos, -side * speed, 0.0, 0.0; // 5th dimension (pause time) = 0
            }
            waypoints_vec.emplace_back();
            waypoints_vec.back().push_back(start);
            waypoints_vec.back().push_back(end);
        }
    }
    return waypoints_vec;
}
