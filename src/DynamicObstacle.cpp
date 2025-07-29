#include <tuple>
#include <DynamicObstacle.h>
#include <nanoflann.h>
#include <Utils.h>

DynamicObstacle::DynamicObstacle(int oid,
                                 std::deque<Eigen::Vector4d> waypoints,
                                 std::shared_ptr<IGeometry> geom,
                                 float elevation)
    : oid_(oid), waypoints_(waypoints),
      geom_(std::move(geom)), elevation_(elevation)
{
    state_ = states_[0] = waypoints_[0];
    waypoints_.pop_front();
    
    auto wp_copy = waypoints_;
    variable_timesteps_ = getVariableTimesteps(globals.T_HORIZON / globals.T0, globals.LOOKAHEAD_MULTIPLE);
    for (int i = 1; i < variable_timesteps_.size(); ++i) {
        int ts = variable_timesteps_[i], prev_ts = variable_timesteps_[i-1];
        states_[ts] = getNextState(states_[prev_ts], (ts - prev_ts)*globals.T0, wp_copy);
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
    // auto horizon = (waypoints_.size() > 0) ? waypoints_[0] : state_;
    // Stop updates if last horizon state is reached
    if (waypoints_.empty()) { completed_ = true; return; }

    state_ = states_[0] = getNextState(state_, globals.TIMESTEP, waypoints_);
    auto wp_copy = waypoints_;
    for (int i = 1; i < variable_timesteps_.size(); ++i) {
        int ts = variable_timesteps_[i], prev_ts = variable_timesteps_[i-1];
        states_[ts] = getNextState(states_[prev_ts], (ts - prev_ts)*globals.T0, wp_copy);
    }
    // Eigen::Vector2d current_vel = state_.segment<2>(2);
    // Eigen::Vector2d target_vel = horizon.segment<2>(2);

    // // Smooth velocity update (exponential smoothing), as we don't know t at horizon
    // static float alpha = globals.TIMESTEP / tau_;
    // Eigen::Vector2d new_vel = current_vel + alpha * (target_vel - current_vel);

    // Eigen::Vector2d direction = horizon({0, 1}) - state_({0, 1});
    // direction.normalize();

    // state_.segment<2>(0) += new_vel.norm() * direction * globals.TIMESTEP;
    // state_.segment<2>(2) = new_vel;

    // float dist_to_horizon = (horizon({0, 1})-state_({0, 1})).norm();
    // if (dist_to_horizon < thresh_)
    // {
    //     if (!waypoints_.empty())
    //         waypoints_.pop_front();
    // }
};

/***************************************************************************************************/
// Compute the current local to world transformation matrix, with additional (optional) offset by delta_t
/***************************************************************************************************/
std::pair<Eigen::Matrix4f, Eigen::Vector4d> DynamicObstacle::getLocalToWorldTransform(const float delta_t) const
{
    // Eigen::Vector4d s = (delta_t == 0) ? state_ : getStateAfterT(delta_t);
    int ts = static_cast<int>(delta_t / globals.T0);
    Eigen::Vector4d s = states_.at(ts);
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf(0, 3) = s[0];
    tf(1, 3) = elevation_;
    tf(2, 3) = s[1];
    return {tf, s};
}


/***************************************************************************************************/
// Returns a vector of k nearest neighbours to query_pt as a vector of (point, squared_dist)
// Computation handled primarily by IGeometry.kNearesNeighbours in Graphics.h, this function
// handles conversion between frames.
/***************************************************************************************************/
std::vector<std::pair<Eigen::Vector3d, double>> DynamicObstacle::getNearestPoints(const Eigen::Vector3d& query_pt, const int k, const float delta_t) const
{
    auto [tf, s] = getLocalToWorldTransform(delta_t);
    Eigen::Matrix4f inv_tf = tf.inverse();
    Eigen::Vector4f world_pt;
    world_pt << query_pt.x(), query_pt.y(), query_pt.z(), 1.0f;
    
    Eigen::Vector4f local_h = inv_tf * world_pt;
    std::vector<std::pair<Eigen::Vector3d, double>> local_nearests = geom_->getNearestPoints(k, Eigen::Vector3d{local_h.x(), local_h.y(), local_h.z()});

    std::vector<std::pair<Eigen::Vector3d, double>> world_results;
    world_results.reserve(local_nearests.size());

    for (const auto& [pt_local, dist_sq_local] : local_nearests)
    {
        // Transform local point back to world coordinates    
        Eigen::Vector4f pt_local_h;
        pt_local_h << pt_local.x(), pt_local.y(), pt_local.z(), 1.0f;
        Eigen::Vector4f pt_world_h = tf * pt_local_h;

        // Planar (x,z) squared distance in world frame
        Eigen::Vector3d pt_world(pt_world_h.x(), pt_world_h.y(), pt_world_h.z());
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
    while (time_remaining > 0.f && !wpts.empty()) {
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
        if (dist_to_wp < thresh_) wpts.pop_front();
        time_remaining -= step;
    }
    return s;
}

/***************************************************************************************************/
// Compute the next state vector based on current waypoint and state
/***************************************************************************************************/
Eigen::Vector4d DynamicObstacle::getNextState(Eigen::Vector4d state, float delta_t, std::deque<Eigen::Vector4d>& waypoints)
{
    Eigen::Vector4d s = state;
    float time_remaining = delta_t;
    const float dt = globals.TIMESTEP;
    while (time_remaining > 0.f && !waypoints.empty()) {
        float step = time_remaining < dt ? time_remaining : dt;
        Eigen::Vector2d current_vel = s.segment<2>(2);
        Eigen::Vector2d target_vel = waypoints.front().segment<2>(2);
        float alpha = dt / tau_;
        Eigen::Vector2d new_vel = current_vel + alpha * (target_vel - current_vel);
        Eigen::Vector2d dir = waypoints.front()({0, 1}) - s({0, 1});
        float dist_to_wp = dir.norm();
        dir.normalize();
        s.segment<2>(0) += new_vel.norm() * dir * step;
        s.segment<2>(2) = new_vel;
        if (dist_to_wp < thresh_) waypoints.pop_front();
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
}; 