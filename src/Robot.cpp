/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <memory>
#include <algorithm>
#include <cmath>

#include <Robot.h>
#include <DynamicObstacle.h>
#include <Utils.h>
#include "Metrics.hpp"

/***************************************************************************/
// Creates a robot. Inputs required are :
//      - Pointer to the simulator
//      - A robot id rid (should be taken from simulator->next_rid_++),
//      - A dequeue of waypoints (5 dimensional - [x, y, xdot, ydot, task_timer]). The last dimension specifies a pause duration at the waypoint.
//      - Scale, for non-spherical robot models
//      - Robot radius
//      - Colour
//      - Type, for id-ing the correct model type (see RobotType in Utils.h)
/***************************************************************************/
Eigen::Vector2d make_RA(Eigen::Ref<const Eigen::Vector2d> x0, Eigen::Ref<const Eigen::Vector2d> g, double d_max)
{
    Eigen::Vector2d dir = g - x0;
    double L = dir.norm();
    if (L < 1e-9)
        return x0;
    return x0 + std::min(L, d_max) * (dir / L);
}

Robot::Robot(Simulator *sim, int rid,
             std::deque<Eigen::VectorXd> waypoints,
             RobotType type, float scale, float radius, Color color)
    : FactorGraph{rid}, sim_(sim), rid_(rid), waypoints_(waypoints),
      robot_type_(type), scale_(scale), robot_radius_(radius), color_(color)
{
    dofs_ = globals.N_DOFS;

    // Check if variable DOF matches robot type
    if (robot_type_ == RobotType::SPHERE && dofs_ > 4)
    {
        std::cerr << "warning: robot " << rid << "has SPHERE type but DOF is set to >4, it will be overridden to 4" << "\n";
        dofs_ = 4;
    }
    else if ((robot_type_ == RobotType::CAR || robot_type_ == RobotType::BUS) && dofs_ < 5)
    {
        std::cerr << "warning: robot " << rid << "has non-sphere type but DOF is set to <5, it will be overridden to 5" << "\n";
        dofs_ = 5;
    }

    // Set height based on robot type - use bounding box center for proper ground placement
    if (robot_type_ == RobotType::SPHERE)
    {
        // For sphere, center is at radius height, and dimensions are symmetric
        height_3D_ = robot_radius_;
        robot_dimensions_ = Eigen::Vector2d(2.0 * robot_radius_, 2.0 * robot_radius_);
    }
    else
    {
        // For custom models, get the Y center of the bounding box
        const auto &modelInfo = sim_->graphics->robotModels_[robot_type_];
        height_3D_ = -modelInfo->boundingBox.min.y * scale_;
        robot_dimensions_ = Eigen::Vector2d(modelInfo->dimensions.x * scale_,
                                            modelInfo->dimensions.z * scale_);
        default_angle_offset_ = modelInfo->orientation_offset;
    }

    // Check if waypoints are empty
    if (waypoints_.size() == 0)
    {
        std::cerr << "warning: waypoints_ empty for robot " << rid << ". Initialising to default (zero vector)" << "\n";
        Eigen::VectorXd wp{{0.0, 0.0, 1.0, 0.0, 0.0}}; // Default non-zero velocity so heading is defined
        waypoints_.push_back(wp);
    }

    // Compute base path length, used for evaluation only
    if (waypoints_.size() > 1)
    {
        for (int i = 0; i < waypoints_.size() - 1; ++i)
        {
            Eigen::Vector2d a = waypoints_[i].head<2>();
            Eigen::Vector2d b = waypoints_[i + 1].head<2>();
            base_path_length_ += std::min(50.0, (b - a).norm());
        }
    }
    // Robot will always set its horizon state to move towards the next waypoint.
    // Once this waypoint has been reached, it pops it from the waypoints
    auto &wp = waypoints_.front();

    // Create start state (5D=[x, y, xdot, ydot, theta], 4D=[x, y, xdot, ydot, theta])
    Eigen::VectorXd start(dofs_);
    double theta_start = 0.0; // default heading is east
    if (dofs_ >= 5)
    {
        theta_start = vel_to_theta(wp(2), wp(3), 0.0);
        if (dofs_ == 5)
        {
            start << wp(0), wp(1), wp(2), wp(3), theta_start;
        }
        else
        {
            start << wp(0), wp(1), wp(2), wp(3), theta_start, 0.0;
        }
    }
    else
    {
        start << wp(0), wp(1), wp(2), wp(3);
    }
    position_ = start;

    // Check if starting waypoint is a task
    if (wp.size() == 5 && wp(4) > 0.0)
    {
        task_active_ = true;
        task_timer_ = float(wp(4));
    }
    waypoints_.pop_front();

    // Get the next waypoint as goal (if exists), otherwise use start
    Eigen::VectorXd goal = start;
    double theta_goal = theta_start;
    if (waypoints_.size() > 0)
    {
        auto &next_wp = waypoints_.front();
        if (dofs_ >= 5)
        {
            theta_goal = vel_to_theta(next_wp(2), next_wp(3), start(4));
            if (dofs_ == 5)
            {
                goal << next_wp(0), next_wp(1), next_wp(2), next_wp(3), theta_goal;
            }
            else
            {
                goal << next_wp(0), next_wp(1), next_wp(2), next_wp(3), theta_goal, 0.0;
            }
        }
        else
        {
            goal << next_wp(0), next_wp(1), next_wp(2), next_wp(3);
        }
    }

    // Initialise the horizon in the direction of the goal, at a distance T_HORIZON * MAX_SPEED from the start.
    const double d_max = globals.T_HORIZON * globals.MAX_SPEED;
    const Eigen::Vector2d start_pos = start.head<2>();
    const Eigen::Vector2d goal_pos = goal.head<2>();
    const Eigen::Vector2d RA = make_RA(start_pos, goal_pos, d_max);
    prev_RA_ = RA;

    Eigen::Vector2d disp = RA - start_pos;
    double T = globals.T_HORIZON * globals.TIMESTEP;
    Eigen::Vector2d vel = disp / std::max(T, 1e-9);
    double speed = vel.norm();
    if (speed > globals.MAX_SPEED)
        vel *= globals.MAX_SPEED / speed;

    Eigen::VectorXd horizon(dofs_);
    double theta_horizon = 0.0;
    if (dofs_ >= 5)
    {
        theta_horizon = vel_to_theta(vel.x(), vel.y(), start(4));
        if (dofs_ == 5)
        {
            horizon << RA.x(), RA.y(), vel.x(), vel.y(), theta_horizon;
        }
        else
        {
            horizon << RA.x(), RA.y(), vel.x(), vel.y(), theta_horizon, 0.0;
        }
    }
    else
    {
        horizon << RA.x(), RA.y(), vel.x(), vel.y();
    }

    // Variables representing the planned path are at timesteps which increase in spacing.
    // eg. (so that a span of 10 timesteps as a planning horizon can be represented by much fewer variables)
    std::vector<int> variable_timesteps = getVariableTimesteps(globals.T_HORIZON / globals.T0, globals.LOOKAHEAD_MULTIPLE);
    num_variables_ = variable_timesteps.size();

    /***************************************************************************/
    /* Create Variables with fixed pose priors on start and horizon variables. */
    /***************************************************************************/
    Color var_color = color_;
    double sigma;
    Eigen::VectorXd mu(dofs_);
    Eigen::VectorXd sigma_list(dofs_);
    double dtheta_unwrapped = 0.0;
    double total_time = std::max(1e-9, double(variable_timesteps.back() * globals.T0));
    double omega = 0.0;
    if (dofs_ >= 5)
    {
        dtheta_unwrapped = angle_diff(horizon(4), start(4));
        omega = dtheta_unwrapped / total_time;
    }

    for (int i = 0; i < num_variables_; i++)
    {
        // Set initial mu and covariance of variable interpolated between start and horizon
        float interp_factor = float(variable_timesteps[i]) / float(variable_timesteps.back());

        if (dofs_ >= 5)
        {
            // For 5D state, interpolate position/velocity separately from orientation
            Eigen::Vector4d start_pv = start.head<4>();     // position and velocity
            Eigen::Vector4d horizon_pv = horizon.head<4>(); // position and velocity
            Eigen::Vector4d interp_pv = start_pv + (horizon_pv - start_pv) * interp_factor;

            double theta_i = wrapAngle(start(4) + interp_factor * dtheta_unwrapped);
            if (dofs_ == 5)
            {
                mu << interp_pv, theta_i;
            }
            else
            { // 6D case
                mu << interp_pv, theta_i, omega;
            }
        }
        else
        {
            // For 4D state, simple linear interpolation works
            mu = start + (horizon - start) * interp_factor;
        }

        // Start and Horizon state variables should be 'fixed' during optimisation at a timestep
        sigma = (i == 0 || i == num_variables_ - 1) ? globals.SIGMA_POSE_FIXED : 0.;
        // sigma = (i == 0) ? globals.SIGMA_POSE_FIXED : (i == num_variables_ - 1) ? 1e-2 :  0. ;
        sigma_list.setConstant(sigma);

        // Create variable and add to robot's factor graph
        auto variable = std::make_shared<Variable>(sim->next_vid_++, rid_, mu, sigma_list, robot_radius_, dofs_, variable_timesteps[i]);
        variables_[variable->key_] = variable;
        variable_keys_.push_back(variable->key_);
    }

    /***************************************************************************/
    /* Create Dynamics factors between variables */
    /***************************************************************************/
    for (int i = 0; i < num_variables_ - 1; i++)
    {
        // T0 is the timestep between the current state and the first planned state.
        float delta_t = globals.T0 * (variable_timesteps[i + 1] - variable_timesteps[i]);
        std::vector<std::shared_ptr<Variable>> variables{getVar(i), getVar(i + 1)};
        auto factor = std::make_shared<DynamicsFactor>(sim->next_fid_++, rid_, variables, dofs_, globals.SIGMA_FACTOR_DYNAMICS, Eigen::VectorXd::Zero(dofs_), delta_t);

        // Add this factor to the variable's list of adjacent factors, as well as to the robot's list of factors
        for (auto var : factor->variables_)
            var->add_factor(factor);
        factors_[factor->key_] = factor;
    }

    /***************************************************************************/
    // Create Obstacle factors for all variables excluding start, excluding horizon
    /***************************************************************************/
    for (int i = 1; i < num_variables_ - 1; i++)
    {
        std::vector<std::shared_ptr<Variable>> variables{getVar(i)};
        auto fac_obs = std::make_shared<ObstacleFactor>(sim, sim->next_fid_++, rid_, variables, globals.SIGMA_FACTOR_OBSTACLE, Eigen::VectorXd::Zero(1), &(sim_->obstacleImg));

        // Add this factor to the variable's list of adjacent factors, as well as to the robot's list of factors
        for (auto var : fac_obs->variables_)
            var->add_factor(fac_obs);
        this->factors_[fac_obs->key_] = fac_obs;
    }
};

/***************************************************************************************************/
/* Destructor */
/***************************************************************************************************/
Robot::~Robot()
{
}

/***************************************************************************************************/
/* Change the prior of the Current state */
/***************************************************************************************************/
void Robot::updateCurrent()
{
    auto curr_var = getVar(0), next_var = getVar(1);

    if (task_timer_ > 0.f)
    {
        task_timer_ = std::max(0.f, task_timer_ - globals.TIMESTEP);
        if (task_timer_ == 0.f)
        {
            task_active_ = false;
        }
        else
        {
            return;
        }
    }

    // Move plan: move plan current state by plan increment
    Eigen::VectorXd increment = (next_var->mu_ - curr_var->mu_) * (globals.TIMESTEP / globals.T0);

    if (dofs_ == 6)
    {
        // -------- 6D: [x, y, vx, vy, theta, omega] --------
        const double dt = globals.TIMESTEP;
        position_.head<4>() += increment.head<4>();

        double theta = position_(4);
        double omega = position_(5);
        const double gamma = 0.0; // raw-rate damping, e.g. 0.2
        double theta_new = wrapAngle(theta + omega * dt);
        double omega_new = (1.0 - gamma * dt) * omega;

        position_(4) = orientation_ = theta_new;
        position_(5) = omega_new;

        Eigen::VectorXd new_mu = curr_var->mu_;
        new_mu.head<4>() += increment.head<4>();
        new_mu(4) = theta_new;
        new_mu(5) = omega_new;
        curr_var->change_variable_prior(new_mu);
    }
    else if (dofs_ == 5)
    {
        // ----------- 5D: [x, y, vx, vy, theta] ------------
        position_.head<4>() += increment.head<4>();

        Eigen::Vector2d vel = position_.segment<2>(2);
        double theta_new = vel_to_theta(vel.x(), vel.y(), position_(4));
        position_(4) = orientation_ = theta_new;

        Eigen::VectorXd new_mu = curr_var->mu_;
        new_mu.head<4>() += increment.head<4>();
        new_mu(4) = theta_new;
        curr_var->change_variable_prior(new_mu);

        // position_.head<4>() += increment.head<4>();
        // // velocity after step
        // Eigen::Vector2d v = position_.segment<2>(2);
        // double vmag = v.norm();

        // // goal direction (use horizon var1 position or next waypoint)
        // Eigen::Vector2d pos = position_.head<2>();
        // Eigen::Vector2d goal = prev_RA_;   // or waypoints_.front().head<2>() if you prefer
        // Eigen::Vector2d dir_goal = goal - pos;
        // double dgn = dir_goal.norm();
        // if (dgn > 1e-9) dir_goal /= dgn; else dir_goal = Eigen::Vector2d::UnitX(); // fallback

        // // headings
        // double th = position_(4);
        // double theta_vel  = (vmag > 1e-9) ? wrapAngle(-std::atan2(v.y(), v.x())) : th; // Y-down
        // double theta_goal = wrapAngle(-std::atan2(dir_goal.y(), dir_goal.x()));

        // // --- continuous weights (no hard gates) ---
        // auto clamp01 = [](double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };

        // // speed weight: 0 at rest, 1 at/above ~0.2*MAX_SPEED (smoothstep)
        // double v_on  = 0.06 * globals.MAX_SPEED;
        // double v_hi  = 0.12 * globals.MAX_SPEED;
        // double s     = clamp01((vmag - v_on) / std::max(v_hi - v_on, 1e-9));
        // double w_speed = s * s * (3.0 - 2.0 * s);

        // // forward-progress weight w.r.t goal (rectified dot)
        // double v_fwd_goal = (vmag > 1e-9) ? v.dot(dir_goal) / vmag : 0.0; // ∈[-1,1]
        // double w_fwd = clamp01(v_fwd_goal);                               // 0 if away, 1 if toward

        // // total velocity-alignment weight (soft): fades in with speed AND only if going toward goal
        // double w_align = std::max(w_speed * std::pow(w_fwd, 1.0), 0.05);    // p=2 sharpens shutdown when slightly backward

        // // small always-on goal-facing bias (prevents “stuck”)
        // double w_goal_floor = 0.05;                         // 10% bias toward goal even when stopped

        // // smoothing gain (seconds to respond)
        // double dt  = globals.TIMESTEP;
        // double tau = 0.15;                                   // response time, tune 0.2–0.5 s
        // double k   = std::clamp(dt / tau, 0.0, 1.0);

        // // compose update: blend toward velocity when appropriate, otherwise gently face goal
        // double d_vel  = angle_diff(theta_vel,  th);
        // double d_goal = angle_diff(theta_goal, th);
        // double d_total = w_align * d_vel + (1.0 - w_align) * w_goal_floor * d_goal;

        // double theta_new = wrapAngle(th + k * d_total);

        // // write back & sync var0 prior
        // position_(4) = orientation_ = theta_new;

        // Eigen::VectorXd new_mu = curr_var->mu_;
        // new_mu.head<4>() += increment.head<4>();
        // new_mu(4) = theta_new;
        // curr_var->change_variable_prior(new_mu);
    }
    else
    {
        // --------------- 4D: [x, y, vx, vy] ---------------
        position_ += increment;
        curr_var->change_variable_prior(curr_var->mu_ + increment);
    }

    // Perform distance check to initiate the task countdown timer
    if (task_active_ && waypoints_.size() > 0)
    {
        auto &wp = waypoints_.front();
        Eigen::Vector2d dist = position_.head<2>() - wp.head<2>();
        if (dist.norm() < robot_radius_)
        {
            task_timer_ = wp(4);
            waypoints_.pop_front();
        }
    }
};

/***************************************************************************************************/
/* Change the prior of the Horizon state */
/***************************************************************************************************/
void Robot::updateHorizon()
{
    // Horizon is moved only if not performing a task
    if (task_active_)
        return;
    task_active_ = false;

    auto horizon = getVar(-1);
    if (!horizon)
        return;

    if (waypoints_.size() > 0)
    {
        auto &wp = waypoints_.front();
        // Check if next waypoint is a task
        next_wp_is_task_ = wp.size() >= 5 && wp(4) > 0.0;

        // --- RA recompute from current robot position (var0) ---
        auto curr_var = getVar(0);
        const Eigen::Vector2d x0 = curr_var->mu_.head<2>();
        const Eigen::Vector2d g = wp.head<2>();
        const double d_max = globals.T_HORIZON * globals.MAX_SPEED;
        const Eigen::Vector2d RA = make_RA(x0, g, d_max);

        if (!std::isfinite(prev_RA_.x()) || !std::isfinite(prev_RA_.y()))
            prev_RA_ = RA;

        // Rate limit RA motino to prevent anchor from teleporting
        const double dt = globals.TIMESTEP;
        const double v_anchor = 1.0 * globals.MAX_SPEED;
        const double cap = v_anchor * dt;
        Eigen::Vector2d step = RA - prev_RA_;
        double L = step.norm();
        prev_RA_ += (L > cap) ? (cap / std::max(L, 1e-12)) * step : step;

        // Move horizon toward (rate-limited) RA with capped velocity
        const Eigen::Vector2d pH = horizon->mu_.head<2>();
        Eigen::Vector2d to_anchor = prev_RA_ - pH;

        Eigen::Vector2d new_vel = Eigen::Vector2d::Zero();
        double dist_to_anchor = to_anchor.norm();
        if (dist_to_anchor > 1e-9)
        {
            double target = dist_to_anchor / std::max(dt, 1e-9);
            double speed = std::min(target, double(globals.MAX_SPEED));
            new_vel = speed * (to_anchor / dist_to_anchor);
        }
        Eigen::Vector2d new_pos = pH + new_vel * dt;

        // ---------- Build horizon new_mu per DOF ----------
        Eigen::VectorXd new_mu(dofs_);
        if (dofs_ == 4)
        {
            new_mu << new_pos, new_vel;
        }
        else if (dofs_ == 5)
        {
            // Update horizon state with new pos, vel, and orientation
            double theta = vel_to_theta(new_vel.x(), new_vel.y(), horizon->mu_(4));
            new_mu << new_pos, new_vel, theta;
        }
        else
        {
            double theta = horizon->mu_(4);
            double omega = horizon->mu_(5);
            const double gamma = 0.0; // yaw-rate damping [s^-1]; keep in sync with DynamicsFactor
            double theta_new = wrapAngle(theta + omega * dt);
            double omega_new = (1.0 - gamma * dt) * omega;
            new_mu << new_pos, new_vel, theta_new, omega_new;
        }

        horizon->change_variable_prior(new_mu);

        // If the horizon has reached the waypoint, handle task or normal waypoint
        Eigen::VectorXd dist_horz_to_goal = wp.head<2>() - new_pos;
        if (dist_horz_to_goal.norm() < robot_radius_)
        {
            if (next_wp_is_task_)
            {
                task_active_ = true;
                // Snap horizon state to task waypoint
                Eigen::VectorXd snap(dofs_);
                if (dofs_ == 4)
                {
                    snap << wp.head<2>(), Eigen::Vector2d::Zero();
                }
                else if (dofs_ >= 5)
                {
                    double theta = horizon->mu_(4); // keep current orientation at task
                    if (dofs_ == 5)
                    {
                        snap << wp.head<2>(), Eigen::Vector2d::Zero(), theta;
                    }
                    else
                    {
                        snap << wp.head<2>(), Eigen::Vector2d::Zero(), theta, 0.0;
                    }
                }
                horizon->change_variable_prior(snap);
            }
            else
            { // Normal waypoint - just pop it
                waypoints_.pop_front();
            }
        }
    }
}

/***************************************************************************************************/
// For new neighbours of a robot, create inter-robot factors if they don't exist.
// Delete existing inter-robot factors for faraway robots
/***************************************************************************************************/
void Robot::updateInterrobotFactors()
{

    // Search through currently connected rids. If any are not in neighbours, delete interrobot factors.
    for (auto rid : connected_r_ids_)
    {
        if (std::find(neighbours_.begin(), neighbours_.end(), rid) == neighbours_.end())
        {
            auto it = sim_->robots_.find(rid);
            if (it != sim_->robots_.end()) {
                deleteInterrobotFactors(sim_->robots_.at(rid));
            }
        };
    }
    // Search through neighbours. If any are not in currently connected rids, create interrobot factors.
    for (auto rid : neighbours_)
    {
        if (std::find(connected_r_ids_.begin(), connected_r_ids_.end(), rid) == connected_r_ids_.end())
        {
            auto it = sim_->robots_.find(rid);
            if (it != sim_->robots_.end()) {
                createInterrobotFactors(sim_->robots_.at(rid));
                if (!sim_->symmetric_factors)
                    sim_->robots_.at(rid)->connected_r_ids_.push_back(rid_);
            }
        };
    }
}

/***************************************************************************************************/
// Create inter-robot factors between this robot and another robot
/***************************************************************************************************/
void Robot::createInterrobotFactors(std::shared_ptr<Robot> other_robot)
{
    // Create Interrobot factors for all timesteps excluding current state
    for (int i = 1; i < num_variables_; i++)
    {
        // Get variables
        std::vector<std::shared_ptr<Variable>> variables{getVar(i), other_robot->getVar(i)};

        // Create the inter-robot factor with a scalar penalty (center-to-center)
        Eigen::VectorXd z = Eigen::VectorXd::Zero(1);
        auto factor = std::make_shared<InterrobotFactor>(sim_->next_fid_++, this->rid_, variables, dofs_,
                                                         globals.SIGMA_FACTOR_INTERROBOT, z,
                                                         0.5 * (this->robot_radius_ + other_robot->robot_radius_),
                                                         this->robot_dimensions_,
                                                         other_robot->robot_dimensions_,
                                                         this->default_angle_offset_,
                                                         other_robot->default_angle_offset_);
        factor->other_rid_ = other_robot->rid_;
        // Add factor the the variable's list of factors, as well as to the robot's list of factors
        for (auto var : factor->variables_)
            var->add_factor(factor);
        this->factors_[factor->key_] = factor;
    }

    // Add the other robot to this robot's list of connected robots, so factor doesn't get duplicated.
    this->connected_r_ids_.push_back(other_robot->rid_);
};

/***************************************************************************************************/
/* Delete interrobot factors between the two robots */
/***************************************************************************************************/
void Robot::deleteInterrobotFactors(std::shared_ptr<Robot> other_robot)
{
    std::vector<Key> facs_to_delete{};
    for (auto &[f_key, fac] : this->factors_)
    {
        if (fac->other_rid_ != other_robot->rid_)
            continue;

        // Only get here if factor is connected to a variable in the other_robot
        for (auto &var : fac->variables_)
        {
            var->delete_factor(f_key);
            facs_to_delete.push_back(f_key);
        }
    }
    for (auto f_key : facs_to_delete)
        this->factors_.erase(f_key);

    // Remove other robot from current robot's connected rids
    auto it = std::find(connected_r_ids_.begin(), connected_r_ids_.end(), other_robot->rid_);
    if (it != connected_r_ids_.end())
    {
        connected_r_ids_.erase(it);
    }
};

/***************************************************************************************************/
// For new obstacles, create dynamic obstacle factors if they don't exist.
// Delete existing inter-robot factors for out-of-bounds obstacles.
/***************************************************************************************************/
void Robot::updateDynamicObstacleFactors()
{
    // If obstacles are removed from simulator then delete factors connected to it
    std::vector<int> to_remove;
    for (int oid : connected_obs_ids_)
    {
        auto it = sim_->obstacles_.find(oid);
        if (it == sim_->obstacles_.end())
        {
            to_remove.push_back(oid);
        }
        else
        {
            // Also remove factors for obstacles that have moved too far away
            auto obs = it->second;
            const double removal_threshold = globals.OBSTALCE_SENSOR_RADIUS + robot_radius_ + 3.0; // Slightly larger than culling threshold

            // Remove factor if obstacle is too far away from both current and future robot positions
            if (getDistToObs(obs) > removal_threshold * removal_threshold)
            {
                to_remove.push_back(oid);
            }
        }
    }
    for (int oid : to_remove)
        deleteDynamicObstacleFactors(oid);

    // Create new dynamic obstacle factors for newly created obstacles
    std::vector<std::shared_ptr<DynamicObstacle>> to_add;
    for (const auto &[oid, obs] : sim_->obstacles_)
    {
        if (std::find(connected_obs_ids_.begin(), connected_obs_ids_.end(), oid) == connected_obs_ids_.end())
        {
            to_add.push_back(obs);
        }
    }
    for (const auto &obs : to_add)
        createDynamicObstacleFactors(obs);
}

/***************************************************************************************************/
// Create dynamic obstacle factors of this obstacle for each robot variable
/***************************************************************************************************/
void Robot::createDynamicObstacleFactors(std::shared_ptr<DynamicObstacle> obs)
{
    // Spatial culling: check if obstacle is close enough to warrant factor creation
    // Use a conservative threshold that's larger than the factor's skip radius
    const double culling_threshold = globals.OBSTALCE_SENSOR_RADIUS + robot_radius_ + 2.0; // Add safety margin
    // Skip factor creation if obstacle is too far away from both current and future robot positions
    if (getDistToObs(obs) > culling_threshold * culling_threshold)
    {
        return;
    }

    for (int i = 1; i < num_variables_ - 1; ++i)
    {
        std::vector<std::shared_ptr<Variable>> variables{getVar(i)};
        if (dofs_ == 4)
        {
        }
        auto do_fac = std::make_shared<DynamicObstacleFactor>(sim_->next_fid_++, rid_, variables, dofs_, globals.SIGMA_FACTOR_DYNAMIC_OBSTACLE,
                                                              Eigen::VectorXd::Zero(1), robot_radius_, obs, robot_dimensions_, default_angle_offset_);
        for (auto var : do_fac->variables_)
            var->add_factor(do_fac);
        factors_[do_fac->key_] = do_fac;
    }
    connected_obs_ids_.push_back(obs->oid_);
}

/***************************************************************************************************/
// Delete dynamic obstacle factors of this obstacle from all robot variables
/***************************************************************************************************/
void Robot::deleteDynamicObstacleFactors(int oid)
{
    std::vector<Key> facs_to_delete;
    for (auto &[f_key, fac] : factors_)
    {
        auto do_fac = std::dynamic_pointer_cast<DynamicObstacleFactor>(fac);
        if (!do_fac || do_fac->obs_->oid_ != oid)
            continue;

        for (auto &var : do_fac->variables_)
            var->delete_factor(f_key);
        facs_to_delete.push_back(f_key);
    }

    for (auto f_key : facs_to_delete)
        factors_.erase(f_key);

    auto it = std::find(connected_obs_ids_.begin(), connected_obs_ids_.end(), oid);
    if (it != connected_obs_ids_.end())
    {
        connected_obs_ids_.erase(it);
    }
}
/***************************************************************************************************/
// Get distance between position and obstacle centre
/***************************************************************************************************/
double Robot::getDistToObs(std::shared_ptr<DynamicObstacle> obs)
{
    // Check current distance between robot and obstacle
    Eigen::Vector2d robot_pos = position_.head<2>();
    Eigen::Vector2d obs_pos_current(obs->state_[0], obs->state_[1]);
    double current_dist_squared = (robot_pos - obs_pos_current).squaredNorm();

    // Also check the distance to the robot's horizon state vs obstacle's future position
    auto horizon_var = getVar(-1);
    Eigen::Vector2d robot_horizon = horizon_var->mu_.head<2>();
    int horizon_ts = horizon_var->ts_;

    // Get obstacle position at the horizon timestep
    Eigen::Vector4d obs_state_future = obs->states_.at(horizon_ts);
    Eigen::Vector2d obs_pos_future(obs_state_future[0], obs_state_future[1]);
    double horizon_dist_squared = (robot_horizon - obs_pos_future).squaredNorm();

    // Use the minimum distance (current or future) for culling decision
    double min_dist_squared = std::min(current_dist_squared, horizon_dist_squared);
    return min_dist_squared;
};

/***************************************************************************************************/
// Drawing functions for the robot.
// We deal with a 2d problem, so the out-of-plane height is set to height_3D_.
/***************************************************************************************************/
void Robot::draw()
{
    const auto &model_info = sim_->graphics->robotModels_[robot_type_];
    
    // Draw planned path
    if (globals.DRAW_PATH)
    {
        for (auto [vid, variable] : variables_)
        {
            const auto &mu = variable->mu_;
            if (!variable->valid_)
                continue;
            if (robot_type_ == RobotType::SPHERE)
            {
                DrawModel(sim_->graphics->robotModels_[robot_type_]->model,
                          Vector3{(float)mu(0), height_3D_, (float)mu(1)},
                          0.5 * robot_radius_, ColorAlpha(color_, 0.5));
            }
            else if (robot_type_ == RobotType::CAR)
            {
                float o_deg = wrapAngle((mu(4) + model_info->orientation_offset)) * (180.0f / M_PI);
                float scale = 0.5 * scale_;
                DrawModelEx(model_info->model,
                            Vector3{(float)mu(0), height_3D_, (float)mu(1)},
                            Vector3{0.0f, 1.0f, 0.0f},
                            o_deg,
                            Vector3{scale, scale, scale},
                            ColorAlpha(color_, 0.5));
            }
        }
        
        for (auto [fid, factor] : factors_)
        {
            if (factor->factor_type_ != DYNAMICS_FACTOR) continue;
            auto variables = factor->variables_;
            Eigen::VectorXd p0 = variables[0]->mu_, p1 = variables[1]->mu_;
            DrawCylinderEx(Vector3{(float)p0(0), globals.ROBOT_RADIUS, (float)p0(1)}, Vector3{(float)p1(0), globals.ROBOT_RADIUS, (float)p1(1)}, 0.1, 0.1, 4, BLACK);
        }
    }

    // Draw connected robots
    if (globals.DRAW_INTERROBOT)
    {
        for (auto rid : connected_r_ids_)
        {
            if (!interrobot_comms_active_ || !sim_->robots_.at(rid)->interrobot_comms_active_)
                continue;
            DrawCylinderEx(Vector3{(float)position_(0), height_3D_, (float)position_(1)},
                           Vector3{(float)(*sim_->robots_.at(rid))[0]->mu_(0), sim_->robots_.at(rid)->height_3D_, (float)(*sim_->robots_.at(rid))[0]->mu_(1)},
                           0.1, 0.1, 4, BLACK);
        }
    }

    // Draw the waypoints of the robot
    if (globals.DRAW_WAYPOINTS)
    {
        for (int wp_idx = 0; wp_idx < waypoints_.size(); wp_idx++)
        {
            DrawCubeV(Vector3{(float)waypoints_[wp_idx](0), height_3D_, (float)waypoints_[wp_idx](1)}, 
                      Vector3{1.f * robot_radius_, 1.f * robot_radius_, 1.f * robot_radius_}, 
                      color_);
        }
    }

    // Draw all factor connections
    if (globals.DRAW_FACTORS)
    {
        for (auto [fid, factor] : factors_)
            factor->draw();
    }
    
    // Draw robots
    if (globals.DRAW_ROBOTS)
    {
        // Draw the robot model based on its type
        if (robot_type_ == RobotType::SPHERE)
        {
            // For sphere type, just draw a sphere (original implementation)
            DrawModel(sim_->graphics->robotModels_[robot_type_]->model,
                      Vector3{(float)position_(0), height_3D_, (float)position_(1)},
                      robot_radius_, color_);
        }
        else
        {
            float o_deg = wrapAngle(orientation_ + model_info->orientation_offset) * (180.0f / M_PI);
            DrawModelEx(model_info->model,
                        Vector3{(float)position_(0), height_3D_, (float)position_(1)},
                        Vector3{0.0f, 1.0f, 0.0f},
                        o_deg,
                        Vector3{scale_, scale_, scale_},
                        color_);
        }
    }
};