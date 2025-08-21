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
Eigen::Vector2d make_RA(Eigen::Ref<const Eigen::Vector2d> x0, Eigen::Ref<const Eigen::Vector2d> g, double d_max) {
    Eigen::Vector2d dir = g - x0;
    double L = dir.norm();
    if (L < 1e-9) return x0;
    return x0 + std::min(L, d_max) * (dir / L);
}

Robot::Robot(Simulator *sim, int rid,
             std::deque<Eigen::VectorXd> waypoints,
             RobotType type, float scale, float radius, Color color) 
    : FactorGraph{rid}, sim_(sim), rid_(rid), waypoints_(waypoints), 
      robot_type_(type), scale_(scale), robot_radius_(radius), color_(color)
{
    dofs_ = globals.N_DOFS;

    // Check if variable DOF matches robot type, as spheres don't have orientation
    if (robot_type_ == RobotType::SPHERE && dofs_ == 5) {
        std::cerr << "warning: robot " << rid << "has SPHERE type but DOF is set to 5, it will be overridden to 4" << "\n";
        dofs_ = 4;
    }

    // Set height based on robot type - use bounding box center for proper ground placement
    if (robot_type_ == RobotType::SPHERE) {
        // For sphere, center is at radius height, and dimensions are symmetric
        height_3D_ = robot_radius_;
        robot_dimensions_ = Eigen::Vector2d(2.0 * robot_radius_, 2.0 * robot_radius_);
    } else {
        // For custom models, get the Y center of the bounding box
        const auto& modelInfo = sim_->graphics->robotModels_[robot_type_];
        height_3D_ = -modelInfo->boundingBox.min.y * scale_;
        
        // Set robot dimensions based on the model's bounding box
        // Scale the dimensions to match the desired robot_radius
        robot_dimensions_ = Eigen::Vector2d(modelInfo->dimensions.x * scale_, 
                                            modelInfo->dimensions.z * scale_);
        // Set default angle offset
        default_angle_offset_ = modelInfo->orientation_offset;
    }

    // Check if waypoints are empty
    if (waypoints_.size() == 0)
    {
        std::cerr << "warning: waypoints_ empty for robot " << rid << ". Initialising to default (zero vector)" << "\n";
        // +1 additional dimension for task_timer, 
        // Orientation is never included in waypoints, as it is computed from velocity
        Eigen::VectorXd wp = Eigen::VectorXd::Zero(5);
        waypoints_.push_back(wp);
    }

    // Compute base path length
    if (waypoints_.size() > 1) {
        for (int i = 0; i < waypoints_.size()-1; ++i) {
            Eigen::Vector2d a = waypoints_[i].head<2>();
            Eigen::Vector2d b = waypoints_[i+1].head<2>();
            base_path_length_ += (b-a).norm();
        }
    }

    // Robot will always set its horizon state to move towards the next waypoint.
    // Once this waypoint has been reached, it pops it from the waypoints
    auto &wp = waypoints_.front();
    // Create start state (5D=[x, y, xdot, ydot, theta], 4D=[x, y, xdot, ydot, theta])
    Eigen::VectorXd start(dofs_);
    if (dofs_ == 5) {
        double initial_theta = 0.0;
        // Only compute orientation from velocity if robot is moving
        if (std::abs(wp(2)) > 1e-6 || std::abs(wp(3)) > 1e-6) {
            initial_theta = -wrapAngle(std::atan2(wp(3), wp(2)));
        }
        start << wp(0), wp(1), wp(2), wp(3), initial_theta; // Exclude the task timer from state
    } else {
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
    if (waypoints_.size() > 0) {
        auto &next_wp = waypoints_.front();
        goal = Eigen::VectorXd(dofs_);
        if (dofs_ == 5) {
            double goal_theta = start(4);
            if (std::abs(next_wp(2)) > 1e-6 || std::abs(next_wp(3)) > 1e-6) {
                goal_theta = -wrapAngle(std::atan2(next_wp(3), next_wp(2)));
            }
            goal << next_wp(0), next_wp(1), next_wp(2), next_wp(3), goal_theta;
        } else {
            goal << next_wp(0), next_wp(1), next_wp(2), next_wp(3);
        }
    }

    // Initialise the horizon in the direction of the goal, at a distance T_HORIZON * MAX_SPEED from the start.
    Eigen::VectorXd horizon(dofs_);
    // if (dofs_ == 5) {
    //     // For 5D state, handle position/velocity and orientation separately
    //     Eigen::Vector4d start_pv = start.head<4>();
    //     Eigen::Vector4d goal_pv = goal.head<4>();
    //     Eigen::Vector4d start2goal_pv = goal_pv - start_pv;
    //     Eigen::Vector4d horizon_pv = start_pv + std::min(start2goal_pv.norm(), double(globals.T_HORIZON * globals.MAX_SPEED)) * start2goal_pv.normalized();
        
    //     // Calculate horizon orientation from velocity direction
    //     double horizon_theta = goal(4);  // Default to goal orientation
    //     if (horizon_pv.segment<2>(2).norm() > 1e-6) {
    //         horizon_theta = -wrapAngle(std::atan2(horizon_pv(2), horizon_pv(3)));  // Negate Y for our coordinate system
    //     }
        
    //     horizon << horizon_pv, horizon_theta;
    // } else {
    //     // For 4D state (no orientation), use simplified logic
    //     Eigen::VectorXd start2goal = goal - start;
    //     horizon = start + std::min(start2goal.norm(), double(globals.T_HORIZON * globals.MAX_SPEED)) * start2goal.normalized();
    // }
    const double d_max = globals.T_HORIZON * globals.MAX_SPEED;
    const Eigen::Vector2d start_pos = start.head<2>();
    const Eigen::Vector2d goal_pos = goal.head<2>();
    const Eigen::Vector2d RA = make_RA(start_pos, goal_pos, d_max);
    prev_RA_ = RA;

    Eigen::Vector2d vel = RA - start_pos;
    double L = vel.norm();
    if (L > 1e-9) vel = (std::min(L, (double)globals.MAX_SPEED)) * (vel / L);
    else vel.setZero();
    
    if (dofs_ == 5) {
        double horizon_theta = start(4);
        if (vel.norm() > 1e-6) horizon_theta = -wrapAngle(std::atan2(vel.y(), vel.x()));

        horizon.resize(5);
        horizon << RA.x(), RA.y(), vel.x(), vel.y(), horizon_theta;
    } else {
        horizon.resize(4);
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

    for (int i = 0; i < num_variables_; i++)
    {
        // Set initial mu and covariance of variable interpolated between start and horizon
        float interp_factor = (float)(variable_timesteps[i] / (float)variable_timesteps.back());
        
        if (dofs_ == 5) {
            // For 5D state, interpolate position/velocity separately from orientation
            Eigen::Vector4d start_pv = start.head<4>();  // position and velocity
            Eigen::Vector4d horizon_pv = horizon.head<4>();  // position and velocity
            Eigen::Vector4d interp_pv = start_pv + (horizon_pv - start_pv) * interp_factor;
            
            // Calculate orientation from interpolated velocity
            double interp_theta = start(4);  // Default to start orientation
            if (interp_pv.tail<2>().norm() > 1e-6) {  // If there's velocity
                interp_theta = -wrapAngle(std::atan2(interp_pv(3), interp_pv(2)));
            }
            
            mu << interp_pv, interp_theta;
        } else {
            // For 4D state, simple linear interpolation works
            mu = start + (horizon - start) * interp_factor;
        }
        // print(variable_timesteps[i], mu.transpose());
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
    auto curr_var = getVar(0);
    
    if (task_timer_ > 0.f)
    {
        task_timer_ = std::max(0.f, task_timer_ - globals.TIMESTEP);
        if (task_timer_ == 0.f) { task_active_ = false; }
        else { return; }
    }
   
    // Move plan: move plan current state by plan increment
    Eigen::VectorXd increment = ((*this)[1]->mu_ - (*this)[0]->mu_) * globals.TIMESTEP / globals.T0;
    
    // Handle orientation separately for 5D state
    if (dofs_ == 5) {
        // Calculate new position and velocity from increment
        Eigen::Vector4d pos_vel_increment = increment.head<4>();
        Eigen::Vector4d new_pos_vel = position_.head<4>() + pos_vel_increment;
        
        // Calculate new orientation based on velocity direction
        double new_theta = position_(4);  // Keep current orientation by default
        Eigen::Vector2d velocity = new_pos_vel.tail<2>();
        if (velocity.norm() > 1e-6) {
            new_theta = -wrapAngle(std::atan2(velocity(1), velocity(0)));  // Negate Y for our coordinate system
        }
        
        // Update position with proper orientation
        position_.head<4>() = new_pos_vel;
        position_(4) = new_theta;
        
        // Update the variable prior similarly
        Eigen::VectorXd new_mu = curr_var->mu_;
        new_mu.head<4>() += increment.head<4>();
        new_mu(4) = new_theta;
        curr_var->change_variable_prior(new_mu);
        
        // Extract orientation for visualization
        orientation_ = new_theta;
    } else {
        // For 4D state, simple increment works
        curr_var->change_variable_prior(curr_var->mu_ + increment);
        position_ = position_ + increment;
    }

    // Perform distance check to initiate the task countdown timer
    if (task_active_ && waypoints_.size() > 0) {
        auto &wp = waypoints_.front();
        Eigen::VectorXd dist_curr_to_goal = position_({0, 1}) - wp({0, 1});
        if (dist_curr_to_goal.norm() <  robot_radius_) {
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
    
    if (waypoints_.size() > 0)
    {
        auto &wp = waypoints_.front();
        // Check if next waypoint is a task
        next_wp_is_task_ = wp.size() >= 5 && wp(4) > 0.0;

        // // Horizon state moves towards the next waypoint.
        // // The Horizon state's velocity is capped at MAX_SPEED
        Eigen::VectorXd dist_horz_to_goal = wp.head<2>() - horizon->mu_.head<2>();
        // // This always moves robot with a new speed
        // Eigen::VectorXd new_vel = dist_horz_to_goal.normalized() * std::min((double)globals.MAX_SPEED, dist_horz_to_goal.norm());
        // Eigen::VectorXd new_pos = horizon->mu_({0, 1}) + new_vel * globals.TIMESTEP;

        // --- NEW ---
        // Recompute a Reachable Anchor (RA) from the *current* robot position
        // so the horizon cannot outrun what the robot can reach within T_HORIZON.
        auto curr_var = getVar(0);
        const Eigen::Vector2d x0 = curr_var->mu_.head<2>();
        const Eigen::Vector2d g  = wp.head<2>();
        const double d_max = globals.T_HORIZON * globals.MAX_SPEED;

        const Eigen::Vector2d RA = make_RA(x0, g, d_max);
        if (!std::isfinite(prev_RA_.x()) || !std::isfinite(prev_RA_.y())) prev_RA_ = RA;
        
        const double v_anchor = 0.8 * globals.MAX_SPEED; // anchor moves a bit slower than robot
        const double cap = v_anchor * globals.TIMESTEP;
        Eigen::Vector2d step = RA - prev_RA_;
        double L = step.norm();
        if (L > cap) prev_RA_ += (cap / L) * step; else prev_RA_ = RA;

        // Move horizon toward (rate-limited) RA with capped velocity
        Eigen::Vector2d to_anchor = prev_RA_ - horizon->mu_.head<2>();
        Eigen::Vector2d new_vel = Eigen::Vector2d::Zero();
        if (to_anchor.norm() > 1e-9) new_vel = (std::min((double)globals.MAX_SPEED, to_anchor.norm()/std::max((double)globals.TIMESTEP,1e-9))) * (to_anchor.normalized());
        Eigen::Vector2d new_pos = horizon->mu_.head<2>() + new_vel * globals.TIMESTEP;

        // Hard clamp: keep horizon inside the reachable disc from x0 (defensive)
        Eigen::Vector2d x0_to_new = new_pos - x0;
        if (x0_to_new.norm() > d_max) new_pos = x0 + (d_max / x0_to_new.norm()) * x0_to_new;

        Eigen::VectorXd new_mu(dofs_);
        if (dofs_ == 5) {
            // Update horizon state with new pos, vel, and orientation
            double new_theta = horizon->mu_(4);  // Keep current orientation by default
            // Only update orientation if there's significant velocity
            if (new_vel.norm() > 1e-6) {
                new_theta = -wrapAngle(std::atan2(new_vel(1), new_vel(0)));  // Negate Y for proper orientation
            }
            new_mu << new_pos, new_vel, new_theta;
        } else {
            new_mu << new_pos, new_vel;
        }
        horizon->change_variable_prior(new_mu);
        // --- NEW ---

        // If the horizon has reached the waypoint, handle task or normal waypoint
        if (dist_horz_to_goal.norm() < robot_radius_)
        {
            if (next_wp_is_task_)
            {
                task_active_ = true;
                // Snap horizon state to task waypoint
                Eigen::VectorXd new_mu(dofs_);
                if (dofs_ == 5) {
                    double task_theta = horizon->mu_(4);  // Keep current orientation at task
                    new_mu << wp.head<2>(), Eigen::VectorXd::Zero(2), task_theta;
                } else {
                    new_mu << wp.head<2>(), Eigen::VectorXd::Zero(2);
                }
                horizon->change_variable_prior(new_mu);
            }
            else
            {
                // Normal waypoint - just pop it
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
            deleteInterrobotFactors(sim_->robots_.at(rid));
        };
    }
    // Search through neighbours. If any are not in currently connected rids, create interrobot factors.
    for (auto rid : neighbours_)
    {
        if (std::find(connected_r_ids_.begin(), connected_r_ids_.end(), rid) == connected_r_ids_.end())
        {
            createInterrobotFactors(sim_->robots_.at(rid));
            if (!sim_->symmetric_factors)
                sim_->robots_.at(rid)->connected_r_ids_.push_back(rid_);
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

        // Create the inter-robot factor with robot dimensions for OBB collision
        Eigen::VectorXd z = Eigen::VectorXd::Zero(variables.front()->n_dofs_);
        auto factor = std::make_shared<InterrobotFactor>(sim_->next_fid_++, this->rid_, variables, dofs_,
                                                         globals.SIGMA_FACTOR_INTERROBOT, z, 
                                                         0.5 * (this->robot_radius_ + other_robot->robot_radius_),
                                                         this->robot_dimensions_,
                                                         other_robot->robot_dimensions_,
                                                         this->default_angle_offset_,
                                                         other_robot->default_angle_offset_);
        factor->other_rid_ = other_robot->rid_;
        // Add factor the the variable's list of factors, as well as to the robot's list of factors
        for (auto var : factor->variables_) var->add_factor(factor);
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
        auto do_fac = std::make_shared<DynamicObstacleFactor>(sim_->next_fid_++, rid_, variables, dofs_, globals.SIGMA_FACTOR_DYNAMIC_OBSTACLE,
                                                              Eigen::VectorXd::Zero(1), robot_radius_, obs);
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
    // Color col = (interrobot_comms_active_) ? color_ : GRAY;
    Color col = color_;
    if (globals.EVAL) {
        // col = interrobot_comms_active_ ? color_ : sim_->metrics->getTrack(rid_).collided ? RED : GRAY;
        col = sim_->metrics->getTrack(rid_).collided ? RED : color_;
    }
    const auto& model_info = sim_->graphics->robotModels_[robot_type_];
    // Draw planned path
    if (globals.DRAW_PATH)
    {
        static int debug = 0;
        for (auto [vid, variable] : variables_)
        {
            const auto& mu = variable->mu_;
            if (!variable->valid_) continue;
            if (robot_type_ == RobotType::SPHERE) {
                DrawModel(sim_->graphics->robotModels_[robot_type_]->model,
                         Vector3{(float)mu(0), height_3D_, (float)mu(1)},
                         0.5 * robot_radius_, ColorAlpha(col, 0.5));
 
            } else if (robot_type_ == RobotType::CAR) {
                float rotation_degrees = (mu(4) + model_info->orientation_offset) * (180.0f / M_PI);
                float adjusted_rotation = std::remainder(rotation_degrees, 360.f);
                float scale = 0.5 * scale_;
                DrawModelEx(model_info->model,
                            Vector3{(float)mu(0), height_3D_, (float)mu(1)},
                            Vector3{0.0f, 1.0f, 0.0f},    // Rotate around Y axis
                            adjusted_rotation,            // Rotation angle in degrees with offset
                            Vector3{scale, scale, scale}, // Uniform scale
                            ColorAlpha(col, 0.5));        // Color tint
            }
        }
        for (auto [fid, factor] : factors_)
        {
            if (factor->factor_type_ != DYNAMICS_FACTOR)
                continue;
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
            // if (!interrobot_comms_active_ || !sim_->robots_.at(rid)->interrobot_comms_active_)
            //     continue;
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
            DrawCubeV(Vector3{(float)waypoints_[wp_idx](0), height_3D_, (float)waypoints_[wp_idx](1)}, Vector3{1.f * robot_radius_, 1.f * robot_radius_, 1.f * robot_radius_}, col);
        }
    }

    // Draw all factor connections
    if (globals.DRAW_FACTORS)
    {
        for (auto [fid, factor] : factors_)
            factor->draw();
    }
    // Draw the robot model based on its type
    if (robot_type_ == RobotType::SPHERE) {
        // For sphere type, just draw a sphere (original implementation)
        DrawModel(sim_->graphics->robotModels_[robot_type_]->model,
                  Vector3{(float)position_(0), height_3D_, (float)position_(1)},
                  robot_radius_, col);
    } else {
        // For car/bus models, apply rotation and proper scaling
        // Convert orientation from radians to degrees for Raylib
        float rotation_degrees = (orientation_ + model_info->orientation_offset) * (180.0f / M_PI);
        float adjusted_rotation = std::remainder(rotation_degrees, 360.f);
        
        // DrawModelEx allows us to specify position, rotation axis, rotation angle, scale, and tint
        DrawModelEx(model_info->model,
                    Vector3{(float)position_(0), height_3D_, (float)position_(1)},
                    Vector3{0.0f, 1.0f, 0.0f},       // Rotate around Y axis
                    adjusted_rotation,               // Rotation angle in degrees with offset
                    Vector3{scale_, scale_, scale_}, // Uniform scale
                    col);                            // Color tint
    }
};