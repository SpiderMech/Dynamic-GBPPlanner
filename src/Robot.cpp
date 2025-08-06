/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <memory>
#include <algorithm>

#include <Robot.h>
#include <DynamicObstacle.h>

/***************************************************************************/
// Creates a robot. Inputs required are :
//      - Pointer to the simulator
//      - A robot id rid (should be taken from simulator->next_rid_++),
//      - A dequeue of waypoints (which are 4 dimensional [x,y,xdot,ydot])
//      - Robot radius
//      - Colour
/***************************************************************************/
Robot::Robot(Simulator *sim,
             int rid,
             std::deque<Eigen::VectorXd> waypoints,
             float size,
             Color color) : FactorGraph{rid},
                            sim_(sim), rid_(rid),
                            waypoints_(waypoints),
                            robot_radius_(size), color_(color)
{

    height_3D_ = robot_radius_; // Height out of plane for 3d visualisation only

    // Initialise robot at
    if (waypoints_.size() == 0)
    {
        std::cerr << "warning: waypoints_ empty for robot " << rid << ". Initialising to default pose ([0, 0, 0, 0])" << "\n";
        Eigen::VectorXd wp{{0., 0., 0., 0., 0.}};
        waypoints_.push_back(wp);
    }

    // Robot will always set its horizon state to move towards the next waypoint.
    // Once this waypoint has been reached, it pops it from the waypoints
    auto &wp = waypoints_.front();
    Eigen::VectorXd wp_no_timer{{wp(0), wp(1), wp(2), wp(3)}};
    Eigen::VectorXd start = position_ = wp_no_timer;
    if (wp.size() >= 5 && wp(4) > 0.0)
    {
        task_active_ = true;
        task_timer_ = float(wp(4));
    }
    waypoints_.pop_front();
    auto goal = (waypoints_.size() > 0) ? wp_no_timer : start;

    // Initialise the horzion in the direction of the goal, at a distance T_HORIZON * MAX_SPEED from the start.
    Eigen::VectorXd start2goal = goal - start;
    Eigen::VectorXd horizon = start + std::min(start2goal.norm(), 1. * globals.T_HORIZON * globals.MAX_SPEED) * start2goal.normalized();

    // Variables representing the planned path are at timesteps which increase in spacing.
    // eg. (so that a span of 10 timesteps as a planning horizon can be represented by much fewer variables)
    std::vector<int> variable_timesteps = getVariableTimesteps(globals.T_HORIZON / globals.T0, globals.LOOKAHEAD_MULTIPLE);
    num_variables_ = variable_timesteps.size();

    /***************************************************************************/
    /* Create Variables with fixed pose priors on start and horizon variables. */
    /***************************************************************************/
    Color var_color = color_;
    double sigma;
    int n = globals.N_DOFS;
    Eigen::VectorXd mu(n);
    Eigen::VectorXd sigma_list(n);

    for (int i = 0; i < num_variables_; i++)
    {
        // Set initial mu and covariance of variable interpolated between start and horizon
        mu = start + (horizon - start) * (float)(variable_timesteps[i] / (float)variable_timesteps.back());
        // Start and Horizon state variables should be 'fixed' during optimisation at a timestep
        sigma = (i == 0 || i == num_variables_ - 1) ? globals.SIGMA_POSE_FIXED : 0.;
        // sigma = (i == 0) ? globals.SIGMA_POSE_FIXED : (i == num_variables_ - 1) ? 1e-2 :  0. ;
        sigma_list.setConstant(sigma);

        // Create variable and add to robot's factor graph
        auto variable = std::make_shared<Variable>(sim->next_vid_++, rid_, mu, sigma_list, robot_radius_, n, variable_timesteps[i]);
        variables_[variable->key_] = variable;
    }

    /***************************************************************************/
    /* Create Dynamics factors between variables */
    /***************************************************************************/
    for (int i = 0; i < num_variables_ - 1; i++)
    {
        // T0 is the timestep between the current state and the first planned state.
        float delta_t = globals.T0 * (variable_timesteps[i + 1] - variable_timesteps[i]);
        std::vector<std::shared_ptr<Variable>> variables{getVar(i), getVar(i + 1)};
        auto factor = std::make_shared<DynamicsFactor>(sim->next_fid_++, rid_, variables, globals.SIGMA_FACTOR_DYNAMICS, Eigen::VectorXd::Zero(globals.N_DOFS), delta_t);

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
    Eigen::VectorXd increment = ((*this)[1]->mu_ - (*this)[0]->mu_) * globals.TIMESTEP / globals.T0;
    // In GBP we do this by modifying the prior on the variable
    // If there is a task active, we don't want to overwrite the strong prior at the task waypoint
    curr_var->change_variable_prior(curr_var->mu_ + increment);
    // Real pose update
    position_ = position_ + increment;

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
    // Prevent horizon from moving beyond a threshold, in case robot is stuck
    Eigen::VectorXd dist_curr_to_horz = horizon->mu_({0, 1}) - getVar(0)->mu_({0, 1});
    if (dist_curr_to_horz.norm() >= globals.MAX_HORIZON_DIST)
        return;

    if (waypoints_.size() > 0)
    {
        auto &wp = waypoints_.front();
        // Check if next waypoint is a task
        next_wp_is_task_ = wp.size() >= 5 && wp(4) > 0.0;

        // Horizon state moves towards the next waypoint.
        // The Horizon state's velocity is capped at MAX_SPEED
        Eigen::VectorXd dist_horz_to_goal = wp({0, 1}) - horizon->mu_({0, 1});
        Eigen::VectorXd new_vel = dist_horz_to_goal.normalized() * std::min((double)globals.MAX_SPEED, dist_horz_to_goal.norm());
        Eigen::VectorXd new_pos = horizon->mu_({0, 1}) + new_vel * globals.TIMESTEP;

        // Update horizon state with new pos and vel
        Eigen::VectorXd new_mu(4);
        new_mu << new_pos, new_vel;
        horizon->change_variable_prior(new_mu);

        // If the horizon has reached the waypoint, handle task or normal waypoint
        if (dist_horz_to_goal.norm() < robot_radius_)
        {
            if (next_wp_is_task_)
            {
                task_active_ = true;
                // Snap horizon state to task waypoint
                Eigen::VectorXd new_mu(4);
                new_mu << wp.head<2>(0), Eigen::VectorXd::Zero(2);
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

        // Create the inter-robot factor
        Eigen::VectorXd z = Eigen::VectorXd::Zero(variables.front()->n_dofs_);
        auto factor = std::make_shared<InterrobotFactor>(sim_->next_fid_++, this->rid_, variables, globals.SIGMA_FACTOR_INTERROBOT, z, 0.5 * (this->robot_radius_ + other_robot->robot_radius_));
        factor->other_rid_ = other_robot->rid_;
        // Add factor the the variable's list of factors, as well as to the robot's list of factors
        for (auto var : factor->variables_)
            var->add_factor(factor);
        this->factors_[factor->key_] = factor;
    }

    // Add the other robot to this robot's list of connected robots.
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
        if (sim_->obstacles_.find(oid) == sim_->obstacles_.end())
        {
            to_remove.push_back(oid);
        }
        else
        {
            // Also remove factors for obstacles that have moved too far away
            auto obs = sim_->obstacles_.at(oid);
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
        auto do_fac = std::make_shared<DynamicObstacleFactor>(sim_->next_fid_++, rid_, variables, globals.SIGMA_FACTOR_DYNAMIC_OBSTACLE,
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
    Color col = (interrobot_comms_active_) ? color_ : GRAY;
    // Draw planned path
    if (globals.DRAW_PATH)
    {
        static int debug = 0;
        for (auto [vid, variable] : variables_)
        {
            if (!variable->valid_)
                continue;
            DrawSphere(Vector3{(float)variable->mu_(0), height_3D_, (float)variable->mu_(1)}, 0.5 * robot_radius_, ColorAlpha(col, 0.5));
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
            DrawCubeV(Vector3{(float)waypoints_[wp_idx](0), height_3D_, (float)waypoints_[wp_idx](1)}, Vector3{1.f * robot_radius_, 1.f * robot_radius_, 1.f * robot_radius_}, col);
        }
    }

    // Draw all factor connections
    if (globals.DRAW_FACTORS)
    {
        for (auto [fid, factor] : factors_)
            factor->draw();
    }
    // Draw the actual position of the robot. This uses the robotModel defined in Graphics.cpp, others can be used.
    DrawModel(sim_->graphics->robotModel_, Vector3{(float)position_(0), height_3D_, (float)position_(1)}, robot_radius_, col);
};