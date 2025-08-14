/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <iostream>
#include <gbp/GBPCore.h>
#include <Simulator.h>
#include <DynamicObstacle.h>
#include <Graphics.h>
#include <Robot.h>
#include <GeometryUtils.h>
#include "Metrics.hpp"
#include <nanoflann.h>
#include "cnpy/cnpy.h"
#include <filesystem>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

/*******************************************************************************/
// Raylib setup
/*******************************************************************************/
Simulator::Simulator()
{
    SetTraceLogLevel(LOG_ERROR);
    if (globals.DISPLAY)
    {
        SetTargetFPS(60);
        InitWindow(globals.SCREEN_SZ, globals.SCREEN_SZ, globals.WINDOW_TITLE);
    }

    // Initialise kdtree for storing robot positions (needed for nearest neighbour check)
    treeOfRobots_ = new KDTree(2, robot_positions_, 50);

    // For display only
    // User inputs an obstacle image where the obstacles are BLACK and background is WHITE.
    obstacleImg = LoadImage(globals.OBSTACLE_FILE.c_str());
    if (obstacleImg.width == 0)
        obstacleImg = GenImageColor(globals.WORLD_SZ, globals.WORLD_SZ, WHITE);

    // However for calculation purposes the image needs to be inverted.
    ImageColorInvert(&obstacleImg);
    graphics = new Graphics(obstacleImg);

    // Initialise helper classes
    scheduler = new TaskScheduler();  // Schedules tasks for robots/ obstacles    
    metrics = new MetricsCollector(); // Collects evaluation metrics

};

/*******************************************************************************/
// Destructor
/*******************************************************************************/
Simulator::~Simulator()
{
    delete treeOfRobots_;
    int n = robots_.size();
    for (int i = 0; i < n; ++i)
        robots_.erase(i);
    if (globals.DISPLAY)
    {
        delete graphics;
        CloseWindow();
    }
    delete scheduler;
    delete metrics;
};

/*******************************************************************************/
// Drawing graphics.
/*******************************************************************************/
void Simulator::draw()
{
    if (!globals.DISPLAY)
        return;

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(graphics->camera3d);
    // Draw Ground
    DrawModel(graphics->groundModel_, graphics->groundModelpos_, 1., WHITE);
    
    // Draw spawn zones if in eval mode
    if (globals.EVAL && !spawn_zones_.empty()) {
        // Pale yellow color with transparency
        Color spawn_zone_color = ColorAlpha(YELLOW, 0.3f);
        float zone_height = 0.1f; // Slightly above ground to avoid z-fighting
        
        for (const auto& zone : spawn_zones_) {
            // Get zone properties (all zones have angle = 0, so no rotation needed)
            Vector3 position = {(float)zone.center.x(), zone_height, (float)zone.center.y()};
            Vector3 size = {(float)(zone.halfExtents.x() * 2.0), 0.2f, (float)(zone.halfExtents.y() * 2.0)};
            
            // Simply draw the spawn zone as an axis-aligned box
            DrawCube(position, size.x, size.y, size.z, spawn_zone_color);
        }
    }
    
    // Draw Dynamic Obstacles
    for (auto [oid, obs] : obstacles_)
        obs->draw();
    // Draw Robots
    for (auto [rid, robot] : robots_)
        robot->draw();
    EndMode3D();
    draw_info(clock_);
    EndDrawing();
};

/*******************************************************************************/
// Timestep loop of simulator.
/*******************************************************************************/
void Simulator::timestep()
{

    if (!(globals.SIM_MODE == Timestep || globals.SIM_MODE == Iterate))
        return;

    // Create and/or destory factors depending on a robot's neighbours
    calculateRobotNeighbours(robots_);
    for (auto [r_id, robot] : robots_)
    {
        robot->updateInterrobotFactors();
        if (globals.USE_DYNAMIC_OBS_FAC) robot->updateDynamicObstacleFactors();
    }

    // If the communications failure rate is non-zero, activate/deactivate robot comms
    setCommsFailure(globals.COMMS_FAILURE_RATE);

    // Perform iterations of GBP. Ideally the internal and external iterations
    // should be interleaved better. Here it is assumed there are an equal number.
    for (int i = 0; i < globals.NUM_ITERS; i++)
    {
        iterateGBP(1, INTERNAL, robots_);
        iterateGBP(1, EXTERNAL, robots_);
    }

    // Update (move) obstacles by one timestep
    for (auto [oid, obs] : obstacles_)
    {
        obs->updateObstacleState();
    }

    // Update the robot current and horizon states by one timestep
    for (auto [r_id, robot] : robots_)
    {
        robot->updateCurrent();
        robot->updateHorizon();
    }

    // Evlaution: Add robot tracks
    if (globals.EVAL) {
        for (const auto& [rid, robot] : robots_) {
            metrics->addSample(rid, clock_ * globals.TIMESTEP, robot->position_.head<2>());
        }
        // Detect and log collisions
        detectCollisions();
    }

    // Increase simulation clock by one timestep
    clock_++;
    if (clock_ * globals.TIMESTEP >= std::min(globals.MAX_TIME, globals.MAX_TIMESTEP * globals.TIMESTEP))
    {
        print("Maximum run time reached, exiting...");
        globals.RUN = false;
    }

    if (globals.SIM_MODE == Iterate && clock_ % globals.ITERATE_STEPS == 0)
    {
        globals.SIM_MODE = SimNone;
        // for (auto [r_id, robot] : robots_) robot->print_graph_info();
    }
};

/*******************************************************************************/
// Use a kd-tree to perform a radius search for neighbours of a robot within comms. range
// (Updates the neighbours_ of a robot)
/*******************************************************************************/
void Simulator::calculateRobotNeighbours(std::map<int, std::shared_ptr<Robot>> &robots)
{
    for (auto [rid, robot] : robots)
    {
        robot_positions_.at(rid) = std::vector<double>{robot->position_(0), robot->position_(1)};
    }
    treeOfRobots_->index->buildIndex();

    for (auto [rid, robot] : robots)
    {
        // Find nearest neighbors in radius
        robot->neighbours_.clear();
        std::vector<double> query_pt = std::vector<double>{robots[rid]->position_(0), robots[rid]->position_(1)};
        const float search_radius = pow(globals.COMMUNICATION_RADIUS, 2.);
        std::vector<nanoflann::ResultItem<size_t, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = true;
        const size_t nMatches = treeOfRobots_->index->radiusSearch(&query_pt[0], search_radius, matches, params);
        for (size_t i = 0; i < nMatches; i++)
        {
            auto it = robots_.begin();
            std::advance(it, matches[i].first);
            if (it->first == rid)
                continue;
            robot->neighbours_.push_back(it->first);
        }
    }
};

/*******************************************************************************/
// Set a proportion of robots to not perform inter-robot communications
/*******************************************************************************/
void Simulator::setCommsFailure(float failure_rate)
{
    if (failure_rate == 0)
        return;
    // Get all the robot ids and then shuffle them
    std::vector<int> range{};
    for (auto &[rid, robot] : robots_)
        range.push_back(rid);
    std::shuffle(range.begin(), range.end(), rng);
    // Set a proportion of the robots as inactive using their interrobot_comms_active_ flag.
    int num_inactive = round(failure_rate * robots_.size());
    for (int i = 0; i < range.size(); i++)
    {
        robots_.at(range[i])->interrobot_comms_active_ = (i >= num_inactive);
    }
}

/*******************************************************************************/
// Handles keypresses and mouse input, and updates camera.
/*******************************************************************************/
void Simulator::eventHandler()
{
    // Deal with Keyboard key press
    int key = GetKeyPressed();
    switch (key)
    {
    case KEY_ESCAPE:
        globals.RUN = false;
        break;
    case KEY_H:
        globals.LAST_SIM_MODE = (globals.SIM_MODE == Help || globals.SIM_MODE == SimNone) ? globals.LAST_SIM_MODE : globals.SIM_MODE;
        globals.SIM_MODE = (globals.SIM_MODE == Help) ? globals.LAST_SIM_MODE : Help;
        break;
    case KEY_SPACE:
        graphics->camera_transition_ = !graphics->camera_transition_;
        break;
    case KEY_P:
        globals.DRAW_PATH = !globals.DRAW_PATH;
        break;
    case KEY_R:
        globals.DRAW_INTERROBOT = !globals.DRAW_INTERROBOT;
        break;
    case KEY_W:
        globals.DRAW_WAYPOINTS = !globals.DRAW_WAYPOINTS;
        break;
    case KEY_O:
        globals.DRAW_OBSTACLES = !globals.DRAW_OBSTACLES;
        break;
    case KEY_F:
        globals.DRAW_FACTORS = !globals.DRAW_FACTORS;
        break;
    case KEY_I:
        if (globals.SIM_MODE == SimNone && globals.LAST_SIM_MODE == Iterate)
        {
            globals.SIM_MODE = Timestep;
            globals.LAST_SIM_MODE = Timestep;
        }
        else
        {
            globals.LAST_SIM_MODE = (globals.SIM_MODE == Iterate) ? globals.LAST_SIM_MODE : Iterate;
            globals.SIM_MODE = (globals.SIM_MODE == Iterate) ? globals.LAST_SIM_MODE : Iterate;
        }
        break;
    case KEY_ENTER:
        globals.SIM_MODE = (globals.SIM_MODE == Timestep) ? SimNone : globals.LAST_SIM_MODE;
        break;
    default:
        break;
    }

    // Mouse input handling
    Ray ray = GetMouseRay(GetMousePosition(), graphics->camera3d);
    Vector3 mouse_gnd = Vector3Add(ray.position, Vector3Scale(ray.direction, -ray.position.y / ray.direction.y));
    Vector2 mouse_pos{mouse_gnd.x, mouse_gnd.z}; // Position on the ground plane
    // Do stuff with mouse here using mouse_pos .eg:
    // if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
    //     do_code
    // }

    // Update the graphics if the camera has moved
    graphics->update_camera();
}

/*******************************************************************************/
// Create new robots if needed. Handles deletion of robots out of bounds.
// New formations must modify the vectors "robots to create" and optionally "robots_to_delete"
// by appending (push_back()) a shared pointer to a Robot class.
/*******************************************************************************/
void Simulator::createOrDeleteRobots()
{
    if (!new_robots_needed_ || !(globals.SIM_MODE == Iterate || globals.SIM_MODE == Timestep))
        return;

    std::vector<std::shared_ptr<Robot>> robots_to_create{};
    std::vector<std::shared_ptr<Robot>> robots_to_delete{};
    Eigen::VectorXd starting, turning, ending; // Waypoints : [x,y,xdot,ydot].
    double bound = double(globals.WORLD_SZ / 2);

    if (globals.FORMATION == "playground")
    {
        new_robots_needed_ = globals.NEW_ROBOTS_NEEDED;
        std::deque<Eigen::VectorXd> wps1{
            Eigen::VectorXd{{-20.0, -0.5, globals.MAX_SPEED * 1.0, 0.0, 0.0}},
            Eigen::VectorXd{{20.0, -0.5, globals.MAX_SPEED * 1.0, 0.0, 0.0}}};
        robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, wps1, RobotType::CAR, 1.f, globals.ROBOT_RADIUS, GREEN));

        std::deque<Eigen::VectorXd> wps2{
            Eigen::VectorXd{{20.0, 0.5, globals.MAX_SPEED * -1.0, 0.0, 0.0}},
            Eigen::VectorXd{{-20.0, 0.5, globals.MAX_SPEED * -1.0, 0.0, 0.0}}};
        robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, wps2, RobotType::CAR, 1.f, globals.ROBOT_RADIUS, RED));
    }

    else if (globals.FORMATION == "layered_walls")
    {
        // Robots in a scenario with a layers of moving walls
        new_robots_needed_ = true;
        if (clock_ % 200 == 0)
        {
            int n_roads = globals.NUM_ROBOTS;
            double lane_width = globals.WORLD_SZ / (n_roads + 1);
            double lane_offset_x;
            for (int i = 1; i < n_roads + 1; ++i)
            {
                lane_offset_x = -globals.WORLD_SZ / 2. + i * lane_width;
                starting = Eigen::VectorXd{{lane_offset_x, -globals.WORLD_SZ / 2., 0., 1. * globals.MAX_SPEED, 0.}};
                ending = Eigen::VectorXd{{lane_offset_x, globals.WORLD_SZ / 2., 0., 1. * globals.MAX_SPEED, 0.}};
                std::deque<Eigen::VectorXd> waypoints{starting, ending};
                float robot_radius = globals.ROBOT_RADIUS;
                Color robot_color = DARKGREEN;
                robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, RobotType::SPHERE, 1.f, robot_radius, robot_color));
            }
        }

        // Delete robots if out of bounds
        for (auto [rid, robot] : robots_)
        {
            if (abs(robot->position_(0)) >= globals.WORLD_SZ / 2 || abs(robot->position_(1)) >= globals.WORLD_SZ / 2)
            {
                robots_to_delete.push_back(robot);
            }
        }
    }

    else if (globals.FORMATION == "circle")
    {
        // Robots must travel to opposite sides of circle
        new_robots_needed_ = false;
        float min_circumference_spacing = 5. * globals.ROBOT_RADIUS;
        double min_radius = 0.25 * globals.WORLD_SZ;
        Eigen::VectorXd centre{{0., 0., 0., 0., 0.}};
        for (int i = 0; i < globals.NUM_ROBOTS; i++)
        {
            // Select radius of large circle to be at least min_radius,
            // Also ensures that robots in the circle are at least min_circumference_spacing away from each other
            float radius_circle = (globals.NUM_ROBOTS == 1) ? min_radius : std::max(min_radius, sqrt(min_circumference_spacing / (2. - 2. * cos(2. * PI / (double)globals.NUM_ROBOTS))));
            Eigen::VectorXd offset_from_centre(5);
            offset_from_centre << radius_circle * cos(2. * PI * i / (float)globals.NUM_ROBOTS), radius_circle * sin(2. * PI * i / (float)globals.NUM_ROBOTS), 0., 0., 0.;
            starting = centre + offset_from_centre;
            ending = centre - offset_from_centre;
            std::deque<Eigen::VectorXd> waypoints{starting, ending};

            // Define robot radius and colour here.
            float robot_radius = globals.ROBOT_RADIUS;
            Color robot_color = ColorFromHSV(i * 360. / (float)globals.NUM_ROBOTS, 1., 0.75);
            robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, RobotType::SPHERE, 1.f, robot_radius, robot_color));
        }
    }

    else if (globals.FORMATION == "junction_twoway")
    {
        // Robots in a two-way junction, turning LEFT (RED), RIGHT (BLUE) or STRAIGHT (GREEN)
        new_robots_needed_ = globals.NEW_ROBOTS_NEEDED; // This is needed so that more robots can be created as the simulation progresses.
        
        // Define constants
        const int n_roads = 4, n_lanes = 2;
        const double lane_width = 4. * globals.ROBOT_RADIUS;

        static bool spawn_zones_initialized = false;
        if (!spawn_zones_initialized) {
            
            
            spawn_zones_initialized = true;
        }

        // Predefine the four finish lines at junction exits
        static std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> finish_lines;
        static bool initialised = false;
        if (!initialised) {
            double junction_exit_dist = 50.0; // Distance from center where finish line is placed
            double line_half_width = n_lanes * lane_width;
            
            // Reference: road 0 going from left to right. Must retain this order.
            // turn = 0 (left), road 1
            finish_lines.push_back({
                Eigen::Vector2d(-line_half_width, -junction_exit_dist),
                Eigen::Vector2d( line_half_width, -junction_exit_dist)
            });
            
            // turn = 1 (straight), road 2
            finish_lines.push_back({
                Eigen::Vector2d(junction_exit_dist, -line_half_width),
                Eigen::Vector2d(junction_exit_dist,  line_half_width)
            });
            
            // turn = 2 (right), road 3
            finish_lines.push_back({
                Eigen::Vector2d(-line_half_width, junction_exit_dist),
                Eigen::Vector2d( line_half_width, junction_exit_dist)
            });

            // Road 0: near start
            finish_lines.push_back({
                Eigen::Vector2d(-junction_exit_dist, -line_half_width),
                Eigen::Vector2d(-junction_exit_dist,  line_half_width)
            });

            // Create spawn zones at each road entrance
            const double zone_length = 15.0;  // Length of spawn zone along the road
            const double zone_width = 2.0 * n_lanes * lane_width;  // Width to cover all lanes, on both roads
            const double bound_offset = 10.0;   // Slight offset beyond world bounds
            const double spawn_offset = globals.WORLD_SZ / 2.0 - zone_length / 2.0 + bound_offset;
            
            // Road 0: Left entrance (horizontal)
            spawn_zones_.emplace_back(
                Eigen::Vector2d(-spawn_offset, 0.0),
                Eigen::Vector2d(zone_length * 0.5, zone_width * 0.5),
                0.0
            );
            
            // Road 1: Top entrance (vertical)  
            spawn_zones_.emplace_back(
                Eigen::Vector2d(0.0, -spawn_offset),
                Eigen::Vector2d(zone_width * 0.5, zone_length * 0.5),
                0.0
            );
            
            // Road 2: Right entrance (horizontal)
            spawn_zones_.emplace_back(
                Eigen::Vector2d(spawn_offset, 0.0),
                Eigen::Vector2d(zone_length * 0.5, zone_width * 0.5),
                0.0
            );
            
            // Road 3: Bottom entrance (vertical)
            spawn_zones_.emplace_back(
                Eigen::Vector2d(0.0, spawn_offset),
                Eigen::Vector2d(zone_width * 0.5, zone_length * 0.5),
                0.0
            );

            initialised = true;
        }
        
        if (clock_ % 20 == 0)
        { // Arbitrary condition on the simulation time to create new robots
            
            int road = random_int(0, n_roads - 1);
            int lane = random_int(0, n_lanes - 1);
            int turn = random_int(0, 2); // 0 = left, 1 = straight, 2 = right

            Eigen::Matrix<double, 5, 5> rot = Eigen::Matrix<double, 5, 5>::Identity();
            double angle = PI / 2. * road;
            double c = std::cos(angle);
            double s = std::sin(angle);
            rot.block<2, 2>(0, 0) << c, -s, s, c;
            rot.block<2, 2>(2, 2) << c, -s, s, c;
           
            double lane_v_offset = (0.5 * (1 - 2. * n_lanes) + lane) * lane_width + 1.0;
            double lane_h_offset = (1 - turn) * (0.5 + lane - n_lanes) * lane_width;

            starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2., lane_v_offset, globals.MAX_SPEED, 0., 0.}};
            turning = rot * Eigen::VectorXd{{lane_h_offset, lane_v_offset, (turn % 2) * globals.MAX_SPEED, (turn - 1) * globals.MAX_SPEED, 0.}};
            ending = rot * Eigen::VectorXd{{lane_h_offset + (turn % 2) * globals.WORLD_SZ * 1., lane_v_offset + (turn - 1) * globals.WORLD_SZ * 1., 0., 0., 0.}};
            std::deque<Eigen::VectorXd> waypoints{starting, turning, ending};
            
            Color robot_color = globals.EVAL ? BLUE : ColorFromHSV(turn * 120., 1., 0.75);
            
            auto robot = std::make_shared<Robot>(this, next_rid_++, waypoints, RobotType::SPHERE, 1.f, globals.ROBOT_RADIUS, robot_color);
            robots_to_create.push_back(robot);
            
            // Set up metrics for this robot
            if (globals.EVAL) {
                metrics->setBaselinePathLength(robot->rid_, robot->base_path_length_);
                int finish_idx = (turn + road) % 4;
                metrics->setFinishLine(robot->rid_, finish_lines[finish_idx].first, finish_lines[finish_idx].second);
            }
        }

        // Delete robots if out of bounds
        for (auto [rid, robot] : robots_)
        {
            if (abs(robot->position_(0)) > globals.WORLD_SZ / 2 || abs(robot->position_(1)) > globals.WORLD_SZ / 2)
            {
                robots_to_delete.push_back(robot);
            }
        }
    }
    
    else
    {
        // Define new formations here!
    }

    // Create and/or delete the robots as necessary.
    for (auto robot : robots_to_create)
    {
        robot_positions_[robot->rid_] = std::vector<double>{robot->waypoints_[0](0), robot->waypoints_[0](1)};
        robots_[robot->rid_] = robot;
        
        // Add initial sample for metrics tracking
        if (globals.EVAL) {
            metrics->addSample(robot->rid_, clock_ * globals.TIMESTEP, robot->position_.head<2>());
        }
    };
    for (auto robot : robots_to_delete)
    {
        deleteRobot(robot);
    };
};

/*******************************************************************************/
// Create new dynamic (moving/static) obstacles. Handles deletion of obstacles out of bounds.
// Works in a similar fashion to robot creation and deletion, but for dynamic obstacles.
/*******************************************************************************/
void Simulator::createOrDeleteObstacles()
{
    if (!new_obstacles_needed_ || !(globals.SIM_MODE == Iterate || globals.SIM_MODE == Timestep))
        return;

    std::vector<std::shared_ptr<DynamicObstacle>> obs_to_create{};
    std::vector<int> obs_to_delete{};

    if (globals.FORMATION == "playground")
    {
        new_obstacles_needed_ = globals.NEW_OBSTACLES_NEEDED;
        std::deque<Eigen::VectorXd> wps;
        Eigen::VectorXd wp1(5), wp2(5), wp3(5);
        wp1 << -10., 0., 1., 0., 0.;
        wp2 << 0., 0., 1., 0., 10.;
        wp3 << 0., -10., 0., -1., 0.;
        wps = {wp1, wp2, wp3};
        auto model = graphics->obstacleModels_[ObstacleType::BUS];
        // auto model = graphics->createBoxObstacleModel(5.f, 5.f, 5.f, 0.0);
        auto obs = std::make_shared<DynamicObstacle>(next_oid_++, wps, model);
        obs_to_create.push_back(obs);
    }

    else if (globals.FORMATION == "layered_walls")
    {
        new_obstacles_needed_ = true;
        static bool obs_initialised = false;
        static std::vector<MotionOptions> motion_options;

        // Define obstacle dimensions and path by defining MotionOptions for each obstacle
        if (!obs_initialised)
        {
            // Motion: left to right
            Eigen::VectorXd mo1_wp1(5), mo1_wp2(5);
            mo1_wp1 << -globals.WORLD_SZ / 2.f, -33.f, 2.f, 0.f, 0.f;
            mo1_wp2 << globals.WORLD_SZ / 2.f, -33.f, 2.f, 0.f, 0.f;
            auto mo1 = MotionOptions(5.f, 5.f, 5.f, 2.5f, 300, graphics, std::deque<Eigen::VectorXd>{mo1_wp1, mo1_wp2});
            // Motion: right to left
            Eigen::VectorXd mo2_wp1(5), mo2_wp2(5);
            mo2_wp1 << globals.WORLD_SZ / 2.f, 0.f, -2.5f, 0.f, 0.f;
            mo2_wp2 << -globals.WORLD_SZ / 2.f, 0.f, -2.5f, 0.f, 0.f;
            auto mo2 = MotionOptions(5.f, 5.f, 5.f, 2.5f, 300, graphics, std::deque<Eigen::VectorXd>{mo2_wp1, mo2_wp2});
            // Motion: left to right
            Eigen::VectorXd mo3_wp1(5), mo3_wp2(5);
            mo3_wp1 << -globals.WORLD_SZ / 2.f, 33.f, 1.5f, 0.f, 0.f;
            mo3_wp2 << globals.WORLD_SZ / 2.f, 33.f, 1.5f, 0.f, 0.f;
            auto mo3 = MotionOptions(5.f, 5.f, 5.f, 2.5f, 200, graphics, std::deque<Eigen::VectorXd>{mo3_wp1, mo3_wp2});

            motion_options = {mo1, mo2, mo3};
            obs_initialised = true;
        }

        for (auto &mo : motion_options)
        {
            if (clock_ - mo.last_spawn_time_ > mo.spawn_interval_)
            {
                auto obs = std::make_shared<DynamicObstacle>(next_oid_++, mo.waypoints_, mo.geom_);
                obs->completed_ = false;
                obs_to_create.push_back(obs);
                mo.last_spawn_time_ = clock_;
            }
        }

        for (auto [oid, obs] : obstacles_)
        {
            if (obs->completed_)
                obs_to_delete.push_back(oid);
        }
    }

    else if (globals.FORMATION == "circle")
    {
        new_obstacles_needed_ = false;
        static bool obs_initialised = false;
        static std::vector<MotionOptions> motion_options;

        if (!obs_initialised)
        {
            // Depends on omega
            int size = 100;
            // Since Y increases downwards, omegas need to be negated for anti-clockwise motion
            std::vector<float> radii = {13.1f, 9.2f, 6.7f};
            std::vector<float> phase_offsets = {-std::atan2(10.f, 8.5f), std::atan2(7.f, -6.f), -std::atan2(6.5f, -1.5f)};
            std::vector<float> omegas = {-0.1f, -0.1f, -0.1f};

            // Cuboids1
            Eigen::VectorXd mo1_wp(5), mo2_wp(5), mo3_wp(5);
            mo1_wp << 8.5, -10., 0., 0., 0.;
            mo2_wp << -6.0, 7.0, 0., 0., 0.;
            mo3_wp << -1.5, -6.5, 0., 0., 0.;

            auto mo1 = MotionOptions(3.f, 5.f, 6.f, 2.5f, 0, graphics, std::deque<Eigen::VectorXd>{mo1_wp});
            auto mo2 = MotionOptions(4.f, 4.f, 4.f, 2.0f, 0, graphics, std::deque<Eigen::VectorXd>{mo2_wp});
            auto mo3 = MotionOptions(3.f, 5.f, 3.f, 2.5f, 0, graphics, std::deque<Eigen::VectorXd>{mo3_wp});

            motion_options = {mo1, mo2, mo3};

            for (int i = 0; i < motion_options.size(); ++i)
            {
                std::deque<Eigen::VectorXd> waypoints;
                float omega = (2.f * PI) / size;
                for (int j = 0; j < size; ++j)
                {
                    float theta = phase_offsets[i] + omega * j;
                    float x = radii[i] * std::cos(theta);
                    float y = radii[i] * std::sin(theta);
                    float vx = -radii[i] * omegas[i] * std::sin(theta);
                    float vy = radii[i] * omegas[i] * std::cos(theta);
                    Eigen::VectorXd wp(5);
                    wp << x, y, vx, vy, 0.0; // 5th dimension (pause time) = 0
                    waypoints.emplace_back(wp);
                }
                motion_options[i].waypoints_ = waypoints;
            }
            obs_initialised = true;
        }

        for (auto &mo : motion_options)
        {
            auto obs = std::make_shared<DynamicObstacle>(next_oid_++, mo.waypoints_, mo.geom_);
            obs_to_create.push_back(obs);
            mo.last_spawn_time_ = clock_;
        }
    }

    else if (globals.FORMATION == "junction_twoway")
    {
        new_obstacles_needed_ = globals.NEW_OBSTACLES_NEEDED;

        float now = clock_ * globals.TIMESTEP;
        static bool initialised = false;

        // Define PoissonSpawners for vehicle obstacles
        static std::vector<PoissonSpawner> spawners = {
            {60.0, 3.0, "r0"}, /* Road 0 */
            {75.0, 30.0, "r1"}, /* Road 1 */
            {50.0, 30.0, "r2"}, /* Road 2 */
            {120.0, 30.0, "r3"} /* Road 3 */
        };

        // Set initial next_spawn, only needs to be called once
        if (!initialised)
        {
            for (auto& spawner : spawners) spawner.schedule_from(now);
            initialised = true;
        }

        // Define helper variables
        int n_roads = 4, n_lanes = 2;
        double lane_width = 4. * globals.ROBOT_RADIUS;
        // Define helper lambdas
        auto makeRotationMatrix5 = [](double road) {
            double angle = PI / 2.0 * road;
            double c = std::cos(angle);
            double s = std::sin(angle);
    
            Eigen::Matrix<double, 5, 5> rot = Eigen::Matrix<double, 5, 5>::Identity();
            rot.block<2, 2>(0, 0) << c, -s, s,  c;
            rot.block<2, 2>(2, 2) << c, -s, s,  c;
            return rot;
        };

        auto lane_v_offset = [n_lanes, lane_width](int lane) {return (0.5 * (1 - 2. * n_lanes) + lane) * lane_width + 1.0;};
        auto lane_h_offset = [n_lanes, lane_width](int turn, int lane) {return (1 - turn) * (0.5 + lane - n_lanes) * lane_width;};

        for (int road = 0; road < spawners.size(); ++road) {
            if (spawners[road].try_spawn(now)) {
                // Random turn (0=left, 1=straight, 2=right) and lane
                int turn = random_int(0, 2);
                int lane = random_int(0, n_lanes - 1);

                
                // Get rotation matrix for this road
                Eigen::Matrix<double, 5, 5> rot = makeRotationMatrix5(road);
                
                // Calculate offsets
                double v_offset = lane_v_offset(lane);
                double h_offset = lane_h_offset(turn, lane);
                
                // Define waypoints adjusted for DynamicObstacle's velocity interpolation behavior
                Eigen::VectorXd starting(5), turning(5), ending(5);
                starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2. - 10., v_offset, 1. * globals.MAX_SPEED, 0., 0.}};
                turning = rot * Eigen::VectorXd{{h_offset, v_offset, 1. * globals.MAX_SPEED, 0., 0.}};
                ending = rot * Eigen::VectorXd{{h_offset + (turn % 2) * globals.WORLD_SZ / 2., v_offset + (turn - 1) * globals.WORLD_SZ / 2., (turn % 2) * globals.MAX_SPEED * 1., (turn - 1) * globals.MAX_SPEED * 1., 0.}};
                
                // Create waypoints deque
                std::deque<Eigen::VectorXd> waypoints{starting, turning, ending};

                // Create the obstacle
                auto model = graphics->obstacleModels_[ObstacleType::BUS];
                auto obs = std::make_shared<DynamicObstacle>(next_oid_++, waypoints, model);
                obs_to_create.push_back(obs);
            }
        }

        // Clean up completed obstacles
        for (auto [oid, obs] : obstacles_)
        {
            if (obs->completed_)
                obs_to_delete.push_back(oid);
        }
    }

    // Create or delete obstacles.
    for (auto obs : obs_to_create) {
        obs->spawn_time_ = clock_ * globals.TIMESTEP;  // Set spawn time when creating obstacle
        obstacles_[obs->oid_] = obs;
    }
    for (const int oid : obs_to_delete)
    {
        scheduler->removeQueue(oid);
        obstacles_.erase(oid);
    }
}

/*******************************************************************************/
// Deletes the robot from the simulator's robots_, as well as any variable/factors associated.
/*******************************************************************************/
void Simulator::deleteRobot(std::shared_ptr<Robot> robot)
{
    auto connected_rids_copy = robot->connected_r_ids_;
    for (auto r : connected_rids_copy)
    {
        robot->deleteInterrobotFactors(robots_.at(r));
        robots_.at(r)->deleteInterrobotFactors(robot);
    }
    robots_.erase(robot->rid_);
    robot_positions_.erase(robot->rid_);
}

/*******************************************************************************/
// Detect and log collisions between robots and obstacles
/*******************************************************************************/
void Simulator::detectCollisions()
{
    // Helper function to create a unique key for a pair of IDs
    static auto pairKey = [](int a, int b) -> uint64_t {
        const uint32_t x = static_cast<uint32_t>(a < b ? a : b);
        const uint32_t y = static_cast<uint32_t>(a < b ? b : a);
        return (static_cast<uint64_t>(x) << 32) | y;
    };
    
    // Static set to track seen collision pairs (persists across calls)
    static std::unordered_set<uint64_t> seen_pairs_;
    
    // Helper lambda to check if collision should be ignored due to spawn zone
    // id_j can be negative (indicating obstacle ID)
    auto shouldIgnoreCollision = [&](int rid_i, int id_j, const Eigen::Vector2d& collision_pos) -> bool {
        if (spawn_zones_.empty()) return false;
        
        double current_time = clock_ * globals.TIMESTEP;
        
        // Get age of first entity (always a robot)
        auto track_i = metrics->getTrack(rid_i);
        double age_i = current_time - track_i.t_start;
        
        // Get age of second entity
        double age_j = spawn_zone_time_threshold_ + 1.0;  // Default: old enough
        if (id_j > 0) {  // It's a robot
            auto track_j = metrics->getTrack(id_j);
            age_j = current_time - track_j.t_start;
        } else if (id_j < 0) {  // It's an obstacle (negative ID)
            int oid = -id_j;  // Convert back to positive obstacle ID
            if (obstacles_.count(oid) > 0) {
                age_j = current_time - obstacles_.at(oid)->spawn_time_;
            }
        }
        
        // Check if at least one entity is young enough to be in spawn grace period
        if (age_i > spawn_zone_time_threshold_ && age_j > spawn_zone_time_threshold_) {
            return false;  // Both are old enough, don't ignore collision
        }
        
        // Check if collision position is in any spawn zone
        for (const auto& zone : spawn_zones_) {
            if (GeometryUtils::pointInOBB(collision_pos, zone)) {
                return true;  // Collision is in spawn zone and at least one entity is young
            }
        }
        
        return false;
    };
    
    // Initialization: determine if robots are spheres or OBBs
    bool robots_are_spheres = true;
    if (!robots_.empty()) {
        // Check the first robot's type
        robots_are_spheres = (robots_.begin()->second->robot_type_ == RobotType::SPHERE);
    }
    
    // Small epsilon for numerical tolerance
    const double eps = 1e-6;
    const double dt = globals.TIMESTEP;
    const double current_time = clock_ * dt;
    
    // Loop 1: Robot-Robot collisions
    for (const auto& [rid_i, robot_i] : robots_) {
        // Get robot i's position and velocity
        Eigen::Vector2d pos_i = robot_i->position_.head<2>();
        Eigen::Vector2d vel_i = robot_i->position_.segment<2>(2);
        double radius_i = robot_i->robot_radius_;
        
        // Calculate search radius for potential collisions
        double query_radius = radius_i + dt * vel_i.norm();
        
        // Use KD-tree to find nearby robots
        std::vector<double> query_pt = {pos_i.x(), pos_i.y()};
        const float search_radius_sq = std::pow(query_radius * 2.0, 2.0); // Extra margin for safety
        std::vector<nanoflann::ResultItem<size_t, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        
        const size_t nMatches = treeOfRobots_->index->radiusSearch(
            &query_pt[0], search_radius_sq, matches, params);
        
        for (size_t k = 0; k < nMatches; k++) {
            // Get the matched robot's ID
            auto it = robots_.begin();
            std::advance(it, matches[k].first);
            int rid_j = it->first;
            
            // Skip self
            if (rid_j == rid_i) continue;
            
            // Skip if we've already checked this pair
            uint64_t pair_id = pairKey(rid_i, rid_j);
            if (seen_pairs_.count(pair_id) > 0) continue;
            
            auto robot_j = robots_.at(rid_j);
            Eigen::Vector2d pos_j = robot_j->position_.head<2>();
            Eigen::Vector2d vel_j = robot_j->position_.segment<2>(2);
            double radius_j = robot_j->robot_radius_;
            
            bool collision = false;
            
            if (robots_are_spheres) {
                // Simple sphere-sphere collision
                double center_dist = (pos_j - pos_i).norm();
                double combined_radius = radius_i + radius_j;
                
                // Check if certainly not colliding
                if (center_dist > combined_radius + eps) {
                    continue;  // No collision
                }
                collision = (center_dist <= combined_radius + eps);
            } else {
                // OBB-OBB collision for car-type robots
                // Get orientations (stored in 5th component if DOF=5, otherwise compute from velocity)
                double theta_i = 0.0;
                double theta_j = 0.0;
                
                if (robot_i->dofs_ == 5) {
                    theta_i = robot_i->position_(4);
                } else if (vel_i.norm() > 1e-6) {
                    theta_i = -wrapAngle(std::atan2(vel_i.y(), vel_i.x()));
                }
                theta_i += robot_i->default_angle_offset_;
                
                if (robot_j->dofs_ == 5) {
                    theta_j = robot_j->position_(4);
                } else if (vel_j.norm() > 1e-6) {
                    Eigen::Vector2d vel_j = robot_j->position_.segment<2>(2);
                    theta_j = -wrapAngle(std::atan2(vel_j.y(), vel_j.x()));
                }
                theta_j += robot_j->default_angle_offset_;
                
                // Create OBBs for both robots
                OBB2D obb_i(pos_i, robot_i->robot_dimensions_ * 0.5, theta_i);
                OBB2D obb_j(pos_j, robot_j->robot_dimensions_ * 0.5, theta_j);
                
                collision = GeometryUtils::overlapsOBB(obb_i, obb_j);
            }
            
            if (collision) {
                // Check if collision should be ignored due to spawn zone
                Eigen::Vector2d collision_pos = (pos_i + pos_j) * 0.5;  // Midpoint of collision
                if (!shouldIgnoreCollision(rid_i, rid_j, collision_pos)) {
                    // Mark collision for both robots
                    bool connected = std::find(robot_i->connected_r_ids_.begin(), robot_i->connected_r_ids_.end(), rid_j) != robot_i->connected_r_ids_.end();
                    printf("CollisionEvent<Robot, Robot, %f>: ids=[%d, %d], pos1=[%f, %f], pos2=[%f, %f], comms_active=[%d, %d], connected=[%d]\n", 
                        current_time, rid_i, rid_j, pos_i.x(), pos_i.y(), pos_j.x(), pos_j.y(), (int)robot_i->interrobot_comms_active_, (int)robot_j->interrobot_comms_active_,
                        (int)connected
                    );
                    globals.LAST_SIM_MODE = Iterate;
                    globals.SIM_MODE = SimNone;
                    seen_pairs_.insert(pair_id);
                    metrics->markCollision(rid_i, current_time);
                    metrics->markCollision(rid_j, current_time);
                }
            }
        }
    }
    
    // Loop 2: Robot-Obstacle collisions
    for (const auto& [oid, obstacle] : obstacles_) {
        // Get obstacle's current state
        Eigen::Vector2d obs_pos = obstacle->state_.head<2>();
        double obs_theta = obstacle->orientation_;
        
        // Get obstacle dimensions from its geometry
        if (!obstacle->geom_) continue;  // Skip if no geometry
        
        // Create OBB for the obstacle
        // Obstacle dimensions are stored in the model info (x=width, z=depth for 2D)
        float width = obstacle->geom_->dimensions.x;
        float depth = obstacle->geom_->dimensions.z;
        Eigen::Vector2d obs_half_extents(width * 0.5, depth * 0.5);
        OBB2D obs_obb(obs_pos, obs_half_extents, obs_theta);
        
        // Calculate a bounding radius for initial culling
        double obs_bounding_radius = obs_obb.getBoundingRadius();
        
        // Check each robot against this obstacle
        for (const auto& [rid, robot] : robots_) {
            Eigen::Vector2d robot_pos = robot->position_.head<2>();
            double robot_radius = robot->robot_radius_;
            
            // Quick distance check for culling
            double center_dist = (obs_pos - robot_pos).norm();
            if (center_dist > obs_bounding_radius + robot_radius + eps) {
                continue;  // Too far apart
            }
            
            bool collision = false;
            
            if (robots_are_spheres) {
                // Sphere-OBB collision
                collision = GeometryUtils::overlapsSphereOBB(
                    robot_pos, robot_radius, obs_obb, eps);
            } else {
                // OBB-OBB collision
                double robot_theta = 0.0;
                if (robot->dofs_ == 5) {
                    robot_theta = robot->position_(4);
                } else {
                    Eigen::Vector2d vel = robot->position_.segment<2>(2);
                    if (vel.norm() > 1e-6) {
                        robot_theta = -wrapAngle(std::atan2(vel.y(), vel.x()));
                    }
                }
                robot_theta += robot->default_angle_offset_;
                
                OBB2D robot_obb(robot_pos, robot->robot_dimensions_ * 0.5, robot_theta);
                collision = GeometryUtils::overlapsOBB(robot_obb, obs_obb);
            }
            
            if (collision) {
                // Create unique key for robot-obstacle pair
                // Use negative oid to distinguish from robot-robot pairs
                uint64_t pair_id = pairKey(rid, -oid);
                
                if (seen_pairs_.count(pair_id) == 0) {
                    // Check if collision should be ignored due to spawn zone
                    if (!shouldIgnoreCollision(rid, -oid, robot_pos)) {
                        printf("CollisionEvent<Robot, Obstacle, %f>: ids=[%d, %d], pos=[%f, %f]\n", current_time, rid, oid, robot_pos.x(), robot_pos.y());
                        globals.LAST_SIM_MODE = Iterate;
                        globals.SIM_MODE = SimNone;
                        seen_pairs_.insert(pair_id);
                        metrics->markCollision(rid, current_time);
                    }
                }
            }
        }
    }
}
