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
// Simulator setup (window initialization moved to main.cpp)
/*******************************************************************************/
Simulator::Simulator()
{
    // Reset sim state
    robots_initialised_ = false;
    obstacles_initialised_ = false;
    next_rid_ = 0;
    next_oid_ = 0;
    next_fid_ = 0;
    next_vid_ = 0;
    clock_ = 0;
    metrics = nullptr;
    
    seen_collision_pairs_.clear();
    motion_options_.clear();
    spawn_gates_.clear();
    robot_spawners_.clear();
    bus_spawners_.clear();
    van_spawners_.clear();
    
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
};

/*******************************************************************************/
// Destructor (window cleanup moved to main.cpp)
/*******************************************************************************/
Simulator::~Simulator()
{
    delete treeOfRobots_;
    
    int n = robots_.size();
    for (int i = 0; i < n; ++i)
        robots_.erase(i);
    
    obstacles_.clear();
    
    if (globals.DISPLAY)
    {
        delete graphics;
    }
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
    
    updateRobotKDTree(robots_);
    processSpawnRequests();
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
        iterateGBP(50, INTERNAL, robots_);
        iterateGBP(10, EXTERNAL, robots_);
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

    // Evaluation: Add robot tracks and sample obstacle counts
    if (globals.EVAL && metrics) {
        double current_time = clock_ * globals.TIMESTEP;
        
        for (const auto& [rid, robot] : robots_) {
            metrics->addSample(rid, current_time, robot->position_.head<2>());
        }
        
        // Sample live obstacle counts per type
        std::unordered_map<ObstacleType, int> live_counts;
        for (const auto& [oid, obs] : obstacles_) {
            live_counts[obs->obstacle_type_]++;
        }
        metrics->updateObstacleSampling(current_time, live_counts);
        
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
// Admit spawn requests for robots and obstacles jointly
/*******************************************************************************/
void Simulator::processSpawnRequests() {
    double now = clock_ * globals.TIMESTEP;

    for (auto& gate : spawn_gates_) {
        gate.process(now, 
            [this](const SpawnRequest& r) { return this->admitSpawnRequest(r); },
            [this](const SpawnRequest& r, double margin) { return this->isSpawnClear(r, margin); }
        );
    }
}

/*******************************************************************************/
// Update robot positions and reindx KD-Tree
/*******************************************************************************/    
void Simulator::updateRobotKDTree(std::map<int, std::shared_ptr<Robot>> &robots){
    for (auto [rid, robot] : robots)
    {
        robot_positions_.at(rid) = std::vector<double>{robot->position_(0), robot->position_(1)};
    }
    treeOfRobots_->index->buildIndex();
}

/*******************************************************************************/
// Use a kd-tree to perform a radius search for neighbours of a robot within comms. range
// (Updates the neighbours_ of a robot)
/*******************************************************************************/
void Simulator::calculateRobotNeighbours(std::map<int, std::shared_ptr<Robot>> &robots)
{
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
    case KEY_M:
        globals.DRAW_ROBOTS = !globals.DRAW_ROBOTS;
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
// Set up environment related structures based on formation
/*******************************************************************************/
void Simulator::setupEnvironment() {
    if (globals.FORMATION == "junction_twoway") {
        const int n_zones = 4; // One zone for each road
        spawn_gates_.clear();
        for (int i = 0; i < n_zones; ++i) {
            SpawnGate g; g.zone_id = i;
            g.min_headway_s = 0.6;
            g.space_margin = 0.3;
            spawn_gates_.push_back(g);
        }
        
        // Initialize spawners for junction_twoway formation
        robot_spawners_ = {
            {globals.ROBOT_SPAWN_MEAN_RATE, globals.ROBOT_SPAWN_MIN_HEADWAY, "robot_r0", globals.VERBOSE},
            {globals.ROBOT_SPAWN_MEAN_RATE, globals.ROBOT_SPAWN_MIN_HEADWAY, "robot_r1", globals.VERBOSE},
            {globals.ROBOT_SPAWN_MEAN_RATE, globals.ROBOT_SPAWN_MIN_HEADWAY, "robot_r2", globals.VERBOSE},
            {globals.ROBOT_SPAWN_MEAN_RATE, globals.ROBOT_SPAWN_MIN_HEADWAY, "robot_r3", globals.VERBOSE},
        };
        
        bus_spawners_ = {
            {globals.BUS_SPAWN_MEAN_RATE, globals.BUS_SPAWN_MIN_HEADWAY, "bus_r0", globals.VERBOSE},
            {globals.BUS_SPAWN_MEAN_RATE, globals.BUS_SPAWN_MIN_HEADWAY, "bus_r1", globals.VERBOSE},
            {globals.BUS_SPAWN_MEAN_RATE, globals.BUS_SPAWN_MIN_HEADWAY, "bus_r2", globals.VERBOSE},
            {globals.BUS_SPAWN_MEAN_RATE, globals.BUS_SPAWN_MIN_HEADWAY, "bus_r3", globals.VERBOSE}
        };
        
        van_spawners_ = {
            {globals.VAN_SPAWN_MEAN_RATE, globals.VAN_SPAWN_MIN_HEADWAY, "van_r0", globals.VERBOSE}
        };
        
        pedestrian_spawner_ = PoissonSpawner{globals.PEDESTRIAN_SPAWN_MEAN_RATE, globals.PEDESTRIAN_SPAWN_MIN_HEADWAY, "pedestrians", globals.VERBOSE};

        if (globals.EVAL) {
            // Set up metrics
            const double lane_width = 4. * globals.ROBOT_RADIUS;
            const double n_lanes = 2;
            const double line_half_width = n_lanes * lane_width;
            const double flow_meter_dist = 40.0;
            // Define flow meters for each direction (NS/SN = North-South/South-North, WE/EW = West-East/East-West)
            FlowMeter flow_in_we(Eigen::Vector2d(-flow_meter_dist, -line_half_width), Eigen::Vector2d(-flow_meter_dist, line_half_width));
            FlowMeter flow_out_we(Eigen::Vector2d(flow_meter_dist, -line_half_width), Eigen::Vector2d(flow_meter_dist, line_half_width));
            FlowMeter flow_in_ns(Eigen::Vector2d(-line_half_width, -flow_meter_dist), Eigen::Vector2d(line_half_width, -flow_meter_dist));
            FlowMeter flow_out_ns(Eigen::Vector2d(-line_half_width, flow_meter_dist), Eigen::Vector2d(line_half_width, flow_meter_dist));
            FlowMeter flow_in_ew(Eigen::Vector2d(flow_meter_dist, -line_half_width), Eigen::Vector2d(flow_meter_dist, line_half_width));
            FlowMeter flow_out_ew(Eigen::Vector2d(-flow_meter_dist, -line_half_width), Eigen::Vector2d(-flow_meter_dist, line_half_width));
            FlowMeter flow_in_sn(Eigen::Vector2d(line_half_width, flow_meter_dist), Eigen::Vector2d(-line_half_width, flow_meter_dist));
            FlowMeter flow_out_sn(Eigen::Vector2d(line_half_width, -flow_meter_dist), Eigen::Vector2d(-line_half_width, -flow_meter_dist));
            
            double simulation_start_time = clock_ * globals.TIMESTEP;
            metrics = new MetricsCollector(simulation_start_time, flow_in_ns, flow_out_ns, flow_in_we, flow_out_we, flow_in_sn, flow_out_sn, flow_in_ew, flow_out_ew);
            metrics->setWarmupTime(double(globals.WARMUP_TIME));

            std::vector<ObstacleType> obs_types = {ObstacleType::BUS, ObstacleType::VAN, ObstacleType::PEDESTRIAN};
            std::unordered_map<ObstacleType, double> obs_areas;
            for (const auto& type : obs_types) {
                auto dims = graphics->obstacleModels_[type]->dimensions;
                obs_areas[type] = dims.x * dims.z;
            }
            metrics->setObstacleAreas(obs_areas);
        }
        
    }
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
    
    if (globals.FORMATION == "playground")
    {
        new_robots_needed_ = globals.NEW_ROBOTS_NEEDED;

        std::deque<Eigen::VectorXd> wps1{
            Eigen::VectorXd{{25.0, 0.0, 0.0, -(double)globals.MAX_SPEED, 0.0}},
            Eigen::VectorXd{{-25.0, 0.0, 0.0,-(double)globals.MAX_SPEED, 0.0}}};
        robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, wps1, RobotType::CAR, 1.f, globals.ROBOT_RADIUS, GREEN));

        std::deque<Eigen::VectorXd> wps2{
            Eigen::VectorXd{{-25.0, 0.0, (double)globals.MAX_SPEED, 0.0, 0.0}},
            Eigen::VectorXd{{ 25.0, 0.0, (double)globals.MAX_SPEED, 0.0, 0.0}}};
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
        float now = clock_ * globals.TIMESTEP;
        const auto carModelInfo = graphics->robotModels_[RobotType::CAR];
        const auto dims = carModelInfo->dimensions;
        const Eigen::Vector2d car_he(dims.x * 0.5, dims.z * 0.5);
        const bool robot_are_spheres = globals.N_DOFS == 4;

        if (!robots_initialised_) {
            // Set initial next_spawn, only needs to be called once
            for (auto& spawner : robot_spawners_) spawner.schedule_from(now);
            robots_initialised_ = true;
        }

        for (int road = 0; road < 4; ++road) {
            if (robot_spawners_[road].try_spawn(clock_ * globals.TIMESTEP)) {
                int lane = random_int(0, 1);
                int turn = 1;
                // int turn = random_int(0, 2); // 1=straight for the paper-style flow curve (no turns)

                Eigen::Matrix<double, 5, 5> rot = Eigen::Matrix<double, 5, 5>::Identity();
                double angle = PI / 2. * road;
                double c = std::cos(angle);
                double s = std::sin(angle);
                rot.block<2, 2>(0, 0) << c, -s, s, c;
                rot.block<2, 2>(2, 2) << c, -s, s, c;
            
                double lane_v_offset = (0.5 * (1 - 2. * n_lanes) + lane) * lane_width;
                double lane_h_offset = (1 - turn) * (0.5 + lane - n_lanes) * lane_width;

                starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2., lane_v_offset, globals.MAX_SPEED, 0., 0.}};
                turning = rot * Eigen::VectorXd{{lane_h_offset, lane_v_offset, (turn % 2) * globals.MAX_SPEED, (turn - 1) * globals.MAX_SPEED, 0.}};
                ending = rot * Eigen::VectorXd{{lane_h_offset + (turn % 2) * (globals.WORLD_SZ * 0.7), lane_v_offset + (turn - 1) * globals.WORLD_SZ * 1., 0., 0., 0.}};
                std::deque<Eigen::VectorXd> waypoints{starting, turning, ending};

                SpawnRequest req;
                req.type = SpawnType::Robot;
                req.zone_id = road;
                req.t_req = clock_ * globals.TIMESTEP;
                req.pos = starting.head<2>();
                req.orientation = wrapAngle(-std::atan2(starting(3), starting(2)) + carModelInfo->orientation_offset); // Used for OBB construction only
                req.half_extents = robot_are_spheres ? Eigen::Vector2d::Constant(globals.ROBOT_RADIUS) : car_he; // Used for OBB construction only
                req.waypoints = waypoints;
                req.robot_type = robot_are_spheres ? RobotType::SPHERE : RobotType::CAR;
                req.radius = robot_are_spheres ? globals.ROBOT_RADIUS : car_he.norm();
                req.color = BLUE;
        
                spawn_gates_[road].enqueue(req);
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
        // Eigen::VectorXd wp1(5), wp2(5), wp3(5), wp4(5);
        Eigen::VectorXd wp1(5);
        wp1 << 0., 1.5, 0., -1., 0.;
        // wp2 <<   0., -5., 1., 0., 0.;
        // wp3 <<   0.,  5., 0., 1., 0.;
        // wp4 << -10.,  5., -1., 0., 0.;
        // wps = {wp1, wp2, wp3, wp4};
        wps = {wp1};
        auto model = graphics->obstacleModels_[ObstacleType::VAN];
        // auto model = graphics->createBoxObstacleModel(5.f, 5.f, 5.f, 0.0);
        auto obs = std::make_shared<DynamicObstacle>(next_oid_++, wps, model);
        obs_to_create.push_back(obs);
    }

    else if (globals.FORMATION == "layered_walls")
    {
        new_obstacles_needed_ = true;

        // Define obstacle dimensions and path by defining MotionOptions for each obstacle
        if (!obstacles_initialised_)
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

            motion_options_ = {mo1, mo2, mo3};
            obstacles_initialised_ = true;
        }

        for (auto &mo : motion_options_)
        {
            if (clock_ - mo.last_spawn_time_ > mo.spawn_interval_)
            {
                auto obs = std::make_shared<DynamicObstacle>(next_oid_++, mo.waypoints_, mo.geom_, mo.color_);
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
        
        if (!obstacles_initialised_)
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

            motion_options_ = {mo1, mo2, mo3};

            for (int i = 0; i < motion_options_.size(); ++i)
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
                motion_options_[i].waypoints_ = waypoints;
            }
            obstacles_initialised_ = true;
        }

        for (auto &mo : motion_options_)
        {
            auto obs = std::make_shared<DynamicObstacle>(next_oid_++, mo.waypoints_, mo.geom_, mo.color_);
            obs_to_create.push_back(obs);
            mo.last_spawn_time_ = clock_;
        }
    }

    else if (globals.FORMATION == "junction_twoway")
    {
        new_obstacles_needed_ = globals.NEW_OBSTACLES_NEEDED;
        float now = clock_ * globals.TIMESTEP;
        
        // Set initial next_spawn, only needs to be called once
        if (!obstacles_initialised_)
        {
            for (auto& spawner : bus_spawners_) spawner.schedule_from(now);
            for (auto& spawner : van_spawners_) spawner.schedule_from(now);
            pedestrian_spawner_.schedule_from(now);
            obstacles_initialised_ = true;
        }

        // Define helper variables
        int n_roads = 4, n_lanes = 2;
        double lane_width = 4. * globals.ROBOT_RADIUS;
        
        // Lambda for creating obstacle spawn requests
        auto makeObstacleSpawnRequest = [](double now, int zone_id, 
                                          const std::deque<Eigen::VectorXd>& waypoints,
                                          std::shared_ptr<ObstacleModelInfo> model) -> SpawnRequest {
        
            const Eigen::VectorXd start = waypoints.front();
            SpawnRequest req;
            req.type = SpawnType::Obstacle;
            req.zone_id = zone_id;
            req.t_req = now;
            req.pos = start.head<2>();
            req.orientation = wrapAngle(-std::atan2(start(3), start(2))+model->orientation_offset);
            req.half_extents = Eigen::Vector2d(model->dimensions.x * 0.5, model->dimensions.z * 0.5);
            req.radius = req.half_extents.norm();
            req.waypoints = waypoints;
            req.model = std::move(model);
            return req;
        };
        
        // Spawn buses (if enabled)
        if (globals.ENABLE_BUSES) {
            for (int road = 0; road < bus_spawners_.size(); ++road) {
                if (bus_spawners_[road].try_spawn(now)) {
                    // Random turn (0=left, 1=straight, 2=right) and lane
                    int turn = random_int(0, 2);
                    int lane = 0; // busses only go on outer lanes
                    
                    // Generate waypoints using the static method
                    auto waypoints = DynamicObstacle::generateBusWaypoints(
                        road, turn, lane, globals.WORLD_SZ, globals.MAX_SPEED, 4. * globals.ROBOT_RADIUS
                    );
                    auto req = makeObstacleSpawnRequest(now, road, waypoints, 
                                                       graphics->obstacleModels_[ObstacleType::BUS]);
                    spawn_gates_[road].enqueue(req);
                }
            }
        }
        
        // Spawn vans with delivery behavior (if enabled)
        if (globals.ENABLE_VANS) {
            for (int road = 0; road < van_spawners_.size(); ++road) {
                if (van_spawners_[road].try_spawn(now)) {
                    auto waypoints = DynamicObstacle::generateVanWaypoints(
                        0, globals.WORLD_SZ, globals.MAX_SPEED, 4. * globals.ROBOT_RADIUS
                    );
                    auto req = makeObstacleSpawnRequest(now, road, waypoints, 
                                                       graphics->obstacleModels_[ObstacleType::VAN]);
                    spawn_gates_[road].enqueue(req);
                }
            }
        }
        
        // Spawn pedestrians (if enabled)
        if (globals.ENABLE_PEDESTRIANS) {
            if (pedestrian_spawner_.try_spawn(now)) {
                // Generate a single pedestrian crossing waypoint
                auto waypoints = DynamicObstacle::generatePedestrianWaypoints(
                    globals.WORLD_SZ, globals.DEFAULT_OBS_SPEED, 4. * globals.ROBOT_RADIUS
                );
                int zone_id = random_int(0, 3); // Random spawn gate for collision checking
                auto req = makeObstacleSpawnRequest(now, zone_id, waypoints, 
                                                   graphics->obstacleModels_[ObstacleType::PEDESTRIAN]);
                spawn_gates_[zone_id].enqueue(req);
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
        // Record obstacle despawn in metrics before deletion
        if (globals.EVAL && metrics) {
            double despawn_time = clock_ * globals.TIMESTEP;
            metrics->addObstacleDespawn(oid, despawn_time);
        }
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
    // Clear seen collision pairs at the start of each simulation
    if (clock_ == 0) seen_collision_pairs_.clear();
    // unique key for a pair of IDs
    auto pairKey = [](int a, int b) -> uint64_t {
        const uint32_t x = static_cast<uint32_t>(a < b ? a : b);
        const uint32_t y = static_cast<uint32_t>(a < b ? b : a);
        return (static_cast<uint64_t>(x) << 32) | y;
    };

    const bool robots_are_spheres = robots_.empty() 
        ? globals.N_DOFS == 4 // surrogate
        : robots_.begin()->second->robot_type_ == RobotType::SPHERE;
    
    const double eps = 1e-9;
    const double dt = globals.TIMESTEP;
    const double now = clock_ * dt;
    
    // Loop 1: Robot-Robot collisions and encounters
    for (const auto& [rid_i, robot_i] : robots_) {
        // robot_i->color_ = BLUE;
        // state slices
        Eigen::Vector2d pos_i = robot_i->position_.head<2>();
        Eigen::Vector2d vel_i = robot_i->position_.segment<2>(2);
        double radius_i = robot_i->robot_radius_;
        double query_radius = radius_i + dt * vel_i.norm();

        // KD-tree radius search
        std::vector<double> query_pt = {pos_i.x(), pos_i.y()};
        const float search_radius_sq = std::pow(query_radius * 3.0, 2.0); // Extra margin for safety
        std::vector<nanoflann::ResultItem<size_t, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        const size_t nMatches = treeOfRobots_->index->radiusSearch(&query_pt[0], search_radius_sq, matches, params);
        
        // Encounter tracking: prepare new inside set for this robot
        std::unordered_set<int> new_robots_inside;
        auto& track = metrics->getTrack(rid_i);
        
        for (size_t k = 0; k < nMatches; k++) {
            // Get the matched robot's ID
            auto it = robots_.begin();
            std::advance(it, matches[k].first);
            int rid_j = it->first;
            // Skip self
            if (rid_j == rid_i) continue;
            
            auto robot_j = robots_.at(rid_j);
            Eigen::Vector2d pos_j = robot_j->position_.head<2>();
            Eigen::Vector2d vel_j = robot_j->position_.segment<2>(2);
            double radius_j = robot_j->robot_radius_;
            double distance = (pos_j - pos_i).norm();
            
            // Calculate interaction radius for this neighbor pair
            double R_in;
            if (robots_are_spheres) {
                R_in = radius_i + radius_j; // combined radii + buffer
            } else {
                double robot_j_diag = robot_j->robot_dimensions_.norm() * 0.5;
                double robot_i_diag = robot_i->robot_dimensions_.norm() * 0.5;
                R_in = robot_i_diag + robot_j_diag; // combined diagonal half extents + buffer
            }
            double buffer =  R_in * 0.5;
            R_in += buffer;
            
            bool in = (distance <= R_in);
            bool was_in = track.robots_inside.count(rid_j) > 0;
            bool entering = (!was_in && in);
            double last_event_time = track.robot_last_event.count(rid_j) > 0 ? track.robot_last_event[rid_j] : -std::numeric_limits<double>::infinity();
            bool cooled = (now - last_event_time >= metrics->interaction_cooldown_threshold_); // refractory period
            
            if (entering && cooled && !metrics->isInWarmupPeriod(now)) {
                track.robot_encounters++;
                track.robot_last_event[rid_j] = now;
            }
            
            if (in) {
                new_robots_inside.insert(rid_j);
            }
            
            // Skip collision check if we've already checked this pair
            uint64_t pair_id = pairKey(rid_i, rid_j);
            if (seen_collision_pairs_.count(pair_id) > 0) continue;
            
            bool collision = false;
            
            if (robots_are_spheres) { /* SPHERE-SPHERE */
                double center_dist = (pos_j - pos_i).norm();
                double combined_radius = radius_i + radius_j;
                collision = (center_dist <= combined_radius + eps);
            } else { /* OBB-OBB */
                double theta_i = wrapAngle(robot_i->orientation_+robot_i->default_angle_offset_);
                double theta_j = wrapAngle(robot_j->orientation_+robot_j->default_angle_offset_);
                OBB2D obb_i(pos_i, robot_i->robot_dimensions_ * 0.5, theta_i);
                OBB2D obb_j(pos_j, robot_j->robot_dimensions_ * 0.5, theta_j);
                collision = GeometryUtils::overlapsOBB(obb_i, obb_j);
            }
            
            if (collision) {
                // Mark collision for both robots
                Eigen::Vector2d collision_pos = (pos_i + pos_j) * 0.5;  // Midpoint of collision
                seen_collision_pairs_.insert(pair_id);
                metrics->markCollision(rid_i, now, collision_pos);
                metrics->markCollision(rid_j, now, collision_pos);
                // Debug
                bool connected = std::find(robot_i->connected_r_ids_.begin(), robot_i->connected_r_ids_.end(), rid_j) != robot_i->connected_r_ids_.end();
                printf("CollisionEvent<Robot, Robot, %f>: ids=[%d, %d], pos1=[%f, %f], pos2=[%f, %f], comms_active=[%d, %d], connected=[%d]\n", 
                    now, rid_i, rid_j, pos_i.x(), pos_i.y(), pos_j.x(), pos_j.y(), (int)robot_i->interrobot_comms_active_, (int)robot_j->interrobot_comms_active_, connected);
                robot_i->color_ = RED;
                robot_j->color_ = RED;
                // globals.LAST_SIM_MODE = Iterate;
                // globals.SIM_MODE = SimNone;
            }
        } 
        // Update robot encounters inside set for this robot
        track.robots_inside.swap(new_robots_inside);
    }
    
    // Loop 2: Robot-Obstacle collisions and encounters
    std::unordered_map<int, std::unordered_set<int>> all_new_obstacle_encounters;
    for (const auto& [oid, obs] : obstacles_) {
        // obs->color_ = GRAY;
        Eigen::Vector2d c = obs->state_.head<2>();
        double o = wrapAngle(obs->orientation_+obs->geom_->orientation_offset);
        const auto dims = obs->geom_->dimensions;
        const Eigen::Vector2d o_he(dims.x*0.5, dims.z*0.5);
        OBB2D obs_obb(c, o_he, o);

        double obs_bounding_radius = obs_obb.getBoundingRadius();

        // KD-tree radius search
        std::vector<double> query_pt = {c.x(), c.y()};
        const float search_radius_sq = std::pow(obs_bounding_radius * 3.0, 2.0); // Extra margin for safety
        std::vector<nanoflann::ResultItem<size_t, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        const size_t nMatches = treeOfRobots_->index->radiusSearch(&query_pt[0], search_radius_sq, matches, params);

        for (size_t k = 0; k < nMatches; k++) {
            // Get the matched robot's ID
            auto it = robots_.begin();
            std::advance(it, matches[k].first);
            int rid = it->first;
            auto robot = robots_.at(rid);
            auto robot_pos = robot->position_.head<2>();
            auto robot_radius = robot->robot_radius_;
            double distance = (robot_pos - c).norm();
            
            // Detect encounters
            auto& track = metrics->getTrack(rid);
            auto& new_obstacle_encounters = all_new_obstacle_encounters[rid];
            
            double R_in;
            if (robots_are_spheres) {
                R_in = robot_radius + obs_bounding_radius;
            } else {
                double robot_diag = robot->robot_dimensions_.norm() * 0.5;
                R_in = robot_diag + obs_bounding_radius;
            }
            double buffer = R_in * 0.5;
            R_in += buffer;
            
            bool in = (distance <= R_in);
            bool was_in = track.obstacles_inside.count(oid) > 0;
            bool entering = (!was_in && in);
            double last_event_time = track.obstacle_last_event.count(oid) > 0 ? track.obstacle_last_event[oid] : -std::numeric_limits<double>::infinity();
            bool cooled = (now - last_event_time >= 2.0); // refractory period
            
            if (entering && cooled && !metrics->isInWarmupPeriod(now)) {
                track.obstacle_encounters++;
                track.obstacle_last_event[oid] = now;
            }
            
            if (in) {
                new_obstacle_encounters.insert(oid);
            }
            
            // Skip collision checks if seen
            uint64_t pair_id = pairKey(-oid, rid);
            if (seen_collision_pairs_.count(pair_id) > 0) continue;

            bool collision = false;
            
            if (robots_are_spheres) { /* OBB-SPHERE */
                collision = GeometryUtils::overlapsSphereOBB(robot_pos, robot_radius, obs_obb, eps);
            } else { /* OBB-OBB */
                double m_theta = wrapAngle(robot->orientation_ + robot->default_angle_offset_);   
                OBB2D robot_obb(robot_pos, robot->robot_dimensions_ * 0.5, m_theta);
                collision = GeometryUtils::overlapsOBB(robot_obb, obs_obb);
            }
            
            if (collision) {
                // Mark robot-obstacle collision
                // For robot-obstacle collisions, use the robot position as collision position
                seen_collision_pairs_.insert(pair_id);
                metrics->markCollision(rid, now, robot_pos);  
                // Debug                 
                printf("CollisionEvent<Robot, Obstacle, %f>: ids=[%d, %d], pos=[%f, %f], theta_o=[%f]\n", now, rid, oid, robot_pos.x(), robot_pos.y(), o);
                obs->color_ = RED;
                robot->color_ = RED;
                // globals.LAST_SIM_MODE = Iterate;
                // globals.SIM_MODE = SimNone;
            }
        }
    }
    for (auto& [rid, new_inside] : all_new_obstacle_encounters) {
        auto& track = metrics->getTrack(rid);
        track.obstacles_inside.swap(new_inside);
    }
}

/*******************************************************************************/
// Check if spawn request location is clear of collisions (class method version)
/*******************************************************************************/
bool Simulator::isSpawnClear(const SpawnRequest& r, double margin) {
    const Eigen::Vector2d p = r.pos;
    const double r_safe = r.radius + margin;

    // Check request against robots
    if (!robots_.empty()) {
        // Perform radius search for collision candidates
        std::vector<double> q = {p.x(), p.y()};
        std::vector<nanoflann::ResultItem<size_t,double>> matches;
        nanoflann::SearchParameters params; params.sorted = false;
        const float R2 = std::pow(r_safe * 5.0, 2.0);
        const size_t nMatches = treeOfRobots_->index->radiusSearch(&q[0], R2, matches, params);
        
        for (size_t k = 0; k < nMatches; k++) {
            auto it = this->robots_.begin();
            std::advance(it, matches[k].first);
            auto match = it->second;
            const bool robot_are_spheres = match->robot_type_ == RobotType::SPHERE;
            const auto match_pos = match->position_.head<2>();
            const auto match_dims = match->robot_dimensions_;
            const auto match_theta = match->orientation_;
            const auto match_radius = match->robot_radius_;
            
            // Request is robot
            if (r.type == SpawnType::Robot) { /* Robot-Robot */
                if (robot_are_spheres) { /* SPHERE-SPHERE */
                    double center_dist = (match_pos - p).norm();
                    double combined_radius = match_radius + r_safe;
                    if (center_dist <= combined_radius) return false;
                } else { /* OBB-OBB */
                    OBB2D match_obb(match_pos, 0.5*match_dims, match_theta);
                    OBB2D spawn_obb(p, r.half_extents + Eigen::Vector2d::Constant(margin), r.orientation);
                    if (GeometryUtils::overlapsOBB(spawn_obb, match_obb)) return false;
                }
            // Request is obstacle
            } else { /* Obstacle-Robot */
                OBB2D obs_obb(r.pos, r.half_extents + Eigen::Vector2d::Constant(margin), r.orientation);
                if (robot_are_spheres) { /* OBB-SPHERE */
                    if (GeometryUtils::overlapsSphereOBB(match_pos, match_radius, obs_obb, 1e-6)) return false;
                } else { /* OBB-OBB */
                    OBB2D match_obb(match_pos, 0.5*match_dims, match_theta);
                    if (GeometryUtils::overlapsOBB(obs_obb, match_obb)) return false;
                }
            }
        }
    }

    // Check request against obstacles
    if (!obstacles_.empty()) {
        for (const auto& [oid, obs] : obstacles_) {
            const Eigen::Vector2d c = obs->state_.head<2>();
            const double o = wrapAngle(obs->orientation_+obs->geom_->orientation_offset);
            const auto dims = obs->geom_->dimensions;
            const Eigen::Vector2d he(dims.x*0.5, dims.z*0.5);
            OBB2D obb(c, he, o);
            if (r.type == SpawnType::Robot) {
                if (GeometryUtils::overlapsSphereOBB(p, r_safe, obb, 1e-6)) return false;
            } else {
                OBB2D spawn_obb(p, r.half_extents + Eigen::Vector2d::Constant(margin), r.orientation);
                if (GeometryUtils::overlapsOBB(spawn_obb, obb)) return false;
            }
        }
    }
    
    // All checks passed -> clear
    return true;
}

/*******************************************************************************/
// Admit spawn request and create robot/obstacle (class method version)
/*******************************************************************************/
bool Simulator::admitSpawnRequest(const SpawnRequest& r) {
    double now = clock_ * globals.TIMESTEP;
    
    if (r.type == SpawnType::Robot) {
        auto robot = std::make_shared<Robot>(this, next_rid_++, r.waypoints, r.robot_type, 1.f, r.radius, r.color);
        robots_[robot->rid_] = robot;
        robot_positions_[robot->rid_] = {robot->position_(0), robot->position_(1)};
        this->treeOfRobots_->index->buildIndex(); // Needed as there may be more than one request created each time
        if (globals.EVAL) {
            metrics->addSample(robot->rid_, now, robot->position_.head<2>());
            metrics->setBaselinePathLength(robot->rid_, robot->base_path_length_);
        }
        return true;
    } else {
        // Determine obstacle type from model
        ObstacleType obs_type = ObstacleType::CUBE;
        if (r.model == graphics->obstacleModels_[ObstacleType::BUS]) {
            obs_type = ObstacleType::BUS;
        } else if (r.model == graphics->obstacleModels_[ObstacleType::VAN]) {
            obs_type = ObstacleType::VAN;
        } else if (r.model == graphics->obstacleModels_[ObstacleType::PEDESTRIAN]) {
            obs_type = ObstacleType::PEDESTRIAN;
        }
        
        auto obs = std::make_shared<DynamicObstacle>(next_oid_++, r.waypoints, r.model, r.color, obs_type);
        obs->spawn_time_ = now;
        obstacles_[obs->oid_] = obs;
        
        // Record obstacle spawn in metrics
        if (globals.EVAL && metrics) {
            metrics->addObstacleSpawn(obs->oid_, obs_type, now);
        }
        return true;
    }
}
