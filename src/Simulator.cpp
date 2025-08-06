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
            for (auto [oid, obs] : obstacles_) obs->draw();
            // Draw Robots
            for (auto [rid, robot] : robots_) robot->draw();
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

    auto t0 = Clock::now();
    
    auto tp1s = Clock::now();
    // Create and/or destory factors depending on a robot's neighbours
    calculateRobotNeighbours(robots_);
    for (auto [r_id, robot] : robots_)
    {
        robot->updateInterrobotFactors();
        robot->updateDynamicObstacleFactors();
    }
    auto tp1e = Clock::now();
    Duration tp1d = tp1e - tp1s;
    // print("TP 1", tp1d.count());

    // If the communications failure rate is non-zero, activate/deactivate robot comms
    setCommsFailure(globals.COMMS_FAILURE_RATE);

    auto tp2s = Clock::now();
    // Perform iterations of GBP. Ideally the internal and external iterations
    // should be interleaved better. Here it is assumed there are an equal number.
    for (int i = 0; i < globals.NUM_ITERS; i++)
    {
        auto intern_start = Clock::now();
        iterateGBP(1, INTERNAL, robots_);
        auto intern_end = Clock::now();
        Duration intern_d = intern_end - intern_start;
        // print("Internal: ", intern_d.count());
        
        auto extern_start = Clock::now();
        iterateGBP(1, EXTERNAL, robots_);
        auto extern_end = Clock::now();
        Duration extern_d = extern_end - extern_start;
        // print("External: ", extern_d.count());
    }
    auto tp2e = Clock::now();
    Duration tp2d = tp2e - tp2s;
    // print("TP 2", tp2d.count());

    auto tp3s = Clock::now();
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
    auto tp3e = Clock::now();
    Duration tp3d = tp3e - tp3s;
    // print("TP 3", tp3d.count());

    // Increase simulation clock by one timestep
    clock_++;
    if (clock_ >= globals.MAX_TIME)
    {
        print("Maximum run time reached, exiting...");
        globals.RUN = false;
    }

    if (globals.SIM_MODE == Iterate && clock_ % globals.ITERATE_STEPS == 0)
    {
        globals.SIM_MODE = SimNone;
        for (auto [r_id, robot] : robots_) robot->print_graph_info();
    }

    auto t1 = Clock::now();
    Duration sim_time = t1 - t0;
    // print("Timestep", sim_time.count(), "\n");


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
    double bound = double(globals.WORLD_SZ / 2 - 1);

    if (globals.FORMATION == "playground")
    {
        new_robots_needed_ = false;
        Eigen::VectorXd wp1(5), wp2(5), wp3(5), wp4(5), wp5(5);
        wp1 << -10., -10.,  1.,  0., 30.;
        wp2 <<  10., -10.,  1.,  0., 5.;
        wp3 <<  10.,  10.,  0.,  1., 5.;
        wp4 << -10.,  10., -1.,  0., 5.;
        wp5 << -10,  -10.,  0., -1., 5.;
        std::deque<Eigen::VectorXd> waypoints{wp1, wp2, wp3, wp4, wp5};

        Color robot_color = ColorFromHSV(5. * 36., 1., 0.75);
        robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, globals.ROBOT_RADIUS, robot_color));
    }


    else if (globals.FORMATION == "circle" || globals.FORMATION == "circle_cluttered_dynamic")
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
            robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, robot_radius, robot_color));
        }
    }

    else if (globals.FORMATION == "junction")
    {
        // Robots in a cross-roads style junction. There is only one-way traffic, and no turning.
        new_robots_needed_ = true; // This is needed so that more robots can be created as the simulation progresses.
        if (clock_ % 20 == 0)
        { // Arbitrary condition on the simulation time to create new robots
            int n_roads = 2;
            int road = random_int(0, n_roads - 1);
            Eigen::Matrix4d rot;
            rot.setZero();
            rot.topLeftCorner(2, 2) << cos(PI / 2. * road), -sin(PI / 2. * road), sin(PI / 2. * road), cos(PI / 2. * road);
            rot.bottomRightCorner(2, 2) << cos(PI / 2. * road), -sin(PI / 2. * road), sin(PI / 2. * road), cos(PI / 2. * road);

            int n_lanes = 2;
            int lane = random_int(0, n_lanes - 1);
            double lane_width = 4. * globals.ROBOT_RADIUS;
            double lane_v_offset = (0.5 * (1 - n_lanes) + lane) * lane_width;
            starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2., lane_v_offset, globals.MAX_SPEED, 0., 0.}};
            ending = rot * Eigen::VectorXd{{(double)globals.WORLD_SZ, lane_v_offset, 0., 0., 0.}};
            std::deque<Eigen::VectorXd> waypoints{starting, ending};
            float robot_radius = globals.ROBOT_RADIUS;
            Color robot_color = DARKGREEN;
            robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, robot_radius, robot_color));
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

    else if (globals.FORMATION == "junction_twoway" || globals.FORMATION == "junction_twoway_dynamic")
    {
        // Robots in a two-way junction, turning LEFT (RED), RIGHT (BLUE) or STRAIGHT (GREEN)
        new_robots_needed_ = true; // This is needed so that more robots can be created as the simulation progresses.
        if (clock_ % 20 == 0)
        { // Arbitrary condition on the simulation time to create new robots
            int n_roads = 4;
            int road = random_int(0, n_roads - 1);
            // We will define one road (the one going from left to right) and then we can rotate the positions for other roads.
            // Eigen::Matrix4d rot;
            // rot.setZero();
            // rot.topLeftCorner(2, 2) << cos(PI / 2. * road), -sin(PI / 2. * road), sin(PI / 2. * road), cos(PI / 2. * road);
            // rot.bottomRightCorner(2, 2) << cos(PI / 2. * road), -sin(PI / 2. * road), sin(PI / 2. * road), cos(PI / 2. * road);

            // 1) Start with identity so diag(4,4)==1 and everything else is 0
            Eigen::Matrix<double,5,5> rot = Eigen::Matrix<double,5,5>::Identity();

            // 2) Compute your 2×2 rotation
            double angle = PI/2. * road;
            double c     = std::cos(angle);
            double s     = std::sin(angle);

            // 3) Plug it into the top‐left 2×2 (x,y) and the next 2×2 (speed,theta)
            rot.block<2,2>(0,0) << c, -s, s,  c;
            rot.block<2,2>(2,2) << c, -s, s,  c;

            // rot5(4,4) is still 1, so wp(4) passes through unchanged
            int n_lanes = 2;
            int lane = random_int(0, n_lanes - 1);
            int turn = random_int(0, 2); // 0 = left, 1 = straight, 2 = right
            double lane_width = 4. * globals.ROBOT_RADIUS;
            double lane_v_offset = (0.5 * (1 - 2. * n_lanes) + lane) * lane_width;
            double lane_h_offset = (1 - turn) * (0.5 + lane - n_lanes) * lane_width;
            starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2., lane_v_offset, globals.MAX_SPEED, 0., 0.}};
            turning = rot * Eigen::VectorXd{{lane_h_offset, lane_v_offset, (turn % 2) * globals.MAX_SPEED, (turn - 1) * globals.MAX_SPEED, 0.}};
            ending = rot * Eigen::VectorXd{{lane_h_offset + (turn % 2) * globals.WORLD_SZ * 1., lane_v_offset + (turn - 1) * globals.WORLD_SZ * 1., 0., 0., 0.}};
            std::deque<Eigen::VectorXd> waypoints;
            
            if (lane == 0 && random_float(0.f, 1.f) < 0.8f) {
                waypoints.push_back(starting);
                int stops_per_segment = 3;
                double pause_time = 10.0;
                for (int i = 1; i < stops_per_segment + 1; ++i) {
                    Eigen::VectorXd wp = starting + (turning - starting) * i / (stops_per_segment+1);
                    wp(4) = pause_time;
                    waypoints.push_back(wp);
                }
                waypoints.push_back(turning);
                for (int i = 1; i < stops_per_segment + 1; ++i){
                    Eigen::VectorXd wp = turning + (ending - turning) * i / (stops_per_segment+1);
                    wp(4) = pause_time;
                    waypoints.push_back(wp);
                }
                waypoints.push_back(ending);
            } else {
                waypoints = {starting, turning, ending};
            }

            float robot_radius = globals.ROBOT_RADIUS;
            Color robot_color = ColorFromHSV(turn * 120., 1., 0.75);
            robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, robot_radius, robot_color));
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

    else if (globals.FORMATION == "moving_walls" || globals.FORMATION == "static_walls" || globals.FORMATION == "layered_walls")
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
                robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, robot_radius, robot_color));
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

    else if (globals.FORMATION == "magic_roundabout")
    {
        // Robots moving in a complex roundabout (modelled after magic roundabout in Swindon, UK)
        new_robots_needed_ = true;
        
        static double speed = double(globals.MAX_SPEED);
        static std::vector<Eigen::VectorXd> entries{
            Eigen::VectorXd{{ bound,  10.5,  -speed,        -0.15 * speed, 0.}},
            Eigen::VectorXd{{ bound,  7.5,   -speed,        -0.15 * speed, 0.}},
            Eigen::VectorXd{{ 30.,   -bound, -0.78 * speed,  speed, 0.}},
            Eigen::VectorXd{{-6.5,   -bound,  0.3  * speed,  speed, 0.}},
            Eigen::VectorXd{{-bound, -34.,    0.8  * speed,  speed, 0.}},
            Eigen::VectorXd{{-bound,  21.,    speed,        -0.15 * speed, 0.}},
            Eigen::VectorXd{{ 3.,     bound, -0.13 * speed, -speed, 0.}},
        };

        static std::vector<Eigen::VectorXd> exits{
            Eigen::VectorXd{{ bound,  3.0,    speed,         0.15 * speed, 0.}},
            Eigen::VectorXd{{ bound,  1.0,    speed,         0.15 * speed, 0.}},
            Eigen::VectorXd{{ 30.  , -bound,  0.78 * speed, -speed, 0.}},
            Eigen::VectorXd{{-6.5,   -bound, -0.3  * speed, -speed, 0.}},
            Eigen::VectorXd{{-bound, -29.,   -0.8  * speed, -speed, 0.}},
            Eigen::VectorXd{{-bound,  25.,   -speed,         0.15 * speed, 0.}},
            Eigen::VectorXd{{ 6.,     bound,  0.13 * speed,  speed, 0.}},
        };

        static std::map<int, Color>color_map {
            {0, DARKGREEN}, {1, DARKGREEN}, {2, DARKBLUE}, {3, PURPLE}, {4, RED} ,
            {5, GRAY}, {6, YELLOW}
        };

        if (clock_ % 500 == 0)
        {
            for (int i = 0; i < globals.NUM_ROBOTS; i++)
            {
                // starting = Eigen::VectorXd{{5., 0., 0., 0. }};
                int entry = random_int(0, entries.size()-1);
                int exit = random_int(0, exits.size()-1);
                while ((exit == 2 || exit == 3) && exit == entry)
                    exit = random_int(0, exits.size()-1);
                robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, 
                    std::deque<Eigen::VectorXd>{entries[entry], exits[exit]}, globals.ROBOT_RADIUS, color_map[exit]));
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

    else
    {
        // print("Shouldn't reach here, formation not defined!");
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
    
        static std::shared_ptr<IGeometry> geom = graphics->GenCubeGeom(5.f, 5.f, 5.f);
        std::deque<Eigen::VectorXd> wps;
        Eigen::VectorXd wp1(5), wp2(5), wp3(5);
        wp1 << -10., -20., 0., -1., 0.;   // Normal waypoint (pause time = 0)
        wp2 << -10.,  20., 0., -1., 0.;     // Pause waypoint (pause for 20 seconds)
        // wp3 << 0., 50., 0., -2., 0.;    // Normal waypoint (pause time = 0)
        wps = {wp1, wp2};
        auto obs = std::make_shared<DynamicObstacle>(next_oid_++, wps, geom, 2.5f);
        obs_to_create.push_back(obs);
    }
    
    else if (globals.FORMATION == "moving_walls")
    {
        new_obstacles_needed_ = globals.NEW_OBSTACLES_NEEDED;
        // Create the obstacle geometry once and reuse it.
        // makeObsModel generates a model and creates a KD-tree in local frame.
        float w = 5.f;
        float h = 5.f;
        float d = 5.f;
        float ele = h / 2.;
        static std::shared_ptr<IGeometry> wall_obs_geom = graphics->GenCubeGeom(w, h, d);
        static uint32_t last_spawn_time = -1000;
        int spawn_interval = 600;

        if (clock_ - last_spawn_time > spawn_interval)
        {
            static float wall_vel_x = 1.f;
            Eigen::VectorXd wp1(5), wp2(5);
            wp1 << -globals.WORLD_SZ / 2.f, 0.f, wall_vel_x, 0.f, 0.f;
            wp2 << globals.WORLD_SZ / 2.f, 0.f, wall_vel_x, 0.f, 0.f;
            std::deque<Eigen::VectorXd> waypoints{wp1, wp2};
            // We deal with a 2D problem: obstacles remains at the same level of elevation and is directly defined here.
            auto wall_obs = std::make_shared<DynamicObstacle>(next_oid_++, waypoints, wall_obs_geom, ele);
            wall_obs->completed_ = false;
            obs_to_create.push_back(wall_obs);
            last_spawn_time = clock_;
        }

        for (auto [oid, obs] : obstacles_)
        {
            if (obs->completed_)
                obs_to_delete.push_back(oid);
        }
    }

    else if (globals.FORMATION == "static_walls")
    {
        new_obstacles_needed_ = false;
        float w = 5.f;
        float h = 5.f;
        float d = 5.f;
        float ele = h / 2.;
        static std::shared_ptr<IGeometry> wall_obs_geom = graphics->GenCubeGeom(w, h, d);
        int n_roads = 8;
        double lane_width = globals.WORLD_SZ / (n_roads + 1);
        double lane_offset_x;
        for (int i = 1; i < n_roads + 1; ++i)
        {
            lane_offset_x = -globals.WORLD_SZ / 2. + i * lane_width;
            Eigen::VectorXd wp(5);
            wp << lane_offset_x, 0., 0., 0., 0.;
            std::deque<Eigen::VectorXd> waypoints{wp};
            auto wall_obs = std::make_shared<DynamicObstacle>(next_oid_++, waypoints, wall_obs_geom, ele);
            obs_to_create.push_back(wall_obs);
        }
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
                auto obs = std::make_shared<DynamicObstacle>(next_oid_++, mo.waypoints_, mo.geom_, mo.elevation_);
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

    else if (globals.FORMATION == "circle_cluttered_dynamic")
    {
        new_obstacles_needed_ = false;
        static bool obs_initialised = false;
        static std::vector<MotionOptions> motion_options;
        
        if (!obs_initialised)
        {
            // Depends on omega
            int size = 100;
            // Since Y increases downwards, omegas need to be negated for anti-clockwise motion
            std::vector<float> radii = {13.1f, 9.2f, 6.7f, 6.3f};
            std::vector<float> phase_offsets = {std::atan2(-10.f, 8.5f), std::atan2(7.f, -6.f), std::atan2(-6.5f, -1.5f), std::atan2(2.f, 6.f)};
            std::vector<float> omegas = {-0.1f, -0.1f, -0.1f, -0.1f};
            
            // Cuboids1
            Eigen::VectorXd mo1_wp(5), mo2_wp(5), mo3_wp(5), mo4_wp(5);
            mo1_wp << 8.5, -10., 0., 0., 0.;
            mo2_wp << -6.0, 7.0, 0., 0., 0.;
            mo3_wp << -1.5, -6.5, 0., 0., 0.;
            mo4_wp << 6.0, 2.0, 0.0, 0.0, 0.;
            
            auto mo1 = MotionOptions(3.f, 5.f, 6.f, 2.5f, 0, graphics, std::deque<Eigen::VectorXd>{mo1_wp});
            auto mo2 = MotionOptions(4.f, 4.f, 4.f, 2.0f, 0, graphics, std::deque<Eigen::VectorXd>{mo2_wp});
            auto mo3 = MotionOptions(3.f, 5.f, 3.f, 2.5f, 0, graphics, std::deque<Eigen::VectorXd>{mo3_wp});
            // Triangles
            Mesh t1 = graphics->genMeshPyramid(3.f, 3.f);
            auto mo4 = MotionOptions(t1, 0, 0.0f, graphics, std::deque<Eigen::VectorXd>{mo4_wp});
            
            motion_options = {mo1, mo2, mo3, mo4};

            for (int i = 0; i < motion_options.size(); ++i) {
                std::deque<Eigen::VectorXd> waypoints;
                float omega = (2.f * PI)/size;
                for (int j = 0; j < size; ++j) {
                    float theta = phase_offsets[i] + omega * j;
                    float x = radii[i] * std::cos(theta);
                    float y = radii[i] * std::sin(theta);
                    float vx = -radii[i] * omegas[i] * std::sin(theta);
                    float vy = radii[i] * omegas[i] * std::cos(theta);
                    Eigen::VectorXd wp(5);
                    wp << x, y, vx, vy, 0.0;  // 5th dimension (pause time) = 0
                    waypoints.emplace_back(wp);
                }
                motion_options[i].waypoints_ = waypoints;
            }
            obs_initialised = true;
        }

        for (auto &mo : motion_options)
        {
            auto obs = std::make_shared<DynamicObstacle>(next_oid_++, mo.waypoints_, mo.geom_, mo.elevation_);
            obs_to_create.push_back(obs);
            mo.last_spawn_time_ = clock_;
        }
    }

    else if (globals.FORMATION == "junction_twoway_dynamic")
    {
        new_obstacles_needed_ = true;
        static bool obs_initialised = false;

        std::vector<float> spawn_probs{0.5f, 0.5f};
        static std::vector<std::shared_ptr<IGeometry>> geoms;
        
        if (!obs_initialised) {
            // Pedestrian geometry
            geoms.push_back(graphics->GenCubeGeom(1.f, 2.f, 1.f));
            // Cyclist geometry
            geoms.push_back(graphics->GenCubeGeom(1.f, 1.f, 2.f));
            obs_initialised = true;
        }

        for (int i = 0; i < geoms.size(); ++i) {
            if (random_float(0.f, 1.f) <= spawn_probs[i]) {
                // Pedestrians
                if (i == 0) {
                    // Generate 1-3 pedestrians each time
                    int n = random_int(1, 3);
                    std::vector<std::deque<Eigen::VectorXd>> wps_vec = DynamicObstacle::GenPedWaypoints(n);
                    for (int j = 0; j < n; ++j) {
                        auto obs = std::make_shared<DynamicObstacle>(next_oid_++, wps_vec[j], geoms[i], 1.f);
                        obs->completed_ = false;
                        obs_to_create.push_back(obs);
                    }
                }
    
                // Cyclists
                else if (i == 1) {
    
                }
            }
        }

        for (auto [oid, obs] : obstacles_) {
            if (obs->completed_)
                obs_to_delete.push_back(oid);
        }
    }

    // Create or delete obstacles.
    for (auto obs : obs_to_create)
        obstacles_[obs->oid_] = obs;
    for (const int oid : obs_to_delete)
        obstacles_.erase(oid);
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

