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

    // Create and/or destory factors depending on a robot's neighbours
    calculateRobotNeighbours(robots_);
    for (auto [r_id, robot] : robots_)
    {
        robot->updateInterrobotFactors();
        robot->updateDynamicObstacleFactors();
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
    std::shuffle(range.begin(), range.end(), gen_uniform);
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
        starting = Eigen::VectorXd{{-bound, 0., double(globals.MAX_SPEED), 0.}};
        ending = Eigen::VectorXd{{ bound, 0., double(globals.MAX_SPEED), 0.}};
        std::deque<Eigen::VectorXd> waypoints{starting, ending};

        Color robot_color = ColorFromHSV(5. * 36., 1., 0.75);
        robots_to_create.push_back(std::make_shared<Robot>(this, next_rid_++, waypoints, globals.ROBOT_RADIUS, robot_color));
    }


    else if (globals.FORMATION == "circle" || globals.FORMATION == "circle_cluttered_dynamic")
    {
        // Robots must travel to opposite sides of circle
        new_robots_needed_ = false;
        float min_circumference_spacing = 5. * globals.ROBOT_RADIUS;
        double min_radius = 0.25 * globals.WORLD_SZ;
        Eigen::VectorXd centre{{0., 0., 0., 0.}};
        for (int i = 0; i < globals.NUM_ROBOTS; i++)
        {
            // Select radius of large circle to be at least min_radius,
            // Also ensures that robots in the circle are at least min_circumference_spacing away from each other
            float radius_circle = (globals.NUM_ROBOTS == 1) ? min_radius : std::max(min_radius, sqrt(min_circumference_spacing / (2. - 2. * cos(2. * PI / (double)globals.NUM_ROBOTS))));
            Eigen::VectorXd offset_from_centre = Eigen::VectorXd{{radius_circle * cos(2. * PI * i / (float)globals.NUM_ROBOTS)},
                                                                 {radius_circle * sin(2. * PI * i / (float)globals.NUM_ROBOTS)},
                                                                 {0.},
                                                                 {0.}};
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
            starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2., lane_v_offset, globals.MAX_SPEED, 0.}};
            ending = rot * Eigen::VectorXd{{(double)globals.WORLD_SZ, lane_v_offset, 0., 0.}};
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

    else if (globals.FORMATION == "junction_twoway")
    {
        // Robots in a two-way junction, turning LEFT (RED), RIGHT (BLUE) or STRAIGHT (GREEN)
        new_robots_needed_ = true; // This is needed so that more robots can be created as the simulation progresses.
        if (clock_ % 20 == 0)
        { // Arbitrary condition on the simulation time to create new robots
            int n_roads = 4;
            int road = random_int(0, n_roads - 1);
            // We will define one road (the one going left) and then we can rotate the positions for other roads.
            Eigen::Matrix4d rot;
            rot.setZero();
            rot.topLeftCorner(2, 2) << cos(PI / 2. * road), -sin(PI / 2. * road), sin(PI / 2. * road), cos(PI / 2. * road);
            rot.bottomRightCorner(2, 2) << cos(PI / 2. * road), -sin(PI / 2. * road), sin(PI / 2. * road), cos(PI / 2. * road);

            int n_lanes = 2;
            int lane = random_int(0, n_lanes - 1);
            int turn = random_int(0, 2);
            double lane_width = 4. * globals.ROBOT_RADIUS;
            double lane_v_offset = (0.5 * (1 - 2. * n_lanes) + lane) * lane_width;
            double lane_h_offset = (1 - turn) * (0.5 + lane - n_lanes) * lane_width;
            starting = rot * Eigen::VectorXd{{-globals.WORLD_SZ / 2., lane_v_offset, globals.MAX_SPEED, 0.}};
            turning = rot * Eigen::VectorXd{{lane_h_offset, lane_v_offset, (turn % 2) * globals.MAX_SPEED, (turn - 1) * globals.MAX_SPEED}};
            ending = rot * Eigen::VectorXd{{lane_h_offset + (turn % 2) * globals.WORLD_SZ * 1., lane_v_offset + (turn - 1) * globals.WORLD_SZ * 1., 0., 0.}};
            std::deque<Eigen::VectorXd> waypoints{starting, turning, ending};
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
                starting = Eigen::VectorXd{{lane_offset_x, -globals.WORLD_SZ / 2., 0., 1. * globals.MAX_SPEED}};
                ending = Eigen::VectorXd{{lane_offset_x, globals.WORLD_SZ / 2., 0., 1. * globals.MAX_SPEED}};
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
            Eigen::VectorXd{{ bound,  10.5,  -speed,        -0.15 * speed}},
            Eigen::VectorXd{{ bound,  7.5,   -speed,        -0.15 * speed}},
            Eigen::VectorXd{{ 30.,   -bound, -0.78 * speed,  speed}},
            Eigen::VectorXd{{-6.5,   -bound,  0.3  * speed,  speed}},
            Eigen::VectorXd{{-bound, -34.,    0.8  * speed,  speed}},
            Eigen::VectorXd{{-bound,  21.,    speed,        -0.15 * speed}},
            Eigen::VectorXd{{ 3.,     bound, -0.13 * speed, -speed}},
        };

        static std::vector<Eigen::VectorXd> exits{
            Eigen::VectorXd{{ bound,  3.0,    speed,         0.15 * speed}},
            Eigen::VectorXd{{ bound,  1.0,    speed,         0.15 * speed}},
            Eigen::VectorXd{{ 30.  , -bound,  0.78 * speed, -speed}},
            Eigen::VectorXd{{-6.5,   -bound, -0.3  * speed, -speed}},
            Eigen::VectorXd{{-bound, -29.,   -0.8  * speed, -speed}},
            Eigen::VectorXd{{-bound,  25.,   -speed,         0.15 * speed}},
            Eigen::VectorXd{{ 6.,     bound,  0.13 * speed,  speed}},
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
        auto obs = std::make_shared<DynamicObstacle>(next_oid_++, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{0., 0., 0., 0.}}}, geom, 2.5f);
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
            Eigen::Vector4d wp1{{-globals.WORLD_SZ / 2.f, 0.f, wall_vel_x, 0.f}};
            Eigen::Vector4d wp2{{globals.WORLD_SZ / 2.f, 0.f, wall_vel_x, 0.f}};
            std::deque<Eigen::Vector4d> waypoints{wp1, wp2};
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
            Eigen::VectorXd wp = Eigen::VectorXd{{lane_offset_x, 0., 0., 0.}};
            std::deque<Eigen::Vector4d> waypoints{wp};
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
            auto mo1 = MotionOptions(5.f, 5.f, 5.f, 2.5f, 300, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{-globals.WORLD_SZ / 2.f, -33.f, 2.f, 0.f}}, Eigen::Vector4d{{globals.WORLD_SZ / 2.f, -33.f, 2.f, 0.f}}}, graphics);
            // Motion: right to left
            auto mo2 = MotionOptions(5.f, 5.f, 5.f, 2.5f, 300, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{globals.WORLD_SZ / 2.f, 0.f, -2.5f, 0.f}}, Eigen::Vector4d{{-globals.WORLD_SZ / 2.f, 0.f, -2.5f, 0.f}}}, graphics);
            // Motion: left to right
            auto mo3 = MotionOptions(5.f, 5.f, 5.f, 2.5f, 200, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{-globals.WORLD_SZ / 2.f, 33.f, 1.5f, 0.f}}, Eigen::Vector4d{{globals.WORLD_SZ / 2.f, 33.f, 1.5f, 0.f}}}, graphics);

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
            auto mo1 = MotionOptions(3.f, 5.f, 6.f, 2.5f, 0, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{ 8.5, -10., 0., 0.}}}, graphics);
            auto mo2 = MotionOptions(4.f, 4.f, 4.f, 2.0f, 0, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{-6.0,  7.0, 0., 0.}}}, graphics);
            auto mo3 = MotionOptions(3.f, 5.f, 3.f, 2.5f, 0, std::deque<Eigen::Vector4d>{Eigen::Vector4d{{-1.5, -6.5, 0., 0.}}}, graphics);
            // Triangles
            Mesh t1 = graphics->genMeshPyramid(3.f, 3.f);
            auto mo4 = MotionOptions(t1, 0, 0.0f, std::deque<Eigen::Vector4d>{{6.0, 2.0, 0.0, 0.0}}, graphics);
            
            motion_options = {mo1, mo2, mo3, mo4};

            for (int i = 0; i < motion_options.size(); ++i) {
                std::deque<Eigen::Vector4d> waypoints;
                float omega = (2.f * PI)/size;
                for (int j = 0; j < size; ++j) {
                    float theta = phase_offsets[i] + omega * j;
                    float x = radii[i] * std::cos(theta);
                    float y = radii[i] * std::sin(theta);
                    float vx = -radii[i] * omegas[i] * std::sin(theta);
                    float vy = radii[i] * omegas[i] * std::cos(theta);
                    waypoints.emplace_back(Eigen::Vector4d{{x, y, vx, vy}});
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

    else
    {
        // print("Shouldn't reach here, formation not defined!");
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

