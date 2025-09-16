/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <Globals.h>
#include <Utils.h>
#include "json.hpp"

/*****************************************************************/
// Simply reads the appropriate sections from the config.json
/*****************************************************************/
void Globals::parse_global_args(std::ifstream& config_file){
    
    // Basic parameters
    nlohmann::json j;
    config_file >> j;
    ASSETS_DIR = j["ASSETS_DIR"];
    N_DOFS = j.value("N_DOFS", 4);
    EVAL = static_cast<bool>((int)j.value("EVAL", 0));
    EXPERIMENT_NAME = j.value("EXPERIMENT_NAME", "");
    VERBOSE = static_cast<bool>((int)j.value("VERBOSE", 0));

    // Display parameters
    DISPLAY = static_cast<bool>((int)j["DISPLAY"]);;
    WORLD_SZ = j["WORLD_SZ"];
    SCREEN_SZ = j["SCREEN_SZ"];
    DRAW_INTERROBOT = static_cast<bool>((int)j["DRAW_INTERROBOT"]);
    DRAW_PATH = static_cast<bool>((int)j["DRAW_PATH"]);
    DRAW_WAYPOINTS = static_cast<bool>((int)j["DRAW_WAYPOINTS"]);
    DRAW_FACTORS = static_cast<bool>((int)j.value("DRAW_FACTORS", 0));
    DRAW_OBSTACLES = static_cast<bool>((int)j.value("DRAW_OBSTACLES", 1));
    DRAW_ROBOTS = static_cast<bool>((int)j.value("DRAW_ROBOTS", 1));
    DRAW_TRAILS = static_cast<bool>((int)j.value("DRAW_TRAILS", 0));

    // Simulation parameters
    SEED = j["SEED"];
    TIMESTEP = j["TIMESTEP"];
    MAX_TIME = static_cast<float>(j["MAX_TIME"]);
    MAX_TIMESTEP = j["MAX_TIMESTEP"];
    WARMUP_TIME = j.value("WARMUP_TIME", 30.f);
    NEW_OBSTACLES_NEEDED = static_cast<bool>((int)j["NEW_OBSTACLES_NEEDED"]);
    NEW_ROBOTS_NEEDED = static_cast<bool>((int)j["NEW_ROBOTS_NEEDED"]);
    NUM_ROBOTS = j["NUM_ROBOTS"];
    T_HORIZON = j["T_HORIZON"];
    MAX_HORIZON_DIST = j.value("MAX_HORIZON_DIST", 15);
    ROBOT_RADIUS = j["ROBOT_RADIUS"];
    COMMUNICATION_RADIUS = j["COMMUNICATION_RADIUS"];
    MAX_SPEED = j["MAX_SPEED"];
    COMMS_FAILURE_RATE = j["COMMS_FAILURE_RATE"];
    FORMATION = j["FORMATION"];
    OBSTACLE_FILE = j["OBSTACLE_FILE"];
    ITERATE_STEPS = j.value("ITERATE_STEPS", 1);

    // GBP parameters
    NUM_ITERS = j["NUM_ITERS"];
    SIGMA_FACTOR_DYNAMICS = j["SIGMA_FACTOR_DYNAMICS"];
    SIGMA_FACTOR_INTERROBOT = j["SIGMA_FACTOR_INTERROBOT"];
    SIGMA_FACTOR_OBSTACLE = j["SIGMA_FACTOR_OBSTACLE"];
    SIGMA_FACTOR_DYNAMIC_OBSTACLE = j.value("SIGMA_FACTOR_DYNAMIC_OBSTACLE", 0.05);
    USE_DYNAMIC_OBS_FAC = static_cast<bool>((int)j.value("USE_DYNAMIC_OBS_FAC", 1));

    // Dynamic Obstacle parameters
    RBF_GAMMA = j.value("RBF_GAMMA", 1.0);
    NUM_NEIGHBOURS = j.value("NUM_NEIGHBOURS", 1);
    OBSTALCE_SENSOR_RADIUS = j.value("OBSTALCE_SENSOR_RADIUS", 3.0);
    DEFAULT_OBS_SPEED = j.value("DEFAULT_OBS_SPEED", 1.0);
    
    // Dynamic Obstacle type toggles
    ENABLE_BUSES = static_cast<bool>((int)j.value("ENABLE_BUSES", 1));
    ENABLE_VANS = static_cast<bool>((int)j.value("ENABLE_VANS", 1));
    ENABLE_PEDESTRIANS = static_cast<bool>((int)j.value("ENABLE_PEDESTRIANS", 1));
    
    // Spawn parameters for junction_twoway formation
    ROBOT_SPAWN_MEAN_RATE = j.value("ROBOT_SPAWN_MEAN_RATE", 5.0);
    ROBOT_SPAWN_MIN_HEADWAY = j.value("ROBOT_SPAWN_MIN_HEADWAY", 1.0);
    BUS_SPAWN_MEAN_RATE = j.value("BUS_SPAWN_MEAN_RATE", 50.0);
    BUS_SPAWN_MIN_HEADWAY = j.value("BUS_SPAWN_MIN_HEADWAY", 20.0);
    VAN_SPAWN_MEAN_RATE = j.value("VAN_SPAWN_MEAN_RATE", 40.0);  // 0.8 * 50.0
    VAN_SPAWN_MIN_HEADWAY = j.value("VAN_SPAWN_MIN_HEADWAY", 16.0);  // 0.8 * 20.0
    PEDESTRIAN_SPAWN_MEAN_RATE = j.value("PEDESTRIAN_SPAWN_MEAN_RATE", 0.01);
    PEDESTRIAN_SPAWN_MIN_HEADWAY = j.value("PEDESTRIAN_SPAWN_MIN_HEADWAY", 0.0);
}

Globals::Globals(){};

/*****************************************************************/
// Allows for parsing of an external config file
/*****************************************************************/
int Globals::parse_global_args(DArgs::DArgs &dargs)
{
    // Argument parser
    this->CONFIG_FILE = dargs("--cfg", "config_file", this->CONFIG_FILE);
    
    if (!dargs.check())
    {
        dargs.print_help();
        print("Incorrect arguments!");
        return EXIT_FAILURE;
    }

    std::ifstream my_config_file(CONFIG_FILE);
    assert(my_config_file && "Couldn't find the config file");
    parse_global_args(my_config_file);
    post_parsing();

    return 0;
};

/*****************************************************************/
// Any checks on the input configs should go here.
/*****************************************************************/
void Globals::post_parsing()
{
    // Cap max speed, since it should be <= ROBOT_RADIUS/2.f / TIMESTEP:
    // In one timestep a robot should not move more than half of its radius
    // (since we plan for discrete timesteps)
    if (MAX_SPEED > ROBOT_RADIUS/2.f/TIMESTEP){
        MAX_SPEED = ROBOT_RADIUS/2.f/TIMESTEP;
        print("Capping MAX_SPEED parameter at ", MAX_SPEED);
    }
    T0 = ROBOT_RADIUS/2.f / MAX_SPEED; // Time between current state and next state of planned path
}
