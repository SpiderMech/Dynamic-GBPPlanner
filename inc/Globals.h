/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <cmath>
#include <raylib.h>
#include <DArgs.h>
#include <fstream>
#include "json.hpp"

// Simulation modes
enum MODES_LIST {SimNone, Timestep, Iterate, Help};

/***********************************************************************************************/
// Global structure. The values here are mostly set using the provided config.json file.
/***********************************************************************************************/
class Globals {
    public:

    // Basic parameters
    const char* WINDOW_TITLE = "Distributing Multirobot Motion Planning with Gaussian Belief Propogation";
    bool RUN = true;
    std::string CONFIG_FILE = "../config/config.json";      // Default config file
    std::string OBSTACLE_FILE;                              // Binary image for obstacles
    std::string ASSETS_DIR;                                 // Directory for Assets
    int N_DOFS;                                             // Degrees of freedom, defaults to 4 (x, y, xdot, ydot), can be 5 (x, y, xdot, ydot)
    MODES_LIST SIM_MODE = Timestep;                         // Simulation mode to begin with
    MODES_LIST LAST_SIM_MODE = Timestep;                    // Storage of Simulation mode (if time is paused eg.)
    bool EVAL = false;                                      // If running in eval mode
    
    // Display parameters
    bool DISPLAY;                                           // Show display or not
    int WORLD_SZ;                                           // [m]
    int SCREEN_SZ;                                          // [pixels]
    bool VERBOSE = false;                                   // Flag for dbg message verbosity
    bool DRAW_INTERROBOT;                                   // Toggle display of inter-robot connections
    bool DRAW_PATH;                                         // Toggle display of planned paths
    bool DRAW_WAYPOINTS;                                    // Toggle display of path planning goals
    bool DRAW_FACTORS;                                      // Toggle display of factors
    bool DRAW_OBSTACLES = true;                             // Toggle display of obstacles
    bool DRAW_ROBOTS = true;                                // Toggle display of robots
    
    // Simulation parameters
    int SEED;                                               // Random Seed 
    float TIMESTEP;                                         // Simulation timestep [s]
    int MAX_TIMESTEP;                                       // Exit simulation if more timesteps than this
    float MAX_TIME;                                         // Exit simulation if more time [s] than this or globals.TIMESTEP * timesteps, which ever is earlier
    bool NEW_OBSTACLES_NEEDED;                              // Whether to generate new obstacles
    bool NEW_ROBOTS_NEEDED;                                 // Whether to generate new robots   
    int NUM_ROBOTS;                                         // Number of robots (if no new robots are to be added)
    float T_HORIZON;                                        // Planning horizon [s]
    float MAX_HORIZON_DIST;                                 // [m]. Maximum distance the horizon state can be away from current state.
    float ROBOT_RADIUS;                                     // [m]
    float COMMUNICATION_RADIUS;                             // [m] Inter-robot factors created if robots are within this range of each other
    float MAX_SPEED;                                        // [m/s]
    float COMMS_FAILURE_RATE;                               // Proportion of robots [0,1] that do not communicate
    int LOOKAHEAD_MULTIPLE = 3;                             // Parameter affecting how planned path is spaced out in time
    std::string FORMATION;                                  // Robot formation (CIRCLE or JUNCTION)
    float T0;                                               // Time between current state and next state of planned path
    int ITERATE_STEPS;                                      // Number of ticks before pausing when in Iterate mode.

    // GBP parameters
    float SIGMA_POSE_FIXED = 1e-15;                         // Sigma for Unary pose factor on current and horizon states
    float SIGMA_FACTOR_DYNAMICS;                            // Sigma for Dynamics factors
    float SIGMA_FACTOR_INTERROBOT;                          // Sigma for Interrobot factor
    float SIGMA_FACTOR_OBSTACLE;                            // Sigma for Static obstacle factors
    float SIGMA_FACTOR_DYNAMIC_OBSTACLE;                    // Sigma for Dynamic obstacle factors
    int NUM_ITERS;                                          // Number of iterations of GBP per timestep
    float DAMPING = 0.;                                     // Damping amount (not used in this work)
    bool USE_DYNAMIC_OBS_FAC = true;                        // Flag for using dynamic obstacle factor                               

    // Dynamic Obstacle parameters
    double RBF_GAMMA;                               // Shape parameter of Gaussian RBFs of obstacle points (gamma = 1/2*sigma^2)
    int NUM_NEIGHBOURS;                             // Number of RBFs to combine for Dynamics Obstacle Factor
    double OBSTALCE_SENSOR_RADIUS;                  // Radius of area where obstacles inside are not skipped.
    double DEFAULT_OBS_SPEED;                       // [m/s]
    
    // Dynamic Obstacle type toggles
    bool ENABLE_BUSES = true;                       // Enable/disable bus obstacles
    bool ENABLE_VANS = true;                        // Enable/disable van obstacles  
    bool ENABLE_PEDESTRIANS = true;                 // Enable/disable pedestrian obstacles
    
    Globals();
    int parse_global_args(DArgs::DArgs& dargs);
    void parse_global_args(std::ifstream& config_file);
    void post_parsing();
};