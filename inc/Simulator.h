/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once

#include <map>
#include <memory>
#include <algorithm>
#include <unordered_set>

#include <Utils.h>
#include <gbp/GBPCore.h>
#include <Graphics.h>
#include <GeometryUtils.h>
#include <gbp/Variable.h>
#include <nanoflann.h>

#include <raylib.h>
#include <rlights.h>
#include <nanoflann.h>
#include <KDTreeMapOfVectorsAdaptor.h>

#include "cnpy/cnpy.h"
#include "Spawn.hpp"

class Robot;
class DynamicObstacle;
class Graphics;
class TreeOfRobots;
class MetricsCollector;
struct MotionOptions;

/************************************************************************************/
// The main Simulator. This is where the magic happens.
/************************************************************************************/
class Simulator {
public:
    friend class Robot;
    friend class Factor;

    // Constructor
    Simulator();
    ~Simulator();

    // Pointer to Graphics class which hold all the camera, graphics and models for display
    Graphics* graphics;

    // kd-tree to store the positions of the robots at each timestep.
    // This is used for calculating the neighbours of robots blazingly fast.
    typedef KDTreeMapOfVectorsAdaptor<std::map<int,std::vector<double>>> KDTree;
    std::map<int, std::vector<double>> robot_positions_{{0,{0.,0.}}};
    KDTree* treeOfRobots_;
    
    // Spawn processing methods (moved from lambdas for efficiency)
    bool isSpawnClear(const SpawnRequest& r, double margin);
    bool admitSpawnRequest(const SpawnRequest& r);

    // Image representing the obstacles in the environment
    Image obstacleImg;
    cnpy::NpyArray obstacleSDF;
    cnpy::NpyArray obstacleJacX;
    cnpy::NpyArray obstacleJacY;
    float* sdf_ptr = nullptr;
    float* jacx_ptr = nullptr;
    float* jacy_ptr = nullptr;
    bool has_sdf  = false;
    bool has_jac  = false;

    int next_rid_ = 0;                                           // New robots will use this rid. It should be ++ incremented when this happens
    int next_vid_ = 0;                                           // New variables will use this vid. It should be ++ incremented when this happens
    int next_fid_ = 0;                                           // New factors will use this fid. It should be ++ incremented when this happens
    int next_oid_ = 0;                                           // New dynamic obstacles will use this oid. It should be ++ incremented when this happens
    uint32_t clock_ = 0;                                         // Simulation clock (timesteps)                   
    std::map<int, std::shared_ptr<Robot>> robots_;               // Map containing smart pointers to all robots, accessed by their rid.
    std::map<int, std::shared_ptr<DynamicObstacle>> obstacles_;  // Map containing smart pointers to all robots, accessed by their rid.
    bool new_robots_needed_ = true;                              // Whether or not to create new robots. (Some formations are dynamicaly changing)
    bool new_obstacles_needed_ = true;                           // Whether or not to create new obstacles. (Some formations are dynamicaly changing)
    bool symmetric_factors = false;                              // If true, when inter-robot factors need to be created between two robots,
                                                                 // a pair of factors is created (one belonging to each robot). This becomes a redundancy.

    MetricsCollector* metrics;                                   // Helper class to record metrics during evaluation
    std::vector<SpawnGate> spawn_gates_;                         // Vector of structures SpawnGate that helps prevent collision-spawning
    bool robots_initialised_;                                    // Flag for initialisations related to robots
    bool obstacles_initialised_;                                 // Flag for initialisations related to robots
    
    std::vector<PoissonSpawner> robot_spawners_;                 // Robot spawners for junction_twoway formation
    std::vector<MotionOptions> motion_options_;                  // Motion options for obstacle formations
    std::vector<PoissonSpawner> bus_spawners_;                   // Bus obstacle spawners
    std::vector<PoissonSpawner> van_spawners_;                   // Van obstacle spawners
    PoissonSpawner pedestrian_spawner_;                          // Pedestrian obstacle spawner
    std::unordered_set<uint64_t> seen_collision_pairs_;         // Track seen collision pairs

    int frame_count = 0;                                        // Frame count for GIF capture

    /*******************************************************************************/
    // Set up environment related structures based on formation
    /*******************************************************************************/
    void setupEnvironment();

    /*******************************************************************************/
    // Admit spawn requests for robots and obstacles jointly
    /*******************************************************************************/
    void processSpawnRequests();

    /*******************************************************************************/
    // Create new robots if needed. Handles deletion of robots out of bounds. 
    // New formations must create spawn requests to spawn gates and optionally "robots_to_delete"
    /*******************************************************************************/    
    void createOrDeleteRobots();

    /*******************************************************************************/
    // Create new dynamic (moving/static) obstacles. Handles deletion of obstacles out of bounds. 
    // Works in a similar fashion to robot creation and deletion, but for dynamic obstacles.
    /*******************************************************************************/    
    void createOrDeleteObstacles();

    /*******************************************************************************/
    // Set a proportion of robots to not perform inter-robot communications
    /*******************************************************************************/
    void setCommsFailure(float failure_rate=globals.COMMS_FAILURE_RATE);

    /*******************************************************************************/
    // Timestep loop of simulator.
    /*******************************************************************************/
    void timestep();

    /*******************************************************************************/
    // Drawing graphics.
    /*******************************************************************************/
    void draw();

    /*******************************************************************************/
    // Use a kd-tree to perform a radius search for neighbours of a robot within comms. range
    // (Updates the neighbours_ of a robot)
    /*******************************************************************************/    
    void calculateRobotNeighbours(std::map<int,std::shared_ptr<Robot>>& robots);
    
    /*******************************************************************************/
    // Update robot positions and reindx KD-Tree
    /*******************************************************************************/    
    void updateRobotKDTree(std::map<int, std::shared_ptr<Robot>> &robots);

    /*******************************************************************************/
    // Handles keypresses and mouse input, and updates camera.
    /*******************************************************************************/
    void eventHandler();

    /*******************************************************************************/
    // Deletes the robot from the simulator's robots_, as well as any variable/factors associated.
    /*******************************************************************************/
    void deleteRobot(std::shared_ptr<Robot> robot);

    /*******************************************************************************/
    // Detect and log collisions between robots and obstacles
    /*******************************************************************************/
    void detectCollisions();
};
