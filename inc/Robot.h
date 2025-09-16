/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include "Simulator.h"
#include <memory>
#include <vector>
#include <deque>
#include <Utils.h>
#include <gbp/GBPCore.h>
#include <gbp/Factor.h>
#include <gbp/Factorgraph.h>
#include <limits>

extern Globals globals;

/***************************************************************************/
// Creates a robot. Inputs required are :
//      - Pointer to the simulator
//      - A robot id rid (should be taken from simulator->next_rid_++),
//      - A dequeue of waypoints (which are 4 dimensional [x,y,xdot,ydot])
//      - Robot radius
//      - Colour
// This is a derived class from the FactorGraph class
/***************************************************************************/
class Robot : public FactorGraph {
public:
    // Constructor
    Robot(Simulator* sim,
          int rid,
          std::deque<Eigen::VectorXd> waypoints,
          RobotType type = RobotType::SPHERE,
          float scale = 1.f,
          float radius = globals.ROBOT_RADIUS,
          Color color =  GREEN);
    ~Robot();


    Simulator* sim_;                            // Pointer to the simulator
    int dofs_;                                  // Degrees of freedom of the robot (4 or 5)
    int rid_ = 0;                               // Robot id
    std::deque<Eigen::VectorXd> waypoints_{};   // Dequeue of waypoints (whenever the robot reaches a point, it is popped off the front of the dequeue)
    float robot_radius_ = 1.;                   // Robot radius
    Color color_ = DARKGREEN;                   // Colour of robot
    RobotType robot_type_ = RobotType::SPHERE;  // Type of robot model to use
    Eigen::Vector2d robot_dimensions_;          // Robot dimensions (width, length) for OBB collision detection

    int num_variables_;                         // Number of variables in the planned path (assumed to be the same for all robots)
    std::vector<int> connected_r_ids_{};        // List of robot ids that are currently connected via inter-robot factors to this robot
    std::vector<int> neighbours_{};             // List of robot ids that are within comms radius of this robot
    std::vector<int> connected_obs_ids_{};      // List of obstacle ids that currently has dynamic factors connected to this robot 
    Image* p_obstacleImage;                     // Pointer to image representing the obstacles in the environment
    float height_3D_ = 0.f;                     // Height out of plane (for 3d visualisation only)
    float scale_ = 1.f;                         // Scale factor of the model
    Eigen::VectorXd position_;                  // Position of the robot (equivalent to taking the [x,y] of the current state of the robot)
    float orientation_ = 0.0f;                  // Current orientation of the robot in radians (for visualization)
    double default_angle_offset_ = 0.0;         // Default angle offset of model
    
    // Task/pause timer functionality
    float task_timer_ = 0.f;                    // Countdown timer for task completion [seconds]
    bool next_wp_is_task_ = false;              // Flag indicating if robot is performing a task
    bool task_active_ = false;                  // Flag indicating if horizon has reached task waypoint

    double base_path_length_ = 0.0;             // Store minimal path length (sum of waypoint segment lengths)
    Eigen::Vector2d prev_RA_ = Eigen::Vector2d(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    
    // Trail functionality
    std::deque<Eigen::Vector2d> position_trail_; // Store previous positions for trail rendering
    static const int MAX_TRAIL_LENGTH = 1000;     // Maximum number of trail positions to store


    /****************************************/
    //Functions
    /****************************************/
    /* Change the prior of the Current state */
    void updateCurrent();

    /* Change the prior of the Horizon state */    
    void updateHorizon();

    /* Detect slign shot states */
    bool isSlignshot();
    
    /* Reinitialise variable chain */
    void reinitialiseVariables(Eigen::Ref<const Eigen::Vector2d> anchor);

    /***************************************************************************************************/
    // For new neighbours of a robot, create inter-robot factors if they don't exist. 
    // Delete existing inter-robot factors for faraway robots
    /***************************************************************************************************/    
    void updateInterrobotFactors();
    void createInterrobotFactors(std::shared_ptr<Robot> other_robot);
    void deleteInterrobotFactors(std::shared_ptr<Robot> other_robot);
    
    /***************************************************************************************************/
    // For new obstacles, create dynamic obstacle factors if they don't exist.
    // Delete existing inter-robot factors for out-of-bounds obstacles.
    /***************************************************************************************************/    
    void updateDynamicObstacleFactors();
    void createDynamicObstacleFactors(std::shared_ptr<DynamicObstacle> obs);
    void deleteDynamicObstacleFactors(int oid);
    double getDistToObs(std::shared_ptr<DynamicObstacle> obs);

    /***************************************************************************************************/    
    // Drawing function
    /***************************************************************************************************/    
    void draw();

    /*******************************************************************************************/   
    // Access operator to get a pointer to a variable from the robot.
    /*******************************************************************************************/   
    std::shared_ptr<Variable>& operator[] (const int& v_id){
        int n = variables_.size();
        int search_vid = ((n + v_id) % n + n) % n;
        auto it = variables_.begin();
        std::advance(it, search_vid);
        return it->second;
    }    


};

