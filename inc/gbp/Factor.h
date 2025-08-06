/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include "Simulator.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Utils.h>
#include <DynamicObstacle.h>
#include <gbp/GBPCore.h>
#include <tuple>
#include <raylib.h>
#include <algorithm>

extern Globals globals;

using Eigen::last;
using Eigen::seq;
using Eigen::seqN;

// Forward declaration
class Variable;

// Types of factors defined. Default is DEFAULT_FACTOR
enum FactorType
{
    DEFAULT_FACTOR = 0,
    DYNAMICS_FACTOR = 1,
    INTERROBOT_FACTOR = 2,
    OBSTACLE_FACTOR = 3,
    DYNAMIC_OBSTACLE_FACTOR = 4,
};
/*****************************************************************************************/
// Factor used in GBP
/*****************************************************************************************/
class Factor
{
public:
    Simulator *sim_;                    // Pointer to simulator
    int f_id_;                          // Factor id
    int r_id_;                          // Robot id this factor belongs to
    Key key_;                           // Factor key = {r_id_, f_id_}
    int other_rid_;                     // id of other connected robot (if this is an inter-robot factor)
    int n_dofs_;                        // n_dofs of the variables connected to this factor
    Eigen::VectorXd z_;                 // Measurement
    Eigen::MatrixXd h_, J_;             // Stored values of measurement function h_func_() and Jacobian J_func_()
    Eigen::VectorXd X_;                 // Stored linearisation point
    Eigen::MatrixXd meas_model_lambda_; // Precision of measurement model
    Mailbox inbox_, outbox_, last_outbox_;
    FactorType factor_type_ = DEFAULT_FACTOR;
    float delta_jac = 1e-8;                              // Delta used for first order jacobian calculation
    bool initialised_ = false;                           // Becomes true when Jacobian calculated for the first time
    bool linear_ = false;                                // True is factor is linear (avoids recomputation of Jacobian)
    bool skip_flag = false;                              // Flag to skip factor update if required
    std::vector<std::shared_ptr<Variable>> variables_{}; // Vector of pointers to the connected variables. Order of variables matters
    virtual bool skip_factor()
    { // Default function to set skip flag
        skip_flag = false;
        return skip_flag;
    };
    bool dbg = false; // Flag to signify if debug message has been printed, for debugging purposes only.

private:
    // Working memory for Cholesky optimization to avoid repeated allocations
    mutable Eigen::LLT<Eigen::MatrixXd> llt_solver_;    // Reusable LLT solver
    mutable Eigen::LDLT<Eigen::MatrixXd> ldlt_solver_;  // Reusable LDLT solver

public:

    // Function declarations
    Factor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
           float sigma, const Eigen::VectorXd &measurement,
           int n_dofs = 4);

    ~Factor();

    virtual void draw() {};

    virtual Eigen::MatrixXd h_func_(const Eigen::VectorXd &X) = 0;

    virtual Eigen::MatrixXd J_func_(const Eigen::VectorXd &X);

    Eigen::MatrixXd jacobianFirstOrder(const Eigen::VectorXd &X0, bool central_diff = false);

    virtual Eigen::VectorXd residual() { return z_ - h_; };

    bool update_factor();

    Message marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx, int marg_idx);
};

/********************************************************************************************/
/********************************************************************************************/
//                      CUSTOM FACTORS SPECIFIC TO THE PROBLEM
// Create a new factor definition as shown with these examples.
// You may create a new factor_type_, in the enum in Factor.h (optional, default type is DEFAULT_FACTOR)
// Create a measurement function h_func_() and optionally Jacobian J_func_().

/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
class DynamicsFactor : public Factor
{
public:
    DynamicsFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
                   float sigma, const Eigen::VectorXd &measurement, float dt);
    // Constant velocity model
    Eigen::MatrixXd h_func_(const Eigen::VectorXd &X) override;
    Eigen::MatrixXd J_func_(const Eigen::VectorXd &X) override;
    void draw() override;
};

/********************************************************************************************/
/* Interrobot factor: for avoidance of other robots */
// This factor results in a high energy or cost if two robots are planning to be in the same
// position at the same timestep (collision). This factor is created between variables of two robots.
// The factor has 0 energy if the variables are further away than the safety distance. skip_ = true in this case.
/********************************************************************************************/
class InterrobotFactor : public Factor
{
    double safety_distance_;

public:
    InterrobotFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
                     float sigma, const Eigen::VectorXd &measurement,
                     float robot_radius);
    Eigen::MatrixXd h_func_(const Eigen::VectorXd &X) override;
    Eigen::MatrixXd J_func_(const Eigen::VectorXd &X) override;
    bool skip_factor() override;
    void draw() override;
};

/********************************************************************************************/
// Obstacle factor for static obstacles in the scene. This factor takes a pointer to the obstacle image from the Simulator.
// Note. in the obstacle image, white areas represent obstacles (as they have a value of 1).
// The input image to the simulator is opposite, which is why it needs to be inverted.
// The delta used in the first order jacobian calculation is chosen such that it represents one pixel in the image.
/********************************************************************************************/
class ObstacleFactor : public Factor
{
    Image *p_obstacleImage_;

public:
    ObstacleFactor(Simulator *sim, int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
                   float sigma, const Eigen::VectorXd &measurement, Image *p_obstacleImage);
    Eigen::MatrixXd h_func_(const Eigen::VectorXd &X);
};

/********************************************************************************************/
// Factor for dynamic obstalces.
/********************************************************************************************/
class DynamicObstacleFactor : public Factor
{
    float delta_t_;
    float robot_radius_;
    double safety_distance_;
    std::vector<std::pair<Eigen::Vector2d, double>> neighbours_;
    
public:
    std::shared_ptr<DynamicObstacle> obs_;
    DynamicObstacleFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
                          float sigma, const Eigen::VectorXd &measurement, float robot_radius, std::shared_ptr<DynamicObstacle> obs);

    double gaussianRBF(double dist_squared) const { return std::exp(-globals.RBF_GAMMA * dist_squared); }
    Eigen::MatrixXd h_func_(const Eigen::VectorXd &X) override;
    Eigen::MatrixXd J_func_(const Eigen::VectorXd &X) override;
    bool skip_factor() override;
    void draw() override;
};