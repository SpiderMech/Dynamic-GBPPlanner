/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <Utils.h>
#include <DynamicObstacle.h>
#include <gbp/GBPCore.h>
#include <gbp/Factor.h>
#include <gbp/Variable.h>
#include <GeometryUtils.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <raylib.h>

// Helper: smallest signed angle difference a-b in (-pi, pi]
static inline double angle_diff(double a, double b) {
    double d = a - b;
    return std::atan2(std::sin(d), std::cos(d));
}

/*****************************************************************************************************/
// Factor constructor
// Inputs:
//  - factor id (taken from simulator->next_f_id_++)
//  - robot id that this factor belongs to.
//  - A vector of pointers to Variables that the factor is to be connected to. Note, the order of the variables matters.
//  - sigma: factor strength. The factor precision Lambda = sigma^-2 * Identity
//  - measurement z: Eigen::VectorXd, must be same size as the output of the measurement function h().
//  - n_dofs is the number of degrees of freedom of the variables this factor is connected to. (eg. 4 for [x,y,xdot,ydot])
/*****************************************************************************************************/
Factor::Factor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
               float sigma, const Eigen::VectorXd &measurement,
               int n_dofs)
    : f_id_(f_id), r_id_(r_id), key_(r_id, f_id), variables_(variables), z_(measurement), n_dofs_(n_dofs)
{

    // Initialise precision of the measurement function
    this->meas_model_lambda_ = Eigen::MatrixXd::Identity(z_.rows(), z_.rows()) / pow(sigma, 2.);

    // Initialise empty inbox and outbox
    int n_dofs_total = 0;
    int n_dofs_var;
    for (auto var : variables_)
    {
        n_dofs_var = var->n_dofs_;
        Message zero_msg(n_dofs_var);
        inbox_[var->key_] = zero_msg;
        outbox_[var->key_] = zero_msg;
        n_dofs_total += n_dofs_var;
    }

    // This parameter useful if the factor is connected to another robot
    other_rid_ = r_id_;

    // Initialise empty linearisation point
    X_ = Eigen::VectorXd::Zero(n_dofs_total);
}

/*****************************************************************************************************/
// Destructor
/*****************************************************************************************************/
Factor::~Factor()
{
}

/*****************************************************************************************************/
// Default measurement function h_func_() is the identity function: it returns the variable.
/*****************************************************************************************************/
Eigen::MatrixXd h_func_(const Eigen::VectorXd &X) { return X; };
/*****************************************************************************************************/
// Default measurement function Jacobian J_func_() is the first order taylor series jacobian by default.
// When defining new factors, custom h_func_() and J_func_() must be defined, otherwise defaults are used.
/*****************************************************************************************************/
Eigen::MatrixXd Factor::J_func_(const Eigen::VectorXd &X) { return this->jacobianFirstOrder(X); };

Eigen::MatrixXd Factor::jacobianFirstOrder(const Eigen::VectorXd &X0, bool central_diff)
{
    Eigen::MatrixXd h0 = h_func_(X0); // Value at lin point
    Eigen::MatrixXd jac_out = Eigen::MatrixXd::Zero(h0.size(), X0.size());
    for (int i = 0; i < X0.size(); i++)
    {
        if (central_diff)
        {
            Eigen::VectorXd X_plus = X0;
            Eigen::VectorXd X_minus = X0;

            X_plus(i) += delta_jac;
            X_minus(i) -= delta_jac;

            Eigen::MatrixXd h_plus = h_func_(X_plus);
            Eigen::MatrixXd h_minus = h_func_(X_minus);

            jac_out(Eigen::all, i) = (h_plus - h_minus) / (2.f * delta_jac);
        }
        else
        {                                                                // forward difference
            Eigen::VectorXd X_copy = X0;                                 // Copy of lin point
            X_copy(i) += delta_jac;                                      // Perturb by delta
            jac_out(Eigen::all, i) = (h_func_(X_copy) - h0) / delta_jac; // Derivative (first order)
        }
    }
    return jac_out;
}

/*****************************************************************************************************/
// Main section: Factor update:
// Messages from connected variables are aggregated. The beliefs are used to create the linearisation point X_.
// The Factor potential is calculated using h_func_ and J_func_
// The factor precision and information is created, and then marginalised to create outgoing messages to its connected variables.
/*****************************************************************************************************/
bool Factor::update_factor()
{

    // Messages from connected variables are aggregated.
    // The beliefs are used to create the linearisation point X_.
    int idx = 0;
    int n_dofs;
    for (int v = 0; v < variables_.size(); v++)
    {
        n_dofs = variables_[v]->n_dofs_;
        auto &[_, __, mu_belief] = this->inbox_[variables_[v]->key_];
        X_(seqN(idx, n_dofs)) = mu_belief;
        idx += n_dofs;
    }

    // *Depending on the problem*, we may need to skip computation of this factor.
    // eg. to avoid extra computation, factor may not be required if two connected variables are too far apart.
    // in which case send out a Zero Message.
    if (this->skip_factor())
    {
        for (auto var : variables_)
        {
            this->outbox_[var->key_] = Message(var->n_dofs_);
        }
        return false;
    }

    // The Factor potential and linearised Factor Precision and Information is calculated using h_func_ and J_func_
    // residual() is by default (z - h_func_(X))
    // Skip calculation of Jacobian if the factor is linear and Jacobian has already been computed once
    h_ = h_func_(X_);
    J_ = (this->linear_ && this->initialised_) ? J_ : this->J_func_(X_);
    Eigen::MatrixXd factor_lam_potential = J_.transpose() * meas_model_lambda_ * J_;
    Eigen::VectorXd factor_eta_potential = (J_.transpose() * meas_model_lambda_) * (J_ * X_ + residual());
    this->initialised_ = true;

    //  Update factor precision and information with incoming messages from connected variables.
    int marginalisation_idx = 0;
    for (int v_out_idx = 0; v_out_idx < variables_.size(); v_out_idx++)
    {
        auto var_out = variables_[v_out_idx];
        // Initialise with factor values
        Eigen::VectorXd factor_eta = factor_eta_potential;
        Eigen::MatrixXd factor_lam = factor_lam_potential;

        // Combine the factor with the belief from other variables apart from the receiving variable
        int idx_v = 0;
        for (int v_idx = 0; v_idx < variables_.size(); v_idx++)
        {
            int n_dofs = variables_[v_idx]->n_dofs_;
            if (variables_[v_idx]->key_ != var_out->key_)
            {
                auto [eta_belief, lam_belief, _] = inbox_[variables_[v_idx]->key_];
                factor_eta(seqN(idx_v, n_dofs)) += eta_belief;
                factor_lam(seqN(idx_v, n_dofs), seqN(idx_v, n_dofs)) += lam_belief;
            }
            idx_v += n_dofs;
        }

        // Marginalise the Factor Precision and Information to send to the relevant variable
        outbox_[var_out->key_] = marginalise_factor_dist(factor_eta, factor_lam, v_out_idx, marginalisation_idx);
        marginalisation_idx += var_out->n_dofs_;
    }

    // debug msg flag
    if (dbg)
    {
        print("********************************************");
        dbg = false;
    }
    return true;
}

/*****************************************************************************************************/
// Marginalise the factor Precision and Information and create the outgoing message to the variable
/*****************************************************************************************************/
Message Factor::marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx, int marg_idx)
{
    // Marginalisation only needed if factor is connected to >1 variables
    int n_dofs = variables_[var_idx]->n_dofs_;
    if (eta.size() == n_dofs)
        return Message{eta, Lam};

    Eigen::VectorXd eta_a(n_dofs), eta_b(eta.size() - n_dofs);
    eta_a = eta(seqN(marg_idx, n_dofs));
    eta_b << eta(seq(0, marg_idx - 1)), eta(seq(marg_idx + n_dofs, last));

    Eigen::MatrixXd lam_aa(n_dofs, n_dofs), lam_ab(n_dofs, Lam.cols() - n_dofs);
    Eigen::MatrixXd lam_ba(Lam.rows() - n_dofs, n_dofs), lam_bb(Lam.rows() - n_dofs, Lam.cols() - n_dofs);
    lam_aa << Lam(seqN(marg_idx, n_dofs), seqN(marg_idx, n_dofs));
    lam_ab << Lam(seqN(marg_idx, n_dofs), seq(0, marg_idx - 1)), Lam(seqN(marg_idx, n_dofs), seq(marg_idx + n_dofs, last));
    lam_ba << Lam(seq(0, marg_idx - 1), seq(marg_idx, marg_idx + n_dofs - 1)), Lam(seq(marg_idx + n_dofs, last), seqN(marg_idx, n_dofs));
    lam_bb << Lam(seq(0, marg_idx - 1), seq(0, marg_idx - 1)), Lam(seq(0, marg_idx - 1), seq(marg_idx + n_dofs, last)),
              Lam(seq(marg_idx + n_dofs, last), seq(0, marg_idx - 1)), Lam(seq(marg_idx + n_dofs, last), seq(marg_idx + n_dofs, last));

    // Use Cholesky decomposition for efficient matrix inversion
    Message marginalised_msg(n_dofs);

    llt_solver_.compute(lam_bb);
    
    if (llt_solver_.info() == Eigen::Success) {
        // Successfully decomposed - solve efficiently
        Eigen::VectorXd solved_eta = llt_solver_.solve(eta_b);
        Eigen::MatrixXd solved_lam = llt_solver_.solve(lam_ba);
        
        marginalised_msg.eta = eta_a - lam_ab * solved_eta;
        marginalised_msg.lambda = lam_aa - lam_ab * solved_lam;
    } else {
        // Fallback to LDLT for indefinite matrices
        ldlt_solver_.compute(lam_bb);
        
        if (ldlt_solver_.info() == Eigen::Success && ldlt_solver_.isPositive()) {
            Eigen::VectorXd solved_eta = ldlt_solver_.solve(eta_b);
            Eigen::MatrixXd solved_lam = ldlt_solver_.solve(lam_ba);
            
            marginalised_msg.eta = eta_a - lam_ab * solved_eta;
            marginalised_msg.lambda = lam_aa - lam_ab * solved_lam;
        } else {
            // Final fallback to direct inverse for problematic cases
            Eigen::MatrixXd lam_bb_inv = lam_bb.inverse();
            marginalised_msg.eta = eta_a - lam_ab * lam_bb_inv * eta_b;
            marginalised_msg.lambda = lam_aa - lam_ab * lam_bb_inv * lam_ba;
        }
    }
    
    if (!marginalised_msg.lambda.allFinite())
        marginalised_msg.setZero();

    return marginalised_msg;
}

/********************************************************************************************/
/********************************************************************************************/
//                      CUSTOM FACTORS SPECIFIC TO THE PROBLEM
// Create a new factor definition as shown with these examples.
// You may create a new factor_type_, in the enum in Factor.h (optional, default type is DEFAULT_FACTOR)
// Create a measurement function h_func_() and optionally Jacobian J_func_().

/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
DynamicsFactor::DynamicsFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables, int n_dofs,
                               float sigma, const Eigen::VectorXd &measurement, float dt)
    : Factor{f_id, r_id, variables, sigma, measurement, n_dofs}, dt_(dt)
{
    factor_type_ = DYNAMICS_FACTOR;
    
    // For 5D state: [x, y, xdot, ydot, theta]
    // With instantaneous alignment model
    if (n_dofs_ == 5) {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2, 2);
        Eigen::MatrixXd Qc_inv = pow(sigma, -2.) * I;
        
        // Build the full 5x5 precision matrix
        Eigen::MatrixXd Qi_inv = Eigen::MatrixXd::Zero(5, 5);
        // Position and velocity components (same as 4D case)
        Qi_inv.block<2,2>(0, 0) = 12. * pow(dt_, -3.) * Qc_inv;
        Qi_inv.block<2,2>(0, 2) = -6. * pow(dt_, -2.) * Qc_inv;
        Qi_inv.block<2,2>(2, 0) = -6. * pow(dt_, -2.) * Qc_inv;
        Qi_inv.block<2,2>(2, 2) = 4. / dt_ * Qc_inv;
        
        // Orientation component - much looser constraint to prevent fighting with velocity
        // The orientation should follow velocity naturally, not be forced
        double sigma_theta = sigma * 10.0;  // Much looser constraint
        double Qc_inv_theta = pow(sigma_theta, -2.);
        Qi_inv(4, 4) = 1. / dt_ * Qc_inv_theta;  // Reduced weight
        
        this->meas_model_lambda_ = Qi_inv;
        
        // For instantaneous alignment, orientation is nonlinear
        this->linear_ = false;
    
    // For 4D state: [x, y, xdot, ydot]
    } else {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2, 2);
        Eigen::MatrixXd O = Eigen::MatrixXd::Zero(2, 2);
        Eigen::MatrixXd Qc_inv = pow(sigma, -2.) * I;

        Eigen::MatrixXd Qi_inv(n_dofs_, n_dofs_);
        Qi_inv << 12. * pow(dt_, -3.) * Qc_inv, -6. * pow(dt_, -2.) * Qc_inv,
                  -6. * pow(dt_, -2.) * Qc_inv, 4. / dt_ * Qc_inv;

        this->meas_model_lambda_ = Qi_inv;
        
        // Store Jacobian as it is linear for 4D
        this->linear_ = true;
        J_ = Eigen::MatrixXd::Zero(n_dofs_, n_dofs_ * 2);
        J_ << I, dt * I, -1 * I, O,
              O, I, O, -1 * I;
    }
}

Eigen::MatrixXd DynamicsFactor::h_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 5) {
        // X contains [x1, y1, xdot1, ydot1, theta1, x2, y2, xdot2, ydot2, theta2]
        Eigen::VectorXd h = Eigen::VectorXd::Zero(5);
        
        // Position constraint: x2 = x1 + xdot1 * dt, use predicted - actual for consistency
        h(0) = X(0) + X(2)*dt_ - X(5);
        h(1) = X(1) + X(3)*dt_ - X(6);
        
        // Velocity constraint: xdot2 = xdot1 (constant velocity)
        h(2) = X(2) - X(7);
        h(3) = X(3) - X(8);
        
        // Orientation constraint: theta should align with velocity direction
        double xdot2 = X(7), ydot2 = X(8);
        double v2 = xdot2*xdot2 + ydot2*ydot2;
        double v_mag = std::sqrt(v2);
        
        // Only apply orientation constraint when moving at reasonable speed
        // This prevents oscillations at low speeds
        double v_min = v0_ * 0.1;  // Minimum velocity to apply constraint
        
        double theta_diff = 0.0;
        if (std::isfinite(X(9)) && v_mag > v_min) {
            // theta should align with velocity direction
            // Since Y is down, we need to negate it for proper angle calculation
            double theta_vel = -wrapAngle(std::atan2(ydot2, xdot2));
            // Use angle_diff for consistent angle wrapping
            theta_diff = angle_diff(theta_vel, X(9));
            
            // Apply smooth transition based on velocity magnitude
            // Use tanh for smoother, bounded transition than sigmoid
            double k = 3.0 / v0_;  // Transition rate
            double w = 0.5 * (1.0 + std::tanh(k * (v_mag - v0_ * 0.3)));
            
            // Scale the error by the weight
            h(4) = w * theta_diff;
        } else {
            // At low speeds, don't constrain orientation
            h(4) = 0.0;
        }
    
        return h;
    } else {
        // Original 4D implementation
        return J_ * X;
    }
}

Eigen::MatrixXd DynamicsFactor::J_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 5) {
        // 5D state Jacobian with instantaneous alignment
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(5, 10);
        
        // Position and velocity constraints Jacobian - same as 4D case
        J(0, 0) = 1;  J(0, 2) = dt_;  J(0, 5) = -1;  // dx
        J(1, 1) = 1;  J(1, 3) = dt_;  J(1, 6) = -1;  // dy
        J(2, 2) = 1;  J(2, 7) = -1;  // dxdot
        J(3, 3) = 1;  J(3, 8) = -1;  // dydot
        
        // Orientation constraint Jacobian (nonlinear due to atan2)
        double xdot2 = X(7), ydot2 = X(8);
        double v2 = xdot2*xdot2 + ydot2*ydot2;
        double v_mag = std::sqrt(v2);
        
        // Minimum velocity threshold
        double v_min = v0_ * 0.1;
        
        if (v_mag > v_min) {
            // Compute weight using tanh for smooth transition
            double k = 3.0 / v0_;
            double tanh_arg = k * (v_mag - v0_ * 0.3);
            double tanh_val = std::tanh(tanh_arg);
            double w = 0.5 * (1.0 + tanh_val);
            
            // d theta_vel / d xdot2, d ydot2 with stronger regularization
            double v2_reg = v2 + v_min * v_min;
            double dthetad_x =  ydot2 / v2_reg;
            double dthetad_y = -xdot2 / v2_reg;
            
            // Product-rule term discarded
            J(4, 7) = w * dthetad_x;  // wrt xdot2
            J(4, 8) = w * dthetad_y;  // wrt ydot2
            J(4, 9) = -w;  // wrt theta2
            
        } else {
            // Below minimum velocity, no constraint
            J.row(4).setZero();
        }
        return J;
    } else {
        // Original 4D implementation
        return J_;
    }
}

void DynamicsFactor::draw()
{
    if (!globals.DRAW_PATH) {
        auto variables = this->variables_;
        Eigen::VectorXd p0 = variables[0]->mu_, p1 = variables[1]->mu_;
        DrawCylinderEx(Vector3{(float)p0(0), globals.ROBOT_RADIUS, (float)p0(1)}, Vector3{(float)p1(0), globals.ROBOT_RADIUS, (float)p1(1)}, 0.1, 0.1, 4, BLACK);
    }
}

/********************************************************************************************/
/* Interrobot factor: for avoidance of other robots */
// This factor results in a high energy or cost if two robots are planning to be in the same
// position at the same timestep (collision). This factor is created between variables of two robots.
// The factor has 0 energy if the variables are further away than the safety distance. skip_ = true in this case.
/********************************************************************************************/

InterrobotFactor::InterrobotFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables, int n_dofs,
                                   float sigma, const Eigen::VectorXd &measurement,
                                   float robot_radius,
                                   const Eigen::Vector2d& robot1_dims,
                                   const Eigen::Vector2d& robot2_dims,
                                   double robot1_angle_offset,
                                   double robot2_angle_offset)
    : Factor{f_id, r_id, variables, sigma, measurement, n_dofs},
      robot1_dimensions_(robot1_dims),
      robot2_dimensions_(robot2_dims),
      robot1_angle_offset_(robot1_angle_offset),
      robot2_angle_offset_(robot2_angle_offset)
{
    factor_type_ = INTERROBOT_FACTOR;
    
    // If dimensions not provided, use sphere approximation
    if (robot1_dimensions_.isZero()) {
        robot1_dimensions_ = Eigen::Vector2d(2 * robot_radius, 2 * robot_radius);
    }
    if (robot2_dimensions_.isZero()) {
        robot2_dimensions_ = Eigen::Vector2d(2 * robot_radius, 2 * robot_radius);
    }
    
    if (n_dofs == 5) {
        double max_ext1 = std::max(robot1_dimensions_(0), robot1_dimensions_(1));
        double max_ext2 = std::max(robot2_dimensions_(0), robot2_dimensions_(1));
        this->safety_distance_ = (max_ext1 + max_ext2) / 2.0 + 0.5;
    } else {
        float eps = 0.2 * robot_radius;
        this->safety_distance_ = 2 * robot_radius + eps;
    }
    this->delta_jac = 1e-3;
}

Eigen::MatrixXd InterrobotFactor::h_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(), z_.cols());

    // For 5D state: [x, y, xdot, ydot, theta]
    if (n_dofs_ == 5) {
        Eigen::Vector2d X_diff = X(seqN(0, 2)) - X(seqN(5, 2));
        X_diff += 1e-6 * r_id_ * Eigen::Vector2d::Ones();

        double r = X_diff.norm();
        if (r <= safety_distance_)
        {
            this->skip_flag = false;
            h(0) = 1.f * (1 - r / safety_distance_);
        }
        else
        {
            this->skip_flag = true;
        }
    } else {
        // Original 4D implementation (sphere approximation)
        Eigen::VectorXd X_diff = X(seqN(0, n_dofs_ / 2)) - X(seqN(n_dofs_, n_dofs_ / 2));
        X_diff += 1e-6 * r_id_ * Eigen::VectorXd::Ones(n_dofs_ / 2);

        double r = X_diff.norm();
        if (r <= safety_distance_)
        {
            this->skip_flag = false;
            h(0) = 1.f * (1 - r / safety_distance_);
        }
        else
        {
            this->skip_flag = true;
        }
    }
    return h;
}

Eigen::MatrixXd InterrobotFactor::J_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), n_dofs_ * 2);
    if (n_dofs_ == 5) {
        Eigen::Vector2d X_diff = X(seqN(0, 2)) - X(seqN(5, 2));
        X_diff += 1e-6 * r_id_ * Eigen::Vector2d::Ones();
        double r = X_diff.norm();
        if (r <= safety_distance_) {
            J(0, seqN(0, 2)) = -1.f / safety_distance_ / r * X_diff;
            J(0, seqN(5, 2)) = 1.f / safety_distance_ / r * X_diff;
        }
    } else {
        // Original 4D implementation
        Eigen::VectorXd X_diff = X(seqN(0, n_dofs_ / 2)) - X(seqN(n_dofs_, n_dofs_ / 2));
        X_diff += 1e-6 * r_id_ * Eigen::VectorXd::Ones(n_dofs_ / 2); // Add a tiny random offset to avoid div/0 errors
        double r = X_diff.norm();
        if (r <= safety_distance_)
        {
            J(0, seqN(0, n_dofs_ / 2)) = -1.f / safety_distance_ / r * X_diff;
            J(0, seqN(n_dofs_, n_dofs_ / 2)) = 1.f / safety_distance_ / r * X_diff;
        }
    }
    return J;
}

bool InterrobotFactor::skip_factor()
{
    if (n_dofs_ == 5) {
        this->skip_flag = ((X_(seqN(0, 2)) - X_(seqN(5, 2))).squaredNorm() >= safety_distance_ * safety_distance_);
    } else {
        // Original 4D implementation
        this->skip_flag = ((X_(seqN(0, n_dofs_ / 2)) - X_(seqN(n_dofs_, n_dofs_ / 2))).squaredNorm() >= safety_distance_ * safety_distance_);
    }
    return this->skip_flag;
}

void InterrobotFactor::draw()
{
    auto v_0 = variables_[0];
    auto v_1 = variables_[1];
    if (!v_0->valid_ || !v_1->valid_)
    {
        return;
    }
    if (!dbg) {
        if (n_dofs_ == 5) {
            Eigen::Vector2d pos1(X_(0), X_(1));
            Eigen::Vector2d pos2(X_(5), X_(6));

            double theta1 = X_(4), theta2 = X_(9);
            double th1 = theta1 + robot1_angle_offset_, th2 = theta2 + robot2_angle_offset_;
            
            OBB2D obb1(pos1, robot1_dimensions_ / 2.0, th1);
            OBB2D obb2(pos2, robot2_dimensions_ / 2.0, th2);

            auto [axis1_1, axis1_2] = obb1.getAxes();
            auto [axis2_1, axis2_2] = obb2.getAxes();

            const auto& [dist, n_hat] = GeometryUtils::getOBBDistance(obb1, obb2);
            
            std::ostringstream oss;
            oss << "ts=" << v_0->ts_ << ", th1=[" << th1 << "], th2=[" << th2 << "]\n";

            oss << "var 1=[";
            for (int i = 0; i < n_dofs_; ++i) {
                oss << X_(i);
                if (i < n_dofs_-1) oss << ",";
            }
            oss << "]\n";

            oss << "axis1_1=[";
            for (int i = 0; i < axis1_1.size(); ++i) {
                oss << axis1_1(i);
                if (i < axis1_1.size() - 1) oss << ",";
            }
            oss << "], axis1_2=[";
            for (int i = 0; i < axis1_2.size(); ++i) {
                oss << axis1_2(i);
                if (i < axis1_2.size() - 1) oss << ",";
            }
            oss << "]\n";

            oss << "var 2=[";
            for (int i = n_dofs_; i < n_dofs_*2; ++i) {
                oss << X_(i);
                if (i < n_dofs_*2-1) oss << ",";
            }
            oss << "]\n";

            oss << "axis2_1=[";
            for (int i = 0; i < axis2_1.size(); ++i) {
                oss << axis2_1(i);
                if (i < axis2_1.size() - 1) oss << ",";
            }
            oss << "], axis2_2=[";
            for (int i = 0; i < axis2_2.size(); ++i) {
                oss << axis2_2(i);
                if (i < axis2_2.size() - 1) oss << ",";
            }
            oss << "]\n";

            oss << "dist=[" << dist << "], sep axis=[";
            for (int i = 0; i < n_hat.size(); ++i) {
                oss << n_hat(i);
                if (i < n_hat.size() - 1) oss << ",";
            }
            oss << "]\n";

            auto h = this->h_func_(X_);
            auto J = this->J_func_(X_);

            oss << "h=[";
            for (int r = 0; r < h.rows(); ++r) {
                for (int c = 0; c < h.cols(); ++c) {
                    oss << h(r, c);
                    if (c < h.cols() - 1) oss << ",";
                }
                if (r < h.rows() - 1) oss << ";";
            }
            oss << "]\n";

            oss << "J=[";
            for (int r = 0; r < J.rows(); ++r) {
                for (int c = 0; c < J.cols(); ++c) {
                    oss << J(r, c);
                    if (c < J.cols() - 1) oss << ",";
                }
                if (r < J.rows() - 1) oss << ";";
            }
            oss << "]\n";

            print(oss.str());
        }
        dbg = true;
    }
    auto diff = v_0->mu_({0, 1}) - v_1->mu_({0, 1});
    if (diff.norm() <= safety_distance_)
    {
        DrawCylinderEx(Vector3{(float)v_0->mu_(0), globals.ROBOT_RADIUS, (float)v_0->mu_(1)},
                       Vector3{(float)v_1->mu_(0), globals.ROBOT_RADIUS, (float)v_1->mu_(1)},
                       0.1, 0.1, 4, RED);
    }
}

/********************************************************************************************/
// Obstacle factor for static obstacles in the scene. This factor takes a pointer to the obstacle image from the Simulator.
// Note. in the obstacle image, white areas represent obstacles (as they have a value of 1).
// The input image to the simulator is opposite, which is why it needs to be inverted.
// The delta used in the first order jacobian calculation is chosen such that it represents one pixel in the image.
/********************************************************************************************/
ObstacleFactor::ObstacleFactor(Simulator *sim, int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
                               float sigma, const Eigen::VectorXd &measurement, Image *p_obstacleImage)
    : Factor{f_id, r_id, variables, sigma, measurement}, p_obstacleImage_(p_obstacleImage)
{
    factor_type_ = OBSTACLE_FACTOR;
    this->delta_jac = 1. * (float)globals.WORLD_SZ / (float)p_obstacleImage->width;
};
Eigen::MatrixXd ObstacleFactor::h_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(1, 1);
    // White areas are obstacles, so h(0) should return a 1 for these regions.
    float scale = p_obstacleImage_->width / (float)globals.WORLD_SZ;
    Vector3 c_hsv = ColorToHSV(GetImageColor(*p_obstacleImage_, (int)((X(0) + globals.WORLD_SZ / 2) * scale), (int)((X(1) + globals.WORLD_SZ / 2) * scale)));
    h(0) = c_hsv.z;
    return h;
};

/********************************************************************************************/
// Factor for dynamic obstalces.
/********************************************************************************************/
DynamicObstacleFactor::DynamicObstacleFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables, int n_dofs,
                                             float sigma, const Eigen::VectorXd &measurement, float robot_radius, std::shared_ptr<DynamicObstacle> obs)
    : Factor{f_id, r_id, variables, sigma, measurement, n_dofs}, robot_radius_(robot_radius), obs_(std::move(obs))
{
    factor_type_ = DYNAMIC_OBSTACLE_FACTOR;
    delta_t_ = variables.front()->ts_ * globals.T0;
    double eps = 0.2f * robot_radius;
    safety_distance_ = robot_radius_ + eps;
    this->delta_jac = 1e-5;
}

Eigen::MatrixXd DynamicObstacleFactor::h_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(), z_.cols());

    for (const auto &[pt_world, dist_squared] : neighbours_)
    {
        double weight = gaussianRBF(dist_squared);
        h(0) += weight;
    }
    h(0) /= double(neighbours_.size());
    return h;
}

Eigen::MatrixXd DynamicObstacleFactor::J_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), X.size());
    Eigen::Vector2d grad_obs{0., 0.};
    for (const auto &[pt_world, dist_squared] : neighbours_)
    {
        Eigen::Vector2d diff = X.segment<2>(0) - pt_world;
        double weight = gaussianRBF(dist_squared);
        grad_obs += -2.0 * globals.RBF_GAMMA * weight * diff;
    }

    J(0, 0) = grad_obs.x();
    J(0, 1) = grad_obs.y();

    J /= double(neighbours_.size());
    return J;
}

bool DynamicObstacleFactor::skip_factor()
{
    neighbours_ = obs_->getNearestPoints(Eigen::Vector2d{X_(0), X_(1)}, globals.NUM_NEIGHBOURS, delta_t_);
    // this->skip_flag = neighbours_.front().second >= globals.OBSTALCE_SENSOR_RADIUS * globals.OBSTALCE_SENSOR_RADIUS;
    this->skip_flag = false;
    return this->skip_flag;
}

void DynamicObstacleFactor::draw()
{
    auto v_0 = variables_[0];
    double dist_sqr = neighbours_.front().second;
    if (dist_sqr <= (safety_distance_ * safety_distance_))
    {
        auto nb = neighbours_.front().first;
        auto state_t = obs_->states_.at(int(delta_t_/globals.T0));
        DrawCylinderEx(Vector3{(float)v_0->mu_(0), globals.ROBOT_RADIUS, (float)v_0->mu_(1)},
        Vector3{(float)nb(0), globals.ROBOT_RADIUS, (float)nb(1)},
        0.1, 0.1, 4, RED);
        
        float rotation = wrapAngle(state_t[4] + obs_->geom_->orientation_offset);
        // Add the model's default offset (similar to robots)
        float rotation_degrees = rotation * (180.0f / M_PI);
        DrawModelEx(obs_->geom_->model,
            Vector3{(float)state_t[0], -obs_->geom_->boundingBox.min.y, (float)state_t[1]},
            Vector3{0.0f, 1.0f, 0.0f},  // Rotate around Y axis
            rotation_degrees,           // Rotation angle in degrees
            Vector3{1.0f, 1.0f, 1.0f},  // Scale
            ColorAlpha(DARKGREEN, 0.2f)
        );               // Color tint
            
        DrawSphere(Vector3{(float)v_0->mu_(0), robot_radius_, (float)v_0->mu_(1)}, 0.8 * robot_radius_, ColorAlpha(DARKGREEN, 0.2f));
        for (const auto& nb : neighbours_) {
            auto nb_pos = nb.first;
            DrawSphere(Vector3{(float)nb_pos(0), robot_radius_, (float)nb_pos(1)}, 0.2 * robot_radius_, ColorAlpha(DARKBLUE, 0.2f));
        }
        if (!dbg)
        {
            print(rotation_degrees);
            dbg = true;
        }
        }
}