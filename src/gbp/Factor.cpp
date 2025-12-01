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
        // print("********************************************");
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

    if (llt_solver_.info() == Eigen::Success)
    {
        // Successfully decomposed - solve efficiently
        Eigen::VectorXd solved_eta = llt_solver_.solve(eta_b);
        Eigen::MatrixXd solved_lam = llt_solver_.solve(lam_ba);

        marginalised_msg.eta = eta_a - lam_ab * solved_eta;
        marginalised_msg.lambda = lam_aa - lam_ab * solved_lam;
    }
    else
    {
        // Fallback to LDLT for indefinite matrices
        ldlt_solver_.compute(lam_bb);

        if (ldlt_solver_.info() == Eigen::Success && ldlt_solver_.isPositive())
        {
            Eigen::VectorXd solved_eta = ldlt_solver_.solve(eta_b);
            Eigen::MatrixXd solved_lam = ldlt_solver_.solve(lam_ba);

            marginalised_msg.eta = eta_a - lam_ab * solved_eta;
            marginalised_msg.lambda = lam_aa - lam_ab * solved_lam;
        }
        else
        {
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
        Qi_inv.block<2, 2>(0, 0) = 12. * pow(dt_, -3.) * Qc_inv;
        Qi_inv.block<2, 2>(0, 2) = -6. * pow(dt_, -2.) * Qc_inv;
        Qi_inv.block<2, 2>(2, 0) = -6. * pow(dt_, -2.) * Qc_inv;
        Qi_inv.block<2, 2>(2, 2) = 4. / dt_ * Qc_inv;

        // Orientation component - much looser constraint to prevent fighting with velocity
        // The orientation should follow velocity naturally, not be forced
        double sigma_theta = sigma * 5.0; // Much looser constraint
        double Qc_inv_theta = pow(sigma_theta, -2.);
        Qi_inv(4, 4) = 1. / dt_ * Qc_inv_theta; // Reduced weight

        this->meas_model_lambda_ = Qi_inv;

        // For instantaneous alignment, orientation is nonlinear
        this->linear_ = false;
    } else {
        // For 4D state: [x, y, xdot, ydot]
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
        // // X contains [x1, y1, xdot1, ydot1, theta1, x2, y2, xdot2, ydot2, theta2]
        Eigen::VectorXd h = Eigen::VectorXd::Zero(5);

        // Position constraint: x2 = x1 + xdot1 * dt, use predicted - actual for consistency
        h(0) = X(0) + X(2) * dt_ - X(5);
        h(1) = X(1) + X(3) * dt_ - X(6);

        // Velocity constraint: xdot2 = xdot1 (constant velocity)
        h(2) = X(2) - X(7);
        h(3) = X(3) - X(8);

        // Orientation constraint: blend of alignment and smoothing
        const double xdot2 = X(7), ydot2 = X(8);
        const double v2    = xdot2*xdot2 + ydot2*ydot2;
        const double vmag  = std::sqrt(v2);

        // Speed band (tune): below v_low → smoothness; above v_high → alignment
        const double v_low  = 0.20 * v0_;
        const double v_high = 0.70 * v0_;
        auto clamp01 = [](double s){ return s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s); };
        double s = (v_high > v_low) ? clamp01((vmag - v_low) / (v_high - v_low)) : 1.0;
        // C1 smoothstep
        double w_align = s*s*(3.0 - 2.0*s);
        double w_smooth_floor = 0.35;
        double w_smooth = std::max(1.0 - w_align, w_smooth_floor);

        // velocity heading (Y-down convention)
        double theta_vel = vel_to_theta(xdot2, ydot2, X(9));
        // a: align θ2 to velocity; b: keep θ2 near θ1
        double a = angle_diff(theta_vel, X(9));   // (θv - θ2)
        double b = angle_diff(X(4),      X(9));   // (θ1 - θ2)

        // blended residual
        h(4) = wrapAngle(w_align * a + w_smooth * b);

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
        J(0, 0) = 1; J(0, 2) = dt_; J(0, 5) = -1; // dx
        J(1, 1) = 1; J(1, 3) = dt_; J(1, 6) = -1; // dy
        J(2, 2) = 1; J(2, 7) = -1; // dxdot
        J(3, 3) = 1; J(3, 8) = -1; // dydot

        // -------- Orientation row (index 4) --------
        const double xdot2 = X(7), ydot2 = X(8);
        const double v2    = xdot2*xdot2 + ydot2*ydot2;
        const double vmag  = std::sqrt(std::max(1e-16, v2));

        const double v_low  = 0.20 * v0_;
        const double v_high = 0.70 * v0_;
        auto clamp01 = [](double s){ return s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s); };
        double s = (v_high > v_low) ? clamp01((vmag - v_low) / (v_high - v_low)) : 1.0;
        double w_align = s*s*(3.0 - 2.0*s);
        double w_smooth_floor = 0.35;
        double w_smooth = std::max(1.0 - w_align, w_smooth_floor);

        const double eps_v = std::pow(0.35 * v0_, 2);

        // derivatives of θv = -atan2(ydot2, xdot2)
        double alpha = clamp01((vmag - v_low) / std::max(v_high - v_low, 1e-9));
        alpha = std::pow(alpha, 1.5);
        const double dthv_dx =  alpha * ( X(8) / (v2 + eps_v));
        const double dthv_dy =  alpha * (-X(7) / (v2 + eps_v));

        // a = angle_diff(θv, θ2) → ∂a/∂θ2 = -1, ∂a/∂xdot2 = dthv_dx, ∂a/∂ydot2 = dthv_dy
        // b = angle_diff(θ1, θ2) → ∂b/∂θ1 = +1, ∂b/∂θ2 = -1
        J(4,7) += w_align * dthv_dx;
        J(4,8) += w_align * dthv_dy;
        J(4,4) += w_smooth * (+1.0);
        J(4,9) += w_align * (-1.0) + w_smooth * (-1.0);
        
        return J;

    } else {
        // Original 4D implementation
        return J_;
    }
}

void DynamicsFactor::draw()
{
    if (!globals.DRAW_PATH)
    {
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
                                   const Eigen::Vector2d &robot1_dims,
                                   const Eigen::Vector2d &robot2_dims,
                                   double robot1_angle_offset,
                                   double robot2_angle_offset)
    : Factor{f_id, r_id, variables, sigma, measurement, n_dofs},
      robot1_dimensions_(robot1_dims),
      robot2_dimensions_(robot2_dims),
      robot1_angle_offset_(robot1_angle_offset),
      robot2_angle_offset_(robot2_angle_offset)
{
    factor_type_ = INTERROBOT_FACTOR;

    // If dimensions not provided, use sphere approximation for debug/drawing only
    if (robot1_dimensions_.isZero())
    {
        robot1_dimensions_ = Eigen::Vector2d(2 * robot_radius, 2 * robot_radius);
    }
    if (robot2_dimensions_.isZero())
    {
        robot2_dimensions_ = Eigen::Vector2d(2 * robot_radius, 2 * robot_radius);
    }

    if (n_dofs >= 5) {
        this->safety_distance_ = 1.5;
    } else {
        float eps = 0.2 * robot_radius;
        this->safety_distance_ = 2 * robot_radius + eps;
    }
    tau_d_ = 0.2 * safety_distance_;
    this->delta_jac = 1e-4;
}

Eigen::MatrixXd InterrobotFactor::h_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 4) {
        Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(),z_.cols());
        Eigen::VectorXd X_diff = X(seqN(0,n_dofs_/2)) - X(seqN(n_dofs_, n_dofs_/2));
        X_diff += 1e-6*r_id_*Eigen::VectorXd::Ones(n_dofs_/2);

        double r = X_diff.norm();
        if (r <= safety_distance_){
            this->skip_flag = false;
            h(0) = 1.f*(1 - r/safety_distance_);
        }
        else {
            this->skip_flag = true;
        }

        return h;
    } else {
        // 1 scalar residual
        Eigen::MatrixXd h(1, 1);
    
        const int n0 = variables_[0]->n_dofs_;
        const int n1 = variables_[1]->n_dofs_;
    
        // State slices
        const Eigen::Vector2d c1 = X.segment<2>(0);
        const Eigen::Vector2d c2 = X.segment<2>(n0);
        const double th1 = (n0 >= 5 ? wrapAngle(X(4)) : 0.0);
        const double th2 = (n1 >= 5 ? wrapAngle(X(n0 + 4)) : 0.0);
    
        // OBB half-extents and heading offsets (keep your flipped-axis convention)
        const Eigen::Vector2d e1 = 0.5 * robot1_dimensions_;
        const Eigen::Vector2d e2 = 0.5 * robot2_dimensions_;
        const double o1 = wrapAngle(th1 + robot1_angle_offset_);
        const double o2 = wrapAngle(th2 + robot2_angle_offset_);
    
        // Build OBBs
        OBB2D obb1(c1, e1, o1);
        OBB2D obb2(c2, e2, o2);
    
        const auto axes1 = obb1.getAxes();
        const auto axes2 = obb2.getAxes();
        const Eigen::Vector2d a1x = axes1.first,  a1y = axes1.second;
        const Eigen::Vector2d a2x = axes2.first,  a2y = axes2.second;
    
        // Smooth |·| and smoothmax params
        const double eps_abs = 1e-6; // for |t|
        const double beta = 3.0;    // softmax hardness
        auto sabs = [&](double t)
        { return std::sqrt(t * t + eps_abs * eps_abs); };
    
        // Signed separation along axis n: s(n) = |(c2 - c1)·n| - (E1(n) + E2(n))
        auto extent = [&](const Eigen::Vector2d &n,
                          const Eigen::Vector2d &ax, const Eigen::Vector2d &ay,
                          const Eigen::Vector2d &e) -> double
        {
            return e.x() * sabs(ax.dot(n)) + e.y() * sabs(ay.dot(n));
        };
    
        auto sep_along = [&](const Eigen::Vector2d &n) -> double
        {
            const Eigen::Vector2d d = c2 - c1;
            const double T = sabs(d.dot(n));
            const double E1 = extent(n, a1x, a1y, e1);
            const double E2 = extent(n, a2x, a2y, e2);
            return T - (E1 + E2);
        };
    
        // Evaluate gaps on the 4 axes
        double g[4];
        g[0] = sep_along(a1x);
        g[1] = sep_along(a1y);
        g[2] = sep_along(a2x);
        g[3] = sep_along(a2y);
    
        // Smoothmax across the 4 gaps → smooth “best” signed separation
        double m = std::max(std::max(g[0], g[1]), std::max(g[2], g[3]));
        double Z = 0.0;
        for (int k = 0; k < 4; ++k) Z += std::exp(beta * (g[k] - m));
        const double phi = m + std::log(std::max(Z, 1e-12)) / beta;
    
        // Smooth hinge residual on (d_safe - phi)
        const double z = safety_distance_ - phi; // >0 ⇒ violation
        // const double eps_h = 1e-6;
        // const double r = std::sqrt(z * z + eps_h * eps_h);
        auto softplus_centered = [](double s){
            const double a = std::abs(s);
            return (std::max(s,0.0) + std::log1p(std::exp(-a))) - std::log(2.0);
        };
    
        const double s = z / tau_d_;
        const double r = tau_d_ * softplus_centered(s);
        const double gate = 1.0 / (1.0 + std::exp((phi - (safety_distance_ + 2.0*tau_d_)) / (0.5*tau_d_)));
    
        // Deactivate (skip/zero) far-away pairs
        // this->skip_flag = (phi > safety_distance_ + 3.0 * tau_d_);
    
        // Output residual, normalised by d_safe to keep scale ~O(1)
        // h(0, 0) = this->skip_flag ? 0.0 : (r / std::max(1e-9, safety_distance_));
        h(0,0) = gate * (r / std::max(1e-9, safety_distance_));
        // h(0, 0) = r / std::max(1e-9, safety_distance_);
        return h;
    }
}

Eigen::MatrixXd InterrobotFactor::J_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 4) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), n_dofs_*2);
        Eigen::VectorXd X_diff = X(seqN(0,n_dofs_/2)) - X(seqN(n_dofs_, n_dofs_/2));
        X_diff += 1e-6*r_id_*Eigen::VectorXd::Ones(n_dofs_/2);// Add a tiny random offset to avoid div/0 errors
        double r = X_diff.norm();
        if (r <= safety_distance_){
            J(0,seqN(0, n_dofs_/2)) = -1.f/safety_distance_/r * X_diff;
            J(0,seqN(n_dofs_, n_dofs_/2)) = 1.f/safety_distance_/r * X_diff;
        }
        return J;
    } else {
        const int n0 = variables_[0]->n_dofs_;
        const int n1 = variables_[1]->n_dofs_;
    
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, n0 + n1);
    
        // --- State slices ---
        const Eigen::Vector2d c1 = X.segment<2>(0);
        const Eigen::Vector2d c2 = X.segment<2>(n0);
        const double th1 = (n0 >= 5 ? wrapAngle(X(4)) : 0.0);
        const double th2 = (n1 >= 5 ? wrapAngle(X(n0 + 4)) : 0.0);
    
        // --- OBB params with offsets ---
        const Eigen::Vector2d e1 = 0.5 * robot1_dimensions_;
        const Eigen::Vector2d e2 = 0.5 * robot2_dimensions_;
        const double o1 = wrapAngle(th1 + robot1_angle_offset_);
        const double o2 = wrapAngle(th2 + robot2_angle_offset_);
    
        OBB2D obb1(c1, e1, o1);
        OBB2D obb2(c2, e2, o2);
    
        const auto axes1 = obb1.getAxes();
        const auto axes2 = obb2.getAxes();
        const Eigen::Vector2d a1x = axes1.first,  a1y = axes1.second;
        const Eigen::Vector2d a2x = axes2.first,  a2y = axes2.second;
    
        const Eigen::Vector2d d = c2 - c1;
    
        // --- Smoothing params ---
        const double eps_abs = 1e-3;    // smoother |·| near t≈0
        const double beta     = 3.0;    // smoothmax sharpness
    
        auto sabs = [&](double t){ return std::sqrt(t*t + eps_abs*eps_abs); };
        auto sgns = [&](double t){ return t / sabs(t); };  // d/dt sabs(t)
    
        // --- per-axis gap + gradients ---
        struct SGrad {
            double s;
            Eigen::RowVector2d dc1, dc2;
            double dth1, dth2;
        };
    
        auto dn_dth_box1 = [&](const Eigen::Vector2d &n)->Eigen::Vector2d{
            if ((n - a1x).squaredNorm() < 1e-18) return -a1y;
            if ((n - a1y).squaredNorm() < 1e-18) return  a1x;
            return Eigen::Vector2d::Zero();
        };
        auto dn_dth_box2 = [&](const Eigen::Vector2d &n)->Eigen::Vector2d{
            if ((n - a2x).squaredNorm() < 1e-18) return -a2y;
            if ((n - a2y).squaredNorm() < 1e-18) return  a2x;
            return Eigen::Vector2d::Zero();
        };
    
        auto extent = [&](const Eigen::Vector2d &n,
                          const Eigen::Vector2d &ax, const Eigen::Vector2d &ay,
                          const Eigen::Vector2d &e)->double{
            return e.x()*sabs(ax.dot(n)) + e.y()*sabs(ay.dot(n));
        };
    
        auto dsep = [&](const Eigen::Vector2d &n)->SGrad{
            SGrad out; out.s=0; out.dc1.setZero(); out.dc2.setZero(); out.dth1=0; out.dth2=0;
    
            const double t  = d.dot(n);
            const double T  = sabs(t);
            const double dt = sgns(t);
    
            const double ax1 = a1x.dot(n), ay1 = a1y.dot(n);
            const double ax2 = a2x.dot(n), ay2 = a2y.dot(n);
            const double E1  = e1.x()*sabs(ax1) + e1.y()*sabs(ay1);
            const double E2  = e2.x()*sabs(ax2) + e2.y()*sabs(ay2);
    
            out.s   = T - (E1 + E2);
            out.dc1 = (-dt) * n.transpose();
            out.dc2 = (+dt) * n.transpose();
    
            // θ-terms kept zero for now (you’re not filling θ columns):
            // out.dth1/out.dth2 could include -dE/dθ etc. later when enabling θ
            return out;
        };
    
        // --- compute gaps/gradients on 4 axes ---
        SGrad gg[4];
        gg[0] = dsep(a1x);
        gg[1] = dsep(a1y);
        gg[2] = dsep(a2x);
        gg[3] = dsep(a2y);
    
        double g[4];
        for (int k=0;k<4;++k) g[k] = gg[k].s;
    
        // ===== (A) φ for residual/gate: from RAW gaps (NO bias, NO EMA) =====
        double mg = std::max(std::max(g[0],g[1]), std::max(g[2],g[3]));
        double Zg = 0.0; for (int k=0;k<4;++k) Zg += std::exp(beta*(g[k]-mg));
        const double phi_g = mg + std::log(std::max(Zg, 1e-12)) / beta;
    
        // ===== (B) Weights for gradient: BIASED gaps + EMA (tie-break + smoothing) =====
        // relative velocity consistent with d = c2 - c1
        const Eigen::Vector2d v1 = X.segment<2>(2);
        const Eigen::Vector2d v2 = X.segment<2>(n0+2);
        const Eigen::Vector2d v_rel = v2 - v1;
    
        double t0 = d.dot(a1x), t1 = d.dot(a1y), t2 = d.dot(a2x), t3 = d.dot(a2y);
        Eigen::Vector2d neff[4] = { sgns(t0)*a1x, sgns(t1)*a1y, sgns(t2)*a2x, sgns(t3)*a2y };
    
        const double dt_bias = 0.15; // s
        double gb[4];
        for (int k=0;k<4;++k){
            const double dgdt = neff[k].dot(v_rel);   // m/s
            gb[k] = g[k] + dt_bias * dgdt;           // biased gap (meters)
        }
    
        double mb = std::max(std::max(gb[0],gb[1]), std::max(gb[2],gb[3]));
        double expv[4], Zb = 0.0;
        for (int k=0;k<4;++k){ expv[k] = std::exp(beta*(gb[k]-mb)); Zb += expv[k]; }
        double w[4]; const double invZb = 1.0 / std::max(Zb, 1e-12);
        for (int k=0;k<4;++k) w[k] = expv[k] * invZb;
    
        // EMA over weights (per-factor storage is ideal; thread_local is okay if needed)
        // static thread_local std::array<double,4> w_prev = {0.25,0.25,0.25,0.25};
        const double alpha = 0.3;
        double w_s[4]; double Sw = 1e-12;
        for (int k=0;k<4;++k){ w_s[k] = alpha*w[k] + (1.0-alpha)*w_prev_[k]; Sw += w_s[k]; }
        for (int k=0;k<4;++k) w[k] = w_s[k] / Sw;
        w_prev_ = {w[0], w[1], w[2], w[3]};
    
        // ===== (C) Blend gradients with (biased+EMA) weights =====
        Eigen::RowVector2d dphi_dc1 = Eigen::RowVector2d::Zero();
        Eigen::RowVector2d dphi_dc2 = Eigen::RowVector2d::Zero();
        double dphi_dth1 = 0.0, dphi_dth2 = 0.0;
        for (int k=0;k<4;++k){
            dphi_dc1 += w[k] * gg[k].dc1;
            dphi_dc2 += w[k] * gg[k].dc2;
            // keep θ grads zero for now (you zero θ columns below)
        }
    
        // ===== (D) Same z/s/sig/gate as residual path (USE φ_g) =====
        const double z   = safety_distance_ - phi_g;
        const double s   = z / tau_d_;
        const double sig = 1.0 / (1.0 + std::exp(-s));
        const double gate = 1.0 / (1.0 + std::exp((phi_g - (safety_distance_ + 2.0*tau_d_)) / (0.5*tau_d_)));
        const double scale = -gate * (sig / std::max(1e-9, safety_distance_));
    
        // ===== (E) Fill J (θ columns kept 0 for now) =====
        J(0, 0)      = scale * dphi_dc1.x();
        J(0, 1)      = scale * dphi_dc1.y();
        if (n0 >= 5) J(0, 4) = 0.0;
    
        J(0, n0+0)   = scale * dphi_dc2.x();
        J(0, n0+1)   = scale * dphi_dc2.y();
        if (n1 >= 5) J(0, n0+4) = 0.0;
    
        if (gate < 1e-6) { w_prev_ = {0.25, 0.25, 0.25, 0.25}; }

        return J;
    }
}

bool InterrobotFactor::skip_factor()
{
    if (n_dofs_ == 4) {
        this->skip_flag = ( (X_(seqN(0,n_dofs_/2)) - X_(seqN(n_dofs_, n_dofs_/2))).squaredNorm() >= safety_distance_*safety_distance_ );
        return this->skip_flag;
    } else {
        this->skip_flag = false;
        return this->skip_flag;
    }
}

void InterrobotFactor::draw()
{
    auto v_0 = variables_[0];
    auto v_1 = variables_[1];
    if (!v_0->valid_ || !v_1->valid_)
    {
        return;
    }

    if (!dbg)
    {
        const int n0 = v_0->n_dofs_;
        const int n1 = v_1->n_dofs_;
        const Eigen::Vector2d c1 = X_.segment<2>(0);
        const Eigen::Vector2d c2 = X_.segment<2>(n0);

        const double th1 = (n0 >= 5 ? wrapAngle(X_(4)) : 0.0);
        const double th2 = (n1 >= 5 ? wrapAngle(X_(n0 + 4)) : 0.0);
        const double o1 = th1 + wrapAngle(robot1_angle_offset_);
        const double o2 = th2 + wrapAngle(robot2_angle_offset_);

        auto h = this->h_func_(X_);
        auto J = this->J_func_(X_);

        auto diff = c1 - c2;
        auto r = diff.norm();
        auto prev_h = r > safety_distance_ ? 0.0 : 1.0 - r / safety_distance_;
        Eigen::MatrixXd prev_J = Eigen::MatrixXd::Zero(1, 10);
        if (r <= safety_distance_) {
            prev_J(0, 0) = -1.0 / safety_distance_ / r * diff(0);
            prev_J(0, 1) = -1.0 / safety_distance_ / r * diff(1);
            prev_J(0, n0+0) = 1.0 / safety_distance_ / r * diff(0);
            prev_J(0, n0+1) = 1.0 / safety_distance_ / r * diff(1);
        }

        // printf("[DEBUG] ts=%d,%d, c1=%f,%f, c2=%f,%f, th=%f,%f, h=%f, J=[%f,%f,%f,%f,%f//%f,%f,%f,%f,%f], prev_h=%f, prev_J=[%f,%f//%f,%f]\n", v_0->ts_, v_1->ts_, c1.x(), c1.y(), c2.x(), c2.y(), o1, o2, h(0, 0), J(0,0), J(0,1), J(0,2), J(0,3), J(0,4), J(0,5), J(0,6), J(0,7), J(0,8), J(0,9), prev_h, prev_J(0, 0), prev_J(0, 1), prev_J(0, n0), prev_J(0, n0+1));
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
// Helper functions for chance-constraint via inflated safety-margins
double sigmaNominalSq()
{ /* Returns the nominal sigma from nominal gamma (globals.RBF_GAMMA) */
    return 0.5 / std::max(1e-12, globals.RBF_GAMMA);
}

double projectedVariance(const Eigen::Matrix2d *Sigma_k, const Eigen::Vector2d &diff)
{ /* Project sigma at timestep k */
    if (!Sigma_k)
        return 0.0;
    double dn = diff.norm();
    if (dn < 1e-9)
    {
        // At the obstacle point → use average variance as a safe fallback
        return 0.5 * Sigma_k->trace();
    }
    Eigen::Vector2d n = diff / dn;
    return (n.transpose() * (*Sigma_k) * n)(0, 0);
}

double gammaEff(const Eigen::Matrix2d *Sigma, const Eigen::Vector2d &diff)
{ /* Get the effective gamma for obstacle timestep k */
    const double s_nom2 = sigmaNominalSq();
    // const double s_obs2 = std::min(1.0, projectedVariance(Sigma, diff));
    const double s_eff2 = s_nom2/* + s_obs2*/;
    return 1.0 / (2.0 * std::max(1e-12, s_eff2));
}

DynamicObstacleFactor::DynamicObstacleFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables, int n_dofs,
                                             float sigma, const Eigen::VectorXd &measurement, float robot_radius, std::shared_ptr<DynamicObstacle> obs,
                                             const Eigen::Vector2d& robot_dims, double robot_a_of)
    : Factor{f_id, r_id, variables, sigma, measurement, n_dofs}, robot_radius_(robot_radius), obs_(std::move(obs)), robot_dimensions_(robot_dims), robot_angle_offset_(robot_a_of)
{
    factor_type_ = DYNAMIC_OBSTACLE_FACTOR;
    delta_t_ = variables.front()->ts_ * globals.T0;
    if (n_dofs == 5) {
        safety_distance_ = 1.5;
    } else {
        double eps = 0.2 * robot_radius;
        safety_distance_ = robot_radius + eps;
    }
    this->delta_jac = 1e-5;
    tau_d_ = 0.2 * safety_distance_;
}

Eigen::MatrixXd DynamicObstacleFactor::h_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 4) {
        Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(), z_.cols());
        const Eigen::Vector2d x = X.segment<2>(0);
    
        double acc = 0.0;
        for (const auto &nb : neighbours_)
        {
            const Eigen::Vector2d diff = x - nb.pt_world;
            const double gamma = gammaEff(nb.Sigma_pos, diff);
            // const double gamma = globals.RBF_GAMMA;
            const double w = std::exp(-gamma * nb.dist_squared);
            acc += w;
        }
        h(0) = (neighbours_.empty() ? 0.0 : acc / double(neighbours_.size()));
        return h;
    
    } else {
        const int n = n_dofs_;
        const Eigen::Vector2d cR = X.segment<2>(0);
        const double thR = (n >= 5) ? wrapAngle(X(4)) : 0.0;
    
        // Robot OBB params
        const Eigen::Vector2d eR = 0.5 * robot_dimensions_;
        const double oR = wrapAngle(thR + robot_angle_offset_);
    
        // Obstacle OBB at this factor’s time (fixed w.r.t. variables)
        const int v_ts = variables_.front()->ts_;
        auto o_state_t = obs_->states_[v_ts];
        const Eigen::Vector2d cO = o_state_t.head<2>();
        const Eigen::Vector2d eO = Eigen::Vector2d(obs_->geom_->dimensions.x*0.5, obs_->geom_->dimensions.z*0.5);;
        const double oO = o_state_t(4);
        const OBB2D O(cO, eO, oO);
        
        // Axes (unit) for both OBBs
        auto axesO = O.getAxes();                 // {aOx, aOy}
        const Eigen::Vector2d aOx = axesO.first;
        const Eigen::Vector2d aOy = axesO.second;
    
        // Robot axes in Y-down convention (match your interrobot factor)
        const double co = std::cos(oR), so = std::sin(oR);
        const Eigen::Vector2d aRx(co, -so);
        const Eigen::Vector2d aRy(so,  co);
    
        // Separation vector (obstacle relative to robot)
        const Eigen::Vector2d d = cO - cR;
    
        // Smooth |·|
        const double eps_abs = 1e-3;
        auto sabs  = [&](double t){ return std::sqrt(t*t + eps_abs*eps_abs); };
    
        // Support/extent of a rectangle along axis n
        auto extent = [&](const Eigen::Vector2d& n,
                          const Eigen::Vector2d& ax, const Eigen::Vector2d& ay,
                          const Eigen::Vector2d& e)->double
        {
            return e.x()*sabs(ax.dot(n)) + e.y()*sabs(ay.dot(n));
        };
    
        // Signed gap along axis n (smooth)
        auto gap = [&](const Eigen::Vector2d& n)->double
        {
            const double T  = sabs(d.dot(n));                         // |(cO-cR)·n|
            const double ER = extent(n, aRx, aRy, eR);                // robot support on n
            const double EO = extent(n, aOx, aOy, eO);                // obstacle support on n
            return T - (ER + EO);                                     // >0: separated, <0: overlapping
        };
    
        // Evaluate on the 4 SAT axes
        double g[4];
        g[0] = gap(aRx);
        g[1] = gap(aRy);
        g[2] = gap(aOx);
        g[3] = gap(aOy);
    
        // Smoothmax across gaps
        const double beta = 3.0;
        double m = std::max(std::max(g[0],g[1]), std::max(g[2],g[3]));
        double Z = 0.0; for (int k=0;k<4;++k) Z += std::exp(beta*(g[k]-m));
        const double phi = m + std::log(std::max(Z,1e-12)) / beta;    // smooth signed separation
    
        // Smooth hinge on (d_safe - phi)
        const double z = safety_distance_ - phi;                      // >0 ⇒ too close/overlap
        auto softplus_centered = [](double s){
            const double a = std::abs(s);
            return (std::max(s,0.0) + std::log1p(std::exp(-a))) - std::log(2.0);
        };
        const double s = z / tau_d_;
        const double r = tau_d_ * softplus_centered(s);
    
        // Far-away gate (fades out factor smoothly)
        const double gate = 1.0 / (1.0 + std::exp((phi - (safety_distance_ + 2.0*tau_d_)) / (0.5*tau_d_)));

        // --- turn-away residual (tiny, only near contact) -----------------
        auto clamp01 = [](double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };

        // proximity gate: 0 far, ->1 when within the margin
        double prox = 1.0 / (1.0 + std::exp((phi - safety_distance_) / (0.25 * tau_d_)));
        double w_turn = prox * prox;                // smooth ramp-in

        // preferred normal from active SAT axes (use raw weights here)
        double m_for_w = m;                          // reuse m from smoothmax above
        double Z_for_w = Z;                          // reuse Z if you like; or recompute raw weights
        double w_raw[4];
        for (int k=0;k<4;++k) w_raw[k] = std::exp(beta*(g[k]-m_for_w)) / std::max(Z_for_w, 1e-12);

        // build a preferred outward normal (unit-ish)
        Eigen::Vector2d n_pref = w_raw[0]*aRx + w_raw[1]*aRy + w_raw[2]*aOx + w_raw[3]*aOy;
        if (n_pref.squaredNorm() > 1e-12) n_pref.normalize();

        // tangent to that normal (Y-down frame)
        Eigen::Vector2d t_pref(n_pref.y(), -n_pref.x());

        // desired heading: along tangent (slide instead of poke)
        double theta_pref = wrapAngle(-std::atan2(t_pref.y(), t_pref.x())); // Y-down
        double d_ang = angle_diff(theta_pref, oR);

        // small urge (radians), also fade by gate so it dies far away
        const double K_TURN = 0.12;                  // tune 0.05–0.25 rad
        double r_turn = (gate * w_turn) * (K_TURN * d_ang);
        // ------------------------------------------------------------------
    
        Eigen::MatrixXd h(1,1);
        // h(0,0) = gate * (r / std::max(1e-9, safety_distance_));
        h(0,0) = gate * (r / std::max(1e-9, safety_distance_)) + r_turn;
        return h;
    }
}

Eigen::MatrixXd DynamicObstacleFactor::J_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 4) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), X.size());
        const Eigen::Vector2d x = X.segment<2>(0);
    
        Eigen::Vector2d grad_obs = Eigen::Vector2d::Zero();
        for (const auto &nb : neighbours_)
        {
            const Eigen::Vector2d diff = x - nb.pt_world;
            const double gamma = gammaEff(nb.Sigma_pos, diff);
            // const double gamma = globals.RBF_GAMMA;
            const double w = std::exp(-gamma * nb.dist_squared);
            grad_obs += -2.0 * gamma * w * diff;
        }
    
        if (!neighbours_.empty())
            grad_obs /= double(neighbours_.size());
        J(0, 0) = grad_obs.x();
        J(0, 1) = grad_obs.y();
        return J;
    
    } else {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, n_dofs_);

        const Eigen::Vector2d cR = X.segment<2>(0);
        const double thR = (n_dofs_ >= 5) ? wrapAngle(X(4)) : 0.0;

        const Eigen::Vector2d eR = 0.5 * robot_dimensions_;
        const double oR = wrapAngle(thR + robot_angle_offset_);

        const int v_ts = variables_.front()->ts_;
        auto o_state_t = obs_->states_[v_ts];
        const Eigen::Vector2d cO = o_state_t.head<2>();
        const Eigen::Vector2d eO = Eigen::Vector2d(obs_->geom_->dimensions.x*0.5, obs_->geom_->dimensions.z*0.5);;
        const double oO = o_state_t(4);
        const OBB2D O(cO, eO, oO);

        auto axesO = O.getAxes();           // obstacle axes
        const Eigen::Vector2d aOx = axesO.first;
        const Eigen::Vector2d aOy = axesO.second;

        // Robot axes (Y-down)
        const double co = std::cos(oR), so = std::sin(oR);
        const Eigen::Vector2d aRx(co, -so);
        const Eigen::Vector2d aRy(so,  co);

        const Eigen::Vector2d d = cO - cR;

        // Smooth |·|
        const double eps_abs = 1e-3;
        auto sabs = [&](double t){ return std::sqrt(t*t + eps_abs*eps_abs); };
        auto sgns = [&](double t){ return t / sabs(t); };

        struct G {
            double s;                        // gap value on this axis
            Eigen::RowVector2d dcR;         // ∂s/∂cR
            double dthR;                     // ∂s/∂θR  (0 if n<5)
        };

        auto extent = [&](const Eigen::Vector2d& n,
                        const Eigen::Vector2d& ax, const Eigen::Vector2d& ay,
                        const Eigen::Vector2d& e)->double{
            return e.x()*sabs(ax.dot(n)) + e.y()*sabs(ay.dot(n));
        };

        // d/dθ of robot axes (Y-down): aRx=(c,-s), aRy=(s,c)
        const Eigen::Vector2d daRx_dth(-so, -co);
        const Eigen::Vector2d daRy_dth( co, -so);

        auto dgap = [&](const Eigen::Vector2d& n)->G{
            G out; out.s=0; out.dcR.setZero(); out.dthR=0;

            const double t  = d.dot(n);           // (cO - cR)·n
            const double T  = sabs(t);
            const double dt = sgns(t);

            // robot + obstacle extents
            const double axR = aRx.dot(n), ayR = aRy.dot(n);
            const double axO = aOx.dot(n), ayO = aOy.dot(n);
            const double ER  = eR.x()*sabs(axR) + eR.y()*sabs(ayR);
            const double EO  = eO.x()*sabs(axO) + eO.y()*sabs(ayO);

            out.s   = T - (ER + EO);

            // ∂T/∂cR = -dt * n^T
            out.dcR = (-dt) * n.transpose();

            if (n_dofs_ >= 5) {
                // ∂ER/∂θ = exR * d|axR|/dθ + eyR * d|ayR|/dθ
                const double daxR_dth = daRx_dth.dot(n);
                const double dayR_dth = daRy_dth.dot(n);
                const double d_axR_dθ = axR / sabs(axR) * daxR_dth;
                const double d_ayR_dθ = ayR / sabs(ayR) * dayR_dth;
                const double dER_dth = eR.x() * d_axR_dθ + eR.y() * d_ayR_dθ;

                out.dthR = - dER_dth;        // minus because s = T - (ER+EO)
            }
            return out;
        };

        // Compute gaps & grads on 4 axes
        G gg[4];
        gg[0] = dgap(aRx);
        gg[1] = dgap(aRy);
        gg[2] = dgap(aOx);
        gg[3] = dgap(aOy);

        double g[4]; for (int k=0;k<4;++k) g[k] = gg[k].s;

        // ===== φ for residual & gate: raw smoothmax over gaps =====
        const double beta = 3.0;
        double mg = std::max(std::max(g[0],g[1]), std::max(g[2],g[3]));
        double Zg = 0.0; for (int k=0;k<4;++k) Zg += std::exp(beta*(g[k]-mg));
        const double phi_g = mg + std::log(std::max(Zg,1e-12))/beta;

        // ===== Blend weights for gradient: biased + EMA (like your interrobot) =====
        // If obstacle velocity available:
        Eigen::Vector2d zero = Eigen::Vector2d::Zero();
        Eigen::Vector2d vR = (n_dofs_ >= 4) ? X.segment<2>(2) : zero;
        Eigen::Vector2d vO = o_state_t.segment<2>(2); // or Zero() if not tracked
        Eigen::Vector2d v_rel = vO - vR;

        // Signed direction along each axis (for bias)
        double t0 = d.dot(aRx), t1 = d.dot(aRy), t2 = d.dot(aOx), t3 = d.dot(aOy);
        Eigen::Vector2d neff[4] = { sgns(t0)*aRx, sgns(t1)*aRy, sgns(t2)*aOx, sgns(t3)*aOy };

        const double dt_bias = 0.15; // seconds (same meaning as in interrobot)
        double gb[4];
        for (int k=0;k<4;++k) {
            const double dgdt = neff[k].dot(v_rel);
            gb[k] = g[k] + dt_bias * dgdt;
        }

        // Softmax weights
        double mb = std::max(std::max(gb[0],gb[1]), std::max(gb[2],gb[3]));
        double expv[4], Zb = 0.0;
        for (int k=0;k<4;++k){ expv[k] = std::exp(beta*(gb[k]-mb)); Zb += expv[k]; }
        double w[4]; const double invZb = 1.0 / std::max(Zb, 1e-12);
        for (int k=0;k<4;++k) w[k] = expv[k] * invZb;

        // EMA over weights (store w_prev_ in the factor like you do in InterrobotFactor)
        const double alpha = 0.3;
        double w_s[4]; double Sw = 1e-12;
        for (int k=0;k<4;++k){ w_s[k] = alpha*w[k] + (1.0-alpha)*w_prev_[k]; Sw += w_s[k]; }
        for (int k=0;k<4;++k) w[k] = w_s[k]/Sw;
        w_prev_ = { w[0], w[1], w[2], w[3] };

        // Blend gradients
        Eigen::RowVector2d dphi_dcR = Eigen::RowVector2d::Zero();
        double dphi_dthR = 0.0;
        for (int k=0;k<4;++k){
            dphi_dcR += w[k] * gg[k].dcR;
            dphi_dthR += w[k] * gg[k].dthR;
        }

        // Residual derivative chain (for the SAT part)
        const double z   = safety_distance_ - phi_g;
        const double s   = z / tau_d_;
        const double sig = 1.0 / (1.0 + std::exp(-s));
        const double gate = 1.0 / (1.0 + std::exp((phi_g - (safety_distance_ + 2.0*tau_d_)) / (0.5*tau_d_)));
        const double scale = -gate * (sig / std::max(1e-9, safety_distance_));

        // Proximity for θ gating
        double prox = 1.0 / (1.0 + std::exp((phi_g - safety_distance_) / (0.25 * tau_d_)));
        double w_theta = prox * prox;

        // Fill x,y
        J(0,0) = scale * dphi_dcR.x();
        J(0,1) = scale * dphi_dcR.y();

        // --- turn-away residual Jacobian wrt theta ---
        const double K_TURN = 0.12;
        double w_turn = prox * prox;
        // r_turn = (gate * w_turn) * (K_TURN * angle_diff(theta_pref, oR))
        // ∂/∂θ angle_diff(theta_pref, oR) = -1  (treat theta_pref constant)
        if (n_dofs_ >= 5) {
            double drturn_dtheta = -(gate * w_turn) * K_TURN;
            // J(0,4) += drturn_dtheta;          // keep this
        }

        // (optional) bounded SAT θ gradient — enable if you want some θ response from SAT
        if (n_dofs_ >= 5) {
            const double JTH_MAX = 0.5;
            double jtheta_sat = scale * w_theta * dphi_dthR;  // SAT θ contribution, gated by proximity
            if (jtheta_sat >  JTH_MAX) jtheta_sat =  JTH_MAX;
            if (jtheta_sat < -JTH_MAX) jtheta_sat = -JTH_MAX;
            // J(0,4) += jtheta_sat;    // <-- add, DO NOT zero
            J(0, 4) = 0.0;
        }

        // Reset EMA when far (optional)
        if (gate < 1e-6) { w_prev_ = {0.25,0.25,0.25,0.25}; }

        return J;
    }
}

bool DynamicObstacleFactor::skip_factor()
{
    if (n_dofs_ == 4) {
        Eigen::Vector2d x = X_.head<2>();
        neighbours_ = obs_->getNearestPointsFromKDTree(x, globals.NUM_NEIGHBOURS, delta_t_);
        // this->skip_flag = neighbours_.size() == 0;
        this->skip_flag = false;
    } else {
        this->skip_flag = false;
    }
    return this->skip_flag;
}

void DynamicObstacleFactor::draw()
{
    if (n_dofs_ == 4) {
        auto v_0 = variables_[0];
        NeighbourHit closestNb = neighbours_.front();
        if (closestNb.dist_squared <= (safety_distance_ * safety_distance_))
        {
            const Eigen::Vector2d nb_pos = closestNb.pt_world;
            auto state_t = obs_->states_.at(int(std::lround(delta_t_ / globals.T0)));
    
            // Draw line to closest neighbour
            DrawCylinderEx(Vector3{(float)v_0->mu_(0), globals.ROBOT_RADIUS, (float)v_0->mu_(1)},
                           Vector3{(float)nb_pos.x(), globals.ROBOT_RADIUS, (float)nb_pos.y()},
                           0.1, 0.1, 4, RED);
    
            // Draw obstacle position at t
            float rotation = wrapAngle(state_t[4] + obs_->geom_->orientation_offset);
            float rotation_degrees = rotation * (180.0f / M_PI);
            DrawModelEx(obs_->geom_->model,
                        Vector3{(float)state_t[0], -obs_->geom_->boundingBox.min.y, (float)state_t[1]},
                        Vector3{0.0f, 1.0f, 0.0f}, // Rotate around Y axis
                        rotation_degrees,          // Rotation angle in degrees
                        Vector3{1.0f, 1.0f, 1.0f}, // Scale
                        ColorAlpha(DARKGREEN, 0.2f));
    
            // Draw violating variable
            DrawSphere(Vector3{(float)v_0->mu_(0), robot_radius_, (float)v_0->mu_(1)}, 0.8 * robot_radius_, ColorAlpha(DARKGREEN, 0.2f));
    
            // Draw neighbours
            for (const auto &nb : neighbours_)
            {
                auto nb_pos = nb.pt_world;
                DrawSphere(Vector3{(float)nb_pos.x(), robot_radius_, (float)nb_pos.y()}, 0.2 * robot_radius_, ColorAlpha(DARKBLUE, 0.2f));
            }
            if (!dbg)
            {
                // auto nbs1 = obs_->getNearestPointsFromKDTree(Eigen::Vector2d{X_(0), X_(1)}, globals.NUM_NEIGHBOURS, delta_t_);
                // auto nbs2 = obs_->getNearestPoints2D(Eigen::Vector2d{X_(0), X_(1)}, delta_t_);
                // print(nbs1.front().dist_squared, nbs2.front().dist_squared);
                // print(nbs1.front().pt_world.transpose(), nbs2.front().pt_world.transpose());
                dbg = true;
            }
        }
    }
}