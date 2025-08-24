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
static inline double angle_diff(double a, double b)
{
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
    if (n_dofs_ == 5)
    {
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
        double sigma_theta = sigma * 10.0; // Much looser constraint
        double Qc_inv_theta = pow(sigma_theta, -2.);
        Qi_inv(4, 4) = 1. / dt_ * Qc_inv_theta; // Reduced weight

        this->meas_model_lambda_ = Qi_inv;

        // For instantaneous alignment, orientation is nonlinear
        this->linear_ = false;

        // For 4D state: [x, y, xdot, ydot]
    }
    else
    {
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
    if (n_dofs_ == 5)
    {
        // X contains [x1, y1, xdot1, ydot1, theta1, x2, y2, xdot2, ydot2, theta2]
        Eigen::VectorXd h = Eigen::VectorXd::Zero(5);

        // Position constraint: x2 = x1 + xdot1 * dt, use predicted - actual for consistency
        h(0) = X(0) + X(2) * dt_ - X(5);
        h(1) = X(1) + X(3) * dt_ - X(6);

        // Velocity constraint: xdot2 = xdot1 (constant velocity)
        h(2) = X(2) - X(7);
        h(3) = X(3) - X(8);

        // Orientation constraint: theta should align with velocity direction
        double xdot2 = X(7), ydot2 = X(8);
        double v2 = xdot2 * xdot2 + ydot2 * ydot2;
        double v_mag = std::sqrt(v2);

        // Only apply orientation constraint when moving at reasonable speed
        // This prevents oscillations at low speeds
        double v_min = v0_ * 0.1; // Minimum velocity to apply constraint

        double theta_diff = 0.0;
        if (std::isfinite(X(9)) && v_mag > v_min)
        {
            // theta should align with velocity direction
            // Since Y is down, we need to negate it for proper angle calculation
            double theta_vel = -wrapAngle(std::atan2(ydot2, xdot2));
            // Use angle_diff for consistent angle wrapping
            theta_diff = angle_diff(theta_vel, X(9));

            // Apply smooth transition based on velocity magnitude
            // Use tanh for smoother, bounded transition than sigmoid
            double k = 3.0 / v0_; // Transition rate
            double w = 0.5 * (1.0 + std::tanh(k * (v_mag - v0_ * 0.3)));

            // Scale the error by the weight
            h(4) = w * theta_diff;
        }
        else
        {
            // At low speeds, don't constrain orientation
            // Issue: this causes free spinning at low speeds?
            h(4) = 0.0;
        }

        return h;
    }
    else
    {
        // Original 4D implementation
        return J_ * X;
    }
}

// Eigen::MatrixXd DynamicsFactor::h_func_(const Eigen::VectorXd &X)
// {
//     if (n_dofs_ == 5)
//     {
//         // X = [x1,y1,xdot1,ydot1,theta1, x2,y2,xdot2,ydot2,theta2]
//         Eigen::VectorXd h = Eigen::VectorXd::Zero(5);

//         // Position (x2 = x1 + xdot1*dt)
//         h(0) = X(0) + X(2) * dt_ - X(5);
//         h(1) = X(1) + X(3) * dt_ - X(6);

//         // Constant velocity (xdot2 = xdot1)
//         h(2) = X(2) - X(7);
//         h(3) = X(3) - X(8);

//         // -------- Orientation (blended) --------
//         const double xdot2 = X(7), ydot2 = X(8);
//         const double v2    = xdot2*xdot2 + ydot2*ydot2;
//         const double vmag  = std::sqrt(v2);

//         // Smooth speed gates: below v_low => pure θ-smoothness; above v_high => pure heading align
//         const double v_low  = 0.05 * v0_;
//         const double v_high = 0.30 * v0_;
//         auto clamp01 = [](double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
//         double s = (v_high > v_low) ? clamp01((vmag - v_low) / (v_high - v_low)) : 1.0;
//         // smoothstep for C1 continuity
//         double w_align = s*s*(3.0 - 2.0*s); // in [0,1]

//         // Heading from velocity (screen Y down -> negate to keep right-handed angle sense)
//         double theta_vel = -wrapAngle(std::atan2(ydot2, xdot2));

//         // Two angle residuals
//         double a = angle_diff(theta_vel, X(9)); // align to velocity
//         double b = angle_diff(X(4),      X(9)); // keep near previous θ

//         // Blended orientation residual
//         h(4) = w_align * a + (1.0 - w_align) * b;

        
//         /* --DEBUG-- */
//         bool in_transition = (w_align > 1e-6 && w_align < 1.0 - 1e-6);
//         bool slow          = (vmag < 0.5 * v0_);
//         if (in_transition || slow) {
//             auto v0 = variables_[0];
//             auto v1 = variables_[1];
            
//             std::ostringstream oss;
//             oss << "[DEBUG] [h] ts=" << v0->ts_ << "," << v1->ts_
//                 << " x2=(" << X(5) << "," << X(6) << ")"
//                 << " v2=(" << X(7) << "," << X(8) << ")"
//                 << " vmag=" << vmag
//                 << " w=" << w_align
//                 << " a(θv-θ2)=" << a
//                 << " b(θ1-θ2)=" << b
//                 << " rθ=" << h(4)
//                 << " θ1=" << X(4)
//                 << " θ2=" << X(9);
//             print(oss.str());
//         }
//         /* --DEBUG-- */

//         return h;
//     }
//     else
//     {
//         return J_ * X; // 4D original
//     }
// }


Eigen::MatrixXd DynamicsFactor::J_func_(const Eigen::VectorXd &X)
{
    if (n_dofs_ == 5)
    {
        // 5D state Jacobian with instantaneous alignment
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(5, 10);

        // Position and velocity constraints Jacobian - same as 4D case
        J(0, 0) = 1;
        J(0, 2) = dt_;
        J(0, 5) = -1; // dx
        J(1, 1) = 1;
        J(1, 3) = dt_;
        J(1, 6) = -1; // dy
        J(2, 2) = 1;
        J(2, 7) = -1; // dxdot
        J(3, 3) = 1;
        J(3, 8) = -1; // dydot

        // Orientation constraint Jacobian (nonlinear due to atan2)
        double xdot2 = X(7), ydot2 = X(8);
        double v2 = xdot2 * xdot2 + ydot2 * ydot2;
        double v_mag = std::sqrt(v2);

        // Minimum velocity threshold
        double v_min = v0_ * 0.1;

        if (v_mag > v_min)
        {
            // Compute weight using tanh for smooth transition
            double k = 3.0 / v0_;
            double tanh_arg = k * (v_mag - v0_ * 0.3);
            double tanh_val = std::tanh(tanh_arg);
            double w = 0.5 * (1.0 + tanh_val);

            // d theta_vel / d xdot2, d ydot2 with stronger regularization
            double v2_reg = v2 + v_min * v_min;
            double dthetad_x = ydot2 / v2_reg;
            double dthetad_y = -xdot2 / v2_reg;

            // Product-rule term discarded
            J(4, 7) = w * dthetad_x; // wrt xdot2
            J(4, 8) = w * dthetad_y; // wrt ydot2
            J(4, 9) = -w;            // wrt theta2
        }
        else
        {
            // Below minimum velocity, no constraint
            J.row(4).setZero();
        }
        return J;
    }
    else
    {
        // Original 4D implementation
        return J_;
    }
}


// Eigen::MatrixXd DynamicsFactor::J_func_(const Eigen::VectorXd &X)
// {
//     if (n_dofs_ == 5)
//     {
//         Eigen::MatrixXd J = Eigen::MatrixXd::Zero(5, 10);

//         // Position & velocity rows
//         J(0,0) = 1;    J(0,2) = dt_; J(0,5) = -1;
//         J(1,1) = 1;    J(1,3) = dt_; J(1,6) = -1;
//         J(2,2) = 1;    J(2,7) = -1;
//         J(3,3) = 1;    J(3,8) = -1;

//         // -------- Orientation row --------
//         const double xdot2 = X(7), ydot2 = X(8);
//         const double v2    = xdot2*xdot2 + ydot2*ydot2;
//         const double vmag  = std::sqrt(std::max(1e-16, v2)); // avoid 0/0

//         const double v_low  = 0.05 * v0_;
//         const double v_high = 0.30 * v0_;
//         auto clamp01 = [](double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
//         double s = (v_high > v_low) ? clamp01((vmag - v_low) / (v_high - v_low)) : 1.0;
//         double w_align = s*s*(3.0 - 2.0*s); // smoothstep

//         // theta_vel = -atan2(ydot2, xdot2)
//         // d(theta_vel)/dx =  y / (x^2 + y^2); d(theta_vel)/dy = -x / (x^2 + y^2)
//         const double v2_reg = v2 + (0.1 * v0_) * (0.1 * v0_); // gentle regulariser
//         const double dthetad_x =  ydot2 / v2_reg;
//         const double dthetad_y = -xdot2 / v2_reg;

//         // Residual parts:
//         // a = angle_diff(theta_vel, θ2) => da/dθ2 = -1, da/dx = dthetad_x, da/dy = dthetad_y
//         // b = angle_diff(θ1, θ2)       => db/dθ1 =  1,  db/dθ2 = -1
//         //
//         // r = w*a + (1-w)*b
//         // ∂r = w∂a + (1-w)∂b + (∂w)(a - b)

//         // Compute a and b for the (∂w)(a-b) term
//         double theta_vel = -wrapAngle(std::atan2(ydot2, xdot2));
//         double a = angle_diff(theta_vel, X(9));
//         double b = angle_diff(X(4),      X(9));
//         double a_minus_b = a - b;

//         // ∂a, ∂b terms
//         J(4,7) += w_align * dthetad_x;   // via a
//         J(4,8) += w_align * dthetad_y;   // via a
//         J(4,9) += w_align * (-1.0)       // ∂a/∂θ2
//                 + (1.0 - w_align) * (-1.0); // ∂b/∂θ2
//         J(4,4) += (1.0 - w_align) * ( 1.0); // ∂b/∂θ1

//         // ∂w terms (optional but helpful around the transition)
//         if (v_high > v_low && vmag > 1e-12)
//         {
//             // smoothstep: w = 3s^2 - 2s^3; dw/ds = 6s - 6s^2 = 6s(1-s)
//             double dwds = 6.0 * s * (1.0 - s);
//             double dsdv = 1.0 / (v_high - v_low); // inside (0,1) region, else 0
//             // guard: only active when 0<s<1 to avoid kinks
//             if (s > 0.0 && s < 1.0)
//             {
//                 double dvdx = (xdot2 / vmag);
//                 double dvdy = (ydot2 / vmag);
//                 double dwdx = dwds * dsdv * dvdx;
//                 double dwdy = dwds * dsdv * dvdy;

//                 J(4,7) += dwdx * a_minus_b;
//                 J(4,8) += dwdy * a_minus_b;
//             }
//         }
        
//         /* -- DEBUG -- */
//         bool in_transition = (w_align > 1e-6 && w_align < 1.0 - 1e-6);
//         bool slow          = (vmag < 0.5 * v0_);
//         if (in_transition || slow) { 
//             const auto row = J.row(4);
//             auto v0 = variables_[0];
//             auto v1 = variables_[1];
    
//             std::ostringstream oss;
//             oss << "[DEBUG] [J] ts=" << v0->ts_ << "," << v1->ts_
//                 << " vmag=" << vmag
//                 << " w=" << w_align
//                 << " dθv/dxdot=" << dthetad_x
//                 << " dθv/dydot=" << dthetad_y
//                 << " (a-b)=" << a_minus_b
//                 << " Jθ-nz{ ";
//             for (int c = 0; c < row.cols(); ++c) {
//                 double val = row(c);
//                 if (std::abs(val) > 1e-12) {
//                     oss << c << ":" << val << " ";
//                 }
//             }
//             oss << "}";
//             print(oss.str());
//         }
//         /* -- DEBUG -- */

//         return J;
//     }
//     else
//     {
//         return J_;
//     }
// }

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

    // Use a center-to-center safety distance irrespective of DOFs.
    // This stabilizes GBP and avoids orientation-induced flips.
    if (n_dofs >= 5) {
        this->safety_distance_ = 1.2;
    } else {
        float eps = 0.5f * robot_radius;
        this->safety_distance_ = 2 * robot_radius + eps; // ~= r1 + r2 + margin
    }
    tau_d_ = 0.2 * safety_distance_;
    // Analytical Jacobian is used (no orientation derivatives).
    this->delta_jac = 1e-4;
}

Eigen::MatrixXd InterrobotFactor::h_func_(const Eigen::VectorXd &X)
{
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
    
    if (dbg) {
        std::ostringstream oss;
        auto v0 = variables_[0];
        auto v1 = variables_[1];
        double Zg = 0.0, w[4];
        for (int k=0;k<4;++k) { w[k] = std::exp(beta*(g[k]-m)); Zg += w[k]; }
        for (int k=0;k<4;++k) w[k] /= std::max(Zg, 1e-12);
        auto vel1 = v0->mu_.segment<2>(2);
        auto vel2 = v1->mu_.segment<2>(2);
        auto v_rel = vel1-vel2;
        oss << "[DEBUG] [h] ts" <<  v0->ts_ << "," << v1->ts_
            << " phi=" << phi
            << " z=" << z
            << " s=" << s
            << " r=" << r
            << " h=" << h(0,0)
            << " gate=" << gate
            << " gaps=["<<g[0]<<","<<g[1]<<","<<g[2]<<","<<g[3]<<"]"
            << " w=["<<w[0]<<","<<w[1]<<","<<w[2]<<","<<w[3]<<"]"
            << " vrel=["<<v_rel.x()<<","<<v_rel.y()<<"]";
        print(oss.str());
    }
    return h;
}

Eigen::MatrixXd InterrobotFactor::J_func_(const Eigen::VectorXd &X)
{
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

    // --- Debug (print the ACTUAL weights used) ---
    if (dbg){
        int kmax = 0; for (int k=1;k<4;++k) if (w[k] > w[kmax]) kmax = k;
        std::ostringstream oss;
        oss << "[DEBUG] [J] ts" << variables_[0]->ts_ << "," << variables_[1]->ts_
            << " phi_g=" << phi_g
            << " z=" << z
            << " s=" << s
            << " sig=" << sig
            << " J_norm=" << J.row(0).norm()
            << " gate=" << gate
            << " gaps=["<<g[0]<<","<<g[1]<<","<<g[2]<<","<<g[3]<<"]"
            << " gb=["<<gb[0]<<","<<gb[1]<<","<<gb[2]<<","<<gb[3]<<"]"
            << " w=["<<w[0]<<","<<w[1]<<","<<w[2]<<","<<w[3]<<"]"
            << " kmax=" << kmax;
        print(oss.str());
    }

    if (gate < 1e-6) { w_prev_ = {0.25, 0.25, 0.25, 0.25}; }

    return J;
}

// Eigen::MatrixXd InterrobotFactor::J_func_(const Eigen::VectorXd &X)
// {
//     const int n0 = variables_[0]->n_dofs_;
//     const int n1 = variables_[1]->n_dofs_;

//     Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, n0 + n1);

//     // State slices
//     const Eigen::Vector2d c1 = X.segment<2>(0);
//     const Eigen::Vector2d c2 = X.segment<2>(n0);
//     const double th1 = (n0 >= 5 ? wrapAngle(X(4)) : 0.0);
//     const double th2 = (n1 >= 5 ? wrapAngle(X(n0 + 4)) : 0.0);

//     // OBB params with your offsets
//     const Eigen::Vector2d e1 = 0.5 * robot1_dimensions_;
//     const Eigen::Vector2d e2 = 0.5 * robot2_dimensions_;
//     const double o1 = wrapAngle(th1 + robot1_angle_offset_);
//     const double o2 = wrapAngle(th2 + robot2_angle_offset_);

//     OBB2D obb1(c1, e1, o1);
//     OBB2D obb2(c2, e2, o2);

//     const auto axes1 = obb1.getAxes();
//     const auto axes2 = obb2.getAxes();
//     const Eigen::Vector2d a1x = axes1.first,  a1y = axes1.second;
//     const Eigen::Vector2d a2x = axes2.first,  a2y = axes2.second;

//     const Eigen::Vector2d d = c2 - c1;

//     // Smoothing constants
//     const double eps_abs = 1e-6;
//     const double eps_h = 1e-6;
//     const double beta = 3.0;

//     auto sabs = [&](double t)
//     { return std::sqrt(t * t + eps_abs * eps_abs); };
//     auto sgns = [&](double t)
//     { return t / std::sqrt(t * t + eps_abs * eps_abs); }; // d/dt |t|

//     // Axis derivative helpers: for your getAxes(), d/dθ of the two axes is:
//     // ∂a_x/∂θ = +a_y,   ∂a_y/∂θ = -a_x
//     auto dn_dth_box1 = [&](const Eigen::Vector2d &n) -> Eigen::Vector2d
//     {
//         // if n==a1x → -a1y; if n==a1y → a1x; else 0
//         if ((n - a1x).squaredNorm() < 1e-18) return -a1y;
//         if ((n - a1y).squaredNorm() < 1e-18) return  a1x;
//         return Eigen::Vector2d::Zero();
//     };
//     auto dn_dth_box2 = [&](const Eigen::Vector2d &n) -> Eigen::Vector2d
//     {
//         // if n==a2x → -a2y; if n==a2y → a2x; else 0
//         if ((n - a2x).squaredNorm() < 1e-18) return -a2y;
//         if ((n - a2y).squaredNorm() < 1e-18) return  a2x;
//         return Eigen::Vector2d::Zero();
//     };

//     // E(ax, ay, e; n) = e.x*|ax·n| + e.y*|ay·n|
//     auto extent = [&](const Eigen::Vector2d &n,
//                       const Eigen::Vector2d &ax, const Eigen::Vector2d &ay,
//                       const Eigen::Vector2d &e) -> double
//     {
//         return e.x() * sabs(ax.dot(n)) + e.y() * sabs(ay.dot(n));
//     };

//     // s(n) = |d·n| - (E1(n)+E2(n))
//     struct SGrad
//     {
//         double s;
//         Eigen::RowVector2d dc1;
//         Eigen::RowVector2d dc2;
//         double dth1;
//         double dth2;
//     };

//     auto dsep = [&](const Eigen::Vector2d &n) -> SGrad
//     {
//         SGrad out;
//         out.s = 0.0;
//         out.dc1.setZero();
//         out.dc2.setZero();
//         out.dth1 = 0.0;
//         out.dth2 = 0.0;

//         // Scalar parts
//         const double t = d.dot(n);
//         const double T = sabs(t);
//         const double dt = sgns(t); // dT/dt

//         // Extents
//         const double ax1 = a1x.dot(n), ay1 = a1y.dot(n);
//         const double ax2 = a2x.dot(n), ay2 = a2y.dot(n);
//         const double E1 = e1.x() * sabs(ax1) + e1.y() * sabs(ay1);
//         const double E2 = e2.x() * sabs(ax2) + e2.y() * sabs(ay2);

//         out.s = T - (E1 + E2);

//         // Position grads (treat n as fixed for pos — standard SAT linearization)
//         // dT/dc1 = dT/dt * d·n/dc1 = dt * (-n),   dT/dc2 = dt * (+n)
//         out.dc1 = (-dt) * n.transpose();
//         out.dc2 = (+dt) * n.transpose();

//         // Heading grads: two effects:
//         // (i) when n comes from that box, n rotates: dT/dθ = dt * d·(dn/dθ)
//         // (ii) E-terms change via d(ax·n)/dθ and d(ay·n)/dθ
//         // Box1:
//         {
//             const Eigen::Vector2d dn1 = dn_dth_box1(n); // zero unless n is a1x/a1y
//             double dE1_dth1 = 0.0;
//             // d(ax1·n)/dθ1 = (∂a1x/∂θ1)·n + a1x·(∂n/∂θ1)
//             // But when n==a1x or a1y, E1 becomes constant wrt θ1 (unit axis aligns),
//             // so we can safely ignore the tiny residual term (kept general below):
//             const double dax1 = (-a1y).dot(n);    // (∂a1x/∂θ1)·n
//             const double day1 = ( a1x).dot(n); // (∂a1y/∂θ1)·n
//             const double sgn_ax1 = sgns(ax1);
//             const double sgn_ay1 = sgns(ay1);
//             // If n rotates (dn1!=0), additional terms from n in dot-products:
//             const double ax1_n = a1x.dot(dn1);
//             const double ay1_n = a1y.dot(dn1);
//             // dE1_dth1 = e1.x() * sgn_ax1 * (dax1 + ax1_n) + e1.y() * sgn_ay1 * (day1 + ay1_n);
//             dE1_dth1 = e1.x()*sgn_ax1 *dax1 + e1.y()*sgn_ay1*day1;

//             // E2 depends on θ1 only via n (if n==a1x/a1y):
//             const double sgn_ax2 = sgns(ax2);
//             const double sgn_ay2 = sgns(ay2);
//             const double ax2_n = a2x.dot(dn1);
//             const double ay2_n = a2y.dot(dn1);
//             const double dE2_dth1 = e2.x() * sgn_ax2 * ax2_n + e2.y() * sgn_ay2 * ay2_n;

//             // T term via n (only if n from box1): dT/dθ1 = dt * d·(dn1)
//             const double dT_dth1 = dt * d.dot(dn1);

//             // out.dth1 = dT_dth1 - (dE1_dth1 + dE2_dth1);
//             out.dth1 =-dE1_dth1;
//         }

//         // Box2:
//         {
//             const Eigen::Vector2d dn2 = dn_dth_box2(n); // zero unless n is a2x/a2y
//             double dE2_dth2 = 0.0;

//             const double dax2 = (-a2y).dot(n);    // (∂a2x/∂θ2)·n
//             const double day2 = ( a2x).dot(n); // (∂a2y/∂θ2)·n
//             const double sgn_ax2 = sgns(ax2);
//             const double sgn_ay2 = sgns(ay2);
//             const double ax2_n = a2x.dot(dn2);
//             const double ay2_n = a2y.dot(dn2);
//             // dE2_dth2 = e2.x() * sgn_ax2 * (dax2 + ax2_n) + e2.y() * sgn_ay2 * (day2 + ay2_n);
//             dE2_dth2 = e2.x()*sgn_ax2*dax2 + e2.y()*sgn_ay2*day2;

//             // E1 depends on θ2 only via n (if n==a2x/a2y):
//             const double sgn_ax1 = sgns(ax1);
//             const double sgn_ay1 = sgns(ay1);
//             const double ax1_n = a1x.dot(dn2);
//             const double ay1_n = a1y.dot(dn2);
//             const double dE1_dth2 = e1.x() * sgn_ax1 * ax1_n + e1.y() * sgn_ay1 * ay1_n;

//             const double dT_dth2 = dt * d.dot(dn2);

//             // out.dth2 = dT_dth2 - (dE1_dth2 + dE2_dth2);
//             out.dth2 = -dE1_dth2;
//         }

//         return out;
//     };

//     // Compute per-axis gaps and grads
//     SGrad gg[4];
//     gg[0] = dsep(a1x);
//     gg[1] = dsep(a1y);
//     gg[2] = dsep(a2x);
//     gg[3] = dsep(a2y);

//     double g[4];
//     for (int k = 0; k < 4; ++k)
//         g[k] = gg[k].s;

//     // Softmax weights w_k for φ = smoothmax(g_k)
//     double m = std::max(std::max(g[0], g[1]), std::max(g[2], g[3]));
//     double Z = 0.0;
//     for (int k = 0; k < 4; ++k)
//         Z += std::exp(beta * (g[k] - m));
//     double w[4];
//     for (int k = 0; k < 4; ++k)
//         w[k] = std::exp(beta * (g[k] - m)) / std::max(Z, 1e-12);

//     // dφ/dx = ∑_k w_k * dg_k/dx
//     Eigen::RowVector2d dphi_dc1 = Eigen::RowVector2d::Zero();
//     Eigen::RowVector2d dphi_dc2 = Eigen::RowVector2d::Zero();
//     double dphi_dth1 = 0.0, dphi_dth2 = 0.0;
//     for (int k = 0; k < 4; ++k)
//     {
//         dphi_dc1 += w[k] * gg[k].dc1;
//         dphi_dc2 += w[k] * gg[k].dc2;
//         dphi_dth1 += w[k] * gg[k].dth1;
//         dphi_dth2 += w[k] * gg[k].dth2;
//     }

//     // Residual r = sqrt((d_safe - φ)^2 + eps_h^2) / d_safe
//     // → dr/dx = (1/d_safe) * ( (z/r) * (-dφ/dx) )
//     // We also skip if φ > d_safe (factor inactive)
//     // Recompute φ to get z (or reuse m,Z above)
//     const double phi = m + std::log(std::max(Z, 1e-12)) / beta;
//     if (phi > safety_distance_ + 3.0 * tau_d_)
//     {
//         this->skip_flag = true;
//         return J; // all zeros
//     }
//     this->skip_flag = false;

//     const double z = safety_distance_ - phi;
//     const double s = z / tau_d_;
//     const double sig = 1.0 / (1.0 + std::exp(-s));
//     // const double scale = -(sig / std::max(1e-9, safety_distance_));
//     const double gate = 1.0 / (1.0 + std::exp((phi - (safety_distance_ + 2.0*tau_d_)) / (0.5*tau_d_)));
//     const double scale = -gate * (sig / std::max(1e-9, safety_distance_));

//     // const double r = std::sqrt(z * z + eps_h * eps_h);
//     // const double inv_sd = 1.0 / std::max(1e-9, safety_distance_);
//     // const double scale = inv_sd * (z / std::max(r, 1e-12)) * (-1.0);
//     // const double alpha = 10.0;
//     // double kth = 1.0 / (1.0 + std::exp(-alpha * z));
//     // double kth = 1.0;

//     // Fill Jacobian (position + heading if present). Velocity columns remain 0.
//     // Box 1: x,y,(vx,vy),theta
//     J(0, 0) = scale * dphi_dc1.x();
//     J(0, 1) = scale * dphi_dc1.y();
//     // if (n0 >= 5) J(0, 4) = scale * dphi_dth1 * kth;
//     if (n0 >= 5) J(0, 4) = 0.0;

//     // Box 2 starts at column n0
//     J(0, n0 + 0) = scale * dphi_dc2.x();
//     J(0, n0 + 1) = scale * dphi_dc2.y();
//     // if (n1 >= 5) J(0, n0 + 4) = scale * dphi_dth2 * kth;
//     if (n1 >= 5) J(0, n0 + 4) = 0.0;

//     if (dbg) {
//         auto v0 = variables_[0];
//         auto v1 = variables_[1];
//         double Zg = 0.0, w[4];
//         for (int k=0;k<4;++k) { w[k] = std::exp(beta*(g[k]-m)); Zg += w[k]; }
//         for (int k=0;k<4;++k) w[k] /= std::max(Zg, 1e-12);
//         int kmax = 0; for (int k=1;k<4;++k) if (w[k] > w[kmax]) kmax = k;
//         std::ostringstream oss;
//         oss << "[DEBUG] [J] ts" <<  v0->ts_ << "," << v1->ts_
//             << " phi=" << phi
//             << " z=" << z
//             << " s=" << s
//             << " sig=" << sig
//             << " J_norm=" << J.row(0).norm()
//             << " gate=" << gate
//             << " gaps=["<<g[0]<<","<<g[1]<<","<<g[2]<<","<<g[3]<<"]"
//             << " w=["<<w[0]<<","<<w[1]<<","<<w[2]<<","<<w[3]<<"]"
//             << " kmax=" << kmax;
//         print(oss.str());
//     }

//     return J;
// }


bool InterrobotFactor::skip_factor()
{
    // const int n0 = variables_[0]->n_dofs_;
    // Eigen::Vector2d p1 = X_.segment<2>(0);
    // Eigen::Vector2d p2 = X_.segment<2>(n0);
    // // Conservative bounding check using OBB circumscribed radii (half-diagonals)
    // Eigen::Vector2d e1 = 0.5 * robot1_dimensions_;
    // Eigen::Vector2d e2 = 0.5 * robot2_dimensions_;
    // double r1 = e1.norm();
    // double r2 = e2.norm();
    // double center_dist = (p1 - p2).norm();
    // double conservative = r1 + r2 + safety_distance_;
    // this->skip_flag = (center_dist > conservative);
    this->skip_flag = false;
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
    const double s_obs2 = projectedVariance(Sigma, diff);
    const double s_eff2 = s_nom2 + s_obs2;
    return 1.0 / (2.0 * std::max(1e-12, s_eff2));
}

DynamicObstacleFactor::DynamicObstacleFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables, int n_dofs,
                                             float sigma, const Eigen::VectorXd &measurement, float robot_radius, std::shared_ptr<DynamicObstacle> obs)
    : Factor{f_id, r_id, variables, sigma, measurement, n_dofs}, robot_radius_(robot_radius), obs_(std::move(obs))
{
    factor_type_ = DYNAMIC_OBSTACLE_FACTOR;
    delta_t_ = variables.front()->ts_ * globals.T0;
    double eps = 0.2 * robot_radius;
    safety_distance_ = robot_radius_ + eps;
    this->delta_jac = 1e-5;
}

Eigen::MatrixXd DynamicObstacleFactor::h_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(), z_.cols());
    const Eigen::Vector2d x = X.segment<2>(0);

    double acc = 0.0;
    for (const auto &nb : neighbours_)
    {
        const Eigen::Vector2d diff = x - nb.pt_world;
        const double gamma = gammaEff(nb.Sigma_pos, diff);
        const double w = std::exp(-gamma * nb.dist_squared);
        acc += w;
    }
    h(0) = (neighbours_.empty() ? 0.0 : acc / double(neighbours_.size()));
    return h;
}

Eigen::MatrixXd DynamicObstacleFactor::J_func_(const Eigen::VectorXd &X)
{
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), X.size());
    const Eigen::Vector2d x = X.segment<2>(0);

    Eigen::Vector2d grad_obs = Eigen::Vector2d::Zero();
    for (const auto &nb : neighbours_)
    {
        const Eigen::Vector2d diff = x - nb.pt_world;
        const double gamma = gammaEff(nb.Sigma_pos, diff);
        const double w = std::exp(-gamma * nb.dist_squared);
        grad_obs += -2.0 * gamma * w * diff;
    }

    if (!neighbours_.empty())
        grad_obs /= double(neighbours_.size());
    J(0, 0) = grad_obs.x();
    J(0, 1) = grad_obs.y();
    return J;
}

bool DynamicObstacleFactor::skip_factor()
{
    Eigen::Vector2d x = X_.head<2>();
    neighbours_ = obs_->getNearestPointsFromKDTree(x, globals.NUM_NEIGHBOURS, delta_t_);
    // neighbours_ = obs_->getNearestPoints2D(x, delta_t_);
    // this->skip_flag = neighbours_.front().dist_squared > (safety_distance_ * safety_distance_);
    this->skip_flag = false;
    return this->skip_flag;
}

void DynamicObstacleFactor::draw()
{
    auto v_0 = variables_[0];
    NeighbourHit closestNb = neighbours_.front();
    if (closestNb.dist_squared <= (safety_distance_ * safety_distance_))
    {
        const Eigen::Vector2d nb_pos = closestNb.pt_world;
        auto state_t = obs_->states_.at(int(delta_t_ / globals.T0));

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