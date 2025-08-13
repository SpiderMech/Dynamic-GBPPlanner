/**************************************************************************************/
// Geometry utilities for Oriented Bounding Box (OBB) collision detection
/**************************************************************************************/
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

/**************************************************************************************/
// Oriented Bounding Box representation
/**************************************************************************************/
struct OBB2D {
    Eigen::Vector2d center;     // Center position
    Eigen::Vector2d halfExtents; // Half-widths in local x and y
    double orientation;          // Rotation angle in radians
    
    OBB2D(const Eigen::Vector2d& c, const Eigen::Vector2d& he, double o)
        : center(c), halfExtents(he), orientation(o) {}
    
    // Get the four corners of the OBB
    std::vector<Eigen::Vector2d> getCorners() const {
        std::vector<Eigen::Vector2d> corners(4);
        double cos_o = std::cos(orientation);
        double sin_o = std::sin(orientation);
        
        // Local to world transformation
        Eigen::Matrix2d R;
        R << cos_o, -sin_o,
             sin_o,  cos_o;
        
        corners[0] = center + R * Eigen::Vector2d( halfExtents.x(),  halfExtents.y());
        corners[1] = center + R * Eigen::Vector2d(-halfExtents.x(),  halfExtents.y());
        corners[2] = center + R * Eigen::Vector2d(-halfExtents.x(), -halfExtents.y());
        corners[3] = center + R * Eigen::Vector2d( halfExtents.x(), -halfExtents.y());
        
        return corners;
    }
    
    // Get the two axes of the OBB (normalized)
    std::pair<Eigen::Vector2d, Eigen::Vector2d> getAxes() const {
        double cos_o = std::cos(orientation);
        double sin_o = std::sin(orientation);
        
        Eigen::Vector2d axis1(cos_o, sin_o);   // Local X axis in world frame
        Eigen::Vector2d axis2(-sin_o, cos_o);  // Local Y axis in world frame
        
        return {axis1, axis2};
    }
};

/**************************************************************************************/
// Separating Axis Theorem (SAT) based collision detection
/**************************************************************************************/
namespace GeometryUtils {
    
    // Project OBB onto an axis and get min/max extents
    inline std::pair<double, double> projectOBB(const OBB2D& obb, const Eigen::Vector2d& axis) {
        double center_proj = obb.center.dot(axis);
        
        // Get the extent of the box when projected onto the axis
        auto [axis1, axis2] = obb.getAxes();
        double extent = std::abs(obb.halfExtents.x() * axis1.dot(axis)) +
                       std::abs(obb.halfExtents.y() * axis2.dot(axis));
        
        return {center_proj - extent, center_proj + extent};
    }
    
    // Check if two OBBs are colliding using SAT
    inline bool checkOBBCollision(const OBB2D& obb1, const OBB2D& obb2) {
        // Get axes from both OBBs (4 axes total for 2D)
        auto [axis1_1, axis1_2] = obb1.getAxes();
        auto [axis2_1, axis2_2] = obb2.getAxes();
        
        std::vector<Eigen::Vector2d> axes = {axis1_1, axis1_2, axis2_1, axis2_2};
        
        // Check separation along each axis
        for (const auto& axis : axes) {
            auto [min1, max1] = projectOBB(obb1, axis);
            auto [min2, max2] = projectOBB(obb2, axis);
            
            // If there's a gap, boxes don't collide
            if (max1 < min2 || max2 < min1) {
                return false;
            }
        }
        
        // No separating axis found, boxes must be colliding
        return true;
    }
    
    // Get minimum distance between two OBBs with penetration depth for overlaps
    // Returns negative distance when overlapping (penetration depth)
    inline std::pair<double, Eigen::Vector2d> getOBBDistance(const OBB2D& obb1, const OBB2D& obb2) {
        // Get axes from both OBBs
        auto [axis1_1, axis1_2] = obb1.getAxes();
        auto [axis2_1, axis2_2] = obb2.getAxes();
        
        std::vector<Eigen::Vector2d> axes = {axis1_1, axis1_2, axis2_1, axis2_2};
        
        // Track minimal separating and overlapping axes
        double best_sep = std::numeric_limits<double>::max();
        Eigen::Vector2d best_sep_axis = Eigen::Vector2d::Zero();
        
        double min_overlap = std::numeric_limits<double>::max();
        Eigen::Vector2d best_ovl_axis = Eigen::Vector2d::Zero();
        bool overlapping = true;

        const Eigen::Vector2d c1 = obb1.center;
        const Eigen::Vector2d c2 = obb2.center;
        const Eigen::Vector2d c12 = c2 - c1;
        
        // Check separation/overlap along each axis
        for (const auto& axis : axes) {
            auto [min1, max1] = projectOBB(obb1, axis);
            auto [min2, max2] = projectOBB(obb2, axis);
            
            // Calculate overlap or separation
            double overlap = std::min(max1, max2) - std::max(min1, min2);
            
            // Separated along this axis, get minimal separation dist
            if (overlap < 0.0) {
                overlapping = false;
                double sep = -overlap;
                if (sep < best_sep) {
                    best_sep = sep;
                    best_sep_axis = axis;
                }
            } 
            // Track minimal overlapping dist and axis
            else {
                if (overlap < min_overlap) {
                    min_overlap = overlap;
                    best_ovl_axis = axis;
                }
            }    
        }
        
        if (!overlapping) {
            Eigen::Vector2d n_hat = best_sep_axis;
            if (n_hat.dot(c12) < 0.0) n_hat = -n_hat;  // ensure pointing 1 -> 2
            return { best_sep, n_hat };
        }

        else {
            // Penetrating: use axis with minimal positive overlap; distance is negative
            Eigen::Vector2d n_hat = best_ovl_axis;
            if (n_hat.squaredNorm() < 1e-24) {
                // Extremely degenerate case (shouldn't happen); fall back to center direction
                n_hat = c12.normalized();
            } else if (n_hat.dot(c12) < 0.0) {
                n_hat = -n_hat;
            }
            return { -min_overlap, n_hat };
        }
    }
    
    // Get smooth pseudo-distance for gradient computation
    // This provides a continuous differentiable approximation
    inline double getSmoothOBBDistance(const OBB2D& obb1, const OBB2D& obb2) {
        // Use ellipsoid approximation for smooth gradients
        Eigen::Vector2d delta = obb2.center - obb1.center;
        
        // Transform to obb1's local frame
        double cos1 = std::cos(-obb1.orientation);
        double sin1 = std::sin(-obb1.orientation);
        Eigen::Vector2d local_delta;
        local_delta.x() = delta.x() * cos1 - delta.y() * sin1;
        local_delta.y() = delta.x() * sin1 + delta.y() * cos1;
        
        // Normalized position in obb1's frame
        Eigen::Vector2d norm_pos1;
        norm_pos1.x() = local_delta.x() / (obb1.halfExtents.x() + 1e-6);
        norm_pos1.y() = local_delta.y() / (obb1.halfExtents.y() + 1e-6);
        
        // Transform to obb2's local frame  
        double angle_diff = obb2.orientation - obb1.orientation;
        double cos_diff = std::cos(-angle_diff);
        double sin_diff = std::sin(-angle_diff);
        Eigen::Vector2d norm_pos2;
        norm_pos2.x() = norm_pos1.x() * cos_diff - norm_pos1.y() * sin_diff;
        norm_pos2.y() = norm_pos1.x() * sin_diff + norm_pos1.y() * cos_diff;
        
        // Ellipsoid distance approximation
        double r1 = norm_pos1.norm();
        double r2 = (obb2.halfExtents.x() + obb2.halfExtents.y()) / 
                   (obb1.halfExtents.x() + obb1.halfExtents.y());
        
        // Smooth approximation that's negative when overlapping
        return (r1 - 1.0 - r2) * obb1.halfExtents.norm();
    }
    
    // Check if a point is inside an OBB
    inline bool isPointInOBB(const Eigen::Vector2d& point, const OBB2D& obb) {
        // Transform point to OBB's local coordinate system
        Eigen::Vector2d local_point = point - obb.center;
        
        double cos_o = std::cos(-obb.orientation);  // Negative for inverse rotation
        double sin_o = std::sin(-obb.orientation);
        
        Eigen::Vector2d rotated_point;
        rotated_point.x() = local_point.x() * cos_o - local_point.y() * sin_o;
        rotated_point.y() = local_point.x() * sin_o + local_point.y() * cos_o;
        
        // Check if point is within the box in local coordinates
        return std::abs(rotated_point.x()) <= obb.halfExtents.x() &&
               std::abs(rotated_point.y()) <= obb.halfExtents.y();
    }
    
    // Get OBB from robot state
    inline OBB2D getRobotOBB(const Eigen::VectorXd& state, const Eigen::Vector2d& dimensions) {
        // State is [x, y, xdot, ydot, theta]
        Eigen::Vector2d center(state(0), state(1));
        Eigen::Vector2d halfExtents = dimensions / 2.0;
        double orientation = state(4);  // theta is at index 4
        
        return OBB2D(center, halfExtents, orientation);
    }
    
    // Normalize angle to [-π, π]
    inline double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
    // Get orientation from velocity
    inline double getOrientationFromVelocity(double vx, double vy, double current_orientation = 0.0) {
        double speed = std::sqrt(vx * vx + vy * vy);
        
        // If nearly stationary, keep current orientation
        if (speed < 0.01) {
            return current_orientation;
        }
        
        return std::atan2(vy, vx);
    }
}
