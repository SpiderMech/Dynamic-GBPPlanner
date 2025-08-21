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
        const double c = std::cos(orientation);
        const double s = std::sin(orientation);

        // y-down, θ clockwise-positive
        Eigen::Matrix2d R;
        R << c,  s,
            -s,  c;

        corners[0] = center + R * Eigen::Vector2d( halfExtents.x(),  halfExtents.y());
        corners[1] = center + R * Eigen::Vector2d(-halfExtents.x(),  halfExtents.y());
        corners[2] = center + R * Eigen::Vector2d(-halfExtents.x(), -halfExtents.y());
        corners[3] = center + R * Eigen::Vector2d( halfExtents.x(), -halfExtents.y());
        return corners;
    }

    // Get the two axes of the OBB (normalized)
    std::pair<Eigen::Vector2d, Eigen::Vector2d> getAxes() const {
        const double c = std::cos(orientation);
        const double s = std::sin(orientation);

        // To comply with world coord-system, R^T is CCW-positive
        Eigen::Vector2d axis1(c,  -s);
        Eigen::Vector2d axis2(s, c);
        return {axis1, axis2};
    }

    // Get the bounding radius of the OOB 
    double getBoundingRadius () const {
        double he_x = halfExtents.x();
        double he_y = halfExtents.y();
        return std::sqrt(he_x*he_x + he_y*he_y);
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
    inline bool overlapsOBB(const OBB2D& obb1, const OBB2D& obb2) {
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

    // Check if a point is inside an OBB
    inline bool pointInOBB(const Eigen::Vector2d& point, const OBB2D& obb) {
        // Transform point to OBB's local coordinate system
        Eigen::Vector2d local_point = point - obb.center;
        
        // Rotate to align with OBB's axes (y-down, clockwise positive)
        double cos_theta = std::cos(obb.orientation);
        double sin_theta = std::sin(obb.orientation);
        Eigen::Vector2d rotated_point;
        rotated_point.x() =  local_point.x() * cos_theta + local_point.y() * -sin_theta;
        rotated_point.y() =  local_point.x() * sin_theta + local_point.y() * cos_theta;
        
        // Check if point is within the box bounds
        return (std::abs(rotated_point.x()) <= obb.halfExtents.x() &&
                std::abs(rotated_point.y()) <= obb.halfExtents.y());
    }

    // Get closest point on an OBB and return point and distance
    inline std::pair<Eigen::Vector2d, double> closestPointOnOBB(Eigen::Ref<const Eigen::Vector2d> p, const OBB2D& obb) { 
        double cos_theta = std::cos(obb.orientation);
        double sin_theta = std::sin(obb.orientation);
        const Eigen::Vector2d ux(cos_theta, -sin_theta);
        const Eigen::Vector2d uy(sin_theta,  cos_theta);
        
        Eigen::Vector2d local_pos = p - obb.center;
        
        Eigen::Vector2d rotated_pos;
        rotated_pos.x() =  local_pos.dot(ux);
        rotated_pos.y() =  local_pos.dot(uy);
        
        // Find the closest point on the OBB to the sphere center (in local coords)
        Eigen::Vector2d closest;
        closest.x() = std::clamp(rotated_pos.x(), -obb.halfExtents.x(), obb.halfExtents.x());
        closest.y() = std::clamp(rotated_pos.y(), -obb.halfExtents.y(), obb.halfExtents.y());

        const Eigen::Vector2d closest_world = obb.center + closest.x()*ux + closest.y()*uy;
        Eigen::Vector2d diff = p - closest_world;
        return {closest_world, diff.squaredNorm()};
    }
    
    // Check if there is overlap between a sphere and an OBB
    inline bool overlapsSphereOBB(Eigen::Ref<const Eigen::Vector2d> c, double r, const OBB2D& B, double eps) {
        // Transform sphere center to OBB's local coordinate system
        Eigen::Vector2d local_center = c - B.center;
        
        // R^T for positive-CCW rotations, so R for reverse
        double cos_theta = std::cos(B.orientation);
        double sin_theta = std::sin(B.orientation);
        Eigen::Vector2d rotated_center;
        rotated_center.x() =  local_center.x() * cos_theta + local_center.y() * -sin_theta;
        rotated_center.y() =  local_center.x() * sin_theta + local_center.y() * cos_theta;
        
        // Find the closest point on the OBB to the sphere center (in local coords)
        Eigen::Vector2d closest;
        closest.x() = std::clamp(rotated_center.x(), -B.halfExtents.x(), B.halfExtents.x());
        closest.y() = std::clamp(rotated_center.y(), -B.halfExtents.y(), B.halfExtents.y());
        
        // Calculate the distance from the sphere center to this closest point
        Eigen::Vector2d diff = rotated_center - closest;
        double dist_squared = diff.squaredNorm();
        
        // Check for overlap (including epsilon tolerance)
        return dist_squared <= (r + eps) * (r + eps);
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
}
