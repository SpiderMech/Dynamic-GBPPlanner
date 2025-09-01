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
        auto [axis1, axis2] = obb.getAxes();
        double extent = std::abs(obb.halfExtents.x() * axis1.dot(axis)) +
                        std::abs(obb.halfExtents.y() * axis2.dot(axis));
        return {center_proj - extent, center_proj + extent};
    }
    
    // Check if two OBBs are colliding using SAT
    inline bool overlapsOBB(const OBB2D& obb1, const OBB2D& obb2) {
        auto axes1 = obb1.getAxes();
        auto axes2 = obb2.getAxes();
        auto ax11 = axes1.first, ax12 = axes1.second;
        auto ax21 = axes2.first, ax22 = axes2.second;        
        std::vector<Eigen::Vector2d> axes = {ax11, ax12, ax21, ax22};
        // Check separation along each axis
        for (const auto& axis : axes) {
            auto p1 = projectOBB(obb1, axis);
            auto p2 = projectOBB(obb2, axis);
            auto min1 = p1.first, max1 = p1.second;
            auto min2 = p2.first, max2 = p2.second;
            // If there's a gap, boxes don't collide
            if (max1 < min2 || max2 < min1) return false;
        }
        return true; // No separating axis found, boxes must be colliding
    }

    // Get closest point on an OBB and return point and distance
    inline std::pair<Eigen::Vector2d, double> closestPointOnOBB(Eigen::Ref<const Eigen::Vector2d> p, const OBB2D& obb) { 
        // Get point in OBB's local frame
        Eigen::Vector2d local_pos = p - obb.center;
        auto axes = obb.getAxes();
        auto ax1 = axes.first, ax2 = axes.second;
        Eigen::Vector2d rotated_pos;
        rotated_pos.x() =  local_pos.dot(ax1);
        rotated_pos.y() =  local_pos.dot(ax2);
        
        // Find the closest point on the OBB in local frame
        Eigen::Vector2d closest;
        closest.x() = std::clamp(rotated_pos.x(), -obb.halfExtents.x(), obb.halfExtents.x());
        closest.y() = std::clamp(rotated_pos.y(), -obb.halfExtents.y(), obb.halfExtents.y());

        const Eigen::Vector2d closest_world = obb.center + closest.x()*ax1 + closest.y()*ax2;
        Eigen::Vector2d diff = p - closest_world;
        return {closest_world, diff.squaredNorm()};
    }
    
    // Check if there is overlap between a sphere and an OBB
    inline bool overlapsSphereOBB(Eigen::Ref<const Eigen::Vector2d> c, double r, const OBB2D& B, double eps) {
        // Transform sphere center to OBB's local coordinate system
        Eigen::Vector2d local_center = c - B.center;
        auto axes = B.getAxes();
        auto ax1 = axes.first, ax2 = axes.second;
        Eigen::Vector2d rotated_center;
        rotated_center.x() =  local_center.dot(ax1);
        rotated_center.y() =  local_center.dot(ax2);
        
        // Find the closest point on the OBB in local frame
        Eigen::Vector2d closest;
        closest.x() = std::clamp(rotated_center.x(), -B.halfExtents.x(), B.halfExtents.x());
        closest.y() = std::clamp(rotated_center.y(), -B.halfExtents.y(), B.halfExtents.y());
        
        Eigen::Vector2d diff = rotated_center - closest;
        double dist_squared = diff.squaredNorm();
        return dist_squared <= (r + eps) * (r + eps);
    }
}
