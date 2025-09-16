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
    
    // Compute signed distance between sphere and OBB
    // Positive when separated, negative when overlapping
    inline double signedDistanceSphereOBB(Eigen::Ref<const Eigen::Vector2d> c, double r, const OBB2D& B) {
        // Transform sphere center to OBB's local coordinate system
        Eigen::Vector2d local_center = c - B.center;
        auto axes = B.getAxes();
        auto ax1 = axes.first, ax2 = axes.second;
        Eigen::Vector2d rotated_center;
        rotated_center.x() = local_center.dot(ax1);
        rotated_center.y() = local_center.dot(ax2);
        
        // Find the closest point on the OBB in local frame
        Eigen::Vector2d closest;
        closest.x() = std::clamp(rotated_center.x(), -B.halfExtents.x(), B.halfExtents.x());
        closest.y() = std::clamp(rotated_center.y(), -B.halfExtents.y(), B.halfExtents.y());
        
        // Calculate distance from sphere center to closest point on OBB
        Eigen::Vector2d diff = rotated_center - closest;
        double distance_to_surface = std::sqrt(diff.squaredNorm());
        
        // Check if sphere center is inside the OBB
        bool center_inside = (std::abs(rotated_center.x()) <= B.halfExtents.x() && 
                             std::abs(rotated_center.y()) <= B.halfExtents.y());
        
        if (center_inside) {
            // Center is inside OBB, return negative distance (penetration depth)
            // Find minimum distance to any edge
            double dist_to_x_edge = B.halfExtents.x() - std::abs(rotated_center.x());
            double dist_to_y_edge = B.halfExtents.y() - std::abs(rotated_center.y());
            double min_dist_to_edge = std::min(dist_to_x_edge, dist_to_y_edge);
            return -(min_dist_to_edge + r);  // Negative for overlap
        } else {
            // Center is outside OBB, return positive distance minus radius
            return distance_to_surface - r;  // Positive when separated, negative when overlapping
        }
    }
    
    // Compute signed distance between two OBBs using SAT
    // Positive when separated, negative when overlapping
    inline double signedDistanceOBB(const OBB2D& obb1, const OBB2D& obb2) {
        auto axes1 = obb1.getAxes();
        auto axes2 = obb2.getAxes();
        auto ax11 = axes1.first, ax12 = axes1.second;
        auto ax21 = axes2.first, ax22 = axes2.second;
        std::vector<Eigen::Vector2d> axes = {ax11, ax12, ax21, ax22};
        
        double min_positive_separation = std::numeric_limits<double>::max();
        double max_negative_penetration = -std::numeric_limits<double>::max();
        bool found_separating_axis = false;
        
        // Check separation along each axis
        for (const auto& axis : axes) {
            auto p1 = projectOBB(obb1, axis);
            auto p2 = projectOBB(obb2, axis);
            double min1 = p1.first, max1 = p1.second;
            double min2 = p2.first, max2 = p2.second;
            
            if (max1 < min2) {
                // obb1 is completely to the left of obb2 on this axis - they are separated
                double separation = min2 - max1;
                found_separating_axis = true;
                min_positive_separation = std::min(min_positive_separation, separation);
            } else if (max2 < min1) {
                // obb2 is completely to the left of obb1 on this axis - they are separated
                double separation = min1 - max2;
                found_separating_axis = true;
                min_positive_separation = std::min(min_positive_separation, separation);
            } else {
                // Overlapping on this axis - calculate penetration depth
                double penetration = -std::min(max1 - min2, max2 - min1);
                max_negative_penetration = std::max(max_negative_penetration, penetration);
            }
        }
        
        // If we found any separating axis, the OBBs are separated
        if (found_separating_axis) {
            return min_positive_separation;  // Return minimum positive separation
        } else {
            return max_negative_penetration; // Return maximum (least) negative penetration
        }
    }
}
