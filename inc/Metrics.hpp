#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <string>
#include <Utils.h>

/***************************************************************************************************/
/* Robot sample structures */
/***************************************************************************************************/
struct Sample
{
    double t;
    Eigen::Vector2d p;
};

/***************************************************************************************************/
/* Encounter tracking data structures */
/***************************************************************************************************/
struct RobotTrack
{
    std::vector<Sample> samples; // In time order
    bool collided = false;
    double t_start = std::numeric_limits<double>::quiet_NaN(); // First sample
    double t_end = std::numeric_limits<double>::quiet_NaN();   // Crash < last sample
    double t_first_collision = std::numeric_limits<double>::quiet_NaN(); // Time of first collision
    Eigen::Vector2d first_collision_pos = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // Position of first collision
    double baseline_path_length = -1.0;                        /* Used for detour ratio */
    
    // Encounter tracking per robot
    std::unordered_set<int> robots_inside;           // robot IDs currently inside interaction radius
    std::unordered_map<int, double> robot_last_event; // refractory timer per robot
    std::unordered_set<int> obstacles_inside;        // obstacle IDs currently inside interaction radius  
    std::unordered_map<int, double> obstacle_last_event; // refractory timer per obstacle
    size_t robot_encounters = 0;     // Count of robot-robot encounter events
    size_t obstacle_encounters = 0;  // Count of robot-obstacle encounter events
};

/***************************************************************************************************/
/* Obstacle sample structures */
/***************************************************************************************************/
struct ObstacleLifecycleInfo
{
    ObstacleType type;
    double spawn_time;
    double despawn_time = std::numeric_limits<double>::quiet_NaN();
    bool completed = false;
};

// Obstacle metrics per type
struct ObstacleTypeMetrics
{
    std::vector<double> lifetimes;             // [s] All instance life times
    std::vector<double> live_obstacle_samples; // Samples of live obstacle counts
    double total_sample_time = 0.0;            // Total time spent sampling
    double area;

    void setArea(double a) {
        if (a > 0.0) area = a;
    }

    double getAverageLifetime() const
    {
        if (lifetimes.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::accumulate(lifetimes.begin(), lifetimes.end(), 0.0) / lifetimes.size();
    }

    double getAverageLiveObstacles() const
    {
        if (live_obstacle_samples.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::accumulate(live_obstacle_samples.begin(), live_obstacle_samples.end(), 0.0) / live_obstacle_samples.size();
    }

    double getDensity(double area) const
    {
        if (live_obstacle_samples.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::accumulate(live_obstacle_samples.begin(), live_obstacle_samples.end(), 0.0) / live_obstacle_samples.size() / area;
    }
};

/***************************************************************************************************/
/* Support stat structures */
/***************************************************************************************************/
struct SummaryStats
{
    double median;
    double iqr; // Q3 - Q1
};

inline SummaryStats medianIQR(std::vector<double> v)
{
    if (v.empty())
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    std::sort(v.begin(), v.end());
    auto q = [&](double q) -> double
    {
        double idx = q * (v.size() - 1);
        size_t i = static_cast<size_t>(std::floor(idx));
        double frac = idx - i;
        if (i + 1 < v.size())
            return v[i] * (1.0 - frac) + v[i + 1] * frac;
        return v[i];
    };
    double med = q(0.5);
    double q1 = q(0.25);
    double q3 = q(0.75);
    return {med, q3 - q1};
}

struct Results
{
    // Distributions
    SummaryStats distance_median_iqr;
    SummaryStats ldj_median_iqr;
    SummaryStats detour_ratio_median_iqr; // if baselines provided (NaN if not available)

    // Scalars
    double normalized_collisions; // collided_robots / total_robots_spawned
    double total_in_flow_rate_per_s;
    double total_out_flow_rate_per_s;
    size_t robots_spawned = 0;
    size_t robots_collided = 0;
    size_t total_collision_events = 0; // event count (can exceed collided robots)
    
    // Encounter statistics
    double avg_robot_encounters_per_second = 0.0;     // Average robot encounters per second
    double avg_obstacle_encounters_per_second = 0.0;  // Average obstacle encounters per second

    // Obstacle metrics per type
    std::unordered_map<ObstacleType, double> obstacle_avg_lifetime;
    std::unordered_map<ObstacleType, double> obstacle_avg_live_count;
    std::unordered_map<ObstacleType, double> obstacle_avg_density;
    std::unordered_map<ObstacleType, double> obstacle_crowdedness;
    
    // Total obstacle metrics (sum across all types)
    double total_obstacle_density = 0.0;     // Sum of all obstacle type densities
    double total_obstacle_crowdedness = 0.0; // Sum of all obstacle type crowdedness
};

/***************************************************************************************************/
/* FlowMeter */
/***************************************************************************************************/
// Simple line segment intersection test (2D)
inline bool segSegIntersect(const Eigen::Vector2d &a, const Eigen::Vector2d &b,
                            const Eigen::Vector2d &c, const Eigen::Vector2d &d)
{
    auto cross = [](const Eigen::Vector2d &u, const Eigen::Vector2d &v)
    { return u.x() * v.y() - u.y() * v.x(); };
    Eigen::Vector2d r = b - a;
    Eigen::Vector2d s = d - c;
    double denom = cross(r, s);
    double numer1 = cross((c - a), r);
    double numer2 = cross((c - a), s);
    if (std::abs(denom) < 1e-12)
    { // parallel or collinear
        if (std::abs(numer1) > 1e-12)
            return false; // parallel non-intersecting
        // collinear: check overlap in projections
        auto on1 = [&](const Eigen::Vector2d &p) -> bool
        {
            double minx = std::min(a.x(), b.x()) - 1e-12, maxx = std::max(a.x(), b.x()) + 1e-12;
            double miny = std::min(a.y(), b.y()) - 1e-12, maxy = std::max(a.y(), b.y()) + 1e-12;
            return (p.x() >= minx && p.x() <= maxx && p.y() >= miny && p.y() <= maxy);
        };
        return on1(c) || on1(d) || on1(a) || on1(b);
    }
    double t = cross((c - a), s) / denom;
    double u = cross((c - a), r) / denom;
    return (t >= -1e-12 && t <= 1.0 + 1e-12 && u >= -1e-12 && u <= 1.0 + 1e-12);
}

struct FlowMeter
{
    Eigen::Vector2d L0, L1; // measurement line endpoints
    double crossings = 0.0; // total crossings counted
    double t0 = std::numeric_limits<double>::quiet_NaN();
    double t1 = std::numeric_limits<double>::quiet_NaN();

    FlowMeter() = default;
    FlowMeter(const Eigen::Vector2d &a, const Eigen::Vector2d &b) : L0(a), L1(b) {}

    // Call per robot per tick with its previous and current center positions
    void accumulate(double t_prev, const Eigen::Vector2d &p_prev,
                    double t_curr, const Eigen::Vector2d &p_curr)
    {
        if (std::isnan(t0))
            t0 = t_prev;
        t1 = t_curr;
        if (segSegIntersect(p_prev, p_curr, L0, L1))
            crossings += 1.0;
    }

    // crossings per second (you can scale to per minute in reporting)
    double ratePerSecond() const
    {
        if (std::isnan(t0) || std::isnan(t1) || t1 <= t0)
            return 0.0;
        return crossings / (t1 - t0);
    }
};

/***************************************************************************************************/
/* Main metrics collector class */
/***************************************************************************************************/
class MetricsCollector
{
private:
    double simulation_start_time_;
    double warmup_time_ = 0.0; // Measurements before this time are excluded
    double t_min_alive = 5.0;  // Min time alive (robots) to exclude collisions being marked
    mutable size_t total_collisions_ = 0;
    const double area_eff = 2944.0; // 16 * 100 * 2 - 16^2 (junction_twoway experiment)
    
    // Encounter tracking parameters
    
    std::unordered_map<int, RobotTrack> tracks_;
    std::unordered_map<int, ObstacleLifecycleInfo> obstacle_tracks_;
    std::unordered_map<ObstacleType, ObstacleTypeMetrics> obstacle_type_metrics_;
    double last_obstacle_sample_time_ = std::numeric_limits<double>::quiet_NaN();
    double obstacle_sample_interval_ = 1.0;
    
    bool flow_enabled_ = true;
    FlowMeter flow_in_ns_;
    FlowMeter flow_out_ns_;
    FlowMeter flow_in_we_;
    FlowMeter flow_out_we_;
    FlowMeter flow_in_sn_;
    FlowMeter flow_out_sn_;
    FlowMeter flow_in_ew_;
    FlowMeter flow_out_ew_;
    
public:
    const double interaction_radius_buffer_ = 0.5;      // [m] Buffer distance for interaction radius
    const double interaction_cooldown_threshold_ = 0.7; // [s] Minimum time between interaction events
    
    MetricsCollector(double simulation_start_time, FlowMeter flow_in_ns = {}, FlowMeter flow_out_ns = {}, FlowMeter flow_in_we = {}, FlowMeter flow_out_we = {}, FlowMeter flow_in_sn = {}, FlowMeter flow_out_sn = {}, FlowMeter flow_in_ew = {}, FlowMeter flow_out_ew = {})
        : simulation_start_time_(simulation_start_time), flow_in_ns_(flow_in_ns), flow_out_ns_(flow_out_ns), flow_in_we_(flow_in_we), flow_out_we_(flow_out_we), flow_in_sn_(flow_in_sn), flow_out_sn_(flow_out_sn), flow_in_ew_(flow_in_ew), flow_out_ew_(flow_out_ew) {}

    // Getters/ Setters
    void setWarmupTime(double warmup_time)
    {
        warmup_time_ = warmup_time;
    }

    void enableFlow(bool on) { flow_enabled_ = on; }

    void setObstacleAreas(const std::unordered_map<ObstacleType, double> areas) {
        for (const auto& [type, area] : areas) {
            obstacle_type_metrics_[type].setArea(area);
        }
    }

    RobotTrack& getTrack(int id)
    {
        auto it = tracks_.find(id);
        if (it == tracks_.end())
            return tracks_[id]; // creates new track
        return it->second;
    }

    void setBaselinePathLength(int robot_id, double L_min)
    {
        tracks_[robot_id].baseline_path_length = L_min;
    }


    bool isInWarmupPeriod(double current_time) const
    {
        return (current_time - simulation_start_time_) < warmup_time_;
    }

    std::pair<double, double> getAverageEncounterRates() const
    {
        double sum_robot_rates = 0.0;
        double sum_obstacle_rates = 0.0;
        size_t valid = 0;

        for (const auto& [robot_id, track] : tracks_) {
            if (track.samples.empty()) continue;

            double t_last = std::isnan(track.t_end) ? track.samples.back().t : track.t_end;
            double duration = t_last - track.t_start;
            if (duration <= 0) continue;

            sum_robot_rates    += track.robot_encounters / duration;
            sum_obstacle_rates += track.obstacle_encounters / duration;
            ++valid;
        }

        if (valid == 0) {
            return {0.0, 0.0};
        }

        return {sum_robot_rates / valid, sum_obstacle_rates / valid};
    }

    // === ROBOT TRACKING API ===
    void addSample(int robot_id, double t, Eigen::Ref<const Eigen::Vector2d> pos)
    {
        // Skip if in warm-up period
        if (isInWarmupPeriod(t))
        {
            return;
        }

        auto &tr = tracks_[robot_id];
        if (tr.samples.empty())
        {
            tr.t_start = t;
        }
        else
        {
            const Sample &prev = tr.samples.back();

            // Update flow rate - classify based on position and direction for bidirectional junction
            if (flow_enabled_)
            {
                Eigen::Vector2d motion = pos - prev.p;
                double abs_dx = std::abs(motion.x());
                double abs_dy = std::abs(motion.y());
                
                // Determine which channel the robot is in based on position
                bool in_horizontal_channel = std::abs(pos.y()) < std::abs(pos.x());
                bool in_vertical_channel = std::abs(pos.x()) < std::abs(pos.y());
                
                if (abs_dy > abs_dx)
                { // Vertical movement
                    if (motion.y() > 0) // moving down
                    {
                        if (pos.y() < 0)
                            flow_in_ns_.accumulate(prev.t, prev.p, t, pos);
                        else
                            flow_out_ns_.accumulate(prev.t, prev.p, t, pos);
                    }
                    else // moving up
                    {
                        if (pos.y() > 0)
                            flow_in_sn_.accumulate(prev.t, prev.p, t, pos);
                        else
                            flow_out_sn_.accumulate(prev.t, prev.p, t, pos);
                    }
                }
                else
                { // Horizontal movement
                    if (motion.x() > 0) // moving right
                    {
                        if (pos.x() < 0) 
                            flow_in_we_.accumulate(prev.t, prev.p, t, pos);
                        else
                            flow_out_we_.accumulate(prev.t, prev.p, t, pos);
                    }
                    else // moving left
                    {
                        if (pos.x() > 0)
                            flow_in_ew_.accumulate(prev.t, prev.p, t, pos);
                        else
                            flow_out_ew_.accumulate(prev.t, prev.p, t, pos);
                    }
                }
            }
        }
        tr.samples.push_back({t, pos});
    }


    void markCollision(int robot_id, double t, const Eigen::Vector2d &collision_pos)
    {
        // Skip if in warm-up period
        if (isInWarmupPeriod(t))
        {
            return;
        }

        auto it = tracks_.find(robot_id);
        if (it == tracks_.end())
            return;
        auto &tr = it->second;
        if (std::abs(t - tr.t_start) < t_min_alive)
            return;
        
        // Track first collision time and position if not already set
        if (std::isnan(tr.t_first_collision))
        {
            tr.t_first_collision = t;
            tr.first_collision_pos = collision_pos;
        }
        
        it->second.collided = true;
        it->second.t_end = std::isnan(it->second.t_end) ? t : it->second.t_end;
        total_collisions_ += 1; // count events; normalized rate uses collided robots fraction
    }

    // === OBSTACLE TRACKING API ===
    void addObstacleSpawn(int obstacle_id, ObstacleType type, double spawn_time)
    {
        obstacle_tracks_[obstacle_id] = ObstacleLifecycleInfo{type, spawn_time};
    }

    void addObstacleDespawn(int obstacle_id, double despawn_time)
    {
        auto it = obstacle_tracks_.find(obstacle_id);
        if (it != obstacle_tracks_.end())
        {
            auto &info = it->second;
            info.despawn_time = despawn_time;
            info.completed = true;

            // Only record lifetime if both spawn and despawn were after warm-up period
            if (!isInWarmupPeriod(info.spawn_time) && !isInWarmupPeriod(despawn_time))
            {
                double lifetime = despawn_time - info.spawn_time;
                obstacle_type_metrics_[info.type].lifetimes.push_back(lifetime);
            }

            obstacle_tracks_.erase(it);
        }
    }

    // Update obstacle sampling (call this regularly to sample live obstacle counts)
    void updateObstacleSampling(double current_time, const std::unordered_map<ObstacleType, int> &live_counts)
    {
        // Skip if in warm-up period
        if (isInWarmupPeriod(current_time))
        {
            return;
        }

        // Sample at regular intervals
        if (std::isnan(last_obstacle_sample_time_) ||
            (current_time - last_obstacle_sample_time_) >= obstacle_sample_interval_)
        {

            for (const auto &[type, count] : live_counts)
            {
                obstacle_type_metrics_[type].live_obstacle_samples.push_back(static_cast<double>(count));
                obstacle_type_metrics_[type].total_sample_time += obstacle_sample_interval_;
            }

            last_obstacle_sample_time_ = current_time;
        }
    }

    // === Compute per-robot primitives ===
    static double pathLength(const std::vector<Sample> &S)
    {
        double L = 0.0;
        for (size_t i = 1; i < S.size(); ++i)
            L += (S[i].p - S[i - 1].p).norm();
        return L;
    }

    // Log Dimensionless Jerk (LDJ); Discrete approximation of: LDJ = ln( (T^5 / L^2) * ∫ ||j(t)||^2 dt )
    static double logDimensionlessJerk(const std::vector<Sample> &S)
    {
        if (S.size() < 5)
            return std::numeric_limits<double>::quiet_NaN();
        // uniform dt assumed (typical in sims); estimate dt as median of differences
        std::vector<double> dts;
        dts.reserve(S.size() - 1);
        for (size_t i = 1; i < S.size(); ++i)
            dts.push_back(S[i].t - S[i - 1].t);
        std::sort(dts.begin(), dts.end());
        double dt = dts[dts.size() / 2];
        if (dt <= 0.0)
            return std::numeric_limits<double>::quiet_NaN();

        // central differences for v, a; then finite diff for j
        const size_t N = S.size();
        std::vector<Eigen::Vector2d> v(N), a(N), j(N);
        // velocity
        v.front().setZero();
        v.back().setZero();
        for (size_t i = 1; i + 1 < N; i++)
            v[i] = (S[i + 1].p - S[i - 1].p) / (2.0 * dt);
        // acceleration
        a.front().setZero();
        a.back().setZero();
        for (size_t i = 1; i + 1 < N; i++)
            a[i] = (v[i + 1] - v[i - 1]) / (2.0 * dt);
        // jerk (use central diff where possible)
        j.front().setZero();
        j.back().setZero();
        for (size_t i = 1; i + 1 < N; i++)
            j[i] = (a[i + 1] - a[i - 1]) / (2.0 * dt);

        double integral_j2 = 0.0; // ∑ ||j||^2 * dt
        for (size_t i = 1; i < N; i++)
        {
            double j2 = j[i].squaredNorm();
            integral_j2 += j2 * dt;
        }

        double T = S.back().t - S.front().t;
        double L = pathLength(S);
        if (T <= 0.0 || L <= 1e-9)
            return std::numeric_limits<double>::quiet_NaN();

        double ldj = std::log((std::pow(T, 5) / (L * L)) * integral_j2);
        return ldj;
    }

    // === Compute final results ===
    Results computeResults() const
    {
        std::vector<double> distances, ldjs, detours;
        distances.reserve(tracks_.size());
        ldjs.reserve(tracks_.size());
        detours.reserve(tracks_.size());

        size_t collided = 0;

        for (const auto &kv : tracks_)
        {
            const RobotTrack &tr = kv.second;
            if (tr.samples.size() < 2)
                continue;

            double L = pathLength(tr.samples);
            distances.push_back(L);

            double ldj = logDimensionlessJerk(tr.samples);
            if (std::isfinite(ldj))
                ldjs.push_back(ldj);

            if (tr.baseline_path_length > 1e-6)
            {
                detours.push_back(L / tr.baseline_path_length); // >= 1.0
            }

            if (tr.collided)
                collided++;
        }

        Results R;
        R.distance_median_iqr = medianIQR(distances);
        R.ldj_median_iqr = medianIQR(ldjs);
        R.detour_ratio_median_iqr = medianIQR(detours); // Will be NaN if detours is empty

        size_t spawned = tracks_.size();
        R.robots_spawned = spawned;
        R.robots_collided = collided;
        R.total_collision_events = total_collisions_;
        R.normalized_collisions = (spawned > 0) ? static_cast<double>(collided) / static_cast<double>(spawned) : 0.0;

        R.total_in_flow_rate_per_s = flow_in_ns_.ratePerSecond() + flow_in_we_.ratePerSecond() + flow_in_sn_.ratePerSecond() + flow_in_ew_.ratePerSecond();
        R.total_out_flow_rate_per_s = flow_out_ns_.ratePerSecond() + flow_out_we_.ratePerSecond() + flow_out_sn_.ratePerSecond() + flow_out_ew_.ratePerSecond();

        // Calculate encounter statistics
        auto avg_encounter_rates = getAverageEncounterRates();
        R.avg_robot_encounters_per_second = avg_encounter_rates.first;
        R.avg_obstacle_encounters_per_second = avg_encounter_rates.second;

        for (const auto &[type, metrics] : obstacle_type_metrics_)
        {
            R.obstacle_avg_lifetime[type] = metrics.getAverageLifetime();
            R.obstacle_avg_live_count[type] = metrics.getAverageLiveObstacles();
            R.obstacle_avg_density[type] = metrics.getDensity(area_eff);
            R.obstacle_crowdedness[type] = metrics.area < 1e-9 ? 0.0 : R.obstacle_avg_live_count[type] * metrics.area / area_eff;
            
            // Sum up total obstacle metrics
            if (!std::isnan(R.obstacle_avg_density[type]))
                R.total_obstacle_density += R.obstacle_avg_density[type];
            if (!std::isnan(R.obstacle_crowdedness[type]))
                R.total_obstacle_crowdedness += R.obstacle_crowdedness[type];
        }

        return R;
    }

    // === CSV Export Functions ===
    static std::string generateTimestampedFilename(const std::string &prefix, const std::string &extension)
    {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::stringstream ss;
        ss << prefix << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
           << "." << extension;
        return ss.str();
    }

    // Export individual samples
    void exportSamplesToCSV(const std::string &experiment_name = "") const
    {
        // Create results directory if it doesn't exist
        std::filesystem::create_directories("results");

        // Generate filename
        std::string prefix = experiment_name.empty() ? "samples" : experiment_name + "_samples";
        std::string filename = "results/" + generateTimestampedFilename(prefix, "csv");

        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }

        // Write CSV header
        file << "robot_id,time_s,pos_x_m,pos_y_m,collided,sample_index\n";

        // Write data for each robot
        for (const auto &[robot_id, track] : tracks_)
        {
            for (size_t i = 0; i < track.samples.size(); ++i)
            {
                const Sample &sample = track.samples[i];
                file << robot_id << ","
                     << std::fixed << std::setprecision(6) << sample.t << ","
                     << std::fixed << std::setprecision(6) << sample.p.x() << ","
                     << std::fixed << std::setprecision(6) << sample.p.y() << ","
                     << (track.collided ? 1 : 0) << ","
                     << i << "\n";
            }
        }

        file.close();
    }

    // Export summary stats
    void exportSummaryToCSV(const std::string &experiment_name = "") const
    {
        // Create results directory if it doesn't exist
        std::filesystem::create_directories("results");

        // Generate filename
        std::string prefix = experiment_name.empty() ? "summary" : experiment_name + "_summary";
        std::string filename = "results/" + generateTimestampedFilename(prefix, "csv");

        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        
        std::cout << "\nExporting metrics to " << filename << "...\n";

        // Write CSV header
        file << "robot_id,collided,t_start_s,t_end_s,t_first_collision_s,first_collision_pos_x_m,first_collision_pos_y_m,duration_s,"
             << "path_length_m,baseline_path_length_m,detour_ratio,ldj,"
             << "robot_encounters,obstacle_encounters,num_samples\n";

        // Write data for each robot
        for (const auto &[robot_id, track] : tracks_)
        {
            if (track.samples.empty())
                continue;

            // Calculate metrics for this robot
            double path_length = pathLength(track.samples);
            double ldj = logDimensionlessJerk(track.samples);
            double duration = (std::isnan(track.t_end) ? track.samples.back().t : track.t_end) - track.t_start;
            double detour_ratio = (track.baseline_path_length > 1e-6) ? path_length / track.baseline_path_length : std::numeric_limits<double>::quiet_NaN();

            file << robot_id << ","
                 << (track.collided ? 1 : 0) << ","
                 << std::fixed << std::setprecision(6) << track.t_start << ","
                 << std::fixed << std::setprecision(6) << (std::isnan(track.t_end) ? track.samples.back().t : track.t_end) << ","
                 << std::fixed << std::setprecision(6) << track.t_first_collision << ","
                 << std::fixed << std::setprecision(6) << track.first_collision_pos.x() << ","
                 << std::fixed << std::setprecision(6) << track.first_collision_pos.y() << ","
                 << std::fixed << std::setprecision(6) << duration << ","
                 << std::fixed << std::setprecision(6) << path_length << ","
                 << std::fixed << std::setprecision(6) << track.baseline_path_length << ","
                 << std::fixed << std::setprecision(6) << detour_ratio << ","
                 << std::fixed << std::setprecision(6) << ldj << ","
                 << track.robot_encounters << ","
                 << track.obstacle_encounters << ","
                 << track.samples.size() << "\n";
        }

        file.close();
        std::cout << "CSV export completed.\n";
    }

    // Export aggregate experiment-level metrics to CSV. This creates a single row with overall experiment statistics
    void exportExperimentMetricsToCSV(const std::string &experiment_name = "") const
    {
        // Create results directory if it doesn't exist
        std::filesystem::create_directories("results");

        // Generate filename
        std::string prefix = experiment_name.empty() ? "experiment" : experiment_name + "_experiment";
        std::string filename = "results/" + generateTimestampedFilename(prefix, "csv");

        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }

        // Compute results
        Results R = computeResults();

        // Write CSV header
        file << "experiment_name,timestamp,robots_spawned,robots_collided,"
             << "total_collision_events,normalized_collisions,total_in_flow_rate_per_s,total_out_flow_rate_per_s,"
             << "avg_robot_encounters_per_s,avg_obstacle_encounters_per_s,"
             << "total_obstacle_density,total_obstacle_crowdedness,"
             << "distance_median_m,distance_iqr_m,"
             << "detour_ratio_median,detour_ratio_iqr,ldj_median,ldj_iqr\n";

        // Generate timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp_ss;
        timestamp_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

        // Write data
        file << (experiment_name.empty() ? "unnamed" : experiment_name) << ","
             << timestamp_ss.str() << ","
             << R.robots_spawned << ","
             << R.robots_collided << ","
             << R.total_collision_events << ","
             << std::fixed << std::setprecision(6) << R.normalized_collisions << ","
             << std::fixed << std::setprecision(6) << R.total_in_flow_rate_per_s << ","
             << std::fixed << std::setprecision(6) << R.total_out_flow_rate_per_s << ","
             << std::fixed << std::setprecision(6) << R.avg_robot_encounters_per_second << ","
             << std::fixed << std::setprecision(6) << R.avg_obstacle_encounters_per_second << ","
             << std::fixed << std::setprecision(6) << R.total_obstacle_density << ","
             << std::fixed << std::setprecision(6) << R.total_obstacle_crowdedness << ","
             << std::fixed << std::setprecision(6) << R.distance_median_iqr.median << ","
             << std::fixed << std::setprecision(6) << R.distance_median_iqr.iqr << ","
             << std::fixed << std::setprecision(6) << R.detour_ratio_median_iqr.median << ","
             << std::fixed << std::setprecision(6) << R.detour_ratio_median_iqr.iqr << ","
             << std::fixed << std::setprecision(6) << R.ldj_median_iqr.median << ","
             << std::fixed << std::setprecision(6) << R.ldj_median_iqr.iqr << "\n";

        file.close();
    }

    // Wrapper to export all at once
    void exportAllCSV(const std::string &experiment_name = "") const
    {
        std::cout << "\nExporting metrics to CSV files...\n";
        // exportSamplesToCSV(experiment_name);
        exportSummaryToCSV(experiment_name);
        exportExperimentMetricsToCSV(experiment_name);
        std::cout << "CSV export completed.\n";
    }
};

// Result print helper
inline void printResults(const Results &R)
{
    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << "                    METRICS SUMMARY\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Robot Statistics
    std::cout << "ROBOT STATISTICS:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::setw(25) << std::left << "  Robots Spawned:"
              << std::setw(10) << std::right << R.robots_spawned << "\n";
    std::cout << std::setw(25) << std::left << "  Robots Collided:"
              << std::setw(10) << std::right << R.robots_collided
              << " (" << std::fixed << std::setprecision(1)
              << (R.robots_spawned > 0 ? 100.0 * R.robots_collided / R.robots_spawned : 0.0)
              << "%)\n";
    std::cout << std::setw(25) << std::left << "  Collision Events:"
              << std::setw(10) << std::right << R.total_collision_events << "\n";
    std::cout << std::setw(25) << std::left << "  Normalized Collisions:"
              << std::setw(10) << std::right << std::fixed << std::setprecision(3)
              << R.normalized_collisions << "\n\n";

    // Flow Metrics
    std::cout << "FLOW METRICS:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::setw(25) << std::left << "  Total In-Flow Rate:"
              << std::setw(10) << std::right << std::fixed << std::setprecision(2)
              << R.total_in_flow_rate_per_s << " /s";
    std::cout << " (" << std::fixed << std::setprecision(1)
              << R.total_in_flow_rate_per_s * 60.0 << " /min)\n";
    std::cout << std::setw(25) << std::left << "  Total Out-Flow Rate:"
              << std::setw(10) << std::right << std::fixed << std::setprecision(2)
              << R.total_out_flow_rate_per_s << " /s";
    std::cout << " (" << std::fixed << std::setprecision(1)
              << R.total_out_flow_rate_per_s * 60.0 << " /min)\n\n";

    // Path Metrics (with NaN handling)
    std::cout << "PATH METRICS:\n";
    std::cout << std::string(40, '-') << "\n";

    auto printStat = [](const std::string &name, double median, double iqr,
                        const std::string &unit = "", int precision = 2)
    {
        std::cout << std::setw(25) << std::left << ("  " + name + ":");
        if (std::isnan(median))
        {
            std::cout << std::setw(10) << std::right << "N/A" << "\n";
        }
        else
        {
            std::cout << std::setw(10) << std::right << std::fixed
                      << std::setprecision(precision) << median << unit;
            if (!std::isnan(iqr) && iqr > 0)
            {
                std::cout << " (IQR: " << std::fixed << std::setprecision(precision)
                          << iqr << unit << ")";
            }
            std::cout << "\n";
        }
    };

    printStat("Distance (median)", R.distance_median_iqr.median,
              R.distance_median_iqr.iqr, " m");
    printStat("Detour Ratio (median)", R.detour_ratio_median_iqr.median,
              R.detour_ratio_median_iqr.iqr, "x", 3);

    // Smoothness Metrics
    std::cout << "\nSMOOTHNESS METRICS:\n";
    std::cout << std::string(40, '-') << "\n";
    printStat("Log Dim. Jerk (median)", R.ldj_median_iqr.median,
              R.ldj_median_iqr.iqr, "", 3);
    
    // Encounter Metrics
    std::cout << "\nENCOUNTER METRICS:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::setw(25) << std::left << "  Avg Robot Encounters:"
              << std::setw(10) << std::right << std::fixed << std::setprecision(2)
              << R.avg_robot_encounters_per_second << " /s";
    std::cout << " (" << std::fixed << std::setprecision(1)
              << R.avg_robot_encounters_per_second * 60.0 << " /min)\n";
    std::cout << std::setw(25) << std::left << "  Avg Obstacle Encounters:"
              << std::setw(10) << std::right << std::fixed << std::setprecision(2)
              << R.avg_obstacle_encounters_per_second << " /s";
    std::cout << " (" << std::fixed << std::setprecision(1)
              << R.avg_obstacle_encounters_per_second * 60.0 << " /min)\n";
    
    // Obstacle Metrics
    if (!R.obstacle_avg_lifetime.empty() || !R.obstacle_avg_live_count.empty())
    {
        std::cout << "\nOBSTACLE METRICS:\n";
        std::cout << std::string(40, '-') << "\n";

        // Helper to convert ObstacleType to string
        auto obstacleTypeToString = [](ObstacleType type) -> std::string
        {
            switch (type)
            {
            case ObstacleType::BUS:
                return "Bus";
            case ObstacleType::VAN:
                return "Van";
            case ObstacleType::PEDESTRIAN:
                return "Pedestrian";
            case ObstacleType::CUBE:
                return "Cube";
            default:
                return "Unknown";
            }
        };

        for (const auto &[type, avg_lifetime] : R.obstacle_avg_lifetime)
        {
            std::string type_name = obstacleTypeToString(type);
            printStat(type_name + " Avg Lifetime", avg_lifetime, std::numeric_limits<double>::quiet_NaN(), " s");
        }

        for (const auto &[type, avg_live] : R.obstacle_avg_live_count)
        {
            std::string type_name = obstacleTypeToString(type);
            printStat(type_name + " Avg Live Count", avg_live, std::numeric_limits<double>::quiet_NaN());
        }
        
        for (const auto &[type, avg_density] : R.obstacle_avg_density)
        {
            std::string type_name = obstacleTypeToString(type);
            printStat(type_name + " Density", avg_density, std::numeric_limits<double>::quiet_NaN(), "/m²", 6);
        }
        
        for (const auto &[type, crowdedness] : R.obstacle_crowdedness)
        {
            std::string type_name = obstacleTypeToString(type);
            printStat(type_name + " Crowdedness", crowdedness, std::numeric_limits<double>::quiet_NaN(), "", 6);
        }
        printStat("Total obstacle density", R.total_obstacle_density, 0.0);
        printStat("Total obstacle crowdedness", R.total_obstacle_crowdedness, 0.0);
    }

    std::cout << "\n"
              << std::string(60, '=') << "\n\n";
}
