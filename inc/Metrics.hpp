// metrics.hpp
#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <optional>

struct Sample {
    double t;
    Eigen::Vector2d p;
};

struct RobotTrack {
    std::vector<Sample> samples; /* In time order */
    bool arrived = false;
    bool collided = false;
    double t_start = std::numeric_limits<double>::quiet_NaN(); /* First sample */
    double t_end   = std::numeric_limits<double>::quiet_NaN(); /* Arrival or last sample */
    double baseline_path_length = -1.0; /* Used for detour ratio */
};

struct SummaryStats {
    double median;
    double iqr; // Q3 - Q1
};

inline SummaryStats medianIQR(std::vector<double> v) {
    if (v.empty()) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    std::sort(v.begin(), v.end());
    auto q = [&](double q)->double {
        double idx = q * (v.size() - 1);
        size_t i = static_cast<size_t>(std::floor(idx));
        double frac = idx - i;
        if (i + 1 < v.size()) return v[i]*(1.0-frac) + v[i+1]*frac;
        return v[i];
    };
    double med = q(0.5);
    double q1  = q(0.25);
    double q3  = q(0.75);
    return {med, q3 - q1};
}

// Simple line segment intersection test (2D)
inline bool segSegIntersect(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                            const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
    auto cross = [](const Eigen::Vector2d& u, const Eigen::Vector2d& v){ return u.x()*v.y() - u.y()*v.x(); };
    Eigen::Vector2d r = b - a;
    Eigen::Vector2d s = d - c;
    double denom = cross(r, s);
    double numer1 = cross((c - a), r);
    double numer2 = cross((c - a), s);
    if (std::abs(denom) < 1e-12) { // parallel or collinear
        if (std::abs(numer1) > 1e-12) return false; // parallel non-intersecting
        // collinear: check overlap in projections
        auto on1 = [&](const Eigen::Vector2d& p)->bool {
            double minx = std::min(a.x(), b.x()) - 1e-12, maxx = std::max(a.x(), b.x()) + 1e-12;
            double miny = std::min(a.y(), b.y()) - 1e-12, maxy = std::max(a.y(), b.y()) + 1e-12;
            return (p.x() >= minx && p.x() <= maxx && p.y() >= miny && p.y() <= maxy);
        };
        return on1(c) || on1(d) || on1(a) || on1(b);
    }
    double t = cross((c - a), s) / denom;
    double u = cross((c - a), r) / denom;
    return (t >= -1e-12 && t <= 1.0+1e-12 && u >= -1e-12 && u <= 1.0+1e-12);
}

// Flow meter through a measurement line
struct FlowMeter {
    Eigen::Vector2d L0, L1;   // measurement line endpoints
    double crossings = 0.0;   // total crossings counted
    double t0 = std::numeric_limits<double>::quiet_NaN();
    double t1 = std::numeric_limits<double>::quiet_NaN();

    FlowMeter() = default;
    FlowMeter(const Eigen::Vector2d& a, const Eigen::Vector2d& b) : L0(a), L1(b) {}

    // Call per robot per tick with its previous and current center positions
    inline void accumulate(double t_prev, const Eigen::Vector2d& p_prev,
                           double t_curr, const Eigen::Vector2d& p_curr) {
        if (std::isnan(t0)) t0 = t_prev;
        t1 = t_curr;
        if (segSegIntersect(p_prev, p_curr, L0, L1)) crossings += 1.0;
    }

    // crossings per second (you can scale to per minute in reporting)
    inline double ratePerSecond() const {
        if (std::isnan(t0) || std::isnan(t1) || t1 <= t0) return 0.0;
        return crossings / (t1 - t0);
    }
};

class MetricsCollector {
public:
    explicit MetricsCollector(FlowMeter fm = {}) : flow_(fm) {}

    // --- Sampling API ---
    inline void addSample(int robot_id, double t, const Eigen::Vector2d& pos) {
        auto& tr = tracks_[robot_id];
        if (tr.samples.empty()) tr.t_start = t;
        else {
            // update flow using the last segment
            const Sample& prev = tr.samples.back();
            if (flow_enabled_) flow_.accumulate(prev.t, prev.p, t, pos);
        }
        tr.samples.push_back({t, pos});
    }

    inline void setBaselinePathLength(int robot_id, double L_min) {
        tracks_[robot_id].baseline_path_length = L_min;
    }

    inline void markArrival(int robot_id, double t) {
        auto it = tracks_.find(robot_id);
        if (it == tracks_.end()) return;
        it->second.arrived = true;
        it->second.t_end = t;
    }

    inline void markCollision(int robot_id, double t) {
        auto it = tracks_.find(robot_id);
        if (it == tracks_.end()) return;
        it->second.collided = true;
        it->second.t_end = std::isnan(it->second.t_end) ? t : it->second.t_end;
        total_collisions_ += 1; // count events; normalized rate uses collided robots fraction
    }

    inline void enableFlow(bool on) { flow_enabled_ = on; }

    // --- Compute per-robot primitives ---
    static double pathLength(const std::vector<Sample>& S) {
        double L = 0.0;
        for (size_t i=1; i<S.size(); ++i) L += (S[i].p - S[i-1].p).norm();
        return L;
    }

    // Log Dimensionless Jerk (LDJ)
    // Discrete approximation of: LDJ = ln( (T^5 / L^2) * ∫ ||j(t)||^2 dt )
    // (Common in motion smoothness literature; here generalized to 2D)
    static double logDimensionlessJerk(const std::vector<Sample>& S) {
        if (S.size() < 5) return std::numeric_limits<double>::quiet_NaN();
        // uniform dt assumed (typical in sims); estimate dt as median of differences
        std::vector<double> dts; dts.reserve(S.size()-1);
        for (size_t i=1;i<S.size();++i) dts.push_back(S[i].t - S[i-1].t);
        std::sort(dts.begin(), dts.end());
        double dt = dts[dts.size()/2];
        if (dt <= 0.0) return std::numeric_limits<double>::quiet_NaN();

        // central differences for v, a; then finite diff for j
        const size_t N = S.size();
        std::vector<Eigen::Vector2d> v(N), a(N), j(N);
        // velocity
        v.front().setZero();
        v.back().setZero();
        for (size_t i=1;i+1<N;i++) v[i] = (S[i+1].p - S[i-1].p) / (2.0*dt);
        // acceleration
        a.front().setZero();
        a.back().setZero();
        for (size_t i=1;i+1<N;i++) a[i] = (v[i+1] - v[i-1]) / (2.0*dt);
        // jerk (use central diff where possible)
        j.front().setZero();
        j.back().setZero();
        for (size_t i=1;i+1<N;i++) j[i] = (a[i+1] - a[i-1]) / (2.0*dt);

        double integral_j2 = 0.0; // ∑ ||j||^2 * dt
        for (size_t i=1;i<N;i++) {
            double j2 = j[i].squaredNorm();
            integral_j2 += j2 * dt;
        }

        double T = S.back().t - S.front().t;
        double L = pathLength(S);
        if (T <= 0.0 || L <= 1e-9) return std::numeric_limits<double>::quiet_NaN();

        double ldj = std::log( (std::pow(T,5) / (L*L)) * integral_j2 );
        return ldj;
    }

    // // --- Final aggregation ---
    // struct Results {
    //     // Distributions
    //     SummaryStats distance_median_iqr;
    //     SummaryStats makespan_median_iqr;
    //     SummaryStats ldj_median_iqr;
    //     std::optional<SummaryStats> detour_ratio_median_iqr; // if baselines provided

    //     // Scalars
    //     double normalized_collisions; // collided_robots / total_robots_spawned
    //     double exit_flow_rate_per_s;  // crossings per second
    //     size_t robots_spawned = 0;
    //     size_t robots_arrived = 0;
    //     size_t robots_collided = 0;
    //     size_t total_collision_events = 0; // event count (can exceed collided robots)
    // };

    // Results computeResults() const {
    //     std::vector<double> distances, makespans, ldjs, detours;
    //     distances.reserve(tracks_.size());
    //     makespans.reserve(tracks_.size());
    //     ldjs.reserve(tracks_.size());
    //     detours.reserve(tracks_.size());

    //     size_t arrived = 0, collided = 0;

    //     for (const auto& kv : tracks_) {
    //         const RobotTrack& tr = kv.second;
    //         if (tr.samples.size() < 2) continue;

    //         double L = pathLength(tr.samples);
    //         distances.push_back(L);

    //         double T = (std::isnan(tr.t_end) ? tr.samples.back().t : tr.t_end) - tr.samples.front().t;
    //         if (T > 0.0) makespans.push_back(T);

    //         double ldj = logDimensionlessJerk(tr.samples);
    //         if (std::isfinite(ldj)) ldjs.push_back(ldj);

    //         if (tr.baseline_path_length.value() > 1e-6) {
    //             detours.push_back(L / tr.baseline_path_length); // >= 1.0
    //         }

    //         if (tr.arrived) arrived++;
    //         if (tr.collided) collided++;
    //     }

    //     Results R;
    //     R.distance_median_iqr = medianIQR(distances);
    //     R.makespan_median_iqr = medianIQR(makespans);
    //     R.ldj_median_iqr      = medianIQR(ldjs);
    //     if (!detours.empty()) R.detour_ratio_median_iqr = medianIQR(detours);

    //     size_t spawned = tracks_.size();
    //     R.robots_spawned = spawned;
    //     R.robots_arrived = arrived;
    //     R.robots_collided = collided;
    //     R.total_collision_events = total_collisions_;
    //     R.normalized_collisions = (spawned > 0) ? static_cast<double>(collided) / static_cast<double>(spawned) : 0.0;

    //     R.exit_flow_rate_per_s = flow_.ratePerSecond();
    //     return R;
    // }

private:
    std::unordered_map<int, RobotTrack> tracks_;
    mutable size_t total_collisions_ = 0;
    FlowMeter flow_;
    bool flow_enabled_ = true;
};
