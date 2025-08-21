/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <gbp/GBPCore.h>
#include <iostream>
#include <chrono>
#include <random>

/*******************************************************************************/
// Robot type enumeration for different vehicle models
/*******************************************************************************/
enum class RobotType {
    SPHERE,  // Default sphere representation (original implementation)
    CAR,
    BUS
};

/*******************************************************************************/
// Dynamic Obstacle type enumeration for different vehicle models
/*******************************************************************************/
enum class ObstacleType {
    CUBE,
    BUS,
    VAN,
    PEDESTRIAN
};

/*******************************************************************************/
// Easy print statement, just use like in python: print(a, b, "blah blah");
// If printing Eigen::Vector or Matrix but after accessing incline functions, you will need to call .eval() first:
// Eigen::VectorXd myVector{{0,1,2}}; print(myVector({0,1}).eval())
/*******************************************************************************/
template <typename T> void print(const T& t) {
    std::cout << t << std::endl;
}
template <typename First, typename... Rest> void print(const First& first, const Rest&... rest) {
    std::cout << first << ", ";
    print(rest...); // recursive call using pack expansion syntax
}

/*******************************************************************************/
// This function draws the FPS and time on the screen, as well as the help screen
/*******************************************************************************/
void draw_info( uint32_t time_cnt);

/*******************************************************************************/
// This function allows you to time events in real-time
// Usage: 
// auto start = std::chrono::steady_clock::now();
// std::cout << "Elapsed(us): " << since(start).count() << std::endl;
/*******************************************************************************/
template <
    class result_t   = std::chrono::microseconds,
    class clock_t    = std::chrono::high_resolution_clock,
    class duration_t = std::chrono::microseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

/*******************************************************************************************/
// Non-linear function for determining the timesteps at which variables in the planned path are placed.
/*******************************************************************************************/
std::vector<int> getVariableTimesteps(int lookahead_horizon, int lookahead_multiple);

/*******************************************************************************************/
// Lineaer function for determining the timesteps at which variables in the planned path are placed.
/*******************************************************************************************/
std::vector<int> getLinearVariableTimesteps(int lookahead_horizon);

/***************************************************************************************************************/
// ANGLE UTILITIES
/***************************************************************************************************************/
// Wrap angle to range [-pi, pi]
inline double wrapAngle(double angle) {
    // Use fmod to wrap to [-2π, 2π] first, then adjust to [-π, π]
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0)
        angle += 2.0 * M_PI;
    return angle - M_PI;
}

/***************************************************************************************************************/
// RANDOM NUMBER GENERATORS
/***************************************************************************************************************/
inline std::mt19937 rng(static_cast<std::mt19937::result_type>(globals.SEED));

int random_int(int lower, int upper);

float random_float(float lower, float upper);

template <typename T>
T random_choice_one(const std::vector<T>& v) {
    std::uniform_int_distribution<size_t> dist(0, v.size() - 1);
    return v[dist(rng)];
}

/***************************************************************************************************************/
// POISSON SPAWNER
/***************************************************************************************************************/
struct PoissonSpawner {
    double mean;
    double hmin;
    double next_spawn;
    std::exponential_distribution<double> exp;
    std::string name;
    bool dbg = false;

    PoissonSpawner(double M, double Hmin, const std::string& n = "", bool verbose = false)
        : mean(M), hmin(Hmin), exp(1.0 / std::max(1e-6, M - Hmin)), next_spawn(0.0), name(n), dbg(verbose) {}

    void set_mean(double M) { mean = M; exp = std::exponential_distribution<double>(1.0 / std::max(1e-6, M - hmin)); }
    void set_hmin(double H) { hmin = H; exp = std::exponential_distribution<double>(1.0 / std::max(1e-6, mean - H)); }
    void set_rate(double lambda) { set_mean(1.0 / lambda); } // λ in spawns/s
    double rate() const { return 1.0 / mean; }

    void schedule_from(double now) {
        next_spawn = now + hmin + exp(rng);
        if (dbg) printf("%s next spawn: %.3f\n", name.c_str(), next_spawn);
    }

    bool try_spawn(double now) {
        if (now >= next_spawn) {
            next_spawn += hmin + exp(rng); // schedule next
            if (dbg) printf("%s next spawn: %.3f\n", name.c_str(), next_spawn);
            return true; // spawn now
        }
        return false;
    }
};


