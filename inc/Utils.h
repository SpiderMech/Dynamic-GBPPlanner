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
// RANDOM NUMBER GENERATORS
/***************************************************************************************************************/
inline std::mt19937 rng(static_cast<std::mt19937::result_type>(globals.SEED));

int random_int(int lower, int upper);

float random_float(float lower, float upper);


