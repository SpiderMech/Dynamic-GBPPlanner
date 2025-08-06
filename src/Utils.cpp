/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <Utils.h>
#include <Globals.h>
extern Globals globals;

/*******************************************************************************************/
// Non-linear function for determining the timesteps at which variables in the planned path are placed.
/*******************************************************************************************/
std::vector<int> getVariableTimesteps(int lookahead_horizon, int lookahead_multiple)
{
    // For a lookahead_multiple of 3, variables are spaced at timesteps:
    // Timesteps
    // 0,    1, 2, 3,    5, 7, 9,    12, 15, 18, ...
    //
    // eg. variables are in groups of size lookahead_multiple.
    // the spacing within a group increases by one each time (1 for the first group, 2 for the second ...)
    // Seems convoluted, but the reasoning was:
    //      the first variable should always be at 1 timestep from the current state (0).
    //      the first few variables should be close together in time
    //      the variables should all be at integer timesteps, but the spacing should sort of increase exponentially.
    std::vector<int> var_list{};
    int N = 1 + int(0.5 * (-1 + sqrt(1 + 8 * (float)lookahead_horizon / (float)lookahead_multiple)));

    for (int i = 0; i < lookahead_multiple * (N + 1); i++)
    {
        int section = int(i / lookahead_multiple);
        int f = (i - section * lookahead_multiple + lookahead_multiple / 2. * section) * (section + 1);
        if (f >= lookahead_horizon)
        {
            var_list.push_back(lookahead_horizon);
            break;
        }
        var_list.push_back(f);
    }

    return var_list;
}

/*******************************************************************************************/
// Lineaer function for determining the timesteps at which variables in the planned path are placed.
/*******************************************************************************************/
std::vector<int> getLinearVariableTimesteps(int lookahead_horizon)
{
    std::vector<int> var_list{};
    for (int i = 0; i < lookahead_horizon; ++i)
        var_list.push_back(i);
    return var_list;
}

/**************************************************************************************/
// Draw the Time and FPS, as well as the help box
/**************************************************************************************/
void draw_info(uint32_t time_cnt){
    static std::map<MODES_LIST, const char*> MODES_MAP = {
        {SimNone,""},
        {Timestep,"Timestep"},
        {Iterate,"Synchronous Iteration"},
        {Help, "Help"},
    };

    int info_box_h = 100, info_box_w = 180;;
    int info_box_x = 10, info_box_y = globals.SCREEN_SZ-40-info_box_h;

    DrawRectangle(0, globals.SCREEN_SZ-30, globals.SCREEN_SZ, 30, RAYWHITE);
    DrawText(TextFormat("Time: %.1f s", time_cnt*globals.TIMESTEP), 5, globals.SCREEN_SZ-20, 20, DARKGREEN);
    DrawText(TextFormat("Press H for Help"), globals.SCREEN_SZ/2-80, globals.SCREEN_SZ-20, 20, DARKGREEN);
    DrawFPS(globals.SCREEN_SZ-80, globals.SCREEN_SZ-20);

    if (globals.SIM_MODE==Help){
        int info_box_h = 500, info_box_w = 500;
        int info_box_x = globals.SCREEN_SZ/2 - info_box_w/2, info_box_y = globals.SCREEN_SZ/2 - info_box_h/2;
        DrawRectangle( info_box_x, info_box_y, info_box_w, info_box_h, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines( info_box_x, info_box_y, info_box_w, info_box_h, BLACK);

        int offset = 40;
        std::vector<std::string> texts{
            "Esc : \t\t\t\t Exit Simulation",
            "H : \t\t\t\t\t\t Close Help",
            "SPACE : \t Camera Transition",
            "ENTER : \t Run/Pause Simulation",
            "P : \t\t\t\t\t\t Toggle Planned paths",
            "R : \t\t\t\t\t\t Toggle Connected Robots",
            "W : \t\t\t\t\t\t Toggle Waypoints",
            "F : \t\t\t\t\t\t Toggle Factors",
            ""         ,
            "Mouse Wheel Scroll : Zoom",
            "Mouse Wheel Drag : Pan",
            "Mouse Wheel Drag + SHIFT : Rotate",
        };

        for (int t=0; t<texts.size(); t++){
            DrawText(texts[t].c_str(), info_box_x + 10, info_box_y + (t+1)*offset, 20, BLACK);
        }
    }
}

/***************************************************************************************************************/
// RANDOM NUMBER GENERATORS
/***************************************************************************************************************/
int random_int(int lower, int upper) {
    return std::uniform_int_distribution<int>(lower, upper)(rng);
}

float random_float(float lower, float upper){
    return std::uniform_real_distribution<float>(lower, upper)(rng);
}

