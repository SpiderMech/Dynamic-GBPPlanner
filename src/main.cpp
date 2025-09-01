/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)

// Define all parameters in the appropriate config file (default: config/config.json)
/**************************************************************************************/
#define RLIGHTS_IMPLEMENTATION // needed to be defined once for the lights shader
#include <iostream>
#include <vector>
#include <string>
#include <Utils.h>

#include <DArgs.h>

#include <Globals.h>
#include <Simulator.h>
#include <Metrics.hpp>

Globals globals;

int main(int argc, char *argv[]){
    
    DArgs::DArgs dargs(argc, argv);
    if (globals.parse_global_args(dargs)) return EXIT_FAILURE;  
    
    std::vector<int> seeds = {globals.SEED , globals.SEED + 1, globals.SEED + 2, globals.SEED + 3, globals.SEED + 4};
    
    for (int seed_idx = 0; seed_idx < seeds.size(); seed_idx++) {
        int current_seed = seeds[seed_idx];
        reseed_rng(current_seed);
        
        std::string experiment_name = globals.EXPERIMENT_NAME + "_" + std::to_string(current_seed);
        printf("Running experiment %s [SEED: %d]\n", globals.EXPERIMENT_NAME.c_str(), current_seed);
        
        Simulator* sim = new Simulator();
        globals.RUN = true;
        sim->setupEnvironment();
        
        // Main simulation loop with timestep limit
        while (globals.RUN) {
            sim->eventHandler();    // Capture keypresses or mouse events             
            sim->createOrDeleteObstacles();
            sim->createOrDeleteRobots();    
            sim->timestep();
            sim->draw();
        }
        
        if (globals.EVAL) {
            auto R = sim->metrics->computeResults();
            printResults(R);
            sim->metrics->exportSummaryToCSV(experiment_name);
        }
        
        delete sim;
    }
    
    print("All experiments completed!");
    return 0;
}    
