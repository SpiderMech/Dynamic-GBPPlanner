/**************************************************************************************/
// Example implementation of the Task Scheduler for different scenarios
// Shows how to use the task scheduling system for factory and road settings
/**************************************************************************************/
#include <TaskScheduler.h>
#include <Simulator.h>
#include <Robot.h>
#include <iostream>

// Example 1: Factory scenario with delivery robots
void setupFactoryScenario(TaskScheduler& scheduler, Simulator* sim) {
    std::cout << "Setting up factory scenario with delivery robots..." << std::endl;
    
    // Define factory locations
    scheduler.addLocation("storage_1", Location(-25, -15, "Storage Area 1", "storage"));
    scheduler.addLocation("storage_2", Location(-25, 15, "Storage Area 2", "storage"));
    scheduler.addLocation("workstation_1", Location(0, -20, "Workstation 1", "workstation"));
    scheduler.addLocation("workstation_2", Location(0, 0, "Workstation 2", "workstation"));
    scheduler.addLocation("workstation_3", Location(0, 20, "Workstation 3", "workstation"));
    scheduler.addLocation("quality_check", Location(20, 0, "Quality Check", "inspection"));
    scheduler.addLocation("packaging", Location(30, -10, "Packaging Station", "packaging"));
    scheduler.addLocation("shipping_dock", Location(30, 10, "Shipping Dock", "shipping"));
    
    // Create delivery robots
    for (int rid = 0; rid < 3; ++rid) {
        // Each robot gets different tasks
        if (rid == 0) {
            // Robot 0: Parts delivery from storage to workstations
            auto storage = scheduler.getLocation("storage_1").value();
            auto workstation1 = scheduler.getLocation("workstation_1").value();
            auto workstation2 = scheduler.getLocation("workstation_2").value();
            
            // Schedule multiple deliveries
            scheduler.scheduleDelivery(rid, storage, workstation1, 10.0, 5.0, TaskPriority::HIGH);
            scheduler.scheduleDelivery(rid, storage, workstation2, 10.0, 5.0, TaskPriority::NORMAL);
            
            // Add a charging stop after deliveries
            auto charging = Location(-30, 0, "Charging Station", "charging");
            scheduler.scheduleStop(rid, charging, 60.0, "charging", TaskPriority::LOW);
        }
        else if (rid == 1) {
            // Robot 1: Quality check and packaging route
            auto workstation3 = scheduler.getLocation("workstation_3").value();
            auto quality = scheduler.getLocation("quality_check").value();
            auto packaging = scheduler.getLocation("packaging").value();
            
            scheduler.scheduleDelivery(rid, workstation3, quality, 5.0, 10.0, TaskPriority::HIGH);
            scheduler.scheduleDelivery(rid, quality, packaging, 5.0, 8.0, TaskPriority::NORMAL);
        }
        else if (rid == 2) {
            // Robot 2: Patrol route for continuous material handling
            std::vector<Location> patrol_stops = {
                scheduler.getLocation("storage_1").value(),
                scheduler.getLocation("workstation_1").value(),
                scheduler.getLocation("workstation_2").value(),
                scheduler.getLocation("storage_2").value(),
                scheduler.getLocation("workstation_3").value()
            };
            std::vector<double> stop_durations = {5.0, 3.0, 3.0, 5.0, 3.0};
            
            scheduler.scheduleRoute(rid, patrol_stops, stop_durations, true, TaskPriority::LOW);
        }
        
        // Get initial waypoints for the robot
        auto waypoints = scheduler.getNextTaskWaypoints(rid);
        if (!waypoints.empty()) {
            // Create robot with task waypoints
            Color robot_color = ColorFromHSV(rid * 120.0f, 1.0f, 0.75f);
            auto robot = std::make_shared<Robot>(sim, rid, waypoints, 
                                                  globals.ROBOT_RADIUS, robot_color);
            sim->robots_[rid] = robot;
        }
    }
}

// Example 2: Road/Traffic scenario with buses and delivery motorcycles
void setupRoadScenario(TaskScheduler& scheduler, Simulator* sim) {
    std::cout << "Setting up road scenario with buses and delivery vehicles..." << std::endl;
    
    // Define road network locations
    // Bus stops along main routes
    scheduler.addLocation("bus_stop_north", Location(0, -40, "North Station", "bus_stop"));
    scheduler.addLocation("bus_stop_central", Location(0, 0, "Central Station", "bus_stop"));
    scheduler.addLocation("bus_stop_south", Location(0, 40, "South Station", "bus_stop"));
    scheduler.addLocation("bus_stop_west", Location(-40, 0, "West Station", "bus_stop"));
    scheduler.addLocation("bus_stop_east", Location(40, 0, "East Station", "bus_stop"));
    
    // Delivery locations
    scheduler.addLocation("restaurant_1", Location(-20, -20, "Pizza Place", "restaurant"));
    scheduler.addLocation("restaurant_2", Location(20, -20, "Burger Joint", "restaurant"));
    scheduler.addLocation("customer_1", Location(-15, 25, "Customer House 1", "residential"));
    scheduler.addLocation("customer_2", Location(25, 15, "Customer House 2", "residential"));
    scheduler.addLocation("customer_3", Location(-25, -10, "Customer House 3", "residential"));
    
    // Bus depot
    scheduler.addLocation("bus_depot", Location(0, -50, "Bus Depot", "depot"));
    
    // Create buses (robots 0-1)
    for (int bus_id = 0; bus_id < 2; ++bus_id) {
        std::vector<Location> bus_route;
        std::vector<double> stop_times;
        
        if (bus_id == 0) {
            // Bus Line 1: North-South route
            bus_route = {
                scheduler.getLocation("bus_depot").value(),
                scheduler.getLocation("bus_stop_north").value(),
                scheduler.getLocation("bus_stop_central").value(),
                scheduler.getLocation("bus_stop_south").value(),
                scheduler.getLocation("bus_stop_central").value(),
                scheduler.getLocation("bus_stop_north").value()
            };
            stop_times = {0.0, 20.0, 15.0, 20.0, 15.0, 20.0}; // Stop times at each station
        } else {
            // Bus Line 2: East-West route
            bus_route = {
                scheduler.getLocation("bus_depot").value(),
                scheduler.getLocation("bus_stop_west").value(),
                scheduler.getLocation("bus_stop_central").value(),
                scheduler.getLocation("bus_stop_east").value(),
                scheduler.getLocation("bus_stop_central").value(),
                scheduler.getLocation("bus_stop_west").value()
            };
            stop_times = {0.0, 25.0, 15.0, 25.0, 15.0, 25.0};
        }
        
        // Schedule the bus route (loops back to depot)
        scheduler.scheduleRoute(bus_id, bus_route, stop_times, true, TaskPriority::NORMAL);
        
        // Create bus robot
        auto waypoints = scheduler.getNextTaskWaypoints(bus_id);
        if (!waypoints.empty()) {
            Color bus_color = (bus_id == 0) ? BLUE : GREEN;
            auto bus = std::make_shared<Robot>(sim, bus_id, waypoints, 
                                               globals.ROBOT_RADIUS * 1.5f, bus_color);
            sim->robots_[bus_id] = bus;
        }
    }
    
    // Create delivery motorcycles (robots 2-4)
    for (int delivery_id = 2; delivery_id < 5; ++delivery_id) {
        int order_num = delivery_id - 2;
        
        if (order_num == 0) {
            // Delivery 1: Pizza delivery
            auto restaurant = scheduler.getLocation("restaurant_1").value();
            auto customer = scheduler.getLocation("customer_1").value();
            scheduler.scheduleDelivery(delivery_id, restaurant, customer, 
                                      30.0, 10.0, TaskPriority::HIGH); // 30s pickup, 10s dropoff
        } else if (order_num == 1) {
            // Delivery 2: Burger delivery
            auto restaurant = scheduler.getLocation("restaurant_2").value();
            auto customer = scheduler.getLocation("customer_2").value();
            scheduler.scheduleDelivery(delivery_id, restaurant, customer, 
                                      20.0, 10.0, TaskPriority::HIGH);
        } else {
            // Delivery 3: Multiple deliveries
            auto restaurant1 = scheduler.getLocation("restaurant_1").value();
            auto customer3 = scheduler.getLocation("customer_3").value();
            auto restaurant2 = scheduler.getLocation("restaurant_2").value();
            auto customer1 = scheduler.getLocation("customer_1").value();
            
            scheduler.scheduleDelivery(delivery_id, restaurant1, customer3, 
                                      25.0, 10.0, TaskPriority::NORMAL);
            scheduler.scheduleDelivery(delivery_id, restaurant2, customer1, 
                                      20.0, 10.0, TaskPriority::NORMAL);
        }
        
        // Create delivery motorcycle robot
        auto waypoints = scheduler.getNextTaskWaypoints(delivery_id);
        if (!waypoints.empty()) {
            Color delivery_color = RED;
            auto motorcycle = std::make_shared<Robot>(sim, delivery_id, waypoints, 
                                                      globals.ROBOT_RADIUS * 0.7f, delivery_color);
            sim->robots_[delivery_id] = motorcycle;
        }
    }
}

// Example 3: Mixed scenario with emergency vehicles
void setupEmergencyScenario(TaskScheduler& scheduler, Simulator* sim) {
    std::cout << "Setting up emergency response scenario..." << std::endl;
    
    // Define emergency locations
    scheduler.addLocation("hospital", Location(0, 0, "Central Hospital", "hospital"));
    scheduler.addLocation("fire_station", Location(-30, -30, "Fire Station", "emergency"));
    scheduler.addLocation("incident_1", Location(25, 15, "Car Accident", "incident"));
    scheduler.addLocation("incident_2", Location(-20, 25, "Fire Emergency", "incident"));
    
    // Ambulance (robot 0) - High priority task
    auto hospital = scheduler.getLocation("hospital").value();
    auto incident = scheduler.getLocation("incident_1").value();
    
    // Emergency response: go to incident, stabilize (30s), return to hospital
    scheduler.scheduleDelivery(0, hospital, incident, 0.0, 30.0, TaskPriority::CRITICAL);
    scheduler.scheduleDelivery(0, incident, hospital, 10.0, 0.0, TaskPriority::CRITICAL);
    
    // Fire truck (robot 1) - High priority task
    auto fire_station = scheduler.getLocation("fire_station").value();
    auto fire_incident = scheduler.getLocation("incident_2").value();
    
    scheduler.scheduleStop(1, fire_incident, 120.0, "firefighting", TaskPriority::CRITICAL);
    scheduler.scheduleDelivery(1, fire_incident, fire_station, 0.0, 0.0, TaskPriority::HIGH);
    
    // Create emergency vehicles
    for (int rid = 0; rid < 2; ++rid) {
        auto waypoints = scheduler.getNextTaskWaypoints(rid);
        if (!waypoints.empty()) {
            Color vehicle_color = (rid == 0) ? WHITE : RED;
            float vehicle_size = globals.ROBOT_RADIUS * 1.2f;
            auto emergency_vehicle = std::make_shared<Robot>(sim, rid, waypoints, 
                                                             vehicle_size, vehicle_color);
            sim->robots_[rid] = emergency_vehicle;
        }
    }
}

// Main function to update robot tasks during simulation
void updateRobotTasks(TaskScheduler& scheduler, Simulator* sim) {
    for (auto& [rid, robot] : sim->robots_) {
        // Check if robot has completed its current waypoints
        if (robot->waypoints_.empty() && !robot->task_active_) {
            // Mark current task as completed
            scheduler.completeTask(rid);
            
            // Get next task if available
            if (scheduler.hasPendingTasks(rid)) {
                auto new_waypoints = scheduler.getNextTaskWaypoints(rid);
                if (!new_waypoints.empty()) {
                    robot->waypoints_ = new_waypoints;
                    std::cout << "Robot " << rid << " starting new task" << std::endl;
                }
            } else {
                std::cout << "Robot " << rid << " has no more tasks" << std::endl;
            }
        }
        
        // Optional: Display current task status
        auto current_task = scheduler.getCurrentTask(rid);
        if (current_task) {
            // Could display task info in the simulator UI
            // For example: current_task->id, current_task->type, etc.
        }
    }
}

// Integration with existing Simulator class
// This would be added to your Simulator.cpp file
void integrateTaskScheduler(Simulator* sim) {
    static TaskScheduler task_scheduler;
    static bool initialized = false;
    
    if (!initialized) {
        // Choose scenario based on formation type
        if (globals.FORMATION == "factory") {
            setupFactoryScenario(task_scheduler, sim);
        } else if (globals.FORMATION == "road_traffic") {
            setupRoadScenario(task_scheduler, sim);
        } else if (globals.FORMATION == "emergency") {
            setupEmergencyScenario(task_scheduler, sim);
        }
        initialized = true;
    }
    
    // Update tasks each timestep
    updateRobotTasks(task_scheduler, sim);
}

// Example of programmatic task creation
void addDynamicTask(TaskScheduler& scheduler, int robot_id) {
    // Example: Add a new delivery task dynamically during simulation
    Location pickup(random_float(-30, 30), random_float(-30, 30), "Dynamic Pickup", "dynamic");
    Location delivery(random_float(-30, 30), random_float(-30, 30), "Dynamic Delivery", "dynamic");
    
    scheduler.scheduleDelivery(robot_id, pickup, delivery, 5.0, 5.0, TaskPriority::NORMAL);
    
    std::cout << "Added dynamic delivery task for robot " << robot_id << std::endl;
}
