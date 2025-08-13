# Task Scheduling System for Robot Motion Planning

## Overview

This task scheduling system provides a clean, flexible interface for managing robot tasks in both factory and road/traffic environments. It extends your existing waypoint system with a higher-level abstraction that makes it easy to define and manage complex robot behaviors.

## Key Features

### 1. **Task Types**
- **Stationary Tasks**: Robot stops at a location for a specified duration (e.g., bus stops, charging stations)
- **Delivery Tasks**: Two-waypoint tasks with pickup and delivery locations
- **Route Tasks**: Multi-stop routes with optional looping (e.g., bus routes, patrol paths)
- **Custom Tasks**: Extensible base class for defining new task types

### 2. **Priority System**
Tasks can be assigned priorities (CRITICAL, HIGH, NORMAL, LOW) to ensure important tasks are executed first.

### 3. **Location Management**
Locations are abstracted with:
- Position (x, y coordinates)
- Semantic names (e.g., "Bus Stop A", "Warehouse 1")
- Type classification (e.g., "bus_stop", "warehouse", "charging_station")
- Arrival radius for task completion detection

### 4. **Task Queue Management**
Each robot has its own task queue that:
- Automatically sorts tasks by priority
- Tracks task status (PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED)
- Maintains task history for analytics
- Supports dynamic task addition and cancellation

## Architecture

```
TaskScheduler (Main Interface)
    ├── TaskQueue (Per Robot)
    │   ├── Pending Tasks (Priority Sorted)
    │   ├── Current Task
    │   └── Completed Tasks (History)
    │
    ├── Location Registry
    │   └── Known Locations (Named positions)
    │
    └── Task Types
        ├── StationaryTask
        ├── DeliveryTask
        ├── RouteTask
        └── CustomTask (User-defined)
```

## Integration with Existing System

The task scheduler seamlessly integrates with your existing waypoint system:

1. **Waypoint Generation**: Each task type knows how to convert itself to the 5D waypoint format (x, y, vx, vy, pause_time)
2. **Pause Timer Compatibility**: Uses the existing 5th dimension for stop durations
3. **Robot Creation**: Works with your existing Robot class constructor

## Usage Examples

### Factory Scenario

```cpp
// Define factory locations
scheduler.addLocation("warehouse", Location(-20, -20, "Main Warehouse", "storage"));
scheduler.addLocation("assembly", Location(0, 0, "Assembly Line", "workstation"));
scheduler.addLocation("shipping", Location(20, 20, "Shipping Dock", "shipping"));

// Schedule a delivery task
scheduler.scheduleDelivery(robot_id, 
    scheduler.getLocation("warehouse").value(),
    scheduler.getLocation("assembly").value(),
    10.0,  // 10 seconds for pickup
    5.0,   // 5 seconds for delivery
    TaskPriority::HIGH);

// Schedule a maintenance stop
Location charging_station(−30, 0, "Charging Station", "charging");
scheduler.scheduleStop(robot_id, charging_station, 60.0, "charging");
```

### Road/Traffic Scenario

```cpp
// Define bus route
std::vector<Location> bus_stops = {
    Location(0, -40, "North Station", "bus_stop"),
    Location(0, 0, "Central Station", "bus_stop"),
    Location(0, 40, "South Station", "bus_stop")
};
std::vector<double> stop_times = {20.0, 15.0, 20.0};

// Schedule the bus route (with looping)
scheduler.scheduleRoute(bus_id, bus_stops, stop_times, true);

// Schedule a delivery motorcycle
Location restaurant(-20, -20, "Pizza Place", "restaurant");
Location customer(15, 25, "Customer House", "residential");
scheduler.scheduleDelivery(motorcycle_id, restaurant, customer, 30.0, 10.0);
```

## Scenario Templates

### 1. Factory Environment
- **Delivery Robots**: Transport materials between storage, workstations, and shipping
- **Patrol Robots**: Continuous movement for monitoring or material handling
- **Maintenance**: Scheduled charging or maintenance stops

### 2. Road/Traffic Environment
- **Buses**: Fixed routes with timed stops at stations
- **Delivery Vehicles**: Food delivery, package delivery with pickup/dropoff times
- **Emergency Vehicles**: High-priority response to incidents

### 3. Mixed Scenarios
- Combine different robot types in the same environment
- Priority-based task execution for emergency situations
- Dynamic task generation based on events

## Advanced Features

### Dynamic Task Addition
```cpp
// Add tasks during simulation
void onNewOrder(TaskScheduler& scheduler, int robot_id) {
    Location pickup = findNearestRestaurant();
    Location delivery = getCustomerLocation();
    scheduler.scheduleDelivery(robot_id, pickup, delivery, 
                              20.0, 10.0, TaskPriority::HIGH);
}
```

### Task Monitoring
```cpp
// Check task status
auto current_task = scheduler.getCurrentTask(robot_id);
if (current_task && current_task->type == TaskType::DELIVERY) {
    auto delivery_task = std::static_pointer_cast<DeliveryTask>(current_task);
    std::cout << "Delivering " << delivery_task->cargo_type << std::endl;
}
```

### Custom Task Types
```cpp
class InspectionTask : public Task {
    std::vector<Location> inspection_points;
    double inspection_time;
    
public:
    std::deque<Eigen::VectorXd> toWaypoints() const override {
        // Convert inspection points to waypoints
    }
};
```

## Benefits

1. **Clean Abstraction**: Separates task logic from low-level waypoint management
2. **Flexibility**: Easy to add new task types and scenarios
3. **Scalability**: Each robot manages its own task queue independently
4. **Maintainability**: Clear separation of concerns and modular design
5. **Reusability**: Same interface works for different environments (factory, road, etc.)

## Future Extensions

Potential enhancements could include:
- Task dependencies (Task B can only start after Task A completes)
- Time windows for tasks (must arrive between certain times)
- Resource constraints (limited cargo capacity, battery life)
- Multi-robot coordination (synchronized tasks)
- Task optimization (route planning, scheduling algorithms)
- Real-time replanning based on obstacles or delays

## Integration Steps

1. Include the header: `#include <TaskScheduler.h>`
2. Create a TaskScheduler instance in your Simulator
3. Define locations for your scenario
4. Schedule tasks for robots
5. Update robot waypoints when tasks complete
6. (Optional) Add UI elements to display task status

The system is designed to be non-invasive and work alongside your existing codebase while providing a powerful abstraction for task management.
