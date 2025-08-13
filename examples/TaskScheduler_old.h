/**************************************************************************************/
// Task Scheduler Interface for Robot Motion Planning
// Supports both delivery tasks (pick-up and drop-off) and stationary tasks (stop at location)
// Suitable for factory and road/traffic scenarios
/**************************************************************************************/
#pragma once
#include <memory>
#include <vector>
#include <deque>
#include <string>
#include <Eigen/Core>
#include <optional>
#include <chrono>

// Forward declarations
class Robot;

/**************************************************************************************/
// Enum definitions for task types and priorities
/**************************************************************************************/
enum class TaskType {
    STATIONARY,     // Task that requires staying at a location (e.g., bus stop, charging station)
    DELIVERY,       // Task that involves pick-up and drop-off
    PATROL,         // Continuous movement between waypoints
    CUSTOM          // User-defined task type
};

enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

enum class TaskStatus {
    PENDING,        // Task not yet started
    IN_PROGRESS,    // Task currently being executed
    COMPLETED,      // Task finished successfully
    FAILED,         // Task failed
    CANCELLED       // Task was cancelled
};

/**************************************************************************************/
// Location structure that can represent both factory positions and road network points
/**************************************************************************************/
struct Location {
    Eigen::Vector2d position;           // x, y coordinates
    std::string name;                   // Optional name (e.g., "Bus Stop A", "Assembly Station 3")
    std::string type;                   // Location type (e.g., "bus_stop", "warehouse", "intersection")
    double arrival_radius = 1.0;        // Radius for considering arrival at location
    
    Location(double x, double y, const std::string& n = "", const std::string& t = "") 
        : position(x, y), name(n), type(t) {}
    
    Location(const Eigen::Vector2d& pos, const std::string& n = "", const std::string& t = "") 
        : position(pos), name(n), type(t) {}
};

/**************************************************************************************/
// Base Task class that all specific tasks inherit from
/**************************************************************************************/
class Task {
public:
    std::string id;                     // Unique task identifier
    TaskType type;                      // Type of task
    TaskPriority priority;              // Task priority
    TaskStatus status;                  // Current status
    
    std::chrono::steady_clock::time_point created_time;
    std::optional<std::chrono::steady_clock::time_point> start_time;
    std::optional<std::chrono::steady_clock::time_point> end_time;
    
    double max_velocity = -1;           // Max velocity for this task (-1 for default)
    std::optional<double> deadline_seconds;  // Optional deadline for task completion
    
    Task(const std::string& task_id, TaskType t, TaskPriority p = TaskPriority::NORMAL)
        : id(task_id), type(t), priority(p), status(TaskStatus::PENDING) {
        created_time = std::chrono::steady_clock::now();
    }
    
    virtual ~Task() = default;
    
    // Convert task to waypoints format (5D: x, y, vx, vy, pause_time)
    virtual std::deque<Eigen::VectorXd> toWaypoints() const = 0;
    
    // Get estimated completion time in seconds
    virtual double getEstimatedDuration() const = 0;
    
    // Validate if task can be executed
    virtual bool isValid() const = 0;
};

/**************************************************************************************/
// Stationary Task - Robot stays at a location for a duration
/**************************************************************************************/
class StationaryTask : public Task {
public:
    Location location;                  // Where to stop
    double duration_seconds;            // How long to stay
    std::string activity;               // What the robot is doing (e.g., "passenger_boarding", "charging")
    
    StationaryTask(const std::string& id, const Location& loc, double duration,
                   const std::string& act = "", TaskPriority p = TaskPriority::NORMAL)
        : Task(id, TaskType::STATIONARY, p), location(loc), 
          duration_seconds(duration), activity(act) {}
    
    std::deque<Eigen::VectorXd> toWaypoints() const override {
        std::deque<Eigen::VectorXd> waypoints;
        Eigen::VectorXd wp(5);
        wp << location.position.x(), location.position.y(), 0.0, 0.0, duration_seconds;
        waypoints.push_back(wp);
        return waypoints;
    }
    
    double getEstimatedDuration() const override {
        return duration_seconds;
    }
    
    bool isValid() const override {
        return duration_seconds > 0;
    }
};

/**************************************************************************************/
// Delivery Task - Pick up from one location and deliver to another
/**************************************************************************************/
class DeliveryTask : public Task {
public:
    Location pickup_location;           // Where to pick up
    Location delivery_location;         // Where to deliver
    double pickup_duration;             // Time needed at pickup (loading)
    double delivery_duration;           // Time needed at delivery (unloading)
    std::string cargo_type;             // What is being delivered
    double cargo_weight = 0;            // Optional weight/size constraints
    
    DeliveryTask(const std::string& id, const Location& pickup, const Location& delivery,
                 double pickup_time = 5.0, double delivery_time = 5.0,
                 const std::string& cargo = "", TaskPriority p = TaskPriority::NORMAL)
        : Task(id, TaskType::DELIVERY, p), 
          pickup_location(pickup), delivery_location(delivery),
          pickup_duration(pickup_time), delivery_duration(delivery_time),
          cargo_type(cargo) {}
    
    std::deque<Eigen::VectorXd> toWaypoints() const override {
        std::deque<Eigen::VectorXd> waypoints;
        
        // Waypoint for pickup location with pause
        Eigen::VectorXd pickup_wp(5);
        pickup_wp << pickup_location.position.x(), pickup_location.position.y(), 
                     0.0, 0.0, pickup_duration;
        waypoints.push_back(pickup_wp);
        
        // Waypoint for delivery location with pause
        Eigen::VectorXd delivery_wp(5);
        delivery_wp << delivery_location.position.x(), delivery_location.position.y(),
                       0.0, 0.0, delivery_duration;
        waypoints.push_back(delivery_wp);
        
        return waypoints;
    }
    
    double getEstimatedDuration() const override {
        // Simple estimate - would need path planning for accurate duration
        double travel_dist = (delivery_location.position - pickup_location.position).norm();
        double travel_time = travel_dist / 2.0;  // Assume 2 m/s average speed
        return pickup_duration + travel_time + delivery_duration;
    }
    
    bool isValid() const override {
        return pickup_duration >= 0 && delivery_duration >= 0 &&
               pickup_location.position != delivery_location.position;
    }
};

/**************************************************************************************/
// Route Task - Follow a predefined route with multiple stops (e.g., bus route)
/**************************************************************************************/
class RouteTask : public Task {
public:
    std::vector<Location> stops;        // Ordered list of stops
    std::vector<double> stop_durations; // Duration at each stop
    bool loop_route;                    // Whether to loop back to start
    int repetitions;                    // Number of times to repeat route (-1 for infinite)
    
    RouteTask(const std::string& id, const std::vector<Location>& route_stops,
              const std::vector<double>& durations, bool loop = false, int reps = 1,
              TaskPriority p = TaskPriority::NORMAL)
        : Task(id, TaskType::PATROL, p), stops(route_stops), 
          stop_durations(durations), loop_route(loop), repetitions(reps) {}
    
    std::deque<Eigen::VectorXd> toWaypoints() const override {
        std::deque<Eigen::VectorXd> waypoints;
        
        for (size_t i = 0; i < stops.size(); ++i) {
            Eigen::VectorXd wp(5);
            double duration = (i < stop_durations.size()) ? stop_durations[i] : 0.0;
            wp << stops[i].position.x(), stops[i].position.y(), 0.0, 0.0, duration;
            waypoints.push_back(wp);
        }
        
        // Add return to start if looping
        if (loop_route && !stops.empty()) {
            Eigen::VectorXd return_wp(5);
            return_wp << stops[0].position.x(), stops[0].position.y(), 0.0, 0.0, 0.0;
            waypoints.push_back(return_wp);
        }
        
        return waypoints;
    }
    
    double getEstimatedDuration() const override {
        double total_duration = 0;
        for (double d : stop_durations) {
            total_duration += d;
        }
        // Add travel time estimate
        for (size_t i = 1; i < stops.size(); ++i) {
            double dist = (stops[i].position - stops[i-1].position).norm();
            total_duration += dist / 2.0;  // Assume 2 m/s
        }
        return total_duration * std::max(1, repetitions);
    }
    
    bool isValid() const override {
        return !stops.empty() && 
               (stop_durations.empty() || stop_durations.size() == stops.size());
    }
};

/**************************************************************************************/
// Task Queue - Manages task scheduling for a robot
/**************************************************************************************/
class TaskQueue {
private:
    std::deque<std::shared_ptr<Task>> pending_tasks;
    std::shared_ptr<Task> current_task;
    std::vector<std::shared_ptr<Task>> completed_tasks;
    size_t max_history = 100;  // Maximum completed tasks to keep
    
public:
    // Add a task to the queue
    void addTask(std::shared_ptr<Task> task) {
        // Insert based on priority
        auto it = pending_tasks.begin();
        while (it != pending_tasks.end() && 
               (*it)->priority >= task->priority) {
            ++it;
        }
        pending_tasks.insert(it, task);
    }
    
    // Get next task and mark as current
    std::shared_ptr<Task> getNextTask() {
        if (pending_tasks.empty()) {
            return nullptr;
        }
        
        current_task = pending_tasks.front();
        pending_tasks.pop_front();
        current_task->status = TaskStatus::IN_PROGRESS;
        current_task->start_time = std::chrono::steady_clock::now();
        return current_task;
    }
    
    // Mark current task as completed
    void completeCurrentTask() {
        if (current_task) {
            current_task->status = TaskStatus::COMPLETED;
            current_task->end_time = std::chrono::steady_clock::now();
            completed_tasks.push_back(current_task);
            
            // Maintain history size
            if (completed_tasks.size() > max_history) {
                completed_tasks.erase(completed_tasks.begin());
            }
            
            current_task = nullptr;
        }
    }
    
    // Cancel a pending task
    bool cancelTask(const std::string& task_id) {
        auto it = std::remove_if(pending_tasks.begin(), pending_tasks.end(),
            [&task_id](const std::shared_ptr<Task>& t) {
                if (t->id == task_id) {
                    t->status = TaskStatus::CANCELLED;
                    return true;
                }
                return false;
            });
        
        bool removed = (it != pending_tasks.end());
        pending_tasks.erase(it, pending_tasks.end());
        return removed;
    }
    
    // Get current task
    std::shared_ptr<Task> getCurrentTask() const {
        return current_task;
    }
    
    // Get all pending tasks
    std::deque<std::shared_ptr<Task>> getPendingTasks() const {
        return pending_tasks;
    }
    
    // Check if queue is empty
    bool isEmpty() const {
        return pending_tasks.empty() && !current_task;
    }
    
    // Get queue size
    size_t size() const {
        return pending_tasks.size() + (current_task ? 1 : 0);
    }
};

/**************************************************************************************/
// Task Scheduler - Main interface for creating and managing tasks
/**************************************************************************************/
class TaskScheduler {
private:
    std::map<int, TaskQueue> robot_queues;  // Task queue per robot
    int next_task_id = 1;
    
    // Predefined locations for common scenarios
    std::map<std::string, Location> known_locations;
    
public:
    TaskScheduler() {
        initializeCommonLocations();
    }
    
    // Initialize common locations for factory/road scenarios
    void initializeCommonLocations() {
        // Example factory locations
        known_locations["warehouse_a"] = Location(Eigen::Vector2d(-20, -20), "Warehouse A", "warehouse");
        known_locations["assembly_1"] = Location(Eigen::Vector2d(0, -10), "Assembly Station 1", "assembly");
        known_locations["shipping"] = Location(Eigen::Vector2d(20, 20), "Shipping Dock", "shipping");
        known_locations["charging_1"] = Location(Eigen::Vector2d(-10, 10), "Charging Station 1", "charging");
        
        // Example road/traffic locations
        known_locations["bus_stop_1"] = Location(Eigen::Vector2d(-15, 0), "Bus Stop 1", "bus_stop");
        known_locations["bus_stop_2"] = Location(Eigen::Vector2d(0, 15), "Bus Stop 2", "bus_stop");
        known_locations["bus_stop_3"] = Location(Eigen::Vector2d(15, 0), "Bus Stop 3", "bus_stop");
        known_locations["depot"] = Location(Eigen::Vector2d(0, -30), "Bus Depot", "depot");
    }
    
    // Add a custom location
    void addLocation(const std::string& key, const Location& location) {
        known_locations[key] = location;
    }
    
    // Get a known location
    std::optional<Location> getLocation(const std::string& key) {
        auto it = known_locations.find(key);
        if (it != known_locations.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    // Generate unique task ID
    std::string generateTaskId(const std::string& prefix = "TASK") {
        return prefix + "_" + std::to_string(next_task_id++);
    }
    
    // Schedule a delivery task for a robot
    void scheduleDelivery(int robot_id, const Location& pickup, const Location& delivery,
                          double pickup_time = 5.0, double delivery_time = 5.0,
                          TaskPriority priority = TaskPriority::NORMAL) {
        auto task = std::make_shared<DeliveryTask>(
            generateTaskId("DELIVERY"), pickup, delivery, 
            pickup_time, delivery_time, "", priority
        );
        robot_queues[robot_id].addTask(task);
    }
    
    // Schedule a stationary task for a robot
    void scheduleStop(int robot_id, const Location& location, double duration,
                      const std::string& activity = "",
                      TaskPriority priority = TaskPriority::NORMAL) {
        auto task = std::make_shared<StationaryTask>(
            generateTaskId("STOP"), location, duration, activity, priority
        );
        robot_queues[robot_id].addTask(task);
    }
    
    // Schedule a route (e.g., bus route) for a robot
    void scheduleRoute(int robot_id, const std::vector<Location>& stops,
                       const std::vector<double>& durations, bool loop = false,
                       TaskPriority priority = TaskPriority::NORMAL) {
        auto task = std::make_shared<RouteTask>(
            generateTaskId("ROUTE"), stops, durations, loop, 1, priority
        );
        robot_queues[robot_id].addTask(task);
    }
    
    // Get next task for a robot (returns waypoints)
    std::deque<Eigen::VectorXd> getNextTaskWaypoints(int robot_id) {
        auto task = robot_queues[robot_id].getNextTask();
        if (task) {
            return task->toWaypoints();
        }
        return std::deque<Eigen::VectorXd>();
    }
    
    // Mark current task as completed
    void completeTask(int robot_id) {
        robot_queues[robot_id].completeCurrentTask();
    }
    
    // Get current task status for a robot
    std::shared_ptr<Task> getCurrentTask(int robot_id) {
        return robot_queues[robot_id].getCurrentTask();
    }
    
    // Cancel a task
    bool cancelTask(int robot_id, const std::string& task_id) {
        return robot_queues[robot_id].cancelTask(task_id);
    }
    
    // Get queue size for a robot
    size_t getQueueSize(int robot_id) {
        return robot_queues[robot_id].size();
    }
    
    // Check if robot has pending tasks
    bool hasPendingTasks(int robot_id) {
        return !robot_queues[robot_id].isEmpty();
    }
};
