/**************************************************************************************/
// Task Scheduler Interface
/**************************************************************************************/
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <chrono>
#include <map>
#include <deque>
#include <vector>
#include <memory>
#include <optional>

// Forward declarations
class Robot;

/**************************************************************************************/
// Enum definitions for task types and priorities
/**************************************************************************************/
enum class TaskType {
    STATIONARY,     // Task which requires robot to stay at a location (i.e., bus stop, charging station)
    DELIVERY,       // Task that involves pick-up and drop-off
    PATROL,         // Continuous, looping movement between waypoints
    CUSTOM          // Any other specific task type
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
// Location structure
/**************************************************************************************/
struct Location {
    Eigen::Vector2d position;      // x, y coordinates
    std::string name;              // Name of the location (i.e., "Bus Stop A", optional)
    std::string type;              // Type of the location (i.e., "bus_stop", optional)
    double arrival_radius = 1.0;   // Radius for considering arrival at location
    Location() 
        : position(0, 0), name(""), type(""), arrival_radius(1.0) {}
    Location(double x, double y, const std::string& n = "", const std::string& t = "");

    Location(const Eigen::Vector2d& pos, const std::string& n = "", const std::string& t = "");
};


/**************************************************************************************/
// Abstract Task class all specific task types inherit from
/**************************************************************************************/
class Task {
public:
    std::string id;
    TaskType type;
    TaskPriority priority;
    TaskStatus status;

    Task(const std::string& task_id, TaskType t, TaskPriority p = TaskPriority::NORMAL);
    virtual ~Task() = default;
    
    // Convert task to waypoints format (5D: x, y, xdot, ydot, pause_time), must be implemented
    virtual std::deque<Eigen::VectorXd> toWaypoints() const = 0;
};

/**************************************************************************************/
// Stationary Task - Robot stays at a location for a duration
/**************************************************************************************/
class StationaryTask : public Task {
public:
    Location location;          // Where to stop
    std::string activity;
    double duration_seconds;    // How long to stop for

    StationaryTask(const std::string& id, const Location loc, double duration, 
                   const std::string& act = "", TaskPriority p = TaskPriority::NORMAL);
    
    std::deque<Eigen::VectorXd> toWaypoints() const override;
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
    
    DeliveryTask(const std::string& id, const Location& pickup, const Location& delivery,
                 double pickup_time = 5.0, double delivery_time = 5.0,
                 TaskPriority p = TaskPriority::NORMAL);
    
    std::deque<Eigen::VectorXd> toWaypoints() const override;
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
                TaskPriority p = TaskPriority::NORMAL);
    
    std::deque<Eigen::VectorXd> toWaypoints() const;
};

/**************************************************************************************/
// Task PriorityQueue - Manages task scheduling for a robot
/**************************************************************************************/
class TaskQueue {
private:
    std::deque<std::shared_ptr<Task>> pending_tasks;
    std::shared_ptr<Task> current_task;
    std::vector<std::shared_ptr<Task>> completed_tasks;
    size_t max_history = 100;  // Maximum completed tasks to keep
    
public:
    // Add a task to the queue
    void addTask(std::shared_ptr<Task> task);
    // Get next task and mark as current
    std::shared_ptr<Task> getNextTask();
    // Mark current task as completed
    void completeCurrentTask();
    // Cancel a pending task
    bool cancelTask(const std::string& task_id);
    // Get current task
    std::shared_ptr<Task> getCurrentTask() const;
    // Get all pending tasks
    std::deque<std::shared_ptr<Task>> getPendingTasks() const;
    // Check if queue is empty
    bool isEmpty() const;
    // Get queue size
    size_t size() const;
};

/**************************************************************************************/
// Task Scheduler - Main interface for creating and managing tasks
/**************************************************************************************/
class TaskScheduler {
public:
    std::map<int, TaskQueue> robot_queues;           // Task queue per robot
    std::map<std::string, Location> known_locations; // map of known locations
    int next_task_id = 1;
    
    TaskScheduler();
    // Initialise common locations
    void initialiseCommonLocations();
    
    // Add a custom location
    void addLocation(const std::string& key, const Location& location);
    
    // Get a known location
    Location getLocation(const std::string& key);
    
    // Generate unique task ID
    std::string generateTaskId(const std::string& prefix = "TASK");
    
    // Schedule a delivery task for a robot
    void scheduleDelivery(int robot_id, const Location& pickup, const Location& delivery,
                          double pickup_time = 5.0, double delivery_time = 5.0,
                          TaskPriority priority = TaskPriority::NORMAL);
    
    // Schedule a stationary task for a robot
    void scheduleStop(int robot_id, const Location& location, double duration,
                      const std::string& activity = "",
                      TaskPriority priority = TaskPriority::NORMAL);
    
    // Schedule a route (e.g., bus route) for a robot
    void scheduleRoute(int robot_id, const std::vector<Location>& stops,
                       const std::vector<double>& durations, bool loop = false,
                       TaskPriority priority = TaskPriority::NORMAL);
    
    // Get next task for a robot (returns waypoints)
    std::deque<Eigen::VectorXd> getNextTaskWaypoints(int robot_id);
    
    // Mark current task as completed
    void completeTask(int robot_id);
    
    // Get current task status for a robot
    std::shared_ptr<Task> getCurrentTask(int robot_id);
    
    // Cancel a task
    bool cancelTask(int robot_id, const std::string& task_id);
    
    // Get queue size for a robot
    size_t getQueueSize(int robot_id);
    
    // Check if robot has pending tasks
    bool hasPendingTasks(int robot_id);

    // Remove entry from robot_queues;
    void removeQueue(int robot_id);
};
    