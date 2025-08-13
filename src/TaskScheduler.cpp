#include <TaskScheduler.h>

/**************************************************************************************/
// Location structure definitions
/**************************************************************************************/
Location::Location(double x, double y, const std::string &n, const std::string &t) 
    : position(x, y), name(n), type(t) {}

Location::Location(const Eigen::Vector2d& pos, const std::string& n, const std::string& t)
    : position(pos), name(n), type(t) {}

/**************************************************************************************/
// Abstract Task class definitions
/**************************************************************************************/
Task::Task(const std::string& task_id, TaskType t, TaskPriority p)
    : id(task_id), type(t), priority(p), status(TaskStatus::PENDING) {}

/**************************************************************************************/
// Stationary Task definitions
/**************************************************************************************/
StationaryTask::StationaryTask(const std::string& id, const Location loc, double duration, 
                               const std::string& act, TaskPriority p)
    : Task(id, TaskType::STATIONARY, p), location(loc), duration_seconds(duration), activity(act) {}

std::deque<Eigen::VectorXd> StationaryTask::toWaypoints() const {
    std::deque<Eigen::VectorXd> waypoints;
    Eigen::VectorXd wp(5);
    wp << location.position.x(), location.position.y(), 0.0, 0.0, duration_seconds;
    waypoints.push_back(wp);
    return waypoints;
}

/**************************************************************************************/
// Delivery Task definitions
/**************************************************************************************/
DeliveryTask::DeliveryTask(const std::string& id, const Location& pickup, const Location& delivery, 
                           double pickup_time, double delivery_time, TaskPriority p)
    : Task(id, TaskType::DELIVERY, p),pickup_location(pickup), delivery_location(delivery), 
      pickup_duration(pickup_time), delivery_duration(delivery_time) {}


std::deque<Eigen::VectorXd> DeliveryTask::toWaypoints() const {
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

/**************************************************************************************/
// Route Task definitions
/**************************************************************************************/
RouteTask::RouteTask(const std::string& id, const std::vector<Location>& route_stops, const std::vector<double>& durations, 
                     bool loop, int reps, TaskPriority p)
    : Task(id, TaskType::PATROL, p), stops(route_stops), 
      stop_durations(durations), loop_route(loop), repetitions(reps) {}

std::deque<Eigen::VectorXd> RouteTask::toWaypoints() const {
    std::deque<Eigen::VectorXd> waypoints;
    
    for (size_t j = 0; j < repetitions; ++j) {
        for (size_t i = 0; i < stops.size(); ++i) {
            Eigen::VectorXd wp(5);
            double duration = (i < stop_durations.size()) ? stop_durations[i] : 0.0;
            
            double vel_x = 0.0, vel_y = 0.0;
            if (i + 1 < stops.size()) {
                Eigen::Vector2d dist_to_next_stop = stops[i+1].position - stops[i].position;
                double speed = std::min(2.0, dist_to_next_stop.norm());
                dist_to_next_stop.normalize();
                vel_x = speed * dist_to_next_stop(0);
                vel_y = speed * dist_to_next_stop(1);
            }
            
            wp << stops[i].position.x(), stops[i].position.y(), vel_x, vel_y, duration;
            waypoints.push_back(wp);
        }
    }
    
    // Add return to start if looping
    if (loop_route && !stops.empty()) {
        Eigen::VectorXd return_wp(5);
        return_wp << stops[0].position.x(), stops[0].position.y(), 0.0, 0.0, 0.0;
        waypoints.push_back(return_wp);
    }
    
    return waypoints;
}

/**************************************************************************************/
// Task PriorityQueue definitions
/**************************************************************************************/
void TaskQueue::addTask(std::shared_ptr<Task> task) {
    // Insert based on priority
    auto it = pending_tasks.begin();
    while (it != pending_tasks.end() && 
            (*it)->priority >= task->priority) {
        ++it;
    }
    pending_tasks.insert(it, task);
}

// Get next task and mark as current
std::shared_ptr<Task> TaskQueue::getNextTask() {
    if (pending_tasks.empty()) {
        return nullptr;
    }
    
    current_task = pending_tasks.front();
    pending_tasks.pop_front();
    current_task->status = TaskStatus::IN_PROGRESS;
    return current_task;
}

// Mark current task as completed
void TaskQueue::completeCurrentTask() {
    if (current_task) {
        current_task->status = TaskStatus::COMPLETED;
        completed_tasks.push_back(current_task);
        
        // Maintain history size
        if (completed_tasks.size() > max_history) {
            completed_tasks.erase(completed_tasks.begin());
        }
        
        current_task = nullptr;
    }
}

// Cancel a pending task
bool TaskQueue::cancelTask(const std::string& task_id) {
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
std::shared_ptr<Task> TaskQueue::getCurrentTask() const {
    return current_task;
}

// Get all pending tasks
std::deque<std::shared_ptr<Task>> TaskQueue::getPendingTasks() const {
    return pending_tasks;
}

// Check if queue is empty
bool TaskQueue::isEmpty() const {
    return pending_tasks.empty() && !current_task;
}

// Get queue size
size_t TaskQueue::size() const {
    return pending_tasks.size() + (current_task ? 1 : 0);
}

/**************************************************************************************/
// Task Scheduler definitions
/**************************************************************************************/
TaskScheduler::TaskScheduler() {
    initialiseCommonLocations();
}

// Initialise common locations
void TaskScheduler::initialiseCommonLocations() {}

// Add a custom location
void TaskScheduler::addLocation(const std::string& key, const Location& loc){
    known_locations[key] = loc;
}

// Get a known location
Location TaskScheduler::getLocation(const std::string& key) {
    // Warning: should never get a not known location
    return known_locations.at(key);
}

// Generate unique task ID
std::string TaskScheduler::generateTaskId(const std::string& prefix) {
    return prefix + "_" + std::to_string(next_task_id++);
}

// Schedule a delivery task for a robot
void TaskScheduler::scheduleDelivery(int robot_id, const Location& pickup, const Location& delivery,
                                     double pickup_time, double delivery_time, TaskPriority priority) 
{
    auto task = std::make_shared<DeliveryTask>(
        generateTaskId("DELIVERY"), pickup, delivery, 
        pickup_time, delivery_time, priority
    );
    robot_queues[robot_id].addTask(task);
}

// Schedule a stationary task for a robot
void TaskScheduler::scheduleStop(int robot_id, const Location& location, double duration,
                                 const std::string& activity, TaskPriority priority) 
{
    auto task = std::make_shared<StationaryTask>(
        generateTaskId("STOP"), location, duration, activity, priority
    );
    robot_queues[robot_id].addTask(task);
}

// Schedule a route (e.g., bus route) for a robot
void TaskScheduler::scheduleRoute(int robot_id, const std::vector<Location>& stops,
                                  const std::vector<double>& durations, bool loop, TaskPriority priority) 
{
    auto task = std::make_shared<RouteTask>(
        generateTaskId("ROUTE"), stops, durations, loop, 1, priority
    );
    robot_queues[robot_id].addTask(task);
}

// Get next task for a robot (returns waypoints)
std::deque<Eigen::VectorXd> TaskScheduler::getNextTaskWaypoints(int robot_id) {
    auto task = robot_queues[robot_id].getNextTask();
    if (task) {
        return task->toWaypoints();
    }
    return std::deque<Eigen::VectorXd>();
}

// Mark current task as completed
void TaskScheduler::completeTask(int robot_id) {
    robot_queues[robot_id].completeCurrentTask();
}

// Get current task status for a robot
std::shared_ptr<Task> TaskScheduler::getCurrentTask(int robot_id) {
    return robot_queues[robot_id].getCurrentTask();
}

// Cancel a task
bool TaskScheduler::cancelTask(int robot_id, const std::string& task_id) {
    return robot_queues[robot_id].cancelTask(task_id);
}

// Get queue size for a robot
size_t TaskScheduler::getQueueSize(int robot_id) {
    return robot_queues[robot_id].size();
}

// Check if robot has pending tasks
bool TaskScheduler::hasPendingTasks(int robot_id) {
    return !robot_queues[robot_id].isEmpty();
}

void TaskScheduler::removeQueue(int robot_id) {
    robot_queues.erase(robot_id);
}