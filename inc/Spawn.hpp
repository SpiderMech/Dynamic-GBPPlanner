#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Utils.h>
/***************************************************************************/
// Structs to help regulate spawning logic and prevent obstacles or robtos
// spawning on top of each other
/***************************************************************************/

extern Globals globals;

enum class SpawnType { Robot, Obstacle };

class ObstacleModelInfo;

struct SpawnRequest {
    SpawnType type;
    int zone_id;
    double t_req; // time of spawn
    Eigen::Vector2d pos; // spawn pos
    double orientation;
    Eigen::Vector2d half_extents;
    std::deque<Eigen::VectorXd> waypoints;
    std::shared_ptr<ObstacleModelInfo> model = nullptr;
    RobotType robot_type = RobotType::SPHERE;
    float radius = globals.ROBOT_RADIUS;
    Color color = GRAY;  // Color for both robot and obstacle instances
};

struct SpawnGate {
    int zone_id; // corresponds to road (starting from the left most road as 0)
    double last_admit_time = -1e-9;
    double min_headway_s = 2.0; // minimum time gap [s]
    double space_margin = globals.ROBOT_RADIUS;
    

    struct Cmp { bool operator()(const SpawnRequest& a, const SpawnRequest& b) const { return a.t_req > b.t_req; } };
    std::priority_queue<SpawnRequest, std::vector<SpawnRequest>, Cmp> pq; // Min-heap by t_req

    void enqueue(const SpawnRequest& r){ pq.push(r); }

    template<class AdmitFn, class ClearanceFn>
    void process(double now, AdmitFn admit, ClearanceFn isClear) 
    {
        while (!pq.empty()) {
            const SpawnRequest top = pq.top();
            
            if (top.t_req > now) break;

            if (now - last_admit_time < min_headway_s) {
                auto r = top; r.t_req = last_admit_time + min_headway_s;
                pq.pop(); pq.push(r);
                break;
            }

            if (!isClear(top, space_margin)) {
                auto r = top; r.t_req = now + 1.0;
                pq.pop(); pq.push(r);
                continue;
            }

            if (admit(top)) {
                last_admit_time = now;
                pq.pop();
            } else {
                auto r = top; r.t_req = now + 0.2;
                pq.pop(); pq.push(r);
            }
        }
    }
};