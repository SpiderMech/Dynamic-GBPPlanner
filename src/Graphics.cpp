/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <Graphics.h>
#include <tuple>


/**************************************************************************/
// Graphics class that deals with the nitty-gritty of display.
// Camera is also included here. You can set different camera positions/trajectories
// and then during simulation cycle through them using the SPACEBAR

// Please note: Raylib camera defines the world with positive X = right, positive Z = down, and positive Y = out-of-plane
// But in our work we use the standard convention of positive X = right, positive Y = down, and positive Z = into-plane
/**************************************************************************/
Graphics::Graphics(Image obstacleImg) : obstacleImg_(ImageCopy(obstacleImg))
{
    if (!globals.DISPLAY)
        return;

    // Reset the static lights count to ensure proper lighting in subsequent simulation runs
    ResetLightsCount();

    // Camera is defined by a forward vector (target - position), as well as an up vector (see raylib for more info)
    // These are vectors for each camera transition. Cycle through them in the simulation with the SPACEBAR
    camera_positions_ = {Vector3{0., 1.f * globals.WORLD_SZ, 0.},
                         (Vector3){20., 15, 20},
                         (Vector3){0., 0.85f * globals.WORLD_SZ, 0.9f * globals.WORLD_SZ}};
    camera_ups_ = {Vector3{0., 0., -1.}, (Vector3){-0.325, 0.9, -0.316}, (Vector3){0., 0., -1.}};
    camera_targets_ = {Vector3{0., 0., 0.}, (Vector3){1.363, 0, 1.463}, (Vector3){0., 0., 0.}};

    camera3d.position = camera_positions_[camera_idx_];
    camera3d.target = camera_targets_[camera_idx_];
    camera3d.up = camera_ups_[camera_idx_];   // Camera up vector
    camera3d.fovy = 60.0f;                    // Camera field-of-view Y
    camera3d.projection = CAMERA_PERSPECTIVE; // Camera mode type

    // Load basic lighting shader
    lightShader_ = LoadShader((globals.ASSETS_DIR + "shaders/base_lighting.vs").c_str(),
                              (globals.ASSETS_DIR + "shaders/lighting.fs").c_str());

    // Get some required shader locations
    lightShader_.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(lightShader_, "viewPos");

    // Ambient light level (some basic lighting)
    int ambientLoc = GetShaderLocation(lightShader_, "ambient");
    float temp[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    SetShaderValue(lightShader_, ambientLoc, temp, SHADER_UNIFORM_VEC4);
    
    // Load all robot and obstacle models with their bounding boxes
    loadRobotModels();
    loadObstacleModels();

    // Height map
    Mesh mesh = GenMeshHeightmap(obstacleImg_, (Vector3){1.f * globals.WORLD_SZ, 1.f * globals.ROBOT_RADIUS, 1.f * globals.WORLD_SZ}); // Generate heightmap mesh (RAM and VRAM)
    ImageColorInvert(&obstacleImg_);                                                                                                   // TEXTURE REQUIRES OBSTACLES ARE BLACK
    texture_img_ = LoadTextureFromImage(obstacleImg_);
    groundModel_ = LoadModelFromMesh(mesh);                                      // Load model from generated mesh
    groundModel_.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = texture_img_; // Set map diffuse texture
    groundModelpos_ = {-globals.WORLD_SZ / 2.f, 0.0f, -globals.WORLD_SZ / 2.f};  // Define model position

    // Create lights
    Light lights[MAX_LIGHTS] = {0};
    Vector3 target = camera3d.target;
    Vector3 position = Vector3{target.x + 10, target.y + 20, target.z + 10};
    lights[0] = CreateLight(LIGHT_POINT, position, target, LIGHTGRAY, lightShader_);
}

Graphics::~Graphics()
{
    // Only cleanup if display was enabled
    if (!globals.DISPLAY)
        return;
        
    // Unload texture
    UnloadTexture(texture_img_);
    
    // Unload ground model
    UnloadModel(groundModel_);
    
    // Unload shader
    UnloadShader(lightShader_);
    
    // Unload obstacle image
    UnloadImage(obstacleImg_);
    
    // Unload all robot models
    for (auto& [type, modelInfo] : robotModels_) {
        if (modelInfo) {
            UnloadModel(modelInfo->model);
        }
    }
    robotModels_.clear();
    
    // Unload all obstacle models
    for (auto& [type, modelInfo] : obstacleModels_) {
        if (modelInfo) {
            UnloadModel(modelInfo->model);
        }
    }
    obstacleModels_.clear();
};

/******************************************************************************************/
// Use captured mouse input and keypresses and modify the camera view.
// Also transition between camera viewframes if necessary.
/******************************************************************************************/
void Graphics::update_camera()
{
    float zoomscale = IsKeyDown(KEY_LEFT_SHIFT) ? 100. : 10.;
    float zoom = -(float)GetMouseWheelMove() * zoomscale;
    CameraMoveToTarget(&camera3d, zoom);
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
    {
        Vector2 del = GetMouseDelta();
        // FOR UP {0,0,-1} and TOWARDS STRAIGHT DOWN
        if (IsKeyDown(KEY_LEFT_SHIFT))
        {
            CameraPitch(&camera3d, -del.y * 0.05, true, true, true);
            // Rotate up direction around forward axis
            camera3d.up = Vector3RotateByAxisAngle(camera3d.up, Vector3{0., 1., 0.}, -0.05 * del.x);
            Vector3 forward = Vector3Subtract(camera3d.target, camera3d.position);
            forward = Vector3RotateByAxisAngle(forward, Vector3{0., 1., 0.}, -0.05 * del.x);
            camera3d.position = Vector3Subtract(camera3d.target, forward);
        }
        else if (IsKeyDown(KEY_LEFT_CONTROL))
        {
            float zoom = del.y * 0.1;
            CameraMoveToTarget(&camera3d, zoom);
        }
        else
        {
            // Camera movement
            CameraMoveRight(&camera3d, -del.x, true);
            Vector3 D = GetCameraUp(&camera3d);
            D.y = 0.;
            D = Vector3Scale(Vector3Normalize(D), del.y);
            camera3d.position = Vector3Add(camera3d.position, D);
            camera3d.target = Vector3Add(camera3d.target, D);
        }
    }
    if (camera_transition_)
    {
        int camera_transition_time = 100;
        if (camera_clock_ == camera_transition_time)
        {
            camera_transition_ = false;
            camera_idx_ = (camera_idx_ + 1) % camera_positions_.size();
            camera_clock_ = 0;
        }
        camera3d.position = Vector3Lerp(camera_positions_[camera_idx_], camera_positions_[(camera_idx_ + 1) % camera_positions_.size()], (camera_clock_ % camera_transition_time) / (float)camera_transition_time);
        camera3d.up = Vector3Lerp(camera_ups_[camera_idx_], camera_ups_[(camera_idx_ + 1) % camera_ups_.size()], (camera_clock_ % camera_transition_time) / (float)camera_transition_time);
        camera3d.target = Vector3Lerp(camera_targets_[camera_idx_], camera_targets_[(camera_idx_ + 1) % camera_targets_.size()], (camera_clock_ % camera_transition_time) / (float)camera_transition_time);
        camera_clock_++;
    }
}

/* Compute the minimal axis-aligned bounding box for a mesh */
BoundingBox Graphics::computeMeshBoundingBox(const Mesh& mesh)
{
    BoundingBox box;
    
    if (mesh.vertexCount == 0) {
        box.min = Vector3{0, 0, 0};
        box.max = Vector3{0, 0, 0};
        return box;
    }
    
    // Initialize with first vertex
    box.min = Vector3{mesh.vertices[0], mesh.vertices[1], mesh.vertices[2]};
    box.max = box.min;
    
    // Find min and max for each axis
    for (int i = 0; i < mesh.vertexCount; i++) {
        float x = mesh.vertices[i * 3 + 0];
        float y = mesh.vertices[i * 3 + 1];
        float z = mesh.vertices[i * 3 + 2];
        
        box.min.x = fminf(box.min.x, x);
        box.min.y = fminf(box.min.y, y);
        box.min.z = fminf(box.min.z, z);
        
        box.max.x = fmaxf(box.max.x, x);
        box.max.y = fmaxf(box.max.y, y);
        box.max.z = fmaxf(box.max.z, z);
    }
    
    return box;
}

/* Load all robot models and compute their bounding boxes */
void Graphics::loadRobotModels()
{
    // Load SPHERE model (default representation)
    {
        Model sphereModel = LoadModelFromMesh(GenMeshSphere(1.0f, 50.f, 50.f));
        sphereModel.materials[0].shader = lightShader_;
        sphereModel.materials[0].maps[0].color = WHITE;
        
        // For a unit sphere, bounding box is simple
        BoundingBox sphereBox = {
            Vector3{-1.0f, -1.0f, -1.0f},  // min
            Vector3{1.0f, 1.0f, 1.0f}       // max
        };
        
        // Dimensions for unit sphere
        Vector3 sphereDims = {2.0f, 2.0f, 2.0f};
        
        // Store in map
        robotModels_[RobotType::SPHERE] = std::make_shared<RobotModelInfo>(sphereModel, sphereBox, sphereDims, 0.0);
    }
    
    // Load vehicle models (CAR, BUS)
    {
        std::vector<std::tuple<RobotType, std::string, double>> model_paths = {
            {RobotType::CAR, globals.ASSETS_DIR + "models/Car.obj", M_PI / 2.0},
            {RobotType::BUS, globals.ASSETS_DIR + "models/Bus.obj", 0.0}
        };

        for (const auto& [type, model_path, of] : model_paths) {
            Model model = LoadModel(model_path.c_str());
            model.materials[0].shader = lightShader_;
            model.materials[0].maps[0].color = WHITE;
            
            // Compute bounding box from the mesh
            BoundingBox bbox = computeMeshBoundingBox(model.meshes[0]);
            
            // Calculate dimensions
            Vector3 dims = {
                bbox.max.x - bbox.min.x,
                bbox.max.y - bbox.min.y,
                bbox.max.z - bbox.min.z
            };
            
            // Store in map
            robotModels_[type] = std::make_shared<RobotModelInfo>(model, bbox, dims, of);
        }
    }
}

void Graphics::loadObstacleModels() {
    // Load vehicle models
    {
        std::vector<std::tuple<ObstacleType, std::string, double>> model_paths = {
            {ObstacleType::BUS, globals.ASSETS_DIR + "models/Bus.obj", 0.0},
            {ObstacleType::VAN, globals.ASSETS_DIR + "models/Van.obj", 0.0},
            {ObstacleType::PEDESTRIAN, globals.ASSETS_DIR + "models/Ped.obj", 0.0},
        };

        for (const auto& [type, model_path, of] : model_paths) {
            // Store in map
            if (type == ObstacleType::PEDESTRIAN) {
                obstacleModels_[type] = createBoxObstacleModel(1.f, 2.f, 1.f, 0.0);
            } else {
                obstacleModels_[type] = createCustomObstacleModel(model_path.c_str(), of);
            }

        }
    }
}

/* Helper method to create box obstacle model with KDTree support */
std::shared_ptr<ObstacleModelInfo> Graphics::createBoxObstacleModel(float width, float height, float depth, double angle_offset, Color color)
{
    Model model = LoadModelFromMesh(GenMeshCube(width, height, depth));
    model.materials[0].shader = lightShader_;
    model.materials[0].maps[0].color = WHITE;
    
    // Compute bounding box
    BoundingBox box = {
        Vector3{-width * 0.5f, -height * 0.5f, -depth * 0.5f},
        Vector3{width * 0.5f, height * 0.5f, depth * 0.5f}
    };
    
    Vector3 dims = {width, height, depth};
    
    auto obstacleInfo = std::make_shared<ObstacleModelInfo>(model, box, dims, angle_offset, color);
    
    // Generate point cloud for KDTree (grid on X-Z plane)
    std::vector<Eigen::Vector2d> points;
    int grid_size = 8;
    auto lerp = [](double a, double b, double t) { return a + (b - a) * t; };
    
    for (int i = 0; i <= grid_size; ++i) {
        for (int j = 0; j <= grid_size; ++j) {
            double u = double(i) / grid_size;
            double v = double(j) / grid_size;
            Eigen::Vector2d pt;
            pt.x() = lerp(-width * 0.5, width * 0.5, u);
            pt.y() = lerp(-depth * 0.5, depth * 0.5, v);  // Z becomes Y in 2D
            points.push_back(pt);
        }
    }
    
    obstacleInfo->initializeKDTree(points);
    return obstacleInfo;
}

/* Helper method to create custom obstacle model from file with KDTree support */
std::shared_ptr<ObstacleModelInfo> Graphics::createCustomObstacleModel(const std::string_view mesh_file, double angle_offset, Color color, bool use_bbox)
{
    if (mesh_file.empty()) {
        throw std::runtime_error("createCustomObstacleModel: mesh_file cannot be empty.");
    }
    
    Model model = LoadModel(mesh_file.data());
    model.materials[0].shader = lightShader_;
    model.materials[0].maps[0].color = WHITE;
    
    Mesh& mesh_ref = model.meshes[0];
    BoundingBox box = computeMeshBoundingBox(mesh_ref);
    Vector3 dims = {
        box.max.x - box.min.x,
        box.max.y - box.min.y,
        box.max.z - box.min.z
    };
    
    auto obstacleInfo = std::make_shared<ObstacleModelInfo>(model, box, dims, angle_offset, color);
    std::vector<Eigen::Vector2d> points;
    
    if (use_bbox) {
        // Generate point cloud from bounding box (grid on X-Z plane)
        int grid_size = 8;
        auto lerp = [](double a, double b, double t) { return a + (b - a) * t; };
        
        for (int i = 0; i <= grid_size; ++i) {
            for (int j = 0; j <= grid_size; ++j) {
                double u = double(i) / grid_size;
                double v = double(j) / grid_size;
                Eigen::Vector2d pt;
                pt.x() = lerp(box.min.x, box.max.x, u);
                pt.y() = lerp(box.min.z, box.max.z, v);  // Z becomes Y in 2D
                points.push_back(pt);
            }
        }
    } else {
        // Extract point cloud from mesh triangles (original behavior)
        for (size_t t = 0; t < mesh_ref.triangleCount; ++t) {
            unsigned short i0 = mesh_ref.indices[3 * t + 0];
            unsigned short i1 = mesh_ref.indices[3 * t + 1];
            unsigned short i2 = mesh_ref.indices[3 * t + 2];
            
            Vector3 v0 = {
                mesh_ref.vertices[3 * i0 + 0],
                mesh_ref.vertices[3 * i0 + 1],
                mesh_ref.vertices[3 * i0 + 2]
            };
            Vector3 v1 = {
                mesh_ref.vertices[3 * i1 + 0],
                mesh_ref.vertices[3 * i1 + 1],
                mesh_ref.vertices[3 * i1 + 2]
            };
            Vector3 v2 = {
                mesh_ref.vertices[3 * i2 + 0],
                mesh_ref.vertices[3 * i2 + 1],
                mesh_ref.vertices[3 * i2 + 2]
            };
            
            // Use triangle centroid in X-Z plane
            Eigen::Vector2d pt;
            pt.x() = (v0.x + v1.x + v2.x) / 3.0;
            pt.y() = (v0.z + v1.z + v2.z) / 3.0;  // Z becomes Y in 2D
            points.push_back(pt);
        }
    }
    obstacleInfo->initializeKDTree(points);
    return obstacleInfo;
}

// Implementation of ObstacleModelInfo methods
void ObstacleModelInfo::initializeKDTree(const std::vector<Eigen::Vector2d>& points)
{
    if (points.empty()) {
        throw std::runtime_error("ObstacleModelInfo::initializeKDTree called with empty point cloud.");
    }
    
    // Copy points to matrix
    mat_.resize(2, points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        mat_.col(i) = points[i];
    }
    
    // Build KDTree
    kdtree_ = std::make_unique<KDTree>(2, std::cref(mat_));
    kdtree_->index_->buildIndex();
}

std::vector<std::pair<Eigen::Vector2d, double>> ObstacleModelInfo::getNearestPoints(int k, const Eigen::Vector2d& query_pt) const
{
    if (!kdtree_) {
        throw std::runtime_error("ObstacleModelInfo::getNearestPoints called before KDTree was built.");
    }
    
    if (mat_.cols() == 0) {
        throw std::runtime_error("ObstacleModelInfo::getNearestPoints called with empty point cloud.");
    }
    
    if (k > mat_.cols()) {
        k = mat_.cols();  // clamp to max available points
    }
    
    std::vector<size_t> ret_indexes(k);
    std::vector<double> out_dists_sqr(k);
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(ret_indexes.data(), out_dists_sqr.data());
    nanoflann::SearchParameters params;
    kdtree_->index_->findNeighbors(resultSet, query_pt.data(), params);
    
    std::vector<std::pair<Eigen::Vector2d, double>> results;
    for (int i = 0; i < k; ++i) {
        results.emplace_back(mat_.col(ret_indexes[i]), out_dists_sqr[i]);
    }
    return results;
}
