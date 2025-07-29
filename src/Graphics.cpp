/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <Graphics.h>

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

    // Assign our lighting shader to robot model
    robotModel_ = LoadModelFromMesh(GenMeshSphere(1., 50.0f, 50.0f));
    robotModel_.materials[0].shader = lightShader_;
    robotModel_.materials[0].maps[0].color = WHITE;

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
    UnloadTexture(texture_img_);
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

// Function to create a cubic Geometry pointer based on specified dimensions
std::shared_ptr<IGeometry> Graphics::GenCubeGeom(float width, float height, float depth, Color color)
{
    Model model = LoadModelFromMesh(GenMeshCube(width, height, depth));
    // Create shared pointer to model with custom deleter for raylib
    auto model_ptr = std::shared_ptr<Model>(
        new Model(model),
        [](Model *m)
        { UnloadModel(*m); delete m; });

    model_ptr->materials[0].shader = lightShader_;
    model_ptr->materials[0].maps[0].color = WHITE;

    return std::make_shared<BoxGeometry>(model_ptr,
                                         Eigen::Vector3d{-width * 0.5f, -height * 0.5f, -depth * 0.5f},
                                         Eigen::Vector3d{ width * 0.5f,  height * 0.5f,  depth * 0.5f},
                                         color);
}

// Function to create a Geometry using imported mesh file
std::shared_ptr<IGeometry> Graphics::GenCustomGeom(const std::string_view mesh_file, Color color)
{
    if (mesh_file.empty())
    {
        throw std::runtime_error("GenCustomGeom:: mesh_file cannot be empty.");
    }

    Model model = LoadModel(mesh_file.data());
    // Create shared pointer to model with custom deleter for raylib
    auto model_ptr = std::shared_ptr<Model>(
        new Model(model),
        [](Model *m)
        { UnloadModel(*m); delete m; });

    model_ptr->materials[0].shader = lightShader_;
    model_ptr->materials[0].maps[0].color = WHITE;

    Mesh &mesh_ref = model_ptr->meshes[0];
    std::vector<Eigen::Vector3d> cloud;
    cloud.reserve(mesh_ref.triangleCount);
    for (size_t t = 0; t < mesh_ref.triangleCount; ++t)
    {
        unsigned short i0 = mesh_ref.indices[3 * t + 0];
        unsigned short i1 = mesh_ref.indices[3 * t + 1];
        unsigned short i2 = mesh_ref.indices[3 * t + 2];

        Vector3 v0{
            mesh_ref.vertices[3 * i0 + 0],
            mesh_ref.vertices[3 * i0 + 1],
            mesh_ref.vertices[3 * i0 + 2]};
        Vector3 v1{
            mesh_ref.vertices[3 * i1 + 0],
            mesh_ref.vertices[3 * i1 + 1],
            mesh_ref.vertices[3 * i1 + 2]};
        Vector3 v2{
            mesh_ref.vertices[3 * i2 + 0],
            mesh_ref.vertices[3 * i2 + 1],
            mesh_ref.vertices[3 * i2 + 2]};

        cloud.emplace_back(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0);
    }
    return std::make_shared<MeshGeometry>(model_ptr, cloud, color);
}

// Function to Create obstacle model directly from self-defined mesh object
std::shared_ptr<IGeometry> Graphics::GenPolyGeom(Mesh &mesh, Color color)
{
    Model model = LoadModelFromMesh(mesh);

    // Create shared pointer to model with custom deleter for raylib
    auto model_ptr = std::shared_ptr<Model>(
        new Model(model),
        [](Model *m)
        { UnloadModel(*m); delete m; });

    model_ptr->materials[0].shader = lightShader_;
    model_ptr->materials[0].maps[0].color = WHITE;

    std::vector<Eigen::Vector3d> cloud;
    cloud.reserve(mesh.triangleCount);

    for (size_t t = 0; t < mesh.triangleCount; ++t)
    {
        unsigned short i0 = mesh.indices[3 * t + 0];
        unsigned short i1 = mesh.indices[3 * t + 1];
        unsigned short i2 = mesh.indices[3 * t + 2];

        Vector3 v0{
            mesh.vertices[3 * i0 + 0],
            mesh.vertices[3 * i0 + 1],
            mesh.vertices[3 * i0 + 2]};
        Vector3 v1{
            mesh.vertices[3 * i1 + 0],
            mesh.vertices[3 * i1 + 1],
            mesh.vertices[3 * i1 + 2]};
        Vector3 v2{
            mesh.vertices[3 * i2 + 0],
            mesh.vertices[3 * i2 + 1],
            mesh.vertices[3 * i2 + 2]};

        cloud.emplace_back(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0);
    }
    return std::make_shared<MeshGeometry>(model_ptr, cloud, color);
}

/* Functions for generating specific meshes */
Mesh Graphics::genMeshPyramid(float base, float height)
{
    Mesh mesh = {0};
    mesh.vertexCount = 5;
    mesh.triangleCount = 6;

    float b2 = base / 2.0f;

    // Allocate memory
    mesh.vertices = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
    mesh.indices = (unsigned short *)MemAlloc(mesh.triangleCount * 3 * sizeof(unsigned short));

    // Define vertices (base triangle: y = -h2, top triangle: y = +h2)
    float verts[15] = {
        -b2, 0.0f, -b2,    // v0: bottom left
        b2, 0.0f, -b2,     // v1: bottom right
        b2, 0.0f, b2,      // v2: top right
        -b2, 0.0f, b2,     // v3: top left
        0.0f, height, 0.0f // v4: apex
    };
    memcpy(mesh.vertices, verts, sizeof(verts));

    // Define triangle indices (each face = 2 triangles except bases)
    unsigned short inds[18] = {
        // base faces (2 triangles), winding reversed for outward normals
        2, 1, 0,
        3, 2, 0,
        // side faces (4 triangles), winding reversed
        4, 1, 0,
        4, 2, 1,
        4, 3, 2,
        4, 0, 3};
    memcpy(mesh.indices, inds, sizeof(inds));

    // --- compute normals for lighting ---
    mesh.normals = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
    // Zero out normals
    memset(mesh.normals, 0, mesh.vertexCount * 3 * sizeof(float));
    // For each triangle, accumulate face normals into vertex normals
    for (int t = 0; t < mesh.triangleCount; ++t)
    {
        unsigned short i0 = inds[3 * t + 0];
        unsigned short i1 = inds[3 * t + 1];
        unsigned short i2 = inds[3 * t + 2];
        Vector3 p0 = {mesh.vertices[3 * i0 + 0], mesh.vertices[3 * i0 + 1], mesh.vertices[3 * i0 + 2]};
        Vector3 p1 = {mesh.vertices[3 * i1 + 0], mesh.vertices[3 * i1 + 1], mesh.vertices[3 * i1 + 2]};
        Vector3 p2 = {mesh.vertices[3 * i2 + 0], mesh.vertices[3 * i2 + 1], mesh.vertices[3 * i2 + 2]};
        Vector3 faceNorm = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(p1, p0), Vector3Subtract(p2, p0)));
        // accumulate normals
        mesh.normals[3 * i0 + 0] += faceNorm.x;
        mesh.normals[3 * i0 + 1] += faceNorm.y;
        mesh.normals[3 * i0 + 2] += faceNorm.z;
        mesh.normals[3 * i1 + 0] += faceNorm.x;
        mesh.normals[3 * i1 + 1] += faceNorm.y;
        mesh.normals[3 * i1 + 2] += faceNorm.z;
        mesh.normals[3 * i2 + 0] += faceNorm.x;
        mesh.normals[3 * i2 + 1] += faceNorm.y;
        mesh.normals[3 * i2 + 2] += faceNorm.z;
    }
    // Normalize vertex normals
    for (int v = 0; v < mesh.vertexCount; ++v)
    {
        Vector3 n = {mesh.normals[3 * v + 0], mesh.normals[3 * v + 1], mesh.normals[3 * v + 2]};
        n = Vector3Normalize(n);
        mesh.normals[3 * v + 0] = n.x;
        mesh.normals[3 * v + 1] = n.y;
        mesh.normals[3 * v + 2] = n.z;
    }
    // --- end normals computation ---

    UploadMesh(&mesh, false);
    return mesh;
}
