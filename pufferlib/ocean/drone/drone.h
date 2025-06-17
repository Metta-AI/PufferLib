// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"

// Width and height for visualisation window
#define WIDTH 1080
#define HEIGHT 720

// Simulation properties
#define GRID_SIZE 10.0f
#define DT 0.02f

// Physical constants for the drone
#define MASS 1.0f       // kg
#define IXX 0.01f       // kgm^2
#define IYY 0.01f       // kgm^2
#define IZZ 0.02f       // kgm^2
#define ARM_LEN 0.1f    // m
#define K_THRUST 3e-5f  // thrust coefficient
#define K_ANG_DAMP 0.2f // angular damping coefficient
#define K_DRAG 1e-6f    // drag (torque) coefficient
#define B_DRAG 0.1f     // linear drag coefficient
#define GRAVITY 9.81f   // m/s^2
#define MAX_RPM 750.0f  // rad/s
#define MAX_VEL 50.0f   // m/s
#define MAX_OMEGA 50.0f // rad/s

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
    float perf;
    float n;
};

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline float dot3(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline void clamp4(float a[4], float min, float max) {
    a[0] = clampf(a[0], min, max);
    a[1] = clampf(a[1], min, max);
    a[2] = clampf(a[2], min, max);
    a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat *q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

typedef struct Client Client;
struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
};

typedef struct Drone Drone;
struct Drone {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    unsigned int tick;
    unsigned int report_interval;
    unsigned int score;
    float episodic_return;

    int n_targets;
    int moves_left;

    Vec3 pos;   // global position (x, y, z)
    Vec3 vel;   // linear velocity (u, v, w)
    Quat quat;  // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega; // angular velocity (p, q, r)

    Vec3 move_target;   // move target position
    Vec3 vec_to_target; // vector to target from drone's current pos

    Client *client;
};

void init(Drone *env) {
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(Drone *env) {
    env->log.score = env->score;
    env->log.episode_return = env->episodic_return;
    env->log.episode_length = env->tick;
    env->log.perf = 0.0f;
    env->log.n += 1.0f;
}

void compute_observations(Drone *env) {
    env->observations[0] = (env->pos.x - env->move_target.x) / GRID_SIZE;
    env->observations[1] = (env->pos.y - env->move_target.y) / GRID_SIZE;
    env->observations[2] = (env->pos.z - env->move_target.z) / GRID_SIZE;

    env->observations[3] = env->pos.x / GRID_SIZE;
    env->observations[4] = env->pos.y / GRID_SIZE;
    env->observations[5] = env->pos.z / GRID_SIZE;

    env->observations[6] = env->quat.w;
    env->observations[7] = env->quat.x;
    env->observations[8] = env->quat.y;
    env->observations[9] = env->quat.z;

    env->observations[10] = env->vel.x / MAX_VEL;
    env->observations[11] = env->vel.y / MAX_VEL;
    env->observations[12] = env->vel.z / MAX_VEL;

    env->observations[13] = env->omega.x / MAX_OMEGA;
    env->observations[14] = env->omega.y / MAX_OMEGA;
    env->observations[15] = env->omega.z / MAX_OMEGA;
}

void c_reset(Drone *env) {
    env->tick = 0;
    env->score = 0;
    env->episodic_return = 0.0f;

    env->n_targets = 5;
    env->moves_left = 1000;

    env->move_target.x = rndf(-9, 9);
    env->move_target.y = rndf(-9, 9);
    env->move_target.z = rndf(-9, 9);

    env->pos.x = rndf(-9, 9);
    env->pos.y = rndf(-9, 9);
    env->pos.z = rndf(-9, 9);

    env->vel.x = 0.0f;
    env->vel.y = 0.0f;
    env->vel.z = 0.0f;

    env->quat.w = 1.0f;
    env->quat.x = 0.0f;
    env->quat.y = 0.0f;
    env->quat.z = 0.0f;

    env->omega.x = 0.0f;
    env->omega.y = 0.0f;
    env->omega.z = 0.0f;

    compute_observations(env);
}

void c_step(Drone *env) {
    clamp4(env->actions, -1.0f, 1.0f);

    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->log.score = 0;

    // motor thrusts
    float T[4];
    for (int i = 0; i < 4; i++) {
        T[i] = K_THRUST * powf((env->actions[i] + 1.0f) * 0.5f * MAX_RPM, 2.0f);
    }

    // body frame net force
    Vec3 F_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // body frame torques
    Vec3 M = {ARM_LEN * (T[1] - T[3]), ARM_LEN * (T[2] - T[0]),
              K_DRAG * (T[0] - T[1] + T[2] - T[3])};

    // applies angular damping to torques
    M.x -= K_ANG_DAMP * env->omega.x;
    M.y -= K_ANG_DAMP * env->omega.y;
    M.z -= K_ANG_DAMP * env->omega.z;

    // body frame force -> world frame force
    Vec3 F_world = quat_rotate(env->quat, F_body);

    // world frame linear drag
    F_world.x -= B_DRAG * env->vel.x;
    F_world.y -= B_DRAG * env->vel.y;
    F_world.z -= B_DRAG * env->vel.z;

    // world frame gravity
    Vec3 accel = {F_world.x / MASS, F_world.y / MASS,
                  (F_world.z / MASS) - GRAVITY};

    // integrates quaternion
    Quat omega_q = {0.0f, env->omega.x, env->omega.y, env->omega.z};
    Quat q_dot = quat_mul(env->quat, omega_q);

    // from the definition of q dot
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // integrations
    env->pos.x += env->vel.x * DT;
    env->pos.y += env->vel.y * DT;
    env->pos.z += env->vel.z * DT;

    env->vel.x += accel.x * DT;
    env->vel.y += accel.y * DT;
    env->vel.z += accel.z * DT;

    env->omega.x += (M.x / IXX) * DT;
    env->omega.y += (M.y / IYY) * DT;
    env->omega.z += (M.z / IZZ) * DT;

    clamp3(&env->vel, -MAX_VEL, MAX_VEL);
    clamp3(&env->omega, -MAX_OMEGA, MAX_OMEGA);

    env->quat.w += q_dot.w * DT;
    env->quat.x += q_dot.x * DT;
    env->quat.y += q_dot.y * DT;
    env->quat.z += q_dot.z * DT;

    quat_normalize(&env->quat);

    // check out of bounds
    bool out_of_bounds = env->pos.x < -GRID_SIZE || env->pos.x > GRID_SIZE ||
                         env->pos.y < -GRID_SIZE || env->pos.y > GRID_SIZE ||
                         env->pos.z < -GRID_SIZE || env->pos.z > GRID_SIZE;

    // give rewards
    if (out_of_bounds) {
        env->rewards[0] -= 1;
        env->episodic_return -= 1;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        compute_observations(env);
        return;
    }

    env->vec_to_target.x = env->pos.x - env->move_target.x;
    env->vec_to_target.y = env->pos.y - env->move_target.y;
    env->vec_to_target.z = env->pos.z - env->move_target.z;

    if (norm3(env->vec_to_target) < 1.5f) {
        env->rewards[0] += 1;
        env->episodic_return += 1;
        env->n_targets -= 1;
        env->score++;

        env->move_target.x = rndf(-9, 9);
        env->move_target.y = rndf(-9, 9);
        env->move_target.z = rndf(-9, 9);
    }

    env->moves_left -= 1;
    if (env->moves_left == 0 || env->n_targets == 0) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(Drone *env) {
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

Client *make_client(Drone *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    InitWindow(WIDTH, HEIGHT, "PufferLib Drone");
    SetTargetFPS(60);

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 40.0f;
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 10.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    return client;
}

void c_render(Drone *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    handle_camera_controls(env->client);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(env->client->camera);

    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_SIZE * 2.0f,
                  GRID_SIZE * 2.0f, GRID_SIZE * 2.0f, WHITE);

    DrawSphere(
        (Vector3){env->move_target.x, env->move_target.y, env->move_target.z},
        0.3f, BLUE);

    DrawSphere((Vector3){env->pos.x, env->pos.y, env->pos.z}, 0.3f, RED);

    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
        T[i] = K_THRUST * rpm * rpm;
    }

    const float rotor_radius = 0.15f;
    const float visual_arm_len = ARM_LEN * 4.0f;

    Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                  {-visual_arm_len, 0.0f, 0.0f},
                                  {0.0f, +visual_arm_len, 0.0f},
                                  {0.0f, -visual_arm_len, 0.0f}};

    Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};

    for (int i = 0; i < 4; i++) {
        Vec3 world_off = quat_rotate(env->quat, rotor_offsets_body[i]);

        Vector3 rotor_pos = {env->pos.x + world_off.x, env->pos.y + world_off.y,
                             env->pos.z + world_off.z};

        float rpm = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
        float intensity = 0.75f + 0.25f * (rpm / MAX_RPM);

        Color rotor_color =
            (Color){(unsigned char)(base_colors[i].r * intensity),
                    (unsigned char)(base_colors[i].g * intensity),
                    (unsigned char)(base_colors[i].b * intensity), 255};

        DrawSphere(rotor_pos, rotor_radius, rotor_color);

        DrawCylinderEx((Vector3){env->pos.x, env->pos.y, env->pos.z}, rotor_pos,
                       0.02f, 0.02f, 8, BLACK);
    }

    Vec3 forward_body = {1.0f, 0.0f, 0.0f};
    Vec3 forward_world = quat_rotate(env->quat, forward_body);
    DrawLine3D((Vector3){env->pos.x, env->pos.y, env->pos.z},
               (Vector3){env->pos.x + forward_world.x * 0.5f,
                         env->pos.y + forward_world.y * 0.5f,
                         env->pos.z + forward_world.z * 0.5f},
               YELLOW);

    if (norm3(env->vel) > 0.1f) {
        DrawLine3D((Vector3){env->pos.x, env->pos.y, env->pos.z},
                   (Vector3){env->pos.x + env->vel.x * 0.1f,
                             env->pos.y + env->vel.y * 0.1f,
                             env->pos.z + env->vel.z * 0.1f},
                   MAGENTA);
    }

    DrawLine3D(
        (Vector3){env->pos.x, env->pos.y, env->pos.z},
        (Vector3){env->move_target.x, env->move_target.y, env->move_target.z},
        ColorAlpha(BLUE, 0.3f));

    EndMode3D();

    DrawText(TextFormat("Targets left: %d", env->n_targets), 10, 10, 20, WHITE);
    DrawText(TextFormat("Moves left: %d", env->moves_left), 10, 40, 20, WHITE);
    DrawText(TextFormat("Episode Return: %.2f", env->episodic_return), 10, 70,
             20, WHITE);

    DrawText("Motor Thrusts:", 10, 110, 20, WHITE);
    DrawText(TextFormat("Front: %.3f", T[0]), 10, 135, 18, ORANGE);
    DrawText(TextFormat("Back:  %.3f", T[1]), 10, 155, 18, PURPLE);
    DrawText(TextFormat("Right: %.3f", T[2]), 10, 175, 18, LIME);
    DrawText(TextFormat("Left:  %.3f", T[3]), 10, 195, 18, SKYBLUE);

    DrawText(TextFormat("Pos: (%.1f, %.1f, %.1f)", env->pos.x, env->pos.y,
                        env->pos.z),
             10, 225, 18, WHITE);
    DrawText(TextFormat("Vel: %.2f m/s", norm3(env->vel)), 10, 245, 18, WHITE);

    DrawText("Left click + drag: Rotate camera", 10, 275, 16, LIGHTGRAY);
    DrawText("Mouse wheel: Zoom in/out", 10, 295, 16, LIGHTGRAY);

    EndDrawing();
}
