/* School: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "simplex.h"

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif

#define MAX_SPEED 0.01f
#define MAX_FACTORY_SPEED 0.001f

#define DRONE 0
#define MOTHERSHIP 1
#define FIGHTER 2
#define BOMBER 3
#define INFANTRY 4
#define TANK 5
#define ARTILLERY 6

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Camera3D camera;
    Model ship;
    Model ground;
    Mesh* mesh;
    Model model;
    Shader terrain_shader;
    Texture2D terrain_texture;
    int terrain_shader_loc;
    unsigned char *terrain_data;
} Client;

typedef struct {
  float w, x, y, z;
} Quat;

typedef struct {
  float x, y, z;
} Vec3;

typedef struct {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float speed;
    float pitch;
    float roll;
    float health;
    float max_turn;
    float max_speed;
    float attack_damage;
    float attack_range;
    Vec3 orientation;
    float yaw;
    int item;
    int unit;
    int target;
    int episode_length;
    float episode_return;
} Entity;

typedef struct {
    Log log;
    Client* client;
    Entity* agents;
    Entity* factories;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    float size_x;
    float size_y;
    float size_z;
    int terrain_width;
    int terrain_height;
    int num_agents;
    int num_factories;
    int num_resources;
    float* terrain;
} School;

int map_idx(School* env, float x, float y) {
    return env->terrain_width*(int)y + (int)x;
}

float ground_height(School* env, float x, float z) {
    int agent_map_x = 128*x + 128*env->size_x;
    int agent_map_z = 128*z + 128*env->size_z;
    if (agent_map_x == 256*env->size_x) {
        agent_map_x -= 1;
    }
    if (agent_map_z == 256*env->size_z) {
        agent_map_z -= 1;
    }
    int idx = map_idx(env, agent_map_x, agent_map_z);
    float terrain_height = env->terrain[idx];
    return (terrain_height - 128.0f*env->size_y) / 128.0f;
}

void perlin_noise(float* map, int width, int height,
        float base_frequency, int octaves, int offset_x, int offset_y, float glob_scale) {
    float frequencies[octaves];
    for (int i = 0; i < octaves; i++) {
        frequencies[i] = base_frequency*pow(2, i);
    }

    float min_value = FLT_MAX;
    float max_value = FLT_MIN;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            for (int oct = 0; oct < octaves; oct++) {
                float freq = frequencies[oct];
                map[adr] += (1.0/pow(2, oct))*noise2(freq*c + offset_x, freq*r + offset_y);
            }
            float val = map[adr];
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
        }
    }

    float scale = 1.0/(max_value - min_value);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            map[adr] = glob_scale * scale * (map[adr] - min_value);
            if (map[adr] < 16.0f) {
                map[adr] = 0.0f;
            } else {
                map[adr] -= 16.0f;
            }
        }
    }
}

void init(School* env) {
    env->agents = calloc(env->num_agents, sizeof(Entity));
    env->factories = calloc(env->num_factories, sizeof(Entity));
    env->terrain_width = 256*env->size_x;
    env->terrain_height = 256*env->size_z;
    env->terrain = calloc(env->terrain_width*env->terrain_height, sizeof(float));
    perlin_noise(env->terrain, env->terrain_width, env->terrain_height, 1.0/128.0, 8, 0, 0, 32);
}

void update_abilities(Entity* agent) {
    if (agent->unit == DRONE) {
        agent->health = 0.4f;
        agent->attack_damage = 0.1f;
        agent->attack_range = 0.15f;
        agent->max_turn = 2.0f;
        agent->max_speed = 1.0f;
    } else if (agent->unit == FIGHTER) {
        agent->health = 1.0f;
        agent->attack_damage = 0.5f;
        agent->attack_range = 0.25f;
        agent->max_turn = 1.0f;
        agent->max_speed = 0.75f;
    } else if (agent->unit == MOTHERSHIP) {
        agent->health = 10.0f;
        agent->attack_damage = 2.0f;
        agent->attack_range = 0.4f;
        agent->max_turn = 0.5f;
        agent->max_speed = 0.5f;
    } else if (agent->unit == BOMBER) {
        agent->health = 1.0f;
        agent->attack_damage = 1.0f;
        agent->attack_range = 0.1f;
        agent->max_turn = 0.5f;
        agent->max_speed = 0.5f;
    } else if (agent->unit == INFANTRY) {
        agent->health = 0.2f;
        agent->attack_damage = 0.2f;
        agent->attack_range = 0.2f;
        agent->max_turn = 2.0f;
        agent->max_speed = 0.25f;
    } else if (agent->unit == TANK) {
        agent->health = 2.0f;
        agent->attack_damage = 0.5f;
        agent->attack_range = 0.25f;
        agent->max_turn = 0.25f;
        agent->max_speed = 0.75f;
    } else if (agent->unit == ARTILLERY) {
        agent->health = 2.0f;
        agent->attack_damage = 2.0f;
        agent->attack_range = 0.7f;
        agent->max_turn = 0.5f;
        agent->max_speed = 0.25f;
    }
}

void respawn(School* env, Entity* agent) {
    //agent->x = randf(-env->size_x, env->size_x);
    //agent->y = randf(-env->size_y, env->size_y);
    //agent->z = randf(-env->size_z, env->size_z);
    int team = agent->item;
    agent->orientation = (Vec3){1, 0, 0};
    if (agent->unit != MOTHERSHIP) {
        int agents_per_team = env->num_agents / env->num_resources;
        int team_mothership_idx = team*agents_per_team;
        agent->x = env->agents[team_mothership_idx].x;
        agent->y = env->agents[team_mothership_idx].y;
        agent->z = env->agents[team_mothership_idx].z;
        if (agent->unit == INFANTRY || agent->unit == TANK || agent->unit == ARTILLERY) {
            agent->y = ground_height(env, agent->x, agent->z);
        }
        return;
    }

    // Find farthest corner to spawn in
    float dists[8];
    for (int i=0; i<8; i++) {
        dists[i] = 999999;
    }

    float sx = env->size_x;
    //float sy = env->size_y;
    float sz = env->size_z;
    //float xx[8] = {-sx, -sx, -sx, -sx, sx, sx, sx, sx};
    //float yy[8] = {-sy, -sy, sy, sy, -sy, -sy, sy, sy};
    //float zz[8] = {-sz, sz, -sz, sz, -sz, sz, -sz, sz};

    float xx[4] = {0, 0, -sx, sx};
    float yy[4] = {0, 0, 0, 0};
    float zz[4] = {-sz, sz, 0, 0};


    // Distance of each corner to nearest opponent
    for (int i=0; i<env->num_agents; i++) {
        Entity* other = &env->agents[i];
        int other_team = other->item;
        if (other_team == team) {
            continue;
        }
        for (int j=0; j<4; j++) {
            float dx = other->x - xx[j];
            float dy = other->y - yy[j];
            float dz = other->z - zz[j];
            float dd = dx*dx + dy*dy + dz*dz;
            if (dd < dists[j]) {
                dists[j] = dd;
            }
        }
    }

    int max_idx = 0;
    float max_dist = 0;
    for (int i=0; i<4; i++) {
        if (dists[i] > max_dist) {
            max_dist = dists[i];
            max_idx = i;
        }
    }

    agent->x = xx[max_idx];
    agent->y = yy[max_idx];
    agent->z = zz[max_idx];

    //agent->x = (rand() % 2 == 0) ? -env->size_x : env->size_x;
    //agent->y = (rand() % 2 == 0) ? -env->size_y : env->size_y;
    //agent->z = (rand() % 2 == 0) ? -env->size_z : env->size_z;

    /*
    if (agent->item == 0) {
        agent->x = -env->size_x;
        agent->y = 0.0f;
        agent->z = 0.0f;
    } else if (agent->item == 1) {
        agent->x = env->size_x;
        agent->y = 0.0f;
        agent->z = 0.0f;
    } else if (agent->item == 2) {
        agent->x = 0.0f;
        agent->y = 0.0f;
        agent->z = -env->size_z;
    } else if (agent->item == 3) {
        agent->x = 0.0f;
        agent->y = 0.0f;
        agent->z = env->size_z;
    }
    */
}


bool attack_air(Entity *agent, Entity *target) {
    float dx = target->x - agent->x;
    float dy = target->y - agent->y;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dy*dy + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    // Unit vec to target
    dy /= dd;
    dz /= dd;
    dx /= dd;

    // Unit forward vec
    float mag = sqrtf(agent->vx*agent->vx + agent->vy*agent->vy + agent->vz*agent->vz);
    float fx = agent->vx / mag;
    float fy = agent->vy / mag;
    float fz = agent->vz / mag;

    // Angle to target
    float angle = acosf(dx*fx + dy*fy + dz*fz);
    if (angle < PI/6) {
        return true;
    }
    return false;
}

bool attack_ground(Entity *agent, Entity *target) {
    if (target->unit == FIGHTER) {
        return false;
    }
    if (target->unit == MOTHERSHIP) {
        return false;
    }
    if (target->unit == BOMBER) {
        return false;
    }

    float dx = target->x - agent->x;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    // Unit vec to target
    dx /= dd;
    dz /= dd;

    // Unit forward vec
    float mag = sqrtf(agent->vx*agent->vx + agent->vz*agent->vz);
    float fx = agent->vx / mag;
    float fz = agent->vz / mag;

    // Angle to target
    float angle = acosf(dx*fx + dz*fz);
    if (angle < PI/6) {
        return true;
    }
    return false;
}

bool attack_bomber(Entity *agent, Entity *target) {
    if (target->unit == DRONE) {
        return false;
    }
    if (target->unit == FIGHTER) {
        return false;
    }
    if (target->unit == MOTHERSHIP) {
        return false;
    }
    if (target->unit == BOMBER) {
        return false;
    }

    float dx = target->x - agent->x;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    return true;
}

bool attack_aa(Entity *agent, Entity *target) {
    if (target->unit == INFANTRY) {
        return false;
    }
    if (target->unit == TANK) {
        return false;
    }
    if (target->unit == ARTILLERY) {
        return false;
    }

    float dx = target->x - agent->x;
    float dy = target->y - agent->y;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    // Angle to target (wrt y)
    float angle = acosf(dy / dd);
    if (angle < PI/6) {
        return true;
    }
    return false;
}

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

static inline int rndi(int a, int b) { return a + rand() % (b - a + 1); }

static inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

// In-place clamp of a vector
static inline void clamp3(Vec3 *vec, float min, float max) {
  vec->x = clampf(vec->x, min, max);
  vec->y = clampf(vec->y, min, max);
  vec->z = clampf(vec->z, min, max);
}

// In-place clamp of a vector
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

static inline void vec3_normalize(Vec3 *v) {
  float n = sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
  v->x /= n;
  v->y /= n;
  v->z /= n;
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
  Quat qv = {0.0f, v.x, v.y, v.z};
  Quat tmp = quat_mul(q, qv);
  Quat q_conj = {q.w, -q.x, -q.y, -q.z};
  Quat res = quat_mul(tmp, q_conj);
  return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float norm = norm3(axis);
    if (norm < 0.0001f) { // Handle zero axis
        return (Quat){1.0f, 0.0f, 0.0f, 0.0f}; // Identity quaternion
    }
    Vec3 norm_axis = {axis.x / norm, axis.y / norm, axis.z / norm};
    float s = sinf(angle / 2.0f);
    float c = cosf(angle / 2.0f);
    Quat q = {c, s * norm_axis.x, s * norm_axis.y, s * norm_axis.z};
    quat_normalize(&q);
    return q;
}

int compare_floats(const void* a, const void* b) {
    return (*(float*)a - *(float*)b) > 0;
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

float clip_angle(float theta) {
    if (theta < -PI) {
        return theta + 2.0f*PI;
    } else if (theta > PI) {
        return theta - 2.0f*PI;
    }
    return theta;
}

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

float randi(int min, int max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

void move_basic(School* env, Entity* agent, int* actions) {
    float d_vx = ((float)actions[0] - 4.0f)/400.0f;
    float d_vy = ((float)actions[1] - 4.0f)/400.0f;
    float d_vz = ((float)actions[2] - 4.0f)/400.0f;

    agent->vx += d_vx;
    agent->vy += d_vy;
    agent->vz += d_vz;

    agent->vx = clip(agent->vx, -MAX_SPEED, MAX_SPEED);
    agent->vy = clip(agent->vy, -MAX_SPEED, MAX_SPEED);
    agent->vz = clip(agent->vz, -MAX_SPEED, MAX_SPEED);

    agent->x += agent->vx;
    agent->y += agent->vy;
    agent->z += agent->vz;

    agent->x = clip(agent->x, -env->size_x, env->size_x);
    agent->y = clip(agent->y, -env->size_y, env->size_y);
    agent->z = clip(agent->z, -env->size_z, env->size_z);
}

void move_ground(School* env, Entity* agent, int* actions) {
    float d_theta = -((float)actions[1] - 4.0f)/40.0f;

    float x = agent->orientation.x;
    float y = agent->orientation.y;
    float z = agent->orientation.z;
    agent->orientation = (Vec3){
        cosf(d_theta)*x - sinf(d_theta)*z,
        y,
        sinf(d_theta)*x + cosf(d_theta)*z
    };
    vec3_normalize(&agent->orientation);

    agent->speed = agent->max_speed * MAX_SPEED;
    agent->vx = agent->speed * agent->orientation.x;
    agent->vz = agent->speed * agent->orientation.z;
    agent->x += agent->vx;
    agent->z += agent->vz;

    agent->x = clip(agent->x, -env->size_x, env->size_x);
    agent->z = clip(agent->z, -env->size_z, env->size_z);
    agent->y = ground_height(env, agent->x, agent->z);
}

void move_ship(School* env, Entity* agent, int* actions) {
    // Compute deltas from actions (same as original)
    float d_pitch = agent->max_turn * ((float)actions[0] - 4.0f) / 40.0f;
    float d_roll = agent->max_turn * ((float)actions[1] - 4.0f) / 40.0f;
    //float d_yaw = agent->max_turn * ((float)actions[2] - 4.0f) / 20.0f;

    // Update speed and clamp
    agent->speed = agent->max_speed * MAX_SPEED;

    // Create rotation quaternions
    float d_yaw = d_roll;
    Quat q_yaw = quat_from_axis_angle((Vec3){0, 1, 0}, d_yaw);
    Quat q_pitch = quat_from_axis_angle((Vec3){1, 0, 0}, d_pitch);
    Quat q_roll = quat_from_axis_angle((Vec3){0, 0, 1}, d_roll);

    Quat q = quat_mul(q_roll, quat_mul(q_pitch, q_yaw));
    quat_normalize(&q);

    Vec3 forward = quat_rotate(q, agent->orientation);
    vec3_normalize(&forward);

    agent->orientation = forward;

    // Get local axes in world coordinates for pitch and roll
    //Vec3 x_axis = quat_rotate(agent->orientation, (Vec3){1.0f, 0.0f, 0.0f}); // Pitch axis
    //Vec3 z_axis = quat_rotate(agent->orientation, (Vec3){0.0f, 0.0f, 1.0f}); // Roll axis
    //Vec3 y_axis = quat_rotate(agent->orientation, (Vec3){0.0f, 1.0f, 0.0f}); // Yaw axis

    // Update orientation: pitch, then roll (no yaw in original)
    /*
    Quat transform = quat_mul(q_roll, q_pitch);
    quat_normalize(&transform);
    agent->orientation = quat_mul(agent->orientation, transform);
    quat_normalize(&agent->orientation);
    //agent->orientation = quat_mul(agent->orientation, q_pitch);
    //agent->orientation = quat_mul(agent->orientation, q_roll);
    //agent->orientation = quat_mul(agent->orientation, q_yaw);
    quat_normalize(&agent->orientation);

    // Update position (move along local z-axis)
    Vec3 forward = quat_rotate(agent->orientation, (Vec3){0.0f, 0.0f, 1.0f});
    //Vec3 forward = {agent->orientation.x, agent->orientation.y, agent->orientation.z};
    */

    agent->x += agent->speed * forward.x;
    agent->y += agent->speed * forward.y;
    agent->z += agent->speed * forward.z;

    // Just for visualization
    agent->vx = agent->speed * forward.x;
    agent->vy = agent->speed * forward.y;
    agent->vz = agent->speed * forward.z;

    // Clamp position to environment bounds
    agent->x = clampf(agent->x, -env->size_x, env->size_x);
    agent->y = clampf(agent->y, -env->size_y, env->size_y);
    agent->z = clampf(agent->z, -env->size_z, env->size_z);
}

void compute_observations(School* env) {
    int obs_idx = 0;
    for (int a=0; a<env->num_agents; a++) {
        Entity* agent = &env->agents[a];
        float dists[env->num_resources];
        for (int i=0; i<env->num_resources; i++) {
            dists[i] = 999999;
        }
        for (int f=0; f<env->num_factories; f++) {
            Entity* factory = &env->factories[f];
            float dx = factory->x - agent->x;
            float dy = factory->y - agent->y;
            float dz = factory->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            int type = f % env->num_resources;
            if (dd < dists[type]) {
                dists[type] = dd;
                env->observations[obs_idx + 3*type] = dx;
                env->observations[obs_idx + 3*type + 1] = dy;
                env->observations[obs_idx + 3*type + 2] = dz;
            }
        }
        obs_idx += 3*env->num_resources;
        env->observations[obs_idx++] = agent->vx/MAX_SPEED;
        env->observations[obs_idx++] = agent->vy/MAX_SPEED;
        env->observations[obs_idx++] = agent->vz/MAX_SPEED;
        env->observations[obs_idx++] = agent->yaw;
        env->observations[obs_idx++] = agent->pitch;
        env->observations[obs_idx++] = agent->roll;
        env->observations[obs_idx++] = agent->x;
        env->observations[obs_idx++] = agent->y;
        env->observations[obs_idx++] = agent->z;
        env->observations[obs_idx++] = env->rewards[a];

        float min_dist = 999999;
        int min_idx = 0;
        for (int j=0; j<env->num_agents; j++) {
            if (j == a) {
                continue;
            }
            Entity* agent = &env->agents[j];
            float dx = agent->x - agent->x;
            float dy = agent->y - agent->y;
            float dz = agent->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            if (dd < min_dist) {
                min_dist = dd;
                min_idx = j;
            }
        }
        Entity* other = &env->agents[min_idx];
        env->observations[obs_idx++] = agent->x - other->x;
        env->observations[obs_idx++] = agent->y - other->y;
        env->observations[obs_idx++] = agent->z - other->z;
        env->observations[obs_idx++] = (float)(agent->item == other->item);
        memset(&env->observations[obs_idx], 0, env->num_resources*sizeof(float));
        env->observations[obs_idx + agent->item] = 1.0f;
        obs_idx += env->num_resources;
    }
}

// Required function
void c_reset(School* env) {
    int agents_per_team = env->num_agents / env->num_resources;
    for (int team=0; team<env->num_resources; team++) {
        for (int i=0; i<agents_per_team; i++) {
            Entity* agent = &env->agents[team*agents_per_team + i];
            if (i == 0) {
                agent->unit = MOTHERSHIP;
            } else if (i % 32 <= 4) {
                agent->unit = TANK;
            } else if (i % 32 <= 8) {
                agent->unit = ARTILLERY;
            } else if (i % 32 <= 12) {
                agent->unit = BOMBER;
            } else if (i % 32 <= 16) {
                agent->unit = FIGHTER;
            } else if (i % 32 <= 26) {
                agent->unit = INFANTRY;
            } else {
                agent->unit = DRONE;
            }

            agent->item = team;
            agent->orientation = (Vec3){1, 0, 0};
            agent->episode_length = 0;
            agent->target = -1;
            update_abilities(agent);
            respawn(env, agent);
        }
    }
    //for (int i=0; i<env->num_factories; i++)    }
    compute_observations(env);
}

void c_step(School* env) {
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    memset(env->terminals, 0, env->num_agents*sizeof(unsigned char));

    for (int i=0; i<env->num_agents; i++) {
        Entity* agent = &env->agents[i];
        agent->episode_length += 1;
        agent->target = -1;

        if (agent->health <= 0) {
            agent->health = 1.0f;
            respawn(env, agent);

            env->log.episode_length += agent->episode_length;
            env->log.episode_return += agent->episode_return;
            env->log.n++;
            agent->episode_length = 0;
            agent->episode_return = 0;
        } else if (agent->unit == DRONE || agent->unit == FIGHTER || agent->unit == BOMBER || agent->unit == MOTHERSHIP) {
            // Crash into terrain
            float terrain_height = ground_height(env, agent->x, agent->z);
            if (agent->y < terrain_height) {
                agent->health = 1.0f;
                respawn(env, agent);
                env->log.episode_length += agent->episode_length;
                env->log.episode_return += agent->episode_return;
                env->log.n++;
                agent->episode_length = 0;
                agent->episode_return = 0;
            }
        }

        //move_basic(env, agent, env->actions + 3*i);
        if (agent->unit == INFANTRY || agent->unit == TANK || agent->unit == ARTILLERY) {
            move_ground(env, agent, env->actions + 3*i);
        } else {
            move_ship(env, agent, env->actions + 3*i);
        }

        if (rand() % env->num_agents == 0) {
            respawn(env, &env->agents[i]);
        }

        // Collision penalty
        /*
        float penalty = 0.0f;
        float same_color_dist = 999999;
        float diff_color_dist = 999999;
        for (int j=0; j<env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            Entity* other = &env->agents[j];
            float dx = other->x - agent->x;
            float dy = other->y - agent->y;
            float dz = other->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            if (other->item == agent->item && dd < same_color_dist) {
                same_color_dist = dd;
            } else if (other->item != agent->item && dd < diff_color_dist) {
                diff_color_dist = dd;
            }
        }
        if (agent->item == 0 && diff_color_dist < same_color_dist) {
            env->rewards[i] -= 0.5f;
        } else if (agent->item == 1 && diff_color_dist < same_color_dist) {
            env->rewards[i] += 0.5f;
        }
        */


        // Distance penalty
        /*
        float dist = (agent->x*agent->x + agent->y*agent->y + agent->z*agent->z);
        if (dist > 0.25 && dist < 0.35) {
            env->rewards[i] += 1.0f;
            agent->episode_return += env->rewards[i];
        }
        */

        //env->rewards[i] -= penalty;
        //agent->episode_return += env->rewards[i];
        /*
        if (agent->episode_length > 256) {
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += agent->episode_length;
            env->log.episode_return += agent->episode_return;
            env->log.n++;
            //env->rewards[i] += 1.0f;
            agent->episode_length = 0;
            agent->episode_return = 0;
        }
        */

        // Target seeking reward
        /*
        for (int f=0; f<env->num_factories; f++) {
            Entity* factory = &env->factories[f];
            float dx = (factory->x - agent->x);
            float dy = (factory->y - agent->y);
            float dz = (factory->z - agent->z);
            float dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0.1) {
                continue;
            }
            if (factory->item == agent->item) {
                agent->item = (agent->item + 1) % env->num_resources;
                update_abilities(agent);
                env->log.perf += 1.0f;
                env->log.score += 1.0f;
                env->log.episode_length += agent->episode_length;
                env->log.n++;
                env->rewards[i] += 1.0f;
                agent->episode_length = 0;
            }
        }
        */
    }
    for (int f=0; f<env->num_factories; f++) {
        Entity* factory = &env->factories[f];
        factory->x += factory->vx;
        factory->y += factory->vy;
        factory->z += factory->vz;

        float factory_x = clip(factory->x, -env->size_x, env->size_x);
        float factory_y = clip(factory->y, -env->size_y, env->size_y);
        float factory_z = clip(factory->z, -env->size_z, env->size_z);

        if (factory_x != factory->x || factory_y != factory->y || factory_z != factory->z) {
            factory->vx = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
            factory->vy = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
            factory->vz = randf(-MAX_FACTORY_SPEED, MAX_FACTORY_SPEED);
            factory->x = factory_x;
            factory->y = factory_y;
            factory->z = factory_z;
        }
    }
    for (int i=0; i<env->num_agents; i++) {
        Entity* agent = &env->agents[i];
        for (int j=0; j<env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            Entity* target = &env->agents[j];
            if (agent->item == target->item) {
                continue;
            }
            bool can_attack = false;
            if (agent->unit == INFANTRY || agent->unit == TANK) {
                can_attack = attack_ground(agent, target);
            } else if (agent->unit == ARTILLERY) {
                can_attack = attack_aa(agent, target);
            } else if (agent->unit == BOMBER) {
                can_attack = attack_bomber(agent, target);
            } else {
                can_attack = attack_air(agent, target);
            }
            if (!can_attack) {
                continue;
            }
            agent->target = j;
            env->rewards[i] += 0.25f;
            agent->episode_return += 0.25f;
            //env->rewards[j] -= 0.25f;
            target->health -= agent->attack_damage;
            break;
        }
    }


    compute_observations(env);
}

Color COLORS[8] = {
    (Color){0, 255, 255, 255},
    (Color){255, 0, 0, 255},
    (Color){0, 255, 0, 255},
    (Color){255, 255, 0, 255},
    (Color){255, 0, 255, 255},
    (Color){0, 0, 255, 255},
    (Color){128, 255, 0, 255},
    (Color){255, 128, 0, 255},
};

Mesh* create_heightmap_mesh(float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    // NOTE: One vertex per pixel
    Mesh* mesh = (Mesh*)calloc(1, sizeof(Mesh));
    mesh->triangleCount = (mapX - 1)*(mapZ - 1)*2;    // One quad every four pixels

    mesh->vertexCount = mesh->triangleCount*3;

    mesh->vertices = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->normals = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->texcoords = (float *)RL_MALLOC(mesh->vertexCount*2*sizeof(float));
    mesh->colors = NULL;
    UploadMesh(mesh, false);
    return mesh;
}

void update_heightmap_mesh(Mesh* mesh, float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    int vCounter = 0;       // Used to count vertices float by float
    int tcCounter = 0;      // Used to count texcoords float by float
    int nCounter = 0;       // Used to count normals float by float

    //Vector3 scaleFactor = { size.x/(mapX - 1), 1.0f, size.z/(mapZ - 1) };
    Vector3 scaleFactor = { 1.0f, 1.0f, 1.0f};

    Vector3 vA = { 0 };
    Vector3 vB = { 0 };
    Vector3 vC = { 0 };
    Vector3 vN = { 0 };

    for (int z = 0; z < mapZ-1; z++)
    {
        for (int x = 0; x < mapX-1; x++)
        {
            // Fill vertices array with data
            //----------------------------------------------------------

            // one triangle - 3 vertex
            mesh->vertices[vCounter] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 1] = heightMap[x + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 2] = (float)z*scaleFactor.z;

            mesh->vertices[vCounter + 3] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 4] = heightMap[x + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 5] = (float)(z + 1)*scaleFactor.z;

            mesh->vertices[vCounter + 6] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 7] = heightMap[(x + 1) + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 8] = (float)z*scaleFactor.z;

            // Another triangle - 3 vertex
            mesh->vertices[vCounter + 9] = mesh->vertices[vCounter + 6];
            mesh->vertices[vCounter + 10] = mesh->vertices[vCounter + 7];
            mesh->vertices[vCounter + 11] = mesh->vertices[vCounter + 8];

            mesh->vertices[vCounter + 12] = mesh->vertices[vCounter + 3];
            mesh->vertices[vCounter + 13] = mesh->vertices[vCounter + 4];
            mesh->vertices[vCounter + 14] = mesh->vertices[vCounter + 5];

            mesh->vertices[vCounter + 15] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 16] = heightMap[(x + 1) + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 17] = (float)(z + 1)*scaleFactor.z;
            vCounter += 18;     // 6 vertex, 18 floats

            // Fill texcoords array with data
            //--------------------------------------------------------------
            mesh->texcoords[tcCounter] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 1] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 2] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 3] = (float)(z + 1)/(mapZ - 1);

            mesh->texcoords[tcCounter + 4] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 5] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 6] = mesh->texcoords[tcCounter + 4];
            mesh->texcoords[tcCounter + 7] = mesh->texcoords[tcCounter + 5];

            mesh->texcoords[tcCounter + 8] = mesh->texcoords[tcCounter + 2];
            mesh->texcoords[tcCounter + 9] = mesh->texcoords[tcCounter + 3];

            mesh->texcoords[tcCounter + 10] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 11] = (float)(z + 1)/(mapZ - 1);
            tcCounter += 12;    // 6 texcoords, 12 floats

            // Fill normals array with data
            //--------------------------------------------------------------
            for (int i = 0; i < 18; i += 9)
            {
                vA.x = mesh->vertices[nCounter + i];
                vA.y = mesh->vertices[nCounter + i + 1];
                vA.z = mesh->vertices[nCounter + i + 2];

                vB.x = mesh->vertices[nCounter + i + 3];
                vB.y = mesh->vertices[nCounter + i + 4];
                vB.z = mesh->vertices[nCounter + i + 5];

                vC.x = mesh->vertices[nCounter + i + 6];
                vC.y = mesh->vertices[nCounter + i + 7];
                vC.z = mesh->vertices[nCounter + i + 8];

                vN = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(vB, vA), Vector3Subtract(vC, vA)));

                mesh->normals[nCounter + i] = vN.x;
                mesh->normals[nCounter + i + 1] = vN.y;
                mesh->normals[nCounter + i + 2] = vN.z;

                mesh->normals[nCounter + i + 3] = vN.x;
                mesh->normals[nCounter + i + 4] = vN.y;
                mesh->normals[nCounter + i + 5] = vN.z;

                mesh->normals[nCounter + i + 6] = vN.x;
                mesh->normals[nCounter + i + 7] = vN.y;
                mesh->normals[nCounter + i + 8] = vN.z;
            }

            nCounter += 18;     // 6 vertex, 18 floats
        }
    }

    // Upload vertex data to GPU (static mesh)
    UpdateMeshBuffer(*mesh, 0, mesh->vertices, mesh->vertexCount * 3 * sizeof(float), 0); // Update vertices
    UpdateMeshBuffer(*mesh, 2, mesh->normals, mesh->vertexCount * 3 * sizeof(float), 0); // Update normals
}


// Required function. Should handle creating the client on first call
void c_render(School* env) {
    if (env->client == NULL) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(env->width, env->height, "PufferLib School");
        SetTargetFPS(30);
        Client* client = (Client*)calloc(1, sizeof(Client));
        env->client = client;
        client->ship = LoadModel("glider.glb");
        client->ground = LoadModelFromMesh(GenMeshCube(1.0f, 1.0f, 1.0f));
        //env->client->ship = LoadModel("resources/puffer.glb");

        Camera3D camera = { 0 };
                                                           //
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
        camera.fovy = 45.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
        camera.position = (Vector3){ 0, 3*env->size_y, -3*env->size_z};
        camera.target = (Vector3){ 0, 0, 0};
        client->camera = camera;

        client->mesh = create_heightmap_mesh(env->terrain, (Vector3){env->terrain_width, 1, env->terrain_height});
        client->model = LoadModelFromMesh(*client->mesh);
        update_heightmap_mesh(client->mesh, env->terrain, (Vector3){env->terrain_width, 1, env->terrain_height});

        client->terrain_shader = LoadShader(
            TextFormat("resources/terraform/shader_%i.vs", GLSL_VERSION),
            TextFormat("resources/terraform/shader_%i.fs", GLSL_VERSION)
        );

        Image img = GenImageColor(env->terrain_width, env->terrain_height, WHITE);
        ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        client->terrain_texture = LoadTextureFromImage(img);
        UnloadImage(img);

        client->terrain_shader_loc = GetShaderLocation(client->terrain_shader, "terrain");
        SetShaderValueTexture(client->terrain_shader, client->terrain_shader_loc, client->terrain_texture);

        client->terrain_data = calloc(4*env->terrain_width*env->terrain_height, sizeof(unsigned char));
        for (int i = 0; i < env->terrain_width*env->terrain_height; i++) {
            client->terrain_data[4*i] = env->terrain[i];
            client->terrain_data[4*i+3] = 255;
        }
        UpdateTexture(client->terrain_texture, client->terrain_data);
        SetShaderValueTexture(client->terrain_shader, client->terrain_shader_loc, client->terrain_texture);

        int shader_width_loc = GetShaderLocation(client->terrain_shader, "width");
        SetShaderValue(client->terrain_shader, shader_width_loc, &env->terrain_width, SHADER_UNIFORM_INT);

        int shader_height_loc = GetShaderLocation(client->terrain_shader, "height");
        SetShaderValue(client->terrain_shader, shader_height_loc, &env->terrain_height, SHADER_UNIFORM_INT);
 
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Client* client = env->client;
    //UpdateCamera(&client->camera, CAMERA_ORBITAL);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(client->camera);

        //BeginShaderMode(client->terrain_shader);
        client->model.materials[0].shader = client->terrain_shader;
        Vector3 pos = {-env->size_x, -env->size_y, -env->size_z};
        DrawModel(client->model, pos, 1.0/128.0f, (Color){156, 50, 20, 255});
        //EndShaderMode();


        for (int f=0; f<env->num_factories; f++) {
            Entity* factory = &env->factories[f];
            DrawSphere((Vector3){factory->x, factory->y, factory->z}, 0.01, COLORS[factory->item]);
        }

        for (int i=0; i<env->num_agents; i++) {
            Entity* agent = &env->agents[i];
            /*
            DrawLine3D(
                (Vector3){agent->x, agent->y, agent->z},
                (Vector3){agent->x + agent->vx, agent->y + agent->vy, agent->z + agent->vz},
                COLORS[agent->item]
            );
            DrawSphere((Vector3){agent->x, agent->y, agent->z}, 0.01, COLORS[agent->item]);
            */

            /*
            Matrix transform = {
                1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy), agent->x,
                2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx), agent->y,
                2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2), agent->z,
                0, 0, 0, 1
            };
            */
            Vector3 pos = {agent->x, agent->y, agent->z};
            Vector3 look = {agent->orientation.x, agent->orientation.y, agent->orientation.z};
            Vector3 up = {0.0f, 1.0f, 0.0f};
            client->ship.transform = MatrixLookAt(pos, look, up);
            client->ground.transform = MatrixLookAt(pos, look, up);

            float v_norm = sqrtf(agent->vx*agent->vx + agent->vy*agent->vy + agent->vz*agent->vz);
            float xx = agent->vx / v_norm;
            float yy = agent->vy / v_norm;
            float zz = agent->vz / v_norm;
            float pitch = asinf(-yy);
            float yaw = atan2f(xx, zz);

            //Vector3 angle = {agent->pitch, agent->yaw, agent->roll};
            Vector3 angle = {pitch, yaw, 0.0f};

            /*
            rlPushMatrix();
            rlTranslatef(0.0f, 0.0f, 0.0f);
            rlRotatef(-90.0f, 0, 0, 90.0f);
            DrawModel(env->client->ship, (Vector3){0, 0, 0}, 1.0f, WHITE);
            rlPopMatrix();
            */

            client->ship.transform = MatrixRotateXYZ(angle);
            client->ground.transform = MatrixRotateXYZ(angle);

            Vector3 scale;
            Model model;
            if (agent->unit == DRONE) {
                scale = (Vector3){0.01f, 0.01f, 0.01f};
                model = client->ship;
            } else if (agent->unit == FIGHTER) {
                scale = (Vector3){0.01f, 0.025f, 0.025f};
                model = client->ship;
            } else if (agent->unit == MOTHERSHIP) {
                scale = (Vector3){0.05f, 0.05f, 0.05f};
                model = client->ship;
            } else if (agent->unit == BOMBER) {
                scale = (Vector3){0.025f, 0.025f, 0.01f};
                model = client->ship;
            } else if (agent->unit == INFANTRY) {
                scale = (Vector3){0.01f, 0.01f, 0.01f};
                model = client->ground;
            } else if (agent->unit == TANK) {
                scale = (Vector3){0.025f, 0.025f, 0.01f};
                model = client->ground;
            } else if (agent->unit == ARTILLERY) {
                scale = (Vector3){0.025f, 0.015f, 0.025f};
                model = client->ground;
            } else {
                scale = (Vector3){0.05f, 0.05f, 0.05f};
                model = client->ground;
            }
            Color color = COLORS[agent->item];
            Vector3 rot = {0.0f, 1.0f, 0.0f};
            DrawModelEx(model, pos, rot, 0, scale, color);

            if (agent->target >= 0) {
                Entity* target = &env->agents[agent->target];
                DrawLine3D(
                    (Vector3){agent->x, agent->y, agent->z},
                    (Vector3){target->x, target->y, target->z},
                    COLORS[agent->item]
                );
            }
        }

        DrawCubeWires(
            (Vector3){0, 0, 0},
            2*env->size_x, 2*env->size_y, 2*env->size_z,
            (Color){0, 255, 255, 128}
        );

    EndMode3D();
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(School* env) {
    free(env->agents);
    free(env->factories);
    if (env->client != NULL) {
        Client* client = env->client;
        //UnloadTexture(client->sprites);
        CloseWindow();
        free(client);
    }
}
