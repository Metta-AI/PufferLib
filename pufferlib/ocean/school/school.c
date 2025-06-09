/* Pure C demo file for School. Build it with:
 * bash scripts/build_ocean.sh school local (debug)
 * bash scripts/build_ocean.sh school fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "school.h"

/* Puffernet is our lightweight cpu inference library that
 * lets you load basic PyTorch model architectures so that
 * you can run them in pure C or on the web via WASM
 */
#include "puffernet.h"

int main() {
    // Weights are exported by running puffer export
    //Weights* weights = load_weights("resources/puffer_school_weights.bin", 137743);

    //int logit_sizes[2] = {9, 5};
    //LinearLSTM* net = make_linearlstm(weights, num_agents, num_obs, logit_sizes, 2);

    School env = {
        .width = 1980,
        .height = 1020,
        .size_x = 2,
        .size_y = 0.5,
        .size_z = 1,
        .num_agents = 1024,
        .num_factories = 4,
        .num_resources = 4,
    };
    init(&env);

    // Allocate these manually since they aren't being passed from Python
    int num_obs = 3*env.num_resources + 14 + env.num_resources;
    env.observations = calloc(env.num_agents*num_obs, sizeof(float));
    env.actions = calloc(3*env.num_agents, sizeof(int));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    int ctrl = 0;

    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            Entity* agent = &env.agents[i];
            int item = agent->item;
            float vx = env.observations[num_obs*i + 3*item];
            float vy = env.observations[num_obs*i + 3*item + 1];
            float vz = env.observations[num_obs*i + 3*item + 2];
            float yaw = env.observations[num_obs*i + 3*item + 3];
            float pitch = env.observations[num_obs*i + 3*item + 4];
            float roll = env.observations[num_obs*i + 3*item + 5];
            float x = env.observations[num_obs*i + 3*item + 6];
            float y = env.observations[num_obs*i + 3*item + 7];
            float z = env.observations[num_obs*i + 3*item + 8];

            if (agent->unit == INFANTRY || agent->unit == TANK || agent->unit == ARTILLERY) {
                env.actions[3*i] = (vx > 0.0f) ? 6 : 2;
                env.actions[3*i + 2] = (vz > 0.0f) ? 6 : 2;
            } else {
                float desired_pitch = atan2f(-y, sqrt(x*x + z*z));
                float pitch_error = desired_pitch - pitch;
                if (pitch_error > 0) {
                    env.actions[3*i] = 6; // pitch up
                } else if (pitch_error < 0) {
                    env.actions[3*i] = 2; // pitch down
                }
                env.actions[3*i] = 4;

                env.actions[3*i + 1] = 4;

                // Roll control
                float desired_yaw = atan2f(-x, -z); // Direction to origin
                float current_yaw = atan2f(vx, vz); // Current velocity direction
                float yaw_error = desired_yaw - yaw;

                // Normalize yaw_error to [-PI, PI]
                if (yaw_error > PI) yaw_error -= 2*PI;
                if (yaw_error < -PI) yaw_error += 2*PI;

                //printf("%f %f\n", yaw_error, yaw);

                if (yaw_error > 0.1f) {
                    env.actions[3*i + 1] = 2; // roll left
                } else if (yaw_error < -0.1f) {
                    env.actions[3*i + 1] = 6; // roll right
                } else {
                    env.actions[3*i + 1] = 4; // neutral roll (assuming 0 is valid)
                }

            }
            //float dpitch = atan2f(dz, sqrtf(dx*dx + dy*dy));
            //float droll = asinf(dz/sqrtf(dx*dx + dy*dy + dz*dz));
            //env.actions[3*i] = 6;
            //env.actions[3*i + 1] = (dpitch > 0.0f) ? 6 : 2;
            //env.actions[3*i + 2] = (droll > 0.0f) ? 6 : 2;
            //env.actions[3*i] = rand() % 9;
            //env.actions[3*i + 1] = rand() % 9;
            //env.actions[3*i + 2] = rand() % 9;
            //env.actions[3*i] = 4.0f;
            //env.actions[3*i + 1] = 4.0f;
            //env.actions[3*i + 2] = 4.0f;
        }

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyPressed(KEY_TAB)) {
                ctrl = (ctrl + 1) % env.num_agents;
            }
            int i = ctrl;
            float vx = env.observations[num_obs*i + 3*env.num_resources];
            float vy = env.observations[num_obs*i + 3*env.num_resources + 1];
            float vz = env.observations[num_obs*i + 3*env.num_resources + 2];
            float x = env.observations[num_obs*i + 3*env.num_resources + 6];
            float y = env.observations[num_obs*i + 3*env.num_resources + 7];
            float z = env.observations[num_obs*i + 3*env.num_resources + 8];

            Camera3D* camera = &(env.client->camera);
            camera->target = (Vector3){x, y, z};

            float dd = sqrtf(vx*vx + vy*vy + vz*vz);
            float forward_x = vx / dd;
            float forward_y = vy / dd;
            float forward_z = vz / dd;

            float dist = 0.5f;
            camera->position = (Vector3){
                x - dist*forward_x,
                y - dist*forward_y + 0.5f,
                z - dist*forward_z
            };

            env.actions[3*i] = 4;
            if (IsKeyDown(KEY_W)) {
                env.actions[3*i] = 6;
            } else if (IsKeyDown(KEY_S)) {
                env.actions[3*i] = 2;
            }

            env.actions[3*i + 1] = 4;
            if (IsKeyDown(KEY_A)) {
                env.actions[3*i + 1] = 6;
            } else if (IsKeyDown(KEY_D)) {
                env.actions[3*i + 1] = 2;
            }
        }


        //forward_linearlstm(net, env.observations, env.actions);
        compute_observations(&env);
        c_step(&env);
        c_render(&env);
    }

    // Try to clean up after yourself
    //free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

