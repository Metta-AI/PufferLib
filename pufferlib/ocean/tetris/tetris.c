#include <time.h>
#include "tetris.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

void demo() {
    Tetris env = {
        .n_rows = 20,
        .n_cols = 10,
        .deck_size=3,
    };
    allocate(&env);
    env.client = make_client(&env);
	SetTargetFPS(10);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)){
            env.actions[0] = 1;
        }
        if (IsKeyDown(KEY_RIGHT)  || IsKeyDown(KEY_D)){
            env.actions[0] = 2;
        }
        if (IsKeyPressed(KEY_R)){
            env.actions[0] = 3;
        }
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
            env.actions[0] = 4;
        }
        if (IsKeyPressed(KEY_ENTER)) {
            env.actions[0] = 5;
        }
        if (IsKeyPressed(KEY_C)) {
            env.actions[0] = 6;
        }
        c_step(&env);
        c_render(&env);
        TraceLog(LOG_INFO, "Reward: %f", env.rewards[0]);

    }
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
