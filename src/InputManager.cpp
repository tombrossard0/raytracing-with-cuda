#include "InputManager.hpp"

void InputManager::update(bool &running) {
    mouse.reset();

    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event); // Make ImGui widgets react to user inputs
        mouse.handleEvent(event);

        if (event.type == SDL_QUIT) running = false;
        if (event.type == SDL_KEYDOWN) keys[event.key.keysym.scancode] = true;
        if (event.type == SDL_KEYUP) keys[event.key.keysym.scancode] = false;
    }
}
