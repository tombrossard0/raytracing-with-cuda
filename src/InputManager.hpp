#pragma once

#include <SDL2/SDL.h>
#include <array>

#include "vec2.cuh"

class InputManager {
    void update();

    bool isKeyDown(SDL_Scancode code) const { return keys[code]; }
    int getMouseWheel() const { return mouseWheel; }

  private:
    std::array<bool, SDL_NUM_SCANCODES> keys{};
    Vec2 mouseDelta;
    int mouseWheel;
};
