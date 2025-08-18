#pragma once

#include <SDL2/SDL.h>
#include <array>
#include <imgui_impl_sdl2.h>

#include "vec2.cuh"

struct MouseButtonState {
    bool pressed = false;
    bool justPressed = false;
    bool justReleased = false;
};

struct MouseState {
    MouseButtonState left, right, middle, x1, x2;
    int x = 0, y = 0;   // position
    int dx = 0, dy = 0; // delta since last frame
    int wheel = 0;

    void reset() {
        // reset only transient states every frame
        left.justPressed = left.justReleased = false;
        right.justPressed = right.justReleased = false;
        middle.justPressed = middle.justReleased = false;
        x1.justPressed = x1.justReleased = false;
        x2.justPressed = x2.justReleased = false;
        dx = dy = 0;
        wheel = 0;
    }

    void handleEvent(const SDL_Event &e) {
        switch (e.type) {
        case SDL_MOUSEBUTTONDOWN:
            setButton(e.button.button, true);
            break;
        case SDL_MOUSEBUTTONUP:
            setButton(e.button.button, false);
            break;
        case SDL_MOUSEMOTION:
            x = e.motion.x;
            y = e.motion.y;
            dx = e.motion.xrel;
            dy = e.motion.yrel;
            break;
        case SDL_MOUSEWHEEL:
            wheel = e.wheel.y;
            break;
        }
    }

  private:
    void setButton(uint8_t button, bool down) {
        // Map the correct button state automatically
        MouseButtonState *b = nullptr;
        if (button == SDL_BUTTON_LEFT)
            b = &left;
        else if (button == SDL_BUTTON_RIGHT)
            b = &right;
        else if (button == SDL_BUTTON_MIDDLE)
            b = &middle;
        else if (button == SDL_BUTTON_X1)
            b = &x1;
        else if (button == SDL_BUTTON_X2)
            b = &x2;

        if (!b) return;

        if (down && !b->pressed) b->justPressed = true;
        if (!down && b->pressed) b->justReleased = true;

        b->pressed = down;
    }
};

class InputManager {
  public:
    MouseState mouse;

    void update(bool &running);
    bool isKeyDown(SDL_Scancode code) const { return keys[code]; }

  private:
    std::array<bool, SDL_NUM_SCANCODES> keys{};
};
