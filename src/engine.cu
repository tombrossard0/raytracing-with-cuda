#include "engine.cuh"

void Engine::initWindow() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to init SDL: " << SDL_GetError() << std::endl;
        throw std::runtime_error("SDL_Init failed");
    }

    this->window =
        SDL_CreateWindow("CUDA Raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, this->window_width,
                         this->window_height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    if (!this->window) {
        std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        throw std::runtime_error("SDL_CreateWindow failed");
    }
}

void Engine::initContext() {
    this->sdl_gl_context = SDL_GL_CreateContext(this->window);
    if (!sdl_gl_context) {
        std::cerr << "Failed to create OpenGL context: " << SDL_GetError() << std::endl;
        throw std::runtime_error("SDL_GL_CreateContext failed");
    }

    SDL_GL_MakeCurrent(this->window, sdl_gl_context);
    SDL_GL_SetSwapInterval(1); // vsync
}

Engine::Engine(int w, int h) : window_width(w), window_height(h) {
    this->initWindow();
    this->initContext();
}

Engine::~Engine() {
    SDL_GL_DeleteContext(this->sdl_gl_context);
    SDL_DestroyWindow(this->window);
    SDL_Quit();
}
