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

void Engine::initImGUI() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(this->window, this->sdl_gl_context);
    ImGui_ImplOpenGL3_Init("#version 330");
}

// --- OpenGL texture for CUDA framebuffer ---
GLuint Engine::createTexture(int w, int h) {
    GLuint texture;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}

Engine::Engine(int w, int h)
    : window_width(w), window_height(h), mouse({false, 0, 0, 0.2f}), running(true),
      lastFrameTime(SDL_GetTicks()), lastFPSTime(lastFrameTime), frameCount(0), fps(0.0f) {
    this->mouse = {false, 0, 0, 0.2f};

    this->initWindow();
    this->initContext();
    this->initImGUI();
}

Engine::~Engine() {
    // --- Cleanup Imgui ---
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    // --- Cleanup Context ---
    SDL_GL_DeleteContext(this->sdl_gl_context);

    // --- Cleanup Window ---
    SDL_DestroyWindow(this->window);
    SDL_Quit();
}
