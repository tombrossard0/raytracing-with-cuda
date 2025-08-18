#include "engine.hpp"

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

void Engine::uploadFbToTexture(Scene &scene) {
    glBindTexture(GL_TEXTURE_2D, scene.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, scene.width, scene.height, GL_RGB, GL_FLOAT, scene.fb);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Engine::clearScreen() {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

void Engine::renderImGui(Scene *scene) {
    ImGui_ImplSDL2_NewFrame();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New")) {}
            if (ImGui::MenuItem("Open...")) {}
            if (ImGui::MenuItem("Exit")) { this->running = false; }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (scene) { scene->renderGUI(scene->texture); }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

Engine::Engine(int w, int h)
    : window_width(w), window_height(h), mouse({false, 0, 0, 0.2f}), running(true),
      currentTime(SDL_GetTicks64()), deltaTime(0), lastFrameTime(currentTime), lastFPSTime(lastFrameTime),
      frameCount(0), fps(0.0f) {
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

void Engine::updateTime() {
    this->currentTime = SDL_GetTicks64();
    this->deltaTime = (this->currentTime - this->lastFrameTime) / 1000.0f; // seconds
    this->lastFrameTime = this->currentTime;
}

void Engine::computeFPS() {
    this->frameCount++;
    if (this->currentTime - this->lastFPSTime >= 1000) { // every second
        this->fps = this->frameCount * 1000.0f / (this->currentTime - this->lastFPSTime);
        this->frameCount = 0;
        this->lastFPSTime = this->currentTime;

        std::string title = "CUDA Raytracer - FPS: " + std::to_string((int)this->fps);
        SDL_SetWindowTitle(this->window, title.c_str());
    }
}

void Engine::processInputs(Scene *scene) {
    const Uint8 *keystate = SDL_GetKeyboardState(NULL);

    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT) running = false;

        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
            mouse.mouseLook = true;
            SDL_GetMouseState(&mouse.lastMouseX, &mouse.lastMouseY);
            SDL_SetRelativeMouseMode(SDL_TRUE);
        }
        if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_RIGHT) {
            mouse.mouseLook = false;
            SDL_SetRelativeMouseMode(SDL_FALSE);
        }
        if (scene && event.type == SDL_MOUSEMOTION && mouse.mouseLook) {
            int dx = event.motion.xrel;
            int dy = -event.motion.yrel;
            scene->yawDeg += dx * mouse.sensitivity;
            scene->pitchDeg += dy * mouse.sensitivity;
            if (scene->pitchDeg > 89.0f) scene->pitchDeg = 89.0f;
            if (scene->pitchDeg < -89.0f) scene->pitchDeg = -89.0f;
        }
        if (scene && event.type == SDL_MOUSEWHEEL) {
            scene->radius -= event.wheel.y * 0.5f;
            if (scene->radius < scene->minRadius) scene->radius = scene->minRadius;
            if (scene->radius > scene->maxRadius) scene->radius = scene->maxRadius;
        }
        if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) { running = false; }
    }

    if (scene) {
        float speed = 10.5f * deltaTime; // movement speed
        float yawRad = scene->yawDeg * M_PI / 180.0f;
        float pitchRad = scene->pitchDeg * M_PI / 180.0f;

        // Forward vector
        Vec3 forward(cosf(pitchRad) * cosf(yawRad), sinf(pitchRad), cosf(pitchRad) * sinf(yawRad));
        forward = forward.normalize();

        // Right vector
        Vec3 right = forward.cross(Vec3(0, 1, 0)).normalize();

        // Up vector
        Vec3 up = right.cross(forward).normalize();

        if (keystate[SDL_SCANCODE_W]) scene->center = scene->center - forward * speed; // forward
        if (keystate[SDL_SCANCODE_S]) scene->center = scene->center + forward * speed; // backward
        if (keystate[SDL_SCANCODE_A]) scene->center = scene->center + right * speed;   // left
        if (keystate[SDL_SCANCODE_D]) scene->center = scene->center - right * speed;   // right
        if (keystate[SDL_SCANCODE_SPACE]) scene->center = scene->center - up * speed;  // up
        if (keystate[SDL_SCANCODE_LCTRL]) scene->center = scene->center + up * speed;  // down
    }
}
