// level script for smile_vis cpp implementations
#include "ddLevelPrototype.h"
#include "smile_vis_graphics.h"
#include "svis_shader_enums.h"

// manipulatible frame data
FrameData frames[2];

// log lua function that can be called in scripts thru this function
void smile_vis_func_register(lua_State *L);

// initilize data strutures for level
void init_data();

// Proxy struct that enables reflection
struct smile_vis_reflect : public ddLvlPrototype {
  smile_vis_reflect() {
    add_lua_function("smile_vis", smile_vis_func_register);

    init_data();
  }
};

void smile_vis_func_register(lua_State *L) {
  // log functions using register_callback_lua
  register_callback_lua(L, "load_graphics", init_gpu_structures);
}

void init_data() {
  // initilize frames
  for (unsigned i = 0; i < 2; i++) {
    frames[i].verts[0] = glm::vec3(-1.0, 1.0, 0.0);
    frames[i].texcoords[0] = glm::vec2(0.0, 1.0);
    frames[i].verts[1] = glm::vec3(-1.0, -1.0, 0.0);
    frames[i].texcoords[1] = glm::vec2(0.0, 0.0);
    frames[i].verts[2] = glm::vec3(1.0, -1.0, 0.0);
    frames[i].texcoords[2] = glm::vec2(1.0, 0.0);
    frames[i].verts[3] = glm::vec3(1.0, 1.0, 0.0);
    frames[i].texcoords[3] = glm::vec2(1.0, 1.0);
  }
}

// log reflection
smile_vis_reflect smile_vis_proxy;
