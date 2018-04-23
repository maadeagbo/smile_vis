// level script for smile_vis cpp implementations
#include "ddLevelPrototype.h"
#include "smile_vis_graphics.h"
#include "svis_shader_enums.h"

// log lua function that can be called in scripts thru this function
void smile_vis_func_register(lua_State *L);

// Proxy struct that enables reflection
struct smile_vis_reflect : public ddLvlPrototype {
  smile_vis_reflect() {
    add_lua_function("smile_vis", smile_vis_func_register);
  }
};

void smile_vis_func_register(lua_State *L) {
  // log functions using register_callback_lua
  register_callback_lua(L, "load_graphics", init_gpu_structures);
  register_callback_lua(L, "smile_UI", load_ui);
}

// log reflection
smile_vis_reflect smile_vis_proxy;
