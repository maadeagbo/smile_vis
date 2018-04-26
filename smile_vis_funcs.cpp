// level script for smile_vis cpp implementations
#include "ddLevelPrototype.h"
#include "smile_vis_graphics.h"
#include "svis_shader_enums.h"

// log lua function that can be called in scripts thru this function
void smile_vis_func_register(lua_State *L);

/** \brief Load new smile data */
int load_smile_data(lua_State *L);

/** \brief Log smile data ground truth */
int log_data_groundtruth(lua_State *L);

/** \brief Log smile data weights and biases truth */
int log_data_weights_biases(lua_State *L);

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
  register_callback_lua(L, "load_folder", load_smile_data);
  register_callback_lua(L, "groundtruth_folder", log_data_groundtruth);
  register_callback_lua(L, "w_b_folders", log_data_weights_biases);

  register_lua_controller(L);
}

int load_smile_data(lua_State *L) {
  // argument contains location of new folder to list files
  const char *directory = luaL_checkstring(L, 1);

  load_files(directory);

  return 0;
}

int log_data_groundtruth(lua_State *L) {
  // argument contains location of new folder to list files
  const char *directory = luaL_checkstring(L, 1);

  load_files(directory, true);

  return 0;
}

int log_data_weights_biases(lua_State *L) {
  // argument contains location of new folder to list files
  const char *directory1 = luaL_checkstring(L, 1);
  const char *directory2 = luaL_checkstring(L, 2);

  load_weights(directory1);
  load_biases(directory2);

  return 0;
}

// log reflection
smile_vis_reflect smile_vis_proxy;
