#pragma once

#include "ddParticleSystem.h"
#include "ddSceneManager.h"
#include "ddShader.h"

/** \brief Data representation of smile feature points */
struct FrameData {
  glm::vec3 verts[4];
  glm::vec2 texcoords[4];
};

/** \brief Create structures for rendering */
int init_gpu_structures(lua_State* L);

/** \brief Modify scene data */
void update_frame_data(const FrameData& data);

/** \brief ImGUI ui for seeing and setting points */
int load_ui(lua_State* L);

/** \brief Sets the list of files visible in menu */
void load_files(const char* directory, const bool ground_truth = false);
