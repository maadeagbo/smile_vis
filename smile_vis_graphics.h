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
int init_gpu_structures(lua_State *L);

/** \brief Modify scene data */
int modify_scene_data(lua_State *L);

/** \brief Updates FrameData for gpu */
void update_frame(const FrameData *frame1, const FrameData *frame2 = nullptr);