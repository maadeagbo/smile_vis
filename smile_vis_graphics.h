#pragma once

#include "ddParticleSystem.h"
#include "ddSceneManager.h"
#include "ddShader.h"

/** \brief Data representation of smile feature points */
struct FrameData {
  dd_array<glm::vec2> input;
};

namespace SVDraw {
  /** \brief Create structures for rendering */
  void init_gpu_structures();

  /** \brief Draws FrameData within bounds of min and max 2D coordinates*/
  void draw_frame(glm::vec2 min_vert, glm::vec2 max_vert, const FrameData frame);
}