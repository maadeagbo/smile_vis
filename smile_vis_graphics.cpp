#include "smile_vis_graphics.h"

#define MAX_LINES 4
#define MAX_INDICES 8

namespace {
  // Particle engine draw
  ddPTask pt_draw_fdata;

  // shader for lines and points
  ddShader linedot_sh;

  // line buffers
  ddVAOData *line_vao = nullptr;
  ddStorageBufferData *line_ssbo = nullptr;
  ddIndexBufferData *line_ebo = nullptr;
  dd_array<unsigned> l_indices;
  dd_array<glm::vec3> l_points;

  // point buffers
  ddVAOData *point_vao = nullptr;
  ddStorageBufferData *corner4_ssbo = nullptr;
  ddStorageBufferData *frame_data_ssbo = nullptr;
}

void SVDraw::init_gpu_structures() {
  // shader init
  cbuff<256> fname;
  linedot_sh.init();
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "LineDot_V.vert");
  linedot_sh.create_vert_shader(fname.str());
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "LineDot_F.frag");
  linedot_sh.create_frag_shader(fname.str());

  // line structures
  ddGPUFrontEnd::create_vao(line_vao);
  ddGPUFrontEnd::create_storage_buffer(line_ssbo,
                                       MAX_LINES * sizeof(float));
  ddGPUFrontEnd::create_index_buffer(line_ebo,
                                     MAX_INDICES * sizeof(unsigned),
                                     &l_indices[0]);
  ddGPUFrontEnd::bind_storage_buffer_atrribute(
      line_vao, line_ssbo, ddAttribPrimitive::FLOAT, 0,
      3, 3 * sizeof(float), 0);
  ddGPUFrontEnd::set_storage_buffer_contents(line_ssbo,
                                             MAX_LINES * sizeof(float), 0,
                                             &l_points[0]);
  ddGPUFrontEnd::bind_index_buffer(line_vao, line_ebo);

  // point structures
  ddGPUFrontEnd::create_vao(point_vao);
  ddGPUFrontEnd::create_storage_buffer(corner4_ssbo, 4 * sizeof(float));
}