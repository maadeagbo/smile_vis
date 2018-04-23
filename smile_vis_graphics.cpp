#include "smile_vis_graphics.h"
#include "ddTerminal.h"
#include "svis_shader_enums.h"

#define MAX_POINTS 4
#define MAX_INDICES 8

namespace {
// Particle engine draw
ddPTask draw_fdata;

// shader for lines and points
ddShader linedot_sh;

// line buffers
ddVAOData *line_vao = nullptr;
ddIndexBufferData *line_ebo = nullptr;
dd_array<unsigned> l_indices = dd_array<unsigned>(MAX_INDICES);
dd_array<glm::vec3> l_points = dd_array<glm::vec3>(MAX_POINTS);
dd_array<glm::vec2> l_texcoords = dd_array<glm::vec2>(MAX_POINTS);

// buffers for drawing primitives
glm::vec3 point_buff[6];
glm::vec2 texcoord_buff[6];

// point buffers
ddVAOData *point_vao = nullptr;
ddStorageBufferData *point_ssbo = nullptr;

// manipulatible frame data
FrameData frames[2];
}  // namespace

/** \brief Draws FrameData for particle task */
void draw_frame();

/** \brief initilize data strutures for level */
void init_data();

/** \brief Refills points and uvs buffer w/ indexed FrameData */
void refill_buffer(const FrameData &data);

/** \brief Set ImGUI style */
void set_imgui_style();

int init_gpu_structures(lua_State *L) {
  // indices buffer
  l_indices[0] = 0;
  l_indices[1] = 1;
  l_indices[2] = 1;
  l_indices[3] = 2;
  l_indices[4] = 2;
  l_indices[5] = 3;
  l_indices[6] = 3;
  l_indices[7] = 0;

  init_data();

	set_imgui_style();

  // shader init
  cbuff<256> fname;
  linedot_sh.init();
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "LineDot_V.vert");
  linedot_sh.create_vert_shader(fname.str());
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "LineDot_F.frag");
  linedot_sh.create_frag_shader(fname.str());

  // line structures
  ddGPUFrontEnd::create_vao(line_vao);
  ddGPUFrontEnd::create_storage_buffer(point_ssbo, l_points.sizeInBytes());
  ddGPUFrontEnd::create_index_buffer(line_ebo, l_indices.sizeInBytes(),
                                     &l_indices[0]);
  ddGPUFrontEnd::bind_storage_buffer_atrribute(line_vao, point_ssbo,
                                               ddAttribPrimitive::FLOAT, 0, 3,
                                               3 * sizeof(float), 0);
  ddGPUFrontEnd::set_storage_buffer_contents(point_ssbo, l_points.sizeInBytes(),
                                             0, &l_points[0]);
  ddGPUFrontEnd::bind_index_buffer(line_vao, line_ebo);

  // point structures
  ddGPUFrontEnd::create_vao(point_vao);

  // register particle task
  draw_fdata.lifespan = 10.f;
  draw_fdata.remain_on_q = true;
  draw_fdata.rfunc = draw_frame;

  ddParticleSys::add_task(draw_fdata);

  return 0;
}

void init_data() {
  // initilize frames (right side)
  frames[0].verts[0] = glm::vec3(-0.2, 1.0, 0.0);   // halfway
  frames[0].texcoords[0] = glm::vec2(0.4, 1.0);     // halfway
  frames[0].verts[1] = glm::vec3(-0.2, -1.0, 0.0);  // halfway
  frames[0].texcoords[1] = glm::vec2(0.4, 0.0);     // halfway
  frames[0].verts[2] = glm::vec3(1.0, -1.0, 0.0);
  frames[0].texcoords[2] = glm::vec2(1.0, 0.0);
  frames[0].verts[3] = glm::vec3(1.0, 1.0, 0.0);
  frames[0].texcoords[3] = glm::vec2(1.0, 1.0);

  update_frame_data(frames[0]);

  // left side
  frames[1].verts[0] = glm::vec3(-1.0, 1.0, 0.0);
  frames[1].texcoords[0] = glm::vec2(0.0, 1.0);
  frames[1].verts[1] = glm::vec3(-1.0, -1.0, 0.0);
  frames[1].texcoords[1] = glm::vec2(0.0, 0.0);
  frames[1].verts[2] = glm::vec3(0.0, -1.0, 0.0);  // halfway
  frames[1].texcoords[2] = glm::vec2(0.5, 0.0);    // halfway
  frames[1].verts[3] = glm::vec3(0.0, 1.0, 0.0);   // halfway
  frames[1].texcoords[3] = glm::vec2(0.5, 1.0);    // halfway
}

void update_frame_data(const FrameData &data) { refill_buffer(data); }

/** \brief Draws FrameData for gpu */
void draw_frame() {
  ddCam *cam = ddSceneManager::get_active_cam();
  const glm::mat4 identity;
  const glm::uvec2 scr_dim = ddSceneManager::get_screen_dimensions();

  if (cam) {
    // switch to separate framebuffer for render to texture
    ddGPUFrontEnd::blit_depth_buffer(ddBufferType::PARTICLE, ddBufferType::XTRA,
                                     scr_dim.x, scr_dim.y);
    ddGPUFrontEnd::bind_framebuffer(ddBufferType::XTRA);
    ddGPUFrontEnd::clear_color_buffer();

    // get camera matrices & activate shader
    const glm::mat4 v_mat = ddSceneManager::calc_view_matrix(cam);
    const glm::mat4 p_mat = ddSceneManager::calc_p_proj_matrix(cam);
    linedot_sh.use();

    // render frame cutout (right side) ****************************************
    // draw feature points
    linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, p_mat * v_mat);
    linedot_sh.set_uniform((int)RE_LineDot::color_v4, glm::vec4(1.f));
    linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, false);
    ddGPUFrontEnd::render_quad();

    // render the background
    linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, identity);
    linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, true);
    linedot_sh.set_uniform((int)RE_LineDot::color_v4,
                           glm::vec4(0.5f, 0.5f, 0.5f, 1.f));
    ddGPUFrontEnd::render_quad();
    linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, false);

    // bind Particle frame buffer & bind texture from last draw calls
    ddGPUFrontEnd::bind_framebuffer(ddBufferType::PARTICLE);
    linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, true);
    ddGPUFrontEnd::bind_pass_texture(ddBufferType::XTRA, 0);
    linedot_sh.set_uniform((int)RE_LineDot::bound_tex_smp2d, 0);

    // render right side cutout
    refill_buffer(frames[0]);
    ddGPUFrontEnd::render_primitive(6, point_buff, texcoord_buff);
    // render border
    linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, false);
    linedot_sh.set_uniform((int)RE_LineDot::color_v4, glm::vec4(1.f));
    ddGPUFrontEnd::draw_indexed_lines_vao(line_vao, l_indices.size(), 0);
  }
}

void refill_buffer(const FrameData &data) {
  for (unsigned i = 0; i < MAX_POINTS; i++) {
    l_points[i] = data.verts[i];
    l_texcoords[i] = data.texcoords[i];
  }
  // set buffers for gpu render
  point_buff[0] = l_points[0];
  texcoord_buff[0] = l_texcoords[0];
  point_buff[1] = l_points[1];
  texcoord_buff[1] = l_texcoords[1];
  point_buff[2] = l_points[2];
  texcoord_buff[2] = l_texcoords[2];

  point_buff[3] = l_points[0];
  texcoord_buff[3] = l_texcoords[0];
  point_buff[4] = l_points[2];
  texcoord_buff[4] = l_texcoords[2];
  point_buff[5] = l_points[3];
  texcoord_buff[5] = l_texcoords[3];
}

void set_imgui_style() {
  ImGuiStyle *style = &ImGui::GetStyle();

  style->WindowPadding = ImVec2(15, 15);
  style->WindowRounding = 5.0f;
  style->FramePadding = ImVec2(5, 5);
  style->FrameRounding = 4.0f;
  style->ItemSpacing = ImVec2(12, 8);
  style->ItemInnerSpacing = ImVec2(8, 6);
  style->IndentSpacing = 25.0f;
  style->ScrollbarSize = 15.0f;
  style->ScrollbarRounding = 9.0f;
  style->GrabMinSize = 5.0f;
  style->GrabRounding = 3.0f;

  style->Colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.83f, 1.00f);
  style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
  style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
  style->Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
  style->Colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
  style->Colors[ImGuiCol_Border] = ImVec4(0.80f, 0.80f, 0.83f, 0.88f);
  style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.92f, 0.91f, 0.88f, 0.00f);
  style->Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
  style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
  style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
  style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 0.98f, 0.95f, 0.75f);
  style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
  style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
  style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
  style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
  style->Colors[ImGuiCol_ScrollbarGrabHovered] =
      ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_ScrollbarGrabActive] =
      ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
  style->Colors[ImGuiCol_ComboBg] = ImVec4(0.19f, 0.18f, 0.21f, 1.00f);
  style->Colors[ImGuiCol_CheckMark] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
  style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
  style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
  style->Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
  style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
  style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_Header] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
  style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
  style->Colors[ImGuiCol_Column] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_ColumnHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
  style->Colors[ImGuiCol_ColumnActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  style->Colors[ImGuiCol_ResizeGripHovered] =
      ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
  style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
  style->Colors[ImGuiCol_CloseButton] = ImVec4(0.40f, 0.39f, 0.38f, 0.16f);
  style->Colors[ImGuiCol_CloseButtonHovered] =
      ImVec4(0.40f, 0.39f, 0.38f, 0.39f);
  style->Colors[ImGuiCol_CloseButtonActive] =
      ImVec4(0.40f, 0.39f, 0.38f, 1.00f);
  style->Colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
  style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
  style->Colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
  style->Colors[ImGuiCol_PlotHistogramHovered] =
      ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
  style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.25f, 1.00f, 0.00f, 0.43f);
  style->Colors[ImGuiCol_ModalWindowDarkening] =
      ImVec4(1.00f, 0.98f, 0.95f, 0.73f);
}

int load_ui(lua_State *L) {
  bool win_on = true;

  // window position and size
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  const glm::uvec2 s_d = ddSceneManager::get_screen_dimensions();
  ImGui::SetNextWindowSize(ImVec2((float)s_d.x * 0.399, (float)s_d.y));

  // window flags
  ImGuiWindowFlags window_flags = 0;
  // window_flags |= ImGuiWindowFlags_MenuBar;
  window_flags |= ImGuiWindowFlags_NoMove;
  window_flags |= ImGuiWindowFlags_NoResize;
  window_flags |= ImGuiWindowFlags_NoCollapse;

  ImGui::Begin("Smile Visualization", &win_on, window_flags);

  ImGui::End();

  return 0;
}