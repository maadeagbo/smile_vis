#include "smile_vis_graphics.h"
#include "ddFileIO.h"
#include "ddTerminal.h"
#include "smile_vis_data.h"
#include "svis_shader_enums.h"

#define MAX_POINTS 4
#define MAX_INDICES 8

struct SController {
  unsigned curr_idx = 0;
  unsigned num_frames = 0;
  float tile_size = 5.f;
  glm::ivec4 ortho_params = glm::ivec4(0, 1920, 1080, 0);
  dd_array<glm::vec3> _input;
  dd_array<glm::vec3> _ground;
  dd_array<glm::vec3> _predicted;
};

namespace {
// Particle engine draw
ddPTask draw_fdata;

// shader for lines and points
ddShader linedot_sh;
ddShader point_sh;

// line buffers
ddVAOData *line_vao = nullptr;
ddIndexBufferData *line_ebo = nullptr;
ddStorageBufferData *line_ssbo = nullptr;
dd_array<unsigned> l_indices = dd_array<unsigned>(MAX_INDICES);
dd_array<glm::vec3> l_points = dd_array<glm::vec3>(MAX_POINTS);
dd_array<glm::vec2> l_texcoords = dd_array<glm::vec2>(MAX_POINTS);

// structures for tracking useable files
int selected_file = 0;
cbuff<512> f_dir;
cbuff<512> gd_dir;
dd_array<cbuff<512>> files;
dd_array<cbuff<64>> file_names;
dd_array<const char *> file_names_ptr;

// buffers for drawing primitives
glm::vec3 point_buff[6];
glm::vec2 texcoord_buff[6];

// point buffers
ddVAOData *point_vao = nullptr;
ddStorageBufferData *point_ssbo = nullptr;

// manipulatible frame data
FrameData frames[2];
SController sctrl;

// data points of current visualization
std::vector<Eigen::VectorXd> input_p;

// data points of ground truth
std::vector<Eigen::VectorXd> groundtr_p;

// weights and biases
std::vector<Eigen::MatrixXd> weights;
std::vector<Eigen::VectorXd> biases;

dd_array<int64_t> i64_bin = dd_array<int64_t>(4);
}  // namespace

//******************************************************************************
//******************************************************************************

#define SCTR_META_NAME "LuaClass.Controller"
#define check_sctrl(L) (SController **)luaL_checkudata(L, 1, SCTR_META_NAME)

const size_t idx_var = getCharHash("idx");
const size_t frames_var = getCharHash("num_frames");
const size_t tile_var = getCharHash("tile");
const size_t ortho_var = getCharHash("ortho");

static int set_val(lua_State *L) {
  SController *ctrl = *check_sctrl(L);

  // check for arguments
  const int args = (int)lua_gettop(L);
  if (args == 3) {
    // arg 2 = name of variable
    cbuff<32> arg_name = (const char *)luaL_checkstring(L, 2);

    if (arg_name.gethash() == idx_var) {
      // set current frame
      ctrl->curr_idx = luaL_checkinteger(L, 3);
    } else if (arg_name.gethash() == ortho_var) {
      // set orthographic matric params
      read_buffer_from_lua(L, i64_bin);
      ctrl->ortho_params.x = i64_bin[0];
      ctrl->ortho_params.y = i64_bin[1];
      ctrl->ortho_params.z = i64_bin[2];
      ctrl->ortho_params.w = i64_bin[3];
    } else if (arg_name.gethash() == tile_var) {
      ctrl->tile_size = luaL_checknumber(L, 3);
    }
  }
  return 0;
}

static int get_val(lua_State *L) {
  SController *ctrl = *check_sctrl(L);
  cbuff<32> arg_name = (const char *)luaL_checkstring(L, 2);

  if (arg_name.gethash() == idx_var) {
    lua_pushinteger(L, ctrl->curr_idx);
  } else if (arg_name.gethash() == frames_var) {
    lua_pushinteger(L, ctrl->num_frames);
  } else if (arg_name.gethash() == tile_var) {
    lua_pushnumber(L, ctrl->tile_size);
  } else if (arg_name.gethash() == ortho_var) {
    push_ivec4_to_lua(L, ctrl->ortho_params.x, ctrl->ortho_params.y,
                      ctrl->ortho_params.y, ctrl->ortho_params.z);
  }

  return 1;
}

static int to_string(lua_State *L) {
  SController *ctrl = *check_sctrl(L);
  std::string buff;

  cbuff<128> out;
  out.format("\n idx: %d, frames: %d", ctrl->curr_idx, ctrl->num_frames);
  buff += out.str();

  lua_pushstring(L, buff.c_str());
  return 1;
}

static const struct luaL_Reg sctrl_methods[] = {{"__index", get_val},
                                                {"__newindex", set_val},
                                                {"__tostring", to_string},
                                                {NULL, NULL}};

const unsigned sctrl_ptr_size = sizeof(SController *);

static int get_sctrl(lua_State *L) {
  // create userdata for instance
  SController **ctrl = (SController **)lua_newuserdata(L, sctrl_ptr_size);

  // assign level controller
  (*ctrl) = &sctrl;

  // set metatable
  luaL_getmetatable(L, SCTR_META_NAME);
  lua_setmetatable(L, -2);

  return 1;
}

static int get_input_data(lua_State *L) {
  DD_FOREACH(glm::vec3, point, sctrl._input) {
    push_vec3_to_lua(L, point.ptr->x, point.ptr->y, point.ptr->z);
  }
  return sctrl._input.size();
}

static int get_ground_data(lua_State *L) {
  DD_FOREACH(glm::vec3, point, sctrl._ground) {
    push_vec3_to_lua(L, point.ptr->x, point.ptr->y, point.ptr->z);
  }
  return sctrl._ground.size();
}

static int get_calc_data(lua_State *L) {
  DD_FOREACH(glm::vec3, point, sctrl._predicted) {
    push_vec3_to_lua(L, point.ptr->x, point.ptr->y, point.ptr->z);
  }
  return sctrl._predicted.size();
}

static const struct luaL_Reg sctrl_lib[] = {
    {"get", get_sctrl},
    {"get_input_data", get_input_data},
    {"get_ground_data", get_ground_data},
    {"get_calc_data", get_calc_data},
    {NULL, NULL}};

int luaopen_sctrl(lua_State *L) {
  // get and log functions in metatable
  luaL_newmetatable(L, SCTR_META_NAME);  // create meta table
  lua_pushvalue(L, -1);                  /* duplicate the metatable */
  lua_setfield(L, -2, "__index");        /* mt.__index = mt */
  luaL_setfuncs(L, sctrl_methods, 0);    /* register metamethods */

  // library functions
  luaL_newlib(L, sctrl_lib);
  return 1;
}

void register_lua_controller(lua_State *L) {
  // log libraries
  luaL_requiref(L, "SController", luaopen_sctrl, 1);
  // clear stack
  int top = lua_gettop(L);
  lua_pop(L, top);
}

//******************************************************************************
//******************************************************************************

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

  point_sh.init();
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "PointRend_V.vert");
  point_sh.create_vert_shader(fname.str());
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "PointRend_G.geom");
  point_sh.create_geom_shader(fname.str());
  fname.format("%s/smile_vis/%s", PROJECT_DIR, "PointRend_F.frag");
  point_sh.create_frag_shader(fname.str());

  // line structures
  ddGPUFrontEnd::create_vao(line_vao);
  ddGPUFrontEnd::create_storage_buffer(line_ssbo, l_points.sizeInBytes());
  ddGPUFrontEnd::create_index_buffer(line_ebo, l_indices.sizeInBytes(),
                                     &l_indices[0]);
  ddGPUFrontEnd::bind_storage_buffer_atrribute(line_vao, line_ssbo,
                                               ddAttribPrimitive::FLOAT, 0, 3,
                                               3 * sizeof(float), 0);
  ddGPUFrontEnd::set_storage_buffer_contents(line_ssbo, l_points.sizeInBytes(),
                                             0, &l_points[0]);
  ddGPUFrontEnd::bind_index_buffer(line_vao, line_ebo);

  // point structures
  ddGPUFrontEnd::create_vao(point_vao);
  ddGPUFrontEnd::create_storage_buffer(point_ssbo, 50 * 3 * sizeof(float));

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

    // data vector
    if (input_p.size() > 0) {
      get_points(input_p, sctrl._input, sctrl.curr_idx, VectorOut::INPUT);
      ddGPUFrontEnd::set_storage_buffer_contents(
          point_ssbo, sctrl._input.sizeInBytes(), 0, &sctrl._input[0]);
    }

    // get camera matrices & activate point shader
    const glm::mat4 v_mat = ddSceneManager::calc_view_matrix(cam);
    const glm::mat4 p_mat = ddSceneManager::calc_o_proj_matrix(
        cam, sctrl.ortho_params.x, sctrl.ortho_params.y, sctrl.ortho_params.z,
        sctrl.ortho_params.w);
    // const glm::mat4 p_mat = ddSceneManager::calc_p_proj_matrix(cam);

    point_sh.use();

    // draw feature points
    glm::mat4 m_mat = glm::scale(glm::mat4(), glm::vec3(1.f, 1.f, 1.f));
    point_sh.set_uniform((int)RE_Point::MV_m4x4, v_mat * m_mat);
    point_sh.set_uniform((int)RE_Point::Proj_m4x4, p_mat);
    point_sh.set_uniform((int)RE_Point::quad_h_width_f, sctrl.tile_size);
    point_sh.set_uniform((int)RE_Point::color_v4, glm::vec4(1.f));

    if (sctrl._input.size() > 0) {
      ddGPUFrontEnd::draw_points(point_vao, point_ssbo,
                                 ddAttribPrimitive::FLOAT, 0, 3, 3, 0, 0,
                                 sctrl._input.size());
    }

    // ground truth
    if (sctrl._ground.size() > 0) {
      point_sh.set_uniform((int)RE_Point::color_v4,
                           glm::vec4(0.f, 1.f, 0.f, 1.f));
      get_points(groundtr_p, sctrl._ground, sctrl.curr_idx, VectorOut::OUTPUT);
      ddGPUFrontEnd::set_storage_buffer_contents(
          point_ssbo, sctrl._ground.sizeInBytes(), 0, &sctrl._ground[0]);
      ddGPUFrontEnd::draw_points(point_vao, point_ssbo,
                                 ddAttribPrimitive::FLOAT, 0, 3, 3, 0, 0,
                                 sctrl._ground.size());
    }

    // predicted
    if (sctrl._predicted.size() > 0) {
      // update calculated points
      get_points(input_p[sctrl.curr_idx], weights, biases, sctrl._predicted);

      point_sh.set_uniform((int)RE_Point::color_v4,
                           glm::vec4(1.f, 0.f, 0.f, 1.f));
      ddGPUFrontEnd::set_storage_buffer_contents(
          point_ssbo, sctrl._predicted.sizeInBytes(), 0, &sctrl._predicted[0]);
      ddGPUFrontEnd::draw_points(point_vao, point_ssbo,
                                 ddAttribPrimitive::FLOAT, 0, 3, 3, 0, 0,
                                 sctrl._predicted.size());
    }

    // render frame cutout (right side) ****************************************
    linedot_sh.use();

    m_mat = glm::translate(glm::mat4(), glm::vec3(5.f, 0.f, 0.f));
    linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, p_mat * v_mat * m_mat);
    linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, false);
    linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, false);
    ddGPUFrontEnd::render_cube();

    // render the background
    linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, identity);
    linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, true);
    linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, false);
    linedot_sh.set_uniform((int)RE_LineDot::color_v4,
                           glm::vec4(0.5f, 0.5f, 0.5f, 1.f));
    ddGPUFrontEnd::render_quad();
    linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, false);

    // bind Particle frame buffer & bind texture from last draw calls
    ddGPUFrontEnd::bind_framebuffer(ddBufferType::PARTICLE);
    linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, true);
    ddGPUFrontEnd::bind_pass_texture(ddBufferType::XTRA, 0);
    linedot_sh.set_uniform((int)RE_LineDot::bound_tex_smp2d, 0);

    // render right side cutout (flip image in y axis)
    refill_buffer(frames[0]);
    ddGPUFrontEnd::toggle_face_cull(false);
    m_mat = glm::scale(glm::mat4(), glm::vec3(1.f, -1.f, 1.f));
    linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, identity * m_mat);
    ddGPUFrontEnd::render_primitive(6, point_buff, texcoord_buff);
    ddGPUFrontEnd::toggle_face_cull(true);
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
  style->Colors[ImGuiCol_Header] = ImVec4(0.23f, 0.23f, 0.23f, 1.00f);
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

  // stop edge clipping
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.7f);

  // show list of selectable files
  ImColor col(1.f, 0.85f, 0.f);
  if (file_names_ptr.size() > 0) {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::Text("IN: %s", f_dir.str());
    ImGui::PopStyleColor();

    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::Text("GT: %s", gd_dir.str());
    ImGui::PopStyleColor();

    ImGui::ListBox("<-- Select data", &selected_file, &file_names_ptr[0],
                   (int)file_names_ptr.size(), 10);

    // button to load data
    if (ImGui::Button("Load selected")) {
      // load input
      input_p = extract_vector2(files[selected_file].str());
      // load ground truth
      std::string temp = gd_dir.str() + std::string("/") +
                         std::string(file_names_ptr[selected_file]);
      groundtr_p = extract_vector2(temp.c_str());

      // set frame count
      sctrl.curr_idx = 0;
      sctrl.num_frames = input_p.size();
      // set array sizes
      get_points(input_p, sctrl._input, sctrl.curr_idx, VectorOut::INPUT);
      get_points(groundtr_p, sctrl._ground, sctrl.curr_idx, VectorOut::OUTPUT);
      get_points(input_p[sctrl.curr_idx], weights, biases, sctrl._predicted);
    }
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::Text("No Folders loaded/Folder not found");
    ImGui::PopStyleColor();
  }
  ImGui::Separator();

  if (sctrl._input.size() > 0) {
    // difference
    unsigned idx = 0;
    // Lateral canthus
    ImGui::Text("Lateral canthus (L)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Lateral canthus (R)
    ImGui::Text("Lateral canthus (R)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Palpebral fissure (RU)
    ImGui::Text("Palpebral fissure (RU)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Palpebral fissure (RL)
    ImGui::Text("Palpebral fissure (RL)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Palpebral fissure (LU)
    ImGui::Text("Palpebral fissure (LU)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Palpebral fissure (LL)
    ImGui::Text("Palpebral fissure (LL)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Depressor (L)
    ImGui::Text("Depressor (L)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Depressor (R)
    ImGui::Text("Depressor (R)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Depressor (M)
    ImGui::Text("Depressor (M)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Iris (M)
    ImGui::Text("Iris (M)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Iris (L)
    ImGui::Text("Iris (L)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Nasal ala (L)
    ImGui::Text("Nasal ala (L)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Nasal ala (R)
    ImGui::Text("Nasal ala (R)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Medial brow (L)
    ImGui::Text("Medial brow (L)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Medial brow (R)
    ImGui::Text("Medial brow (R)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Malar eminence (L)
    ImGui::Text("Malar eminence (L)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
    // Malar eminence (R)
    ImGui::Text("Malar eminence (R)");
    ImGui::InputFloat2("ground", &sctrl._ground[idx][0]);
    ImGui::InputFloat2("predict", &sctrl._predicted[idx][0]);
    idx++;
  }

  ImGui::PopItemWidth();
  ImGui::End();

  return 0;
}

void load_files(const char *directory, const bool ground_truth) {
  if (ground_truth) {
    // log ground truth directory
    gd_dir = directory;
  } else {
    // open folder & extract files
    f_dir = directory;
    ddIO folder_handle;
    folder_handle.open(directory, ddIOflag::DIRECTORY);
    dd_array<cbuff<512>> unfiltered = folder_handle.get_directory_files();

    // check if file contains _s_out.csv or _v_out.csv
    dd_array<unsigned> valid_files(unfiltered.size());
    unsigned files_found = 0;
    DD_FOREACH(cbuff<512>, file, unfiltered) {
      // capture index of matching files
      if (file.ptr->contains("_s_out") || file.ptr->contains("_v_out")) {
        valid_files[files_found] = file.i;
        files_found++;
      }
    }

    // create visible list of files
    files.resize(files_found);
    file_names.resize(files_found);
    file_names_ptr.resize(files_found);
    for (unsigned i = 0; i < files_found; i++) {
      files[i] = unfiltered[valid_files[i]];

      std::string _s = unfiltered[valid_files[i]].str();
      file_names[i] = _s.substr(_s.find_last_of("\\/") + 1).c_str();
      file_names_ptr[i] = file_names[i].str();
    }
  }
}

void load_weights(const char *directory) {
  // open folder & extract files
  ddIO folder_handle;
  folder_handle.open(directory, ddIOflag::DIRECTORY);
  dd_array<cbuff<512>> unfiltered = folder_handle.get_directory_files();

  // check if file contains .csv
  DD_FOREACH(cbuff<512>, file, unfiltered) {
    // load up matching files
    if (file.ptr->contains(".csv")) {
      weights.push_back(extract_matrix(file.ptr->str()));
    }
  }
}

void load_biases(const char *directory) {
  // open folder & extract files
  ddIO folder_handle;
  folder_handle.open(directory, ddIOflag::DIRECTORY);
  dd_array<cbuff<512>> unfiltered = folder_handle.get_directory_files();

  // check if file contains .csv
  DD_FOREACH(cbuff<512>, file, unfiltered) {
    // load up matching files
    if (file.ptr->contains(".csv")) {
      biases.push_back(extract_vector(file.ptr->str()));
    }
  }
}