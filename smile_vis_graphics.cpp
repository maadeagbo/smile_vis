#include "smile_vis_graphics.h"
#include "ddFileIO.h"
#include "ddTerminal.h"
#include "smile_vis_data.h"
#include "svis_shader_enums.h"

#define MAX_POINTS 4
#define MAX_INDICES 8

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

// data points of current visualization
std::vector<Eigen::VectorXd> input_p;

// data points of ground truth
std::vector<Eigen::VectorXd> groundtruth_p;

// data points of calculated network
std::vector<Eigen::VectorXd> calc_p;
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

    // get camera matrices & activate point shader
    const glm::mat4 v_mat = ddSceneManager::calc_view_matrix(cam);
    //const glm::mat4 p_mat =
         //ddSceneManager::calc_o_proj_matrix(cam, 1250, 1360, 730, 710);
    const glm::mat4 p_mat = ddSceneManager::calc_p_proj_matrix(cam);
    point_sh.use();

    // draw feature points
    glm::mat4 m_mat = glm::scale(glm::mat4(), glm::vec3(0.2f));
    point_sh.set_uniform((int)RE_Point::MV_m4x4, v_mat * m_mat);
    point_sh.set_uniform((int)RE_Point::Proj_m4x4, p_mat);
    point_sh.set_uniform((int)RE_Point::quad_h_width_f, 10.f);
    point_sh.set_uniform((int)RE_Point::color_v4, glm::vec4(1.f));

    if (input_p.size() > 0) {
      std::vector<glm::vec3> data_i = std::move(get_points(input_p, 0, VectorOut::INPUT));
      //dd_array<glm::vec3> data_i(1); data_i[0] = glm::vec3(0.f, 0.f, 0.f);
      ddGPUFrontEnd::set_storage_buffer_contents(point_ssbo, data_i.size() * sizeof(glm::vec3),
                                             0, &data_i[0]);
      ddGPUFrontEnd::draw_points(point_vao, point_ssbo, ddAttribPrimitive::FLOAT, 0, 3, 3, 0, 0, data_i.size());
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
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.75f);

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
      input_p = extract_vector2(files[selected_file].str());
      std::string temp = gd_dir.str() + std::string("/") + 
        std::string(file_names_ptr[selected_file]);
      groundtruth_p = extract_vector2(temp.c_str());
    }
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::Text("No Folders loaded/Folder not found");
    ImGui::PopStyleColor();
  }
  ImGui::Separator();

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
