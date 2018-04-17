#include "smile_vis_graphics.h"
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
void refill_buffer(const FrameData& data);

int init_gpu_structures(lua_State *L) {
	// indices buffer
	l_indices[0] = 0; l_indices[1] = 1;
	l_indices[2] = 1; l_indices[3] = 2;
	l_indices[4] = 2; l_indices[5] = 3;
	l_indices[6] = 3; l_indices[7] = 0;

	init_data();

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
  ddGPUFrontEnd::set_storage_buffer_contents(
		point_ssbo, l_points.sizeInBytes(), 0, &l_points[0]);
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
	frames[0].verts[0] = glm::vec3(0.0, 1.0, 0.0);		// halfway
	frames[0].texcoords[0] = glm::vec2(0.5, 1.0);		// halfway
	frames[0].verts[1] = glm::vec3(0.0, -1.0, 0.0);		// halfway
	frames[0].texcoords[1] = glm::vec2(0.5, 0.0);		// halfway
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
	frames[1].verts[2] = glm::vec3(0.0, -1.0, 0.0);		// halfway
	frames[1].texcoords[2] = glm::vec2(0.5, 0.0);		// halfway
	frames[1].verts[3] = glm::vec3(0.0, 1.0, 0.0);		// halfway
	frames[1].texcoords[3] = glm::vec2(0.5, 1.0);		// halfway
}

void update_frame_data(const FrameData& data) {
	refill_buffer(data);
}

/** \brief Draws FrameData for gpu */
void draw_frame() {
	ddCam *cam = ddSceneManager::get_active_cam();
	const glm::mat4 identity;

	if (cam) {
		linedot_sh.use();
		ddGPUFrontEnd::toggle_depth_mask(true);

		// get camera matrices
		const glm::mat4 v_mat = ddSceneManager::calc_view_matrix(cam);
		const glm::mat4 p_mat = ddSceneManager::calc_p_proj_matrix(cam);

		// wipe background
		linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, identity);
		linedot_sh.set_uniform((int)RE_LineDot::color_v4, glm::vec4(0.f, 0.f, 0.f, 1.f));
		linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, false);
		linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, true);
		ddGPUFrontEnd::render_quad();
		linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, false);

		// draw feature points
		//linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, identity);
		linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, p_mat * v_mat);
		linedot_sh.set_uniform((int)RE_LineDot::color_v4, glm::vec4(1.f));
		ddGPUFrontEnd::render_quad();

		// render the background (right side)
		linedot_sh.set_uniform((int)RE_LineDot::MVP_m4x4, identity);
		linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, true);
		linedot_sh.set_uniform((int)RE_LineDot::color_v4, glm::vec4(0.5f, 0.5f, 0.5f, 1.f));
		ddGPUFrontEnd::render_quad();
		linedot_sh.set_uniform((int)RE_LineDot::send_to_back_b, false);

		// render frame cutout (right side)
		//ddGPUFrontEnd::toggle_depth_test(false);
		linedot_sh.set_uniform((int)RE_LineDot::color_v4, glm::vec4(1.f));

		ddGPUFrontEnd::draw_indexed_lines_vao(line_vao, l_indices.size(), 0);

		// bind texture from last draw calls
		linedot_sh.set_uniform((int)RE_LineDot::render_to_tex_b, true);
		ddGPUFrontEnd::bind_pass_texture(ddBufferType::PARTICLE, 0, 1);
		linedot_sh.set_uniform((int)RE_LineDot::bound_tex_smp2d, 0);

		//ddGPUFrontEnd::render_quad();
		refill_buffer(frames[1]);
		ddGPUFrontEnd::render_primitive(6, point_buff, texcoord_buff);

		ddGPUFrontEnd::toggle_depth_mask(false);
	}
}

void refill_buffer(const FrameData& data) {
	for (unsigned i = 0; i < MAX_POINTS; i++) {
		l_points[i] = data.verts[i];
		l_texcoords[i] = data.texcoords[i];
	}
	// set buffers for gpu render
	point_buff[0] = l_points[0]; texcoord_buff[0] = l_texcoords[0];
	point_buff[1] = l_points[1]; texcoord_buff[1] = l_texcoords[1];
	point_buff[2] = l_points[2]; texcoord_buff[2] = l_texcoords[2];

	point_buff[3] = l_points[0]; texcoord_buff[3] = l_texcoords[0];
	point_buff[4] = l_points[2]; texcoord_buff[4] = l_texcoords[2];
	point_buff[5] = l_points[3]; texcoord_buff[5] = l_texcoords[3];
}
