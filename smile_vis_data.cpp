#include "smile_vis_data.h"
#include "ddFileIO.h"
#include "ddTerminal.h"
#include <iostream>

namespace {
// log keys for easy indexing
std::map<string64, unsigned> input_keys;
std::map<unsigned, float> input_time;
std::map<string64, unsigned> output_keys;
std::map<unsigned, float> output_time;
}  // namespace

std::vector<double> feedForward(Eigen::VectorXd &inputs,
                                std::vector<Eigen::MatrixXd> &weights,
                                std::vector<Eigen::VectorXd> &biases) {
  // output is class id
  int layers = weights.size();
  // inputs are assumed normalized where appropriate
  // convert to row vector
  Eigen::VectorXd layerin = inputs;
  for (int i = 0; i < layers; i++) {
    if (weights[i].rows() != layerin.size()) {
      std::cout << "Input and weights have incompatible dimensions at layer "
                << i << "!!";
      std::cout << "Input size: " << layerin.size()
                << ", expected size: " << weights[i].rows() << std::endl;
      POW2_VERIFY_MSG(
          weights[i].rows() != layerin.size(),
          "Input and weights have incompatible dimensions at layer %d", i);
    }
    layerin = weights[i].transpose() * layerin + biases[i];
    // component wise RELU
    if (i < layers - 1) {
      for (int j = 0; j < layerin.size(); j++) {
        layerin[j] = std::max(0.0, layerin[j]);
      }
    }
  }
  std::vector<double> output(layerin.size());
  for (int i = 0; i < layerin.size(); i++) {
    output[i] = layerin[i];
  }
  return output;
}

Eigen::VectorXd extract_vector(const char *in_file) {
  Eigen::VectorXd out_vec;
  ddIO vec_io;

  bool success = vec_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get vector size
    unsigned long vec_size = 0;
    const char *line = vec_io.readNextLine();
    if (line && *line) vec_size = std::strtoul(line, NULL, 10);

    printf("    Creating new vector (%lu)...\n", vec_size);
    out_vec = Eigen::VectorXd::Zero(vec_size);

    // populate vector
    line = vec_io.readNextLine();
    unsigned idx = 0;
    while (line && *line) {
      out_vec(idx) = std::strtod(line, NULL);

      line = vec_io.readNextLine();
      idx++;
    }
    // std::cout << out_vec << "\n";
  }

  return out_vec;
}

std::vector<Eigen::VectorXd> extract_vector2(const char *in_file,
                                             const VectorOut type) {
  std::vector<Eigen::VectorXd> out_vec;
  ddIO vec_io;

  bool success = vec_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get vector size
    const char *line = vec_io.readNextLine();
    dd_array<string64> indices;

    switch (type) {
      case VectorOut::INPUT:
        // get input keys
        indices = StrLib::tokenize2<64>(line, ",");
        if (input_keys.size() == 0) {
          DD_FOREACH(string64, _key, indices) {
            // set offset if time column is present (must be 1st column)
            if (_key.ptr->contains("time")) {
							input_time[_key.i] = 0.f;
            }
            input_keys[*_key.ptr] = _key.i;
          }
        }
        // skip to next line in file
        line = vec_io.readNextLine();
        break;
      case VectorOut::OUTPUT:
        // get output keys
        indices = StrLib::tokenize2<64>(line, ",");
        if (output_keys.size() == 0) {
          DD_FOREACH(string64, _key, indices) {
            if (_key.ptr->contains("time")) {
							output_time[_key.i] = 0.f;
            }
            output_keys[*_key.ptr] = _key.i;
          }
        }
        // skip to next line in file
        line = vec_io.readNextLine();
        break;
      case VectorOut::INPUT_C:
        indices = StrLib::tokenize2<64>(line, " ");
        break;
      case VectorOut::OUTPUT_C:
        indices = StrLib::tokenize2<64>(line, " ");
        break;
      default:
        break;
    }
    const unsigned vec_size = indices.size();

    // ddTerminal::f_post("Creating new input vectors(%lu)...", vec_size);
    // populate vector
    unsigned idx = 0;
    while (line && *line) {
      out_vec.push_back(Eigen::VectorXd::Zero(vec_size));

      // loop thru columns per row
      unsigned r_idx = 0;
      const char *curr_row = line;
      while (*curr_row) {
        char *nxt_dbl = nullptr;
        // printf("%s\n", curr_row);
        out_vec[idx](r_idx) = std::strtod(curr_row, &nxt_dbl);
        curr_row = nxt_dbl;

        r_idx++;
      }
      // std::cout << out_vec[idx] << "\n\n";

      line = vec_io.readNextLine();
      idx++;
    }
  }

  return out_vec;
}

Eigen::MatrixXd extract_matrix(const char *in_file) {
  Eigen::MatrixXd out_mat;
  ddIO mat_io;

  bool success = mat_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get matrix size
    unsigned long mat_size[2] = {0, 0};
    const char *line = mat_io.readNextLine();
    char *nxt_num = nullptr;
    if (line && *line) {
      // doesn't work if line doesn't have 2 number white-space separated
      mat_size[0] = std::strtoul(line, &nxt_num, 10);
      mat_size[1] = std::strtoul(nxt_num, NULL, 10);
    }

    printf("    Creating new matrix (%lu, %lu)...\n", mat_size[0], mat_size[1]);
    out_mat = Eigen::MatrixXd::Zero(mat_size[0], mat_size[1]);

    // populate matrix
    line = mat_io.readNextLine();
    unsigned r_idx = 0;
    while (line && *line) {
      // loop thru columns per row
      unsigned c_idx = 0;
      const char *curr_row = line;
      while (*curr_row) {
        char *nxt_dbl = nullptr;
        // printf("%s\n", curr_row);
        out_mat(r_idx, c_idx) = std::strtod(curr_row, &nxt_dbl);
        curr_row = nxt_dbl;

        c_idx++;
      }

      line = mat_io.readNextLine();
      r_idx++;
    }
    // std::cout << out_mat << "\n";
  }

  return out_mat;
}

void get_points(std::vector<Eigen::VectorXd> &v_bin,
                dd_array<glm::vec3> &out_bin, const unsigned idx,
                const VectorOut type) {
  if (type == VectorOut::INPUT) {
    // use all values
    if ((int)out_bin.size() != (v_bin[idx].size() / 2)) {
      out_bin.resize(v_bin[idx].size() / 2);
    }

    unsigned c_idx = 0;

    while (c_idx < out_bin.size()) {
      out_bin[c_idx] =
          glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
      c_idx++;
    }

    // Oral commisure (L) x,Oral commisure (L) y,
    // Oral commisure (R) x,Oral commisure (R) y,
    // Dental show (Top) x,Dental show (Top) y,
    // Dental show (Bottom) x, Dental show (Bottom) y
    // Iris (M) x,Iris (M) y,
    // Iris (L) x,Iris (L) y,
  } else {
    // use all values
		unsigned c_idx = 0;
    if ((int)out_bin.size() != ((v_bin[idx].size()) / 2)) {
      out_bin.resize((v_bin[idx].size()) / 2);
    }

    while ((c_idx) < out_bin.size()) {
      out_bin[c_idx] =
          glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
      c_idx++;
    }

    // Lateral canthus (L) x,Lateral canthus (L) y,
    // Lateral canthus (R) x,Lateral canthus (R) y,
    // Palpebral fissure (RU) x,Palpebral fissure (RU) y,
    // Palpebral fissure (RL) x,Palpebral fissure (RL) y,
    // Palpebral fissure (LU) x,Palpebral fissure (LU) y,
    // Palpebral fissure (LL) x,Palpebral fissure (LL) y,
    // Depressor (L) x,Depressor (L) y,
    // Depressor (R) x,Depressor (R) y,
    // Depressor (M) x,Depressor (M) y,
    // Nasal ala (L) x,Nasal ala (L) y,
    // Nasal ala (R) x,Nasal ala (R) y,
    // Medial brow (L) x,Medial brow (L) y,
    // Medial brow (R) x,Medial brow (R) y,
    // Malar eminence (L) x,Malar eminence (L) y,
    // Malar eminence (R) x,Malar eminence (R) y
  }
}

void get_points(Eigen::VectorXd &input, std::vector<Eigen::MatrixXd> &weights,
                std::vector<Eigen::VectorXd> &biases,
                dd_array<glm::vec3> &output) {
  std::vector<double> out_d = feedForward(input, weights, biases);

  // skip 1st 4 values
  if (output.size() != ((out_d.size()) / 2)) {
    output.resize((out_d.size()) / 2);
  }
  unsigned c_idx = 0;

  while ((c_idx) < output.size()) {
    output[c_idx] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
    c_idx++;
  }

  // Lateral canthus (L) x,Lateral canthus (L) y,
  // Lateral canthus (R) x,Lateral canthus (R) y,
  // Palpebral fissure (RU) x,Palpebral fissure (RU) y,
  // Palpebral fissure (RL) x,Palpebral fissure (RL) y,
  // Palpebral fissure (LU) x,Palpebral fissure (LU) y,
  // Palpebral fissure (LL) x,Palpebral fissure (LL) y,
  // Depressor (L) x,Depressor (L) y,
  // Depressor (R) x,Depressor (R) y,
  // Depressor (M) x,Depressor (M) y,
  // Nasal ala (L) x,Nasal ala (L) y,
  // Nasal ala (R) x,Nasal ala (R) y,
  // Medial brow (L) x,Medial brow (L) y,
  // Medial brow (R) x,Medial brow (R) y,
  // Malar eminence (L) x,Malar eminence (L) y,
  // Malar eminence (R) x,Malar eminence (R) y
}

std::map<string64, unsigned> &get_input_keys() { return input_keys; }

std::map<string64, unsigned> &get_output_keys() { return output_keys; }

void export_canonical_data(dd_array<glm::vec3> &input,
                           dd_array<glm::vec3> &ground, const char *dir,
                           const char *gdir, const char *file_id,
                           const glm::vec2 canonical_iris_pos,
                           const float canonical_iris_dist, const bool append) {
  // create new file
	string512 f_id = file_id;
  f_id = f_id.trim(0, 7);
  string512 out_f_name, out_fg_name;
  out_f_name.format("%s/%s_canon.csv", dir, f_id.str());
  out_fg_name.format("%s/%s_canon.csv", gdir, f_id.str());
  // ddTerminal::f_post("Creating: %s", out_f_name.str());

  // get translation offset (Iris (M) x, Iris (M) y)
  string64 map_idx = "Iris (M) x";
  const unsigned iris_m_idx = input_keys[map_idx]/2;
	map_idx = "Iris (L) x";
  const unsigned iris_l_idx = input_keys[map_idx]/2;
  // glm::vec2 delta_pos = glm::vec2(-input[iris_l_idx]);

  // palpebral fissure delta and center
	map_idx = "Palpebral fissure (RL) x";
  const unsigned pf_r_l = output_keys[map_idx] / 2;
	map_idx = "Palpebral fissure (LL) x";
  const unsigned pf_l_l = output_keys[map_idx] / 2;
  glm::vec2 delta_pos = glm::vec2(-ground[pf_r_l]);
  // ddTerminal::f_post("PF R L: %.3f", -delta_pos.y);

  // apply delta translation to all points
  dd_array<glm::vec2> input_n(input.size());
  dd_array<glm::vec2> ground_n(ground.size());

  DD_FOREACH(glm::vec3, vec, input) {  // input
    input_n[vec.i] = glm::vec2(*vec.ptr) + delta_pos;
  }
  DD_FOREACH(glm::vec3, vec, ground) {  // ground truth
    ground_n[vec.i] = glm::vec2(*vec.ptr) + delta_pos;
  }

  // get rotation offset b/t lateral & medial iris
  const float rot_offset = atan2(ground_n[pf_l_l].y, ground_n[pf_l_l].x);
  glm::mat2 r_mat;
  r_mat[0][0] = glm::cos(-rot_offset);
  r_mat[0][1] = glm::sin(-rot_offset);
  r_mat[1][0] = -glm::sin(-rot_offset);
  r_mat[1][1] = glm::cos(-rot_offset);

  // apply negative rotation to all points (at the current pos)
  DD_FOREACH(glm::vec2, vec, input_n) {  // input
    input_n[vec.i] = r_mat * (*vec.ptr);
  }
  DD_FOREACH(glm::vec2, vec, ground_n) {  // ground
    ground_n[vec.i] = r_mat * (*vec.ptr);
  }

  // scale points so that iris distance is set to a canonical distance
  const float dist = glm::distance(ground_n[pf_r_l], ground_n[pf_l_l]);
  const float scale_factor = canonical_iris_dist / dist;
  glm::mat2 s_mat;
  s_mat[0][0] = s_mat[1][1] = scale_factor;
  s_mat[0][1] = s_mat[1][0] = 0.f;

  DD_FOREACH(glm::vec2, vec, input_n) {  // input
    input_n[vec.i] = s_mat * (*vec.ptr);
  }
  DD_FOREACH(glm::vec2, vec, ground_n) {  // ground
    ground_n[vec.i] = s_mat * (*vec.ptr);
  }

  // apply translation to all points to move iris to canonical position
  DD_FOREACH(glm::vec2, vec, input_n) {
    // ddTerminal::f_post("#%u : %.3f, %.3f", vec.i, vec.ptr->x, vec.ptr->y);
		*vec.ptr += canonical_iris_pos;
  }
  DD_FOREACH(glm::vec2, vec, ground_n) {
    // ddTerminal::f_post("----> %.3f, %.3f", input_n[vec.i].x,
    // input_n[vec.i].y);
    *vec.ptr += canonical_iris_pos;
  }

  // write out input and ground file
  ddIO i_out, g_out;
  if (append) {
    i_out.open(out_f_name.str(), ddIOflag::APPEND);
  } else {
    i_out.open(out_f_name.str(), ddIOflag::WRITE);
  }

  string512 out_str;
  string8 x_val, y_val;
  DD_FOREACH(glm::vec2, vec, input_n) {
		x_val.format(" %.5f", vec.ptr->x);
		y_val.format(" %.5f", vec.ptr->y);
		out_str = out_str + x_val + y_val;
  }
  out_str.format("%s\n", out_str.str(1));
  // ddTerminal::post(out_str.c_str());
  i_out.writeLine(out_str.str());

  if (append) {
    g_out.open(out_fg_name.str(), ddIOflag::APPEND);
  } else {
    g_out.open(out_fg_name.str(), ddIOflag::WRITE);
  }

  out_str = "";
  DD_FOREACH(glm::vec2, vec, ground_n) {
		x_val.format(" %.5f", vec.ptr->x);
		y_val.format(" %.5f", vec.ptr->y);
		out_str = out_str + x_val + y_val;
  }
	out_str.format("%s\n", out_str.str(1));
  g_out.writeLine(out_str.str());
}

void export_canonical(const char *input_dir, const char *ground_dir,
                      const glm::vec2 canonical_iris_pos,
                      const float canonical_iris_dist) {
  // export input files
  ddIO io_input, io_ground;
  bool success = io_input.open(input_dir, ddIOflag::DIRECTORY);
  success |= io_ground.open(ground_dir, ddIOflag::DIRECTORY);
  if (success) {
    // for each file:
    dd_array<string512> i_files = io_input.get_directory_files();
    dd_array<string512> g_files = io_ground.get_directory_files();
    ddTerminal::f_post("Opening in dir: %s..", input_dir);
    ddTerminal::f_post("Opening ground dir: %s..", ground_dir);
    DD_FOREACH(string512, file, i_files) {
      const char *g_file = g_files[file.i].str();
      // get name of file
			dd_array<unsigned> token_idx = StrLib::tokenize(file.ptr->str(), "\\/");
      const unsigned idx = token_idx[token_idx.size() - 1];
      const string32 f_name = file.ptr->str(idx + 1);

      if (!f_name.contains("canon")) {
        ddTerminal::f_post("  Exporting: %s", f_name.str());

        // extract contents of each file and convert to glm vectors
        std::vector<Eigen::VectorXd> i_vec =
            extract_vector2(file.ptr->str(), VectorOut::INPUT);
        std::vector<Eigen::VectorXd> g_vec =
            extract_vector2(g_file, VectorOut::OUTPUT);

        // loop thru lines fo each and write to output file
        for (size_t j = 0; j < i_vec.size(); j++) {
          dd_array<glm::vec3> i_p, g_p;
          get_points(i_vec, i_p, j, VectorOut::INPUT);
          get_points(g_vec, g_p, j, VectorOut::OUTPUT);

          const bool append = (j == 0) ? false : true;
          export_canonical_data(i_p, g_p, input_dir, ground_dir, f_name.str(),
                                canonical_iris_pos, canonical_iris_dist,
                                append);
        }
        ddTerminal::post("---> Done.");
      }
    }
  }
}
