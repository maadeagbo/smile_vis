#include "smile_vis_data.h"
#include "ddFileIO.h"
#include "ddTerminal.h"
#include <iostream>

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

std::vector<Eigen::VectorXd> extract_vector2(const char *in_file) {
  std::vector<Eigen::VectorXd> out_vec;
  ddIO vec_io;

  bool success = vec_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get vector size
    const char *line = vec_io.readNextLine();
    dd_array<cbuff<64>> indices = StrSpace::tokenize1024<64>(line, ",");
    const std::string _f = in_file;
    if (_f.find("canon") != std::string::npos) {
      indices = StrSpace::tokenize1024<64>(line, " ");
    } else {
      line = vec_io.readNextLine();  // skip header line
    }
    const unsigned vec_size = indices.size();

    // ddTerminal::f_post("Creating new input vectors(%lu)...", vec_size);
    // populate vector
    line = vec_io.readNextLine();
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
    // skip 1st 4 values (remove delta values)
    unsigned c_idx = 2;
    if (type == VectorOut::OUTPUT_C) {
      c_idx = 0;
    }
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

void export_canonical_data(dd_array<glm::vec3> &input,
                           dd_array<glm::vec3> &ground, const char *dir,
                           const char *gdir, const char *file_id,
                           const glm::vec2 canonical_iris_pos,
                           const float canonical_iris_dist) {
  // create new file
  std::string f_id = file_id;
  f_id = f_id.substr(0, 7);
  cbuff<512> out_f_name, out_fg_name;
  out_f_name.format("%s/%s_canon.csv", dir, f_id.c_str());
  out_fg_name.format("%s/%s_canon.csv", gdir, f_id.c_str());
  // ddTerminal::f_post("Creating: %s", out_f_name.str());

  // get translation offset (Iris (M) x, Iris (M) y)
  const unsigned iris_m_idx = 2;
  const unsigned iris_l_idx = 3;
  // glm::vec2 delta_pos = glm::vec2(-input[iris_l_idx]);

  // palpebral fissure delta and center
  const unsigned pf_r_l = 5;
  const unsigned pf_l_l = 7;
  glm::vec2 delta_pos = glm::vec2(-ground[pf_r_l]);
  ddTerminal::f_post("PF R L: %.3f", -delta_pos.y);

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
  DD_FOREACH(glm::vec3, vec, input) {
    // ddTerminal::f_post("#%u : %.3f, %.3f", vec.i, vec.ptr->x, vec.ptr->y);
    input_n[vec.i] = input_n[vec.i] + canonical_iris_pos;
  }
  DD_FOREACH(glm::vec3, vec, ground) {
    // ddTerminal::f_post("----> %.3f, %.3f", input_n[vec.i].x,
    // input_n[vec.i].y);
    ground_n[vec.i] = ground_n[vec.i] + canonical_iris_pos;
  }

  // write out input and ground file
  ddIO i_out, g_out;
  i_out.open(out_f_name.str(), ddIOflag::APPEND);
  std::string out_str;
  std::string _sp(" ");
  DD_FOREACH(glm::vec2, vec, input_n) {
    out_str +=
        std::to_string(vec.ptr->x) + _sp + std::to_string(vec.ptr->y) + _sp;
  }
  out_str.pop_back();
  out_str += "\n";
  // ddTerminal::post(out_str.c_str());
  i_out.writeLine(out_str.c_str());

  g_out.open(out_fg_name.str(), ddIOflag::APPEND);
  out_str = "";
  DD_FOREACH(glm::vec2, vec, ground_n) {
    out_str +=
        std::to_string(vec.ptr->x) + _sp + std::to_string(vec.ptr->y) + _sp;
  }
  out_str.pop_back();
  out_str += "\n";
  g_out.writeLine(out_str.c_str());
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
    dd_array<cbuff<512>> i_files = io_input.get_directory_files();
    dd_array<cbuff<512>> g_files = io_ground.get_directory_files();
    ddTerminal::f_post("Opening in dir: %s..", input_dir);
    ddTerminal::f_post("Opening ground dir: %s..", ground_dir);
    DD_FOREACH(cbuff<512>, file, i_files) {
      const char *g_file = g_files[file.i].str();
      // get name of file
      const std::string temp = file.ptr->str();
      const size_t idx = temp.find_last_of("\\/");
      const std::string f_name = temp.substr(idx + 1);

      if (f_name.find("canon") == std::string::npos) {
        ddTerminal::f_post("  Exporting: %s", f_name.c_str());

        // extract contents of each file and convert to glm vectors
        std::vector<Eigen::VectorXd> i_vec = extract_vector2(file.ptr->str());
        std::vector<Eigen::VectorXd> g_vec = extract_vector2(g_file);

        // loop thru lines fo each and write to output file
        for (size_t j = 0; j < i_vec.size(); j++) {
          dd_array<glm::vec3> i_p, g_p;
          get_points(i_vec, i_p, j, VectorOut::INPUT);
          get_points(g_vec, g_p, j, VectorOut::OUTPUT);

          export_canonical_data(i_p, g_p, input_dir, ground_dir, f_name.c_str(),
                                canonical_iris_pos, canonical_iris_dist);
        }
        ddTerminal::post("---> Done.");
      }
    }
  }
}
