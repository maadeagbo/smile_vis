#include "smile_vis_data.h"
#include <iostream>
#include "ddFileIO.h"
#include "ddTerminal.h"

std::vector<double> feedForward(Eigen::VectorXd& inputs,
                                std::vector<Eigen::MatrixXd>& weights,
                                std::vector<Eigen::VectorXd>& biases) {
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

Eigen::VectorXd extract_vector(const char* in_file) {
  Eigen::VectorXd out_vec;
  ddIO vec_io;

  bool success = vec_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get vector size
    unsigned long vec_size = 0;
    const char* line = vec_io.readNextLine();
    if (line && *line) vec_size = std::strtoul(line, NULL, 10);

    printf("    Creating new vector (%lu)...\n", vec_size);
    out_vec = Eigen::VectorXd::Zero(vec_size);

    // populate vector
    line = vec_io.readNextLine();
    unsigned idx = 0;
    while (line && *line) {
      out_vec(idx) = std::strtof(line, NULL);

      line = vec_io.readNextLine();
      idx++;
    }
    // std::cout << out_vec << "\n";
  }

  return out_vec;
}

std::vector<Eigen::VectorXd> extract_vector2(const char* in_file) {
  std::vector<Eigen::VectorXd> out_vec;
  ddIO vec_io;

  bool success = vec_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get vector size
    const char* line = vec_io.readNextLine();
    dd_array<cbuff<64>> indices = StrSpace::tokenize1024<64>(line, ",");
    const unsigned vec_size = indices.size();

    ddTerminal::f_post("Creating new input vectors(%lu)...", vec_size);
    // populate vector
    line = vec_io.readNextLine();
    unsigned idx = 0;
    while (line && *line) {
      out_vec.push_back(Eigen::VectorXd::Zero(vec_size));

      // loop thru columns per row
      unsigned r_idx = 0;
      const char* curr_row = line;
      while (*curr_row) {
        char* nxt_float = nullptr;
        // printf("%s\n", curr_row);
        out_vec[idx](r_idx) = std::strtof(curr_row, &nxt_float);
        curr_row = nxt_float;

        r_idx++;
      }
      // std::cout << out_vec[idx] << "\n\n";

      line = vec_io.readNextLine();
      idx++;
    }
  }

  return out_vec;
}

Eigen::MatrixXd extract_matrix(const char* in_file) {
  Eigen::MatrixXd out_mat;
  ddIO mat_io;

  bool success = mat_io.open(in_file, ddIOflag::READ);

  if (success) {
    // get matrix size
    unsigned long mat_size[2] = {0, 0};
    const char* line = mat_io.readNextLine();
    char* nxt_num = nullptr;
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
      const char* curr_row = line;
      while (*curr_row) {
        char* nxt_float = nullptr;
        // printf("%s\n", curr_row);
        out_mat(r_idx, c_idx) = std::strtof(curr_row, &nxt_float);
        curr_row = nxt_float;

        c_idx++;
      }

      line = mat_io.readNextLine();
      r_idx++;
    }
    // std::cout << out_mat << "\n";
  }

  return out_mat;
}

void get_points(std::vector<Eigen::VectorXd>& v_bin,
                dd_array<glm::vec3>& out_bin, const unsigned idx,
                const VectorOut type) {
  if (type == VectorOut::INPUT) {
    // use all values
    if ((int)out_bin.size() != (v_bin[idx].size() / 2)) {
      out_bin.resize(v_bin[idx].size() / 2);
    }

    unsigned c_idx = 0;

    // Oral commisure (L) x,Oral commisure (L) y,
    out_bin[c_idx] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Oral commisure (R) x,Oral commisure (R) y,
    out_bin[c_idx] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Dental show (Top) x,Dental show (Top) y,
    out_bin[c_idx] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Dental show (Bottom) x, Dental show (Bottom) y
    out_bin[c_idx] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
  } else {
    // skip 1st 4 values (remove delta values)
    if ((int)out_bin.size() != ((v_bin[idx].size() - 4) / 2)) {
      out_bin.resize((v_bin[idx].size() - 4) / 2);
    }
    unsigned c_idx = 2;

    // Lateral canthus (L) x,Lateral canthus (L) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Lateral canthus (R) x,Lateral canthus (R) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Palpebral fissure (RU) x,Palpebral fissure (RU) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Palpebral fissure (RL) x,Palpebral fissure (RL) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Palpebral fissure (LU) x,Palpebral fissure (LU) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Palpebral fissure (LL) x,Palpebral fissure (LL) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Depressor (L) x,Depressor (L) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Depressor (R) x,Depressor (R) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Depressor (M) x,Depressor (M) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Iris (M) x,Iris (M) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Iris (L) x,Iris (L) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Nasal ala (L) x,Nasal ala (L) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Nasal ala (R) x,Nasal ala (R) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Medial brow (L) x,Medial brow (L) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Medial brow (R) x,Medial brow (R) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Malar eminence (L) x,Malar eminence (L) y,
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
    c_idx++;
    // Malar eminence (R) x,Malar eminence (R) y
    out_bin[c_idx - 2] =
        glm::vec3(v_bin[idx](c_idx * 2), v_bin[idx](c_idx * 2 + 1), 0.f);
  }
}

void get_points(Eigen::VectorXd& input, std::vector<Eigen::MatrixXd>& weights,
                std::vector<Eigen::VectorXd>& biases,
                dd_array<glm::vec3>& output) {
  std::vector<double> out_d = feedForward(input, weights, biases);

  // skip 1st 4 values
  if (output.size() != ((out_d.size() - 4) / 2)) {
    output.resize((out_d.size() - 4) / 2);
  }
  unsigned c_idx = 2;

  // Lateral canthus (L) x,Lateral canthus (L) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Lateral canthus (R) x,Lateral canthus (R) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Palpebral fissure (RU) x,Palpebral fissure (RU) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Palpebral fissure (RL) x,Palpebral fissure (RL) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Palpebral fissure (LU) x,Palpebral fissure (LU) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Palpebral fissure (LL) x,Palpebral fissure (LL) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Depressor (L) x,Depressor (L) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Depressor (R) x,Depressor (R) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Depressor (M) x,Depressor (M) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Iris (M) x,Iris (M) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Iris (L) x,Iris (L) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Nasal ala (L) x,Nasal ala (L) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Nasal ala (R) x,Nasal ala (R) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Medial brow (L) x,Medial brow (L) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Medial brow (R) x,Medial brow (R) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Malar eminence (L) x,Malar eminence (L) y,
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
  c_idx++;
  // Malar eminence (R) x,Malar eminence (R) y
  output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
}