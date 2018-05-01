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
      out_vec(idx) = std::strtod(line, NULL);

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
        char* nxt_dbl = nullptr;
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
        char* nxt_dbl = nullptr;
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

void get_points(std::vector<Eigen::VectorXd>& v_bin,
                dd_array<glm::vec3>& out_bin, const unsigned idx,
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
    if ((int)out_bin.size() != ((v_bin[idx].size() - 4) / 2)) {
      out_bin.resize((v_bin[idx].size() - 4) / 2);
    }
    unsigned c_idx = 2;

		while ((c_idx - 2) < out_bin.size()) {
			out_bin[c_idx - 2] =
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

void get_points(Eigen::VectorXd& input, std::vector<Eigen::MatrixXd>& weights,
                std::vector<Eigen::VectorXd>& biases,
                dd_array<glm::vec3>& output) {
  std::vector<double> out_d = feedForward(input, weights, biases);

  // skip 1st 4 values
  if (output.size() != ((out_d.size() - 4) / 2)) {
    output.resize((out_d.size() - 4) / 2);
  }
  unsigned c_idx = 2;

	while ((c_idx - 2) < output.size()) {
		output[c_idx - 2] = glm::vec3(out_d[c_idx * 2], out_d[c_idx * 2 + 1], 0.f);
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
