#include "Container.h"
#include "Eigen/Core"
#include "ddIncludes.h"
#include <vector>

enum class VectorOut { INPUT, OUTPUT, CALC, OUTPUT_C };

/** \brief Pipe input thru neural net matrices */
std::vector<double> feedForward(Eigen::VectorXd &inputs,
                                std::vector<Eigen::MatrixXd> &weights,
                                std::vector<Eigen::VectorXd> &biases);

/** \brief Get 1D eigen vector from input file */
Eigen::VectorXd extract_vector(const char *in_file);

/** \brief Get vector of 1D eigen vector from input file */
std::vector<Eigen::VectorXd> extract_vector2(const char *in_file);

/** \brief Get 2D eigen matrix from input file */
Eigen::MatrixXd extract_matrix(const char *in_file);

/** \brief Convert eigen vector to array of glm::vec3 */
void get_points(std::vector<Eigen::VectorXd> &v_bin,
                dd_array<glm::vec3> &out_bin, const unsigned idx,
                const VectorOut type);

/** \brief Get calculated points */
void get_points(Eigen::VectorXd &input, std::vector<Eigen::MatrixXd> &weights,
                std::vector<Eigen::VectorXd> &biases,
                dd_array<glm::vec3> &output);

/** \brief Export data into calibrated space */
void export_canonical_data(dd_array<glm::vec3> &input,
                           dd_array<glm::vec3> &ground, const char *dir,
                           const char *gdir, const char *file_id,
                           const glm::vec2 canonical_iris_pos,
                           const float canonical_iris_dist);

/** \brief Export data into calibrated space by folder */
void export_canonical(const char *input_dir, const char *ground_dir,
                      const glm::vec2 canonical_iris_pos,
                      const float canonical_iris_dist);
