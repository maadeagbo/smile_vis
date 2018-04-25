#include <vector>
#include "Eigen/Core"
#include "ddIncludes.h"

enum class VectorOut { INPUT, OUTPUT };

/** \brief Pipe input thru neural net matrices */
std::vector<double> feedForward(Eigen::VectorXd& inputs,
                                std::vector<Eigen::MatrixXd>& weights,
                                std::vector<Eigen::VectorXd>& biases);

/** \brief Get 1D eigen vector from input file */
Eigen::VectorXd extract_vector(const char* in_file);

/** \brief Get vector of 1D eigen vector from input file */
std::vector<Eigen::VectorXd> extract_vector2(const char* in_file);

/** \brief Get 2D eigen matrix from input file */
Eigen::MatrixXd extract_matrix(const char* in_file);

/** \brief Convert eigen vector to array of glm::vec3 */
std::vector<glm::vec3> get_points(std::vector<Eigen::VectorXd>& v_bin,
                                  const unsigned idx, const VectorOut type);