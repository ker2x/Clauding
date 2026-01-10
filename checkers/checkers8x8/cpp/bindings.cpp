
#include "game.h"
#include "mcts.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(checkers_cpp, m) {
  m.doc() = "Fast Checkers C++ Extension";

  // Bind Game Class
  py::class_<checkers::Game>(m, "Game")
      .def(py::init<>())
      .def(py::init<const checkers::Game &>())
      .def("get_legal_actions", &checkers::Game::get_legal_actions)
      .def("make_action", &checkers::Game::make_action)
      .def("is_terminal", &checkers::Game::is_terminal)
      .def("get_result", &checkers::Game::get_result)
      .def("to_neural_input", &checkers::Game::to_neural_input)
      .def("clone", &checkers::Game::clone)
      // Accessors for debugging if needed
      .def_readwrite("move_count", &checkers::Game::move_count)
      .def_readwrite("current_player", &checkers::Game::current_player)
      .def_readwrite("player_men", &checkers::Game::player_men)
      .def_readwrite("player_kings", &checkers::Game::player_kings)
      .def_readwrite("opponent_men", &checkers::Game::opponent_men)
      .def_readwrite("opponent_kings", &checkers::Game::opponent_kings)
      .def_readwrite("position_history", &checkers::Game::position_history);

  // Bind MCTS Class
  py::class_<checkers::MCTS>(m, "MCTS")
      .def(py::init<float, int, float, float>(), py::arg("c_puct"),
           py::arg("num_simulations"), py::arg("dirichlet_alpha"),
           py::arg("dirichlet_epsilon"))
      .def("start_search", &checkers::MCTS::start_search)
      .def("is_finished", &checkers::MCTS::is_finished)
      .def("find_leaves", &checkers::MCTS::find_leaves)
      .def("process_results", &checkers::MCTS::process_results)
      .def("get_policy", &checkers::MCTS::get_policy)
      .def("get_root_visit_count", &checkers::MCTS::get_root_visit_count);
}
