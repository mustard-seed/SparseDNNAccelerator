#include "aocl_utils_cpp.hpp"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <vector>

namespace aocl_utils_cpp {

Options::Options() {
}

Options::Options(int num, char *argv[]) {
  addFromCommandLine(num, argv);
}

bool Options::has(const std::string &name) const {
  return m_options.find(name) != m_options.end();
}

std::string &Options::get(const std::string &name) {
  return m_options[name];
}

const std::string &Options::get(const std::string &name) const {
  OptionMap::const_iterator it = m_options.find(name);
  if(it == m_options.end()) {
    errorNonExistent(name);
    std::cerr << "Option '" << name << "' does not exist.\n";
    exit(1);
  }
  return it->second;
}

void Options::addFromCommandLine(int num, char *argv[]) {
  for(int i = 1; i < num; ++i) {
    const std::string arg = argv[i];

    // Look for the first '-'.
    if(arg.size() > 1 && arg[0] == '-') {
      size_t eq = arg.find('=');
      size_t name_start = 1;

      // Check if there's a second '-'.
      if(arg.size() > 2 && arg[1] == '-') {
        name_start = 2;
      }

      if(eq == std::string::npos) {
        // No '='; treat as a boolean option.
        set(arg.substr(name_start), true);
      }
      else if(eq == name_start) {
        // No name?!
        errorNameless();
      }
      else {
        set(arg.substr(name_start, eq - name_start), arg.substr(eq + 1));
      }
    }
    else {
      // Not an option.
      m_nonoptions.push_back(arg);
    }
  }
}

void Options::errorNameless() const {
  std::cerr << "No name provided for option.\n";
  exit(1);
}

void Options::errorNonExistent(const std::string &name) const {
  std::cerr << "Option '" << name << "' does not exist.\n";
  exit(1);
}

void Options::errorWrongType(const std::string &name) const {
  std::cerr << "Value for option '" << name << "' is not of the right type (value = '"
            << get(name) << "').\n";
  exit(1);
}

} // ns aocl_utils_cpp

