#include <fstream>
#include <vector>

int main(int, char **) {
  // read input
  std::vector<int> target;
  std::ifstream ifs("input", std::ifstream::in);
  while (!ifs.eof()) {
    int val;
    ifs >> val;
    target.push_back(val);
  }
  ifs.close();

  // my results
  std::vector<int> result = {1, 2, 3};

  for (size_t i = 0; i < target.size(); ++i) {
    printf("%lu: %i vs %i\n", i, result[i], target[i]);
  }

  // check if result matches target
  if (target == result)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}