#include <vector>
#include <algorithm>
/*
  This is an emulation of the itertools.combinations function of python
  Based in an answers in StackOverflow
*/
using namespace std;


vector<vector<int> > getCombination(int n, int r) {
  vector<bool> v(n);
  vector<vector<int> > combinations;
  fill(v.begin(), v.begin() + r, true);
  do {
    vector<int> one_combination;
    for (int i = 0; i < n; ++i) {
      if (v[i]) {
          one_combination.push_back(i);
      }
    }
    combinations.push_back(one_combination);
  } while (prev_permutation(v.begin(), v.end()));
  return combinations;
}
