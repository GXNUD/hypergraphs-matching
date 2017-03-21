#include <bits/stdc++.h>

using namespace std;

vector<vector<double> > getPermutation(vector<double> data) {
  vector<vector<double> > perms;
  sort(data.begin(), data.end());
  do {
    perms.push_back(data);
  } while(next_permutation(data.begin(), data.end()));
  return perms;
}

int main() {
  vector<double> sines(3);
  sines[0] = 45.0;
  sines[1] = 15.0;
  sines[2] = 25.0;
  vector<vector<double> > perms = getPermutation(sines);
  for (int i = 0; i < perms.size(); i++) {
    for (int j = 0; j < perms[i].size(); j++) {
      cout << perms[i][j] << " ";
    }
    cout << endl;
  }
  return 0;
}
