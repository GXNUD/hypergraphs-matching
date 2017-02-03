
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char *argv[])
{
    char buffer[80];
    vector<double> a;
    vector<double> b;
    vector<double> c;

    a.push_back(999.25);
    a.push_back(888.50);
    a.push_back(777.25);

    b.push_back(999.25);
    b.push_back(888.50);
    b.push_back(777.25);

    c.push_back(0.0);
    c.push_back(0.0);
    c.push_back(0.0);



    for(int i = 0; i < c.size(); i++)
    {
        c[i]=a[i]+b[i];
      cout << c[i] << endl;
    }

    cout << "----------" << endl;
    return EXIT_SUCCESS;
}
