CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : main.cpp
  g++ -std=c++11 $(CFLAGS) main.cpp $(LIBS) -o hyper.out
