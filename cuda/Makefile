CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : main.cpp
	g++ $(CFLAGS) main.cpp $(LIBS) -o hyper.out
clean:
	rm hyper.out
