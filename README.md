# Hypergrah Matching

## Linea de compilación del código en "Cpp hyperMatching code":
```sh
g++ `pkg-config --cflags --libs opencv` main.cpp -o hiper.out
```

## Compilar con c++11:

```sh
g++ -std=c++11 `pkg-config --cflags --libs opencv` main.cpp -o hiper.out
```
## or
```sh
g++ -std=c++11 `pkg-config --cflags opencv` main.cpp `pkg-config --libs opencv` -o hiper.out
```
#Lanzar antes de compilar un programa en cuda 
```sh
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
```
##&
```
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
#Compilar en CUDA
```sh
nvcc programa.cu -o programa
```

#Profiling using GPROF.
```sh
g++ -std=c++11 -Wall -pg `pkg-config --cflags opencv` main.cpp `pkg-config --libs opencv` -o hiper.out
```
##Ejecutar 
```sh
./hiper.out 
```
##Crear archivo de profiling
```sh
gprof hiper.out gmon.out > analysis.txt
```
#Crear gráfico 
```sh
perf record -g -- ./test_gprof 
```
##guardar en png
```sh
perf script | c++filt | python -m gprof2dot -f perf | dot -Tpng -o profiling.png
```
