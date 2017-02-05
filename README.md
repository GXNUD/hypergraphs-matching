# Hypergrah Matching

## Linea de compilación del código en "Cpp hyperMatching code":
```sh
g++ `pkg-config --cflags --libs opencv` hipergraphMatching.cpp -o hiper.out
```

## Compilar con c++11:

```sh
g++ -std=c++11 `pkg-config --cflags --libs opencv` hipergraphMatching.cpp -o hiper.out
```
## or
```sh
g++ -std=c++11 `pkg-config --cflags opencv` hipergraphMatching.cpp `pkg-config --libs opencv` -o hiper.out
```
#Lanzar antes de compilar un programa en cuda 
```sh
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
```
##&
```
$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
#Compilar en CUDA
```sh
nvcc programa.cu -o programa
```


