# Hypergrah Matching

Algoritmo de correspondencia entre imágenes usando hipergrafos.

---
# Dependencias

## Dependencias Generales

* Instalar Opencv 2.4.13 como se indica [aquí](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)

## Dependencia extra para código en python

* Instalar pip como se idica [aquí](https://pip.pypa.io/en/stable/installing/)

* Instalar Numpy con pip así:
```sh
sudo pip install numpy
```
---
# Compilar y ejecutar código secuencial en C/C++

Desde la carpeta del proyecto

```sh
cd cpp_hyperMatching_code
```

luego compilar usando simplemente:

```sh
make
```

si por alguna razón no se tiene ``make`` usar:

```sh
g++ `pkg-config --cflags --libs opencv` main.cpp -o hiper.out
```

finalmente para ejecutar el código
```sh
./hiper.out
```
---
# Ejecutar código secuencial en python

Desde la carpeta del proyecto

```sh
cd python_hyperMatching_code
```

luego ejecutar:
```sh
python main.py
```
---
# Compilar y correr código con CUDA

## Lanzar antes de compilar un programa en cuda
```sh
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
```
## &
```
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
## Compilar en CUDA
```sh
nvcc programa.cu -o programa
```
---
# Profiling using GPROF.
```sh
g++ -std=c++11 -Wall -pg `pkg-config --cflags opencv` main.cpp `pkg-config --libs opencv` -o hiper.out
```
## Ejecutar
```sh
./hiper.out
```
## Crear archivo de profiling
```sh
gprof hiper.out gmon.out > analysis.txt
```
## Crear gráfico
```sh
perf record -g -- ./hiper.out
```
## guardar en png
```sh
perf script | c++filt | python -m gprof2dot -f perf | dot -Tpng -o profiling.png
```
---
# Autores

* Leiver Campeón - [leiverandres](https://github.com/leiverandres)
* Yensy Gomez - [YensyGomez](https://github.com/YensyGomez)
* John Osorio - [kala855](https://github.com/kala855)
* Sebastián Vega - [sebasvega95](https://github.com/sebasvega95)
