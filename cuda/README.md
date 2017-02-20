# Hypergraph Matching

## CUDA Version

In this folder you can compile and run the CUDA version of our
Hypergraph Matching Code, the goal is to have an incremental CUDA
implementation.

We are going to start using the OpenCV-CUDA version of some algorithms
like SURF, and trying to re-create new versions of functions like
"delaunay triangulation".

Remember always when you compile the code have the libraries and
binaries of *nvcc* correctly added to the path.

```sh
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
```
```sh
export
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```


