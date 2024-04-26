# Evolving Soft Robots

This repository contains an evolutionary algorithm designed to evolve soft robots. The framework is built using basic NumPy operations, employing Numba decorators for optimization, and integrates <a href="https://www.taichi-lang.org/">Taichi</a> for the simulation part, enhancing computational efficiency.

## Dependencies
The following libraries are required:
| Package               | Version (>=) |
|-----------------------|--------------|
| numpy                 | 1.25.2       |
| taichi                | 1.6.0        |
| numba                 | 0.58.0       |
| matplotlib            | 3.7.1        |
| gmsh                  | 4.11.1       |

## Evolution process
The objective of the EA is to maximize the speed of a robot 

For the spring-mass simulator, the following parameters were used: dt = 1e-4 [sec], ground spring Kg = 100.000 [N/m] , gravity g = -9.81 [m/s2], global frequency ω = 2π [Hz], static friction coeficient μs = 0.74 and kinetic friction coeficient μk = 0.57. Dampening is included, where the velocity of each mass at each time step is reduced by a factor of 0.999.The rest length L0 of each spring in the robot is updated according to the following equation: L0 = L0 ∗(a+bsin(ωt+c)) , where a, b, c are spring parameters and ω the global frequency.

## Hyperparameters
The default set of hyperparameters is :

- population size = 100 
-  

## Robot Generation
The basic geometry consists of a cuboid structure described as (n×cubesx, m×cubesy, l×cubesz). Additionally, there is an option to load a geometry from a .msh file. Two files are provided: one for a human body and another for a bb8-type robot from Star Wars.

## Evolved robots 

### (2x2x2) cuboid structure
![GIF](2x2.gif)

### Crawling humanoid
![GIF](human.gif)

