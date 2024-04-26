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
| gmsh (optional)       | 4.11.1       |

## Evolution process
The objective of the EA is to maximize the speed of a robot. The robot motion is simulated using a spring-mass simulator, where the rest length L0 of each spring in the robot is updated according to : L0 = L0 ∗(a+bsin(ωt+c)), where a, b, c are spring parameters and ω the global frequency. A predefined set of material for a,b,c and spring constant k is prescribed as well as a centroid for each one of them. The springs of the robot adopt the material with the closest euclidean distance. An additional material with zero values for all the parameters is included to prescribe voids in the robot structure. In EA algorithm the centroids of the set of materials is iteratively updated to maximize the speed of the robot.

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

