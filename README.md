# Evolving Soft Robots

This repository contains an evolutionary algorithm designed to evolve soft robots. The framework is built using basic NumPy operations, employing Numba decorators for optimization, and integrates Taichi for the simulation part, enhancing computational efficiency.

## Dependencies
The following libraries are required:
| Package               | Version (>=) |
|-----------------------|--------------|
| numpy                 | 1.25.2       |
| taichi                | 1.6.0        |
| numba                 | 0.58.0       |
| matplotlib            | 3.7.1        |
| gmsh                  | 4.11.1       |

## Robot Generation
The basic geometry consists of a cuboid structure described as (n×cubesx, m×cubesy, l×cubesz). Additionally, there is an option to load a geometry from a .msh file. Two files are provided: one for a human body structure and another for a star wars bb8-type structure.




<!-- 
Non-geometric evolved robots
![GIF](human.gif)

Geometric evolved robots
![GIF](2x2.gif)
-->
