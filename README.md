# Evolving Soft Robots

This repository contains an evolutionary algorithm designed to evolve soft robots. The framework is built using basic NumPy operations, employing Numba decorators for optimization, and integrates <a href="https://www.taichi-lang.org/">Taichi</a> for the simulation part.

## Dependencies
The following libraries are required:
| Package               | Version (>=) |
|-----------------------|--------------|
| numpy                 | 1.25.2       |
| taichi                | 1.6.0        |
| numba                 | 0.58.0       |
| matplotlib (optional) | 3.7.1        |
| gmsh (optional)       | 4.11.1       |

## Evolution process
The objective of the Evolutionary Algorithm is to optimize the speed of a robot. The robot's motion is simulated using a spring-mass system, wherein the rest length (L0) of each spring is dynamically adjusted based on the formula: $$L0 = L0 * (a + b * sin(ωt + c))$$, where 'a', 'b', and 'c' represent spring parameters, and ω denotes the global frequency. A predefined set of materials is provided, each characterized by specific values for 'a', 'b', 'c', and the spring constant 'k', along with a centroid defining its position in the 3D space. Before the simulation, each robot spring selects a material from the material set based on the closest Euclidean distance. Furthermore, an additional material, with zero values for all parameters, is included to designate voids within the robot structure. Within the EA algorithm, the centroids of the material set are iteratively updated to optimize the robot's speed.

## Usage
To run the Evolutionary Algorithm (EA) algorithm, execute `main.py`. 

To visualize the saved robots, execute `test.py`.
 

## Robot Generation
The basic geometry consists of a cuboid structure described as (n×cubesx, m×cubesy, l×cubesz). Additionally, there is an option to load a geometry from a .msh file. Two files are provided: one for a human body and another for a bb8-type robot from Star Wars.

## Evolved robots 

### (2x2x2) cuboid structure
![GIF](2x2.gif)

### Crawling humanoid
![GIF](human.gif)

