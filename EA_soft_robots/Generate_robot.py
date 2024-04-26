import numpy as np 
import gmsh
import sys
from typing import Union,List
'''
Functions to generate an initial
structure for the robot
'''
#Load mesh generated from gmsh
#----------------------------------------------------------------------
def Load_mesh():
    file = './robot_mesh/bb8_true.msh'   #bb8 like robot 
    #file = './robot_mesh/88_elements.msh'
    np.set_printoptions(threshold=sys.maxsize)
    gmsh.initialize()
    gmsh.open(file)

    cube_indices = (gmsh.model.mesh.get_elements_by_type(5)[1]).reshape(-1,8)-1
    coords_id = gmsh.model.mesh.get_nodes()[0] -1 
    coords = (gmsh.model.mesh.get_nodes()[1]).reshape(-1,3)[:,:3] 

    coord_dic = {f'{coords_id[i]}':i for i in range(coords_id.shape[0])}
    #Fix order
    for i in range(cube_indices.shape[0]):
        for j in range(cube_indices.shape[1]):
            cube_indices[i,j] = coord_dic[f'{cube_indices[i,j]}']
    return cube_indices,coords

#Add springs to the domain
def get_springs(indices,coords):
    mesh_faces = []
    edges = []
    for cube_indices in indices:
        edges.extend(cube_edges(cube_indices))
                
        cube_faces = [
            cube_indices[0], cube_indices[1], cube_indices[2],
            cube_indices[0], cube_indices[2], cube_indices[3],
            cube_indices[4], cube_indices[5], cube_indices[6],
            cube_indices[4], cube_indices[6], cube_indices[7],
            cube_indices[0], cube_indices[4], cube_indices[7],
            cube_indices[0], cube_indices[7], cube_indices[3],
            cube_indices[1], cube_indices[5], cube_indices[2],
            cube_indices[2], cube_indices[6], cube_indices[5],
            cube_indices[0], cube_indices[1], cube_indices[4],
            cube_indices[4], cube_indices[5], cube_indices[1],
            cube_indices[3], cube_indices[2], cube_indices[7],
            cube_indices[7], cube_indices[6], cube_indices[2]
        ]
        mesh_faces.extend(cube_faces)


    edges = np.array(edges,dtype=np.int32)
    coords = np.array(coords,dtype=np.float32)
    faces = np.array(mesh_faces,dtype=np.int32)
    
    spring_L0 = np.array([np.sqrt(np.sum((coords[i[0]]-coords[i[1]])**2)) for i in edges])
    node_con = np.ones((coords.shape[0],10*7),dtype=np.int32)*(-1)
    node_con_sign = np.zeros((coords.shape[0],10*7),dtype=np.float32)

    for i in range(coords.shape[0]):
        f,sign = np.where(edges==i)
        sign = np.where(sign==0,-1,1)
        node_con[i,:f.shape[0]] = f
        node_con_sign[i,:f.shape[0]] = sign

    structure = {
        'coords':coords,
        'edges':edges,
        'springs_L0':spring_L0,
        'structure_faces':faces,
        'particle_con':node_con,
        'particle_con_sign':node_con_sign
        }
    return structure

def cube_edges(indices):
    edges = []
    for i in range(8):
        for j in range(i):
            if indices[j]!=indices[i]:
                edge = (indices[j],indices[i])
                edges.append(edge)
    return edges

def rotate_coordinates(coordinates, theta_x=0, theta_y=0 ,theta_z=0):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]],dtype=np.float32)
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]],dtype=np.float32)
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]],dtype=np.float32)
    
    rotated_coords = coordinates.dot(Rx).dot(Ry).dot(Rz)
    return rotated_coords

#Load robot
def load_robot()->dict:
    cube_indices,coords = Load_mesh()
    structure = get_springs(cube_indices,rotate_coordinates(coords))
    return structure
#----------------------------------------------------------------------


#Create a robot of size (n_elx,n_ely,n_elz)
#----------------------------------------------------------------------
def create_robot(domain_size:Union[tuple,List])->dict:
    mesh_vertices = []
    mesh_faces = []
    vertex_dict = {}
    edges = []

    def get_or_add_vertex_index(vertex):
        if vertex not in vertex_dict:
            vertex_dict[vertex] = len(mesh_vertices)
            mesh_vertices.append(vertex)
        return vertex_dict[vertex]
    
    for x in range(domain_size[0]):
        for y in range(domain_size[1]):
            for z in range(domain_size[2]):
                # Define the vertices of the cube
                cube_vertices = [
                    (x, y, z),
                    (x + 1, y, z),
                    (x + 1, y + 1, z),
                    (x, y + 1, z),
                    (x, y, z + 1),
                    (x + 1, y, z + 1),
                    (x + 1, y + 1, z + 1),
                    (x, y + 1, z + 1)
                ]
                # Append the vertices and indices to the mesh
                cube_indices = [get_or_add_vertex_index(v) for v in cube_vertices]
                edges.extend(cube_edges(cube_indices))
                
                cube_faces = [
                    cube_indices[0], cube_indices[1], cube_indices[2],
                    cube_indices[0], cube_indices[2], cube_indices[3],
                    cube_indices[4], cube_indices[5], cube_indices[6],
                    cube_indices[4], cube_indices[6], cube_indices[7],
                    cube_indices[0], cube_indices[4], cube_indices[7],
                    cube_indices[0], cube_indices[7], cube_indices[3],
                    cube_indices[1], cube_indices[5], cube_indices[2],
                    cube_indices[2], cube_indices[6], cube_indices[5],
                    cube_indices[0], cube_indices[1], cube_indices[4],
                    cube_indices[4], cube_indices[5], cube_indices[1],
                    cube_indices[3], cube_indices[2], cube_indices[7],
                    cube_indices[7], cube_indices[6], cube_indices[2]
                ]
                mesh_faces.extend(cube_faces)
    
    edges = np.array(edges,dtype=np.int32)
    coords = np.array(mesh_vertices,dtype=np.float32)
    faces = np.array(mesh_faces,dtype=np.int32)
    spring_L0 = np.array([np.sqrt(np.sum((coords[i[0]]-coords[i[1]])**2)) for i in edges])

    #Get the node conectivity
    node_con = np.ones((coords.shape[0],7*10),dtype=np.int32)*(-1)
    node_con_sign = np.zeros((coords.shape[0],7*10),dtype=np.float32)

    for i in range(coords.shape[0]):
        f,sign = np.where(edges==i)
        sign = np.where(sign==0,-1,1)
        node_con[i,:f.shape[0]] = f
        node_con_sign[i,:f.shape[0]] = sign
    
    structure ={
        'coords':coords,
        'edges':edges,
        'springs_L0':spring_L0,
        'structure_faces':faces,
        'particle_con':node_con,
        'particle_con_sign':node_con_sign
        }
    return structure
#----------------------------------------------------------------------

#Functions to create a floor for visualization
#----------------------------------------------------------------------
def create_floor(X,Y,num):
    x,y,z = np.meshgrid(np.linspace(-X/2,X/2,num+1),np.linspace(-Y/2,Y/2,num+1),[0]) 
    coords = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)),axis=1,dtype=np.float32)
    vertices = {}
    mesh_faces = []
    
    for id,c in enumerate(coords):
        key = (c[0],c[1],c[2])
        vertices[key] = id

    for x in range(num):
        for y in range(num):
                # Define the vertices of the cube
                square_vertices = [(x-num/2, y-num/2,0 ),
                                    (x + 1-num/2, y-num/2, 0),
                                    (x + 1-num/2, y + 1-num/2, 0),
                                    (x-num/2, y + 1-num/2,0)]
                square_indices = [vertices[i] for i in square_vertices]
                cube_faces = [
                square_indices[0], square_indices[1], square_indices[2],
                square_indices[2], square_indices[3], square_indices[0]]
                mesh_faces.extend(cube_faces)
   
    faces = np.array(mesh_faces,dtype=np.int32)
    return coords,faces

#Create a colored floor 
def create_floor_2(X,Y,num):
    x,y,z = np.meshgrid(np.linspace(-X/2,X/2,num+1),np.linspace(-Y/2,Y/2,num+1),[0]) 
    coords = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)),axis=1,dtype=np.float32)
    vertices = {}
    mesh_faces = []
    faces_color = []
    for id,c in enumerate(coords):
        key = (c[0],c[1],c[2])
        vertices[key] = id

    for x in range(num):
        for y in range(num):
                # Define the vertices of the cube
                square_vertices = [(x-num/2, y-num/2,0 ),
                                    (x + 1-num/2, y-num/2, 0),
                                    (x + 1-num/2, y + 1-num/2, 0),
                                    (x-num/2, y + 1-num/2,0)]
                square_indices = [vertices[i] for i in square_vertices]
                cube_faces = [
                square_indices[0], square_indices[1], square_indices[2],
                square_indices[2], square_indices[3], square_indices[0]]
                mesh_faces.extend(cube_faces)
   
    faces = np.array(mesh_faces,dtype=np.int32)

    id = 1 
    jd = 0
    for _ in range(0,num+1):
        if (jd//2)%2==0:
            id=1
        else:
            id=2
        for _ in range(0,num,2):
            if id%2==0:
                color = (1,1,1)
            else:
                color = (0.0,0.1,0.1)
            id+=1
            faces_color.append(color)
            faces_color.append(color)
        jd +=1
    
    faces_color = np.array(faces_color,dtype=np.float32)

    floor_structure = {
        'floor_coords':coords,
        'floor_faces':faces,
        'floor_vertices_color':faces_color
        }
    return floor_structure
#----------------------------------------------------------------------