import taichi as ti 
import numpy as np 
from Generate_robot import create_floor,create_robot
import time 

#With rendering 
@ti.data_oriented
class Simulator:
    def __init__(self,
                 coords:np.ndarray,
                 edges:np.ndarray,
                 springs_L0:np.ndarray,
                 particle_con:np.ndarray,
                 particle_con_sign:np.ndarray,
                 rendering:bool=False,
                 structure_faces:np.ndarray=None,
                 floor_coords:np.ndarray=None,
                 floor_faces:np.ndarray=None,
                 floor_vertices_color:np.ndarray=None,
                 material:np.ndarray=None):
        
        self.rendering = rendering
        #Simulator param
        self.dt = 1e-4
        self.Kg = 1e5
        self.gravity = -9.81
        self.a =1
        self.omega =  2*ti.math.pi
        ####################
        #Initial parameters
        self.coords = coords
        self.edges = edges
        self.L0 = springs_L0
        if structure_faces is not None:
            self.structure_faces = structure_faces

        n_particles = coords.shape[0]
        n_springs = edges.shape[0]
        ####################
        #Particles attributes
        self.n_particles = n_particles
        self.mass = ti.field(dtype=ti.float32, shape=n_particles)
        self.pos = ti.Vector.field(3,dtype=ti.float32, shape=(n_particles))                                     # position
        self.vel = ti.Vector.field(3,dtype=ti.float32, shape=(n_particles))                                     # velocity
        self.Force = ti.Vector.field(3,dtype=ti.float32, shape=(n_particles))                                   # forces 
        self.Con = ti.Vector.field(particle_con.shape[1],dtype=ti.int32,shape=(n_particles))                    # connectivity
        self.Con_direction =  ti.Vector.field(particle_con_sign.shape[1],dtype=ti.float32,shape=(n_particles))  # connectivity direction
        self.pos_proj = ti.Vector.field(3,dtype=ti.float32, shape=(n_particles))
        ####################
        #Spring attributes
        self.springs = ti.Vector.field(2,dtype=ti.int32, shape=(n_springs))
        self.springs_L0 = ti.field(dtype=ti.float32, shape=(n_springs))
        self.k_springs = ti.field(dtype=ti.float32, shape=(n_springs))
        self.b_springs = ti.field(dtype=ti.float32, shape=(n_springs))
        self.c_springs = ti.field(dtype=ti.float32, shape=(n_springs))
        
        #Rendering
        if rendering:
            self.faces = ti.field(ti.int32, structure_faces.shape[0]) 
            self.floor_pos = ti.Vector.field(3,dtype=ti.float32, shape=(floor_coords.shape[0])) 
            self.floor_face = ti.field(ti.int32, floor_faces.shape[0]) 
            self.floor_color = ti.Vector.field(3,dtype=ti.float32, shape=(floor_vertices_color.shape[0])) 

            #Fill the fields with np arrays
            self.faces.from_numpy(structure_faces)
            self.floor_pos.from_numpy(floor_coords)
            self.Con.from_numpy(particle_con)
            self.Con_direction.from_numpy(particle_con_sign)
            self.floor_face.from_numpy(floor_faces)
            self.floor_color.from_numpy(floor_vertices_color)

            self.material = material
            #To visualize material centroids
            self.material = ti.Vector.field(3,dtype=ti.float32, shape=(material.shape[0]))
            self.material_color = ti.Vector.field(3,dtype=ti.float32, shape=(material.shape[0]))
            self.material.from_numpy(material)
            self.material_color.from_numpy(np.array([[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,0,0]],dtype=np.float32))
            self.step = 500
        else:
            self.step = 1

    def initialize(self):
        #Load initial structure
        z0 = 0.0
        p_mass = 0.125
        self.mass.fill(p_mass)
        self.springs.from_numpy(self.edges)
        self.pos.from_numpy(self.coords+np.array([0.0,0.0,z0],dtype=np.float32))
        self.springs_L0.from_numpy(self.L0)
        self.vel.fill(0.0)
        self.Force.fill(0.0)
        self.T = 0.0
    
    def load_parameters(self,k,b,c):

        self.k_springs.from_numpy(k)
        self.b_springs.from_numpy(b)
        self.c_springs.from_numpy(c)
        self.k = k
        act = np.unique(self.edges[k>0].reshape(-1,))
        #Create a boolean mask for active particles
        self.active = np.zeros(self.n_particles,dtype=bool)
        self.active[act] = True 

    def get_active_structure(self,edge_id):
        #Crete active edges
        active_edge = self.k[self.k>0].shape
        self.active_springs = ti.Vector.field(2,dtype=ti.int32, shape=(active_edge))
        self.active_springs.from_numpy(self.edges[self.k>0])
        
        self.active_springs_proj = ti.Vector.field(2,dtype=ti.int32, shape=(active_edge))
        self.active_springs_proj.from_numpy(self.edges[self.k>0])

        #Crete active particles    
        act = np.unique(self.edges[self.k>0].reshape(-1,))
        #Create a boolean mask for active particles
        self.active = np.zeros(self.n_particles,dtype=bool)
        self.active[act] = True

        self.active_pos = ti.Vector.field(3,dtype=ti.float32, shape=(act.shape[0]))
        self.active_pos_proj = ti.Vector.field(3,dtype=ti.float32, shape=(act.shape[0]))
        
        active_face = []
        for i,face in enumerate(self.structure_faces.reshape(-1,3)):
            if set(face).issubset(set(act)):
                active_face.append(face[0])
                active_face.append(face[1])
                active_face.append(face[2])
        active_face = np.array(active_face,dtype=np.int32)
        self.active_faces = ti.field(ti.int32, active_face.shape[0]) 
        self.active_faces.from_numpy(active_face)

        self.active_faces_proj = ti.field(ti.int32, active_face.shape[0]) 
        self.active_faces_proj.from_numpy(active_face)

        # colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0)]*self.material.shape[0]
        # edge_color = []
        # for i,id in enumerate(edge_id):
        #     edge_color.append(colors[id])
        
        # edge_color = np.array(edge_color,dtype=np.float32)
        # self.edge_color = ti.Vector.field(3,dtype=ti.float32, shape=(edge_color.shape[0])) 
        # self.edge_color.from_numpy(edge_color)

    @ti.kernel
    def step_ti(self,T:float):
        # #Initialize by zeroing 
        self.Force.fill(0.0)
        for x in self.mass:
            for y in ti.static(range(70)):
                if self.Con[x][y]!=-1:
                    edge = self.Con[x][y]
                    edge_sign =  self.Con_direction[x][y]
                    L = (self.pos[self.springs[edge][1]] - self.pos[self.springs[edge][0]]).norm()
                    L0 = (self.a+self.b_springs[edge]*ti.sin(self.omega*T+self.c_springs[edge]))*self.springs_L0[edge]
                    #L0 = self.springs_L0[edge]
                    spring_force = self.k_springs[edge]*(L-L0)
                    direction = edge_sign*(self.pos[self.springs[edge][0]] - self.pos[self.springs[edge][1]])
                    self.Force[x] += direction * spring_force
                       
        for x in self.mass:
            self.Force[x] += ti.Vector([0,0,self.gravity])*self.mass[x]
            if self.pos[x][2]<=0:
                Fn = ti.Vector([0.0,0.0,self.Force[x][2]])
                Fh = ti.Vector([self.Force[x][0],self.Force[x][1],0.0])
                if Fh.norm()<0.74*Fn.norm():
                    self.Force[x] -= Fh
                else:
                    self.Force[x] -=0.57*Fn.norm()*Fh/Fh.norm()
                self.Force[x][2] -= self.pos[x][2]*self.Kg
            Acceleration =  self.Force[x]/self.mass[x]
            self.vel[x] += self.dt*Acceleration
            self.vel[x] *= 0.999
            self.pos[x] += self.dt*self.vel[x]
            self.pos_proj[x] = ti.Vector([self.pos[x][0],self.pos[x][1],0.0001])

    def run_par(self,iterations:int,pop:int,eval:bool):
        if not eval:
            prev_pos = self.pos.to_numpy()

        if self.rendering:
            window = ti.ui.Window("Simulator", (768, 768),show_window=True)
            canvas = window.get_canvas()
            scene = ti.ui.Scene()
            camera = ti.ui.Camera()
            camera.position(5, 5, 5) 
            camera.lookat(0, 0, 0)     
            camera.up(0, 0, 1)          
            scene.set_camera(camera)
        
        for _ in range(iterations):
            for _ in range(self.step):
                self.step_ti(self.T)
                self.T += self.dt
            if self.rendering:
                #Only_Active_structure
                scene.ambient_light((0.8, 0.8, 0.8))
                scene.point_light(pos=(1.5, 1.5, 1.5), color=(1, 1, 1))
                scene.mesh(self.floor_pos,indices=self.floor_face,per_vertex_color = self.floor_color,show_wireframe=False,normals=False)
                
                self.active_pos_proj.from_numpy(self.pos_proj.to_numpy()[self.active])
                self.active_pos.from_numpy((self.pos.to_numpy()[self.active]))
                scene.particles(self.active_pos,color = (0.5, 0, 0), radius = 0.02)
                scene.particles(self.active_pos_proj, color = (0, 0, 0), radius = 0.02)
                
                #scene.lines(vertices=self.pos,indices=self.active_springs,color=(1,0.2,0), width = 2.0)
                #scene.lines(vertices=self.pos_proj,indices=self.active_springs_proj,color = (0, 0, 0), width = 2.0)
                scene.mesh(self.pos,indices=self.active_faces,color = (1,0.5,0),show_wireframe=False)
                scene.mesh(self.pos_proj,indices=self.active_faces_proj,color = (0,0.0,0.0),show_wireframe=False)

                #All_structure
                #scene.particles(self.pos, color = (0.68, 0.26, 0.19), radius = 0.02)
                #scene.lines(vertices=self.pos,indices=self.springs,color = (0.28, 0.68, 0.99), width = 3.0)
                #scene.mesh(self.pos,indices=self.faces,color = (0,0.5,0.5),show_wireframe=False)
            
                #Material centroids
                #scene.particles(self.material,per_vertex_color = self.material_color, radius = 0.2)
                canvas.scene(scene)
                window.show()

        if not eval:
            dist = np.zeros(pop,dtype=np.float32)
            for i in range(pop):
                start_index = i * int(self.n_particles/pop)
                end_index = (i + 1) * int(self.n_particles/pop)
                active_p = self.active[start_index:end_index]
                x_cm_prev = np.mean(prev_pos[start_index:end_index][active_p],axis=0)
                x_cm_new = np.mean(self.pos.to_numpy()[start_index:end_index][active_p],axis=0)
                dist[i] = np.linalg.norm(x_cm_prev[:2]-x_cm_new[:2])
            return  dist
        else:
            return None