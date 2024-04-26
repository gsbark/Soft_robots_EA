import numpy as np 
from numba import jit,prange
from simulator import Simulator
from Generate_robot import load_robot,create_robot
import time
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
from typing import Union,List,Any

#Helper functions
#-------------------------------------------------------
def Assemble_pop(
        robot: np.ndarray,
        coords: np.ndarray,
        edges: np.ndarray,
        pop_num:int,
        Material:np.ndarray,
        num_material:int=10,
        ):
    
    '''
    Assemble identical population, for testing perposes only
    '''
    #mat : k , b, c
    Material = Material.repeat(num_material//5,axis=0)

    k_spring = np.empty(shape=(edges.shape[0]),dtype=np.float32)
    b_param = np.empty(shape=(edges.shape[0]),dtype=np.float32)
    c_param = np.empty(shape=(edges.shape[0]),dtype=np.float32)
    edges_id = np.empty(shape=(edges.shape[0]),dtype=np.int32)

    for ide,s in enumerate(edges):
        a = np.zeros(3)
        for ii in range(3):
            test = np.array([coords[s[0]][ii],coords[s[1]][ii]])
            a[ii] = min(test)
        center = np.abs(coords[s[0]]-coords[s[1]])/2 + a
        id = find_closest_center(center,robot)
        edges_id[ide]=id
        mat = Material[id]
        k_spring[ide] = mat[0]
        b_param[ide] = mat[1]
        c_param[ide] = mat[2]
    
    k_spring = np.tile(k_spring,(pop_num,1))
    b_param = np.tile(b_param,(pop_num,1))
    c_param = np.tile(c_param,(pop_num,1))
    
    return k_spring,b_param,c_param,edges_id

@jit(nopython = True)
def find_closest_center(main_center:np.ndarray,other_centers:np.ndarray)->int:
    '''
    Iterate all the centers of material for a given spring
    and find the closest one
    '''
    min_distance = np.inf
    for id,center in enumerate(other_centers):
        distance = np.sqrt(np.sum((main_center-center)**2))
        if distance < min_distance:
            min_distance = distance
            mat = id
    return mat

def get_parallel_structure(
        coords:np.ndarray,
        edges:np.ndarray,
        springs_L0:np.ndarray,
        particle_con:np.ndarray,
        particle_con_sign:np.ndarray,
        structure_faces:np.ndarray,
        population_num:int
        )-> dict:
    
    '''
    Create a structure to evaluate all the population 
    in paralllel
    '''

    par_coords = np.tile(coords,(population_num,1))
    par_springs_L0 = np.tile(springs_L0,(population_num))
    particle_con_sign = np.tile(particle_con_sign,(population_num,1))
    par_edges = edges.copy()
    par_faces = structure_faces.copy()
    par_particle_con = particle_con.copy()

    for _ in range(1, population_num):
        par_edges = np.concatenate([par_edges, 1+np.max(par_edges)+edges], axis=0)  
        par_faces =  np.concatenate([par_faces, 1+np.max(par_faces)+structure_faces], axis=0)  
        
        mask = particle_con!=-1
        new_conn = particle_con.copy()
        new_conn[mask] += 1+np.max(par_particle_con)
        par_particle_con =  np.concatenate([par_particle_con,new_conn], axis=0)  
    
    parallel_structure = {'coords':par_coords,
                          'edges':par_edges,
                          'springs_L0':par_springs_L0,
                          'structure_faces':par_faces,
                          'particle_con':par_particle_con,
                          'particle_con_sign':particle_con_sign}
    
    return parallel_structure

#GA functions
#------------------------------------------------------------------------
def init_population(num:int,coords:np.ndarray,num_material:int=5)->np.ndarray:
    '''
    The population consists of num number of arrays of length
    equal to N representing the centroids of each material
    '''
    pop = np.empty((num,num_material,3),dtype=np.float32)
    for i in range(num):
        centroids = np.array(np.random.uniform(np.amin(coords,axis=0),np.amax(coords,axis=0)))[None]
        for _ in range(num_material-1):
            c_mat = np.random.uniform(np.amin(coords,axis=0),np.amax(coords,axis=0))
            centroids = np.concatenate((centroids,c_mat[None]),axis=0,dtype=np.float32)
        pop[i] = centroids
    return pop


@jit(nopython = True,parallel=True)
def assemble_pop(
    population:np.ndarray,
    coords:np.ndarray,
    edges:np.ndarray,
    Material
    ):
    '''
    To evaluate population we iterate all the population 
    and for each spring of the robot we find the closest 
    material and assign the corresponding properties to 
    each spring
    '''
    #mat : k , b, c
    
    k_spring = np.empty(shape=(population.shape[0],edges.shape[0]),dtype=np.float32)
    b_param = np.empty(shape=(population.shape[0],edges.shape[0]),dtype=np.float32)
    c_param = np.empty(shape=(population.shape[0],edges.shape[0]),dtype=np.float32)
    
    # edges_id = np.empty(shape=(population.shape[0],edges.shape[0]),dtype=np.int32)
    for i in prange(population.shape[0]):
        for ide,s in enumerate(edges):
            a = np.zeros(3)
            for ii in range(3):
                test = np.array([coords[s[0]][ii],coords[s[1]][ii]])
                a[ii] = min(test)
            center = np.abs(coords[s[0]]-coords[s[1]])/2 + a
            id = find_closest_center(center,population[i])
            # edges_id[i,ide]=id
            mat = Material[id]
            k_spring[i,ide] = mat[0]
            b_param[i,ide] = mat[1]
            c_param[i,ide] = mat[2]
    return k_spring,b_param,c_param

def eval_population(
        spring_sim,
        k_springs:np.ndarray,
        b_springs:np.ndarray,
        c_springs:np.ndarray,
        parallel_eval:bool,
        edge_id:np.ndarray=None,
        eval:bool=False
        )->np.ndarray:
    '''
    Run taichi simulator to evaluate population
    '''
    #Parallel evaluation
    if parallel_eval:
        spring_sim.initialize()
        spring_sim.load_parameters(k=k_springs.reshape(-1,),b=b_springs.reshape(-1,),c=c_springs.reshape(-1,))
        if eval ==True:
            spring_sim.get_active_structure(edge_id)
        dist=spring_sim.run_par(iterations=10000,pop=k_springs.shape[0],eval=eval)  
    
    else:
    #Serial evaluation 
        dist = np.zeros(k_springs.shape[0],dtype=np.float32)
        for i in range(k_springs.shape[0]):
            spring_sim.initialize()
            spring_sim.load_parameters(k=k_springs[i],b=b_springs[i],c=c_springs[i])
            dist[i] = spring_sim.run(10000) 

    return dist

@jit(nopython = True)
def select_tour(
    population:np.ndarray,
    score:np.ndarray,
    tour_size:int,
    p_size:int,
    el_size:int,
    num_material:int
    ):
    '''
    Tournament selection method: the parent pool is filled
    by iteratively drawing a number of candidates from the 
    population and selecting the best individual each time
    '''
    fitness = score
    elites = population[np.argsort(fitness)[-el_size:]]
    #Normalize fitness
    fitness = fitness/np.sum(fitness)
    #Fit rest of the parents according to their fitness
    parents = np.empty((p_size,num_material,3),dtype=np.float32)
    #Normalize
    fitness = fitness/np.sum(fitness)
    for k in range(parents.shape[0]):
        samples = np.random.choice(score.shape[0],size=tour_size,replace=False)
        winner = np.argmax(fitness[samples])
        parents[k] = population[samples[winner]]
    return parents,elites

@jit(nopython = True)
def variation(
    population:np.ndarray,
    m_rate:float,
    c_rate:float,
    num:int,
    num_material:int
    ):
    
    #Crossover operator
    new_population = np.empty((num,*population.shape[1:]),dtype=np.float32)
    for ii in range(num):
        if np.random.rand() > c_rate:
            idx = np.random.randint(0,population.shape[0],size=1) 
            new_population[ii] = np.copy(population[idx])
        else:
            parent1, parent2 = population[np.random.choice(population.shape[0],size=2,replace=False)]
            idx = np.random.randint(0,num_material, size=2)
            start, end = np.min(idx), np.max(idx)
            new_population[ii,start:end] = np.copy(parent1[start:end])
            new_population[ii,:start] =  np.copy(parent2[:start])
            new_population[ii,end:] =  np.copy(parent2[end:])
    #Mutation operator
    for ii in range(num):
        if np.random.rand() > m_rate:
               idx = np.random.randint(0,num_material)
               dx1 = np.random.uniform(-np.abs(new_population[ii,idx,0])*0.1,np.abs(new_population[ii,idx,0])*0.1)
               dx2 = np.random.uniform(-np.abs(new_population[ii,idx,1])*0.1,np.abs(new_population[ii,idx,1])*0.1)
               dx3 = np.random.uniform(-np.abs(new_population[ii,idx,2])*0.1,np.abs(new_population[ii,idx,2])*0.1)
               new_population[ii,idx] += np.array([dx1,dx2,dx3]) 
    return new_population

################################################################
################################################################
def run_GA(
        domain:Union[str,tuple],
        runs:int,
        population_num:int,
        Material:np.ndarray,
        save_interval:int=100,
        log_interval:int=10,
        parallel_eval:bool=True,
        par_size:float = 0.5,
        elite_size:float = 0.1,
        crossover_rate:float=0.5,
        mutation_rate:float = 0.5,
        num_material:int = 10
        ):
        
    history = np.empty((runs,population_num),dtype=np.float32)
    population_dist = np.empty(population_num,dtype=np.float32)

    elite_pop = int(np.floor(population_num*elite_size)) 
    parents_pop = int(np.floor(population_num*par_size))  
    
    Material = Material.repeat(num_material//5,axis=0)
    #Load domain
    if domain == 'robot':
        structure = load_robot()
    else:
        structure = create_robot(domain)
    
    #Initialize population (create material centroids)
    population = init_population(num=population_num,
                                 coords=structure['coords'],
                                 num_material=num_material)

    if parallel_eval:
        sim_structure = get_parallel_structure(**structure,population_num=population_num)
    else:
        sim_structure = structure

    #Initialize simulator
    Mass_spring_simulator = Simulator(**sim_structure)
    #Initialize logger 
    writer = SummaryWriter()
    #Start Evolution process
    generation = 1
    
    for ii in range(0,runs):
        time_start = time.perf_counter()
        #Step 1: Assemble/Evaluate Population  
        k_springs,b_springs,c_springs = assemble_pop(population=population,
                                                     coords=structure['coords'],
                                                     edges=structure['edges'],
                                                     Material=Material)       

        population_dist = eval_population(spring_sim=Mass_spring_simulator,
                                          k_springs=k_springs,
                                          b_springs=b_springs,
                                          c_springs =c_springs,
                                          parallel_eval=parallel_eval)
        #Step 2: Create parent set
        parents,elites = select_tour(population=population,
                                     score=population_dist,
                                     tour_size=8,
                                     p_size=parents_pop,
                                     el_size=elite_pop,
                                     num_material=num_material)
        
        #Step 3: Perform variation
        new_population = variation(population=parents,
                                   m_rate=mutation_rate,
                                   c_rate=crossover_rate,
                                   num=population_num-elites.shape[0],
                                   num_material=num_material)
        
        #Update population
        population = np.concatenate((new_population,elites),axis=0)
        #Save robot
        if generation%save_interval==0:
            id = np.argmax(population_dist)
            np.save(f'./saved_robots/gen_{generation}.npy',population[id,:])
        
        #Save logs
        if generation%log_interval==0:
            writer.add_scalar(tag='robot_speed',scalar_value = np.max(population_dist),
                              global_step = generation,walltime=(time.perf_counter() - time_start)/60)
            writer.flush()
        
        if generation%10==0:
            print(f'Generation : {generation}',np.max(population_dist))

        history[ii,:] = population_dist
        generation +=1
    writer.close()
    return history
