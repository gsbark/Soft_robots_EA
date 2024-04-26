import numpy as np 
from Generate_robot import load_robot,create_robot,create_floor_2
from GA_utils import assemble_pop,get_parallel_structure,eval_population,Assemble_pop
from simulator import Simulator
import taichi as ti

def test_robots(num=1):
    file = './2x2x2/gen_2000.npy'
    mat_file = './2x2x2/Material.npy'
    best_robot = np.load(file)
    Material = np.load(mat_file)
    robot_structure = create_robot((2,2,2))
    #robot_structure = load_robot()
    sim_structure = get_parallel_structure(**robot_structure,population_num=num)

    sim_structure['coords'][:,0] += np.repeat(np.arange(0, num * 5, 5), robot_structure['coords'].shape[0])

    floor_structure = create_floor_2(111,111,111)
    #Initialize taichi 
    ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)
    spring_sim1 = Simulator(**sim_structure,**floor_structure,material=best_robot,rendering=True)
    
    k,b,c,edge_id = Assemble_pop(robot=best_robot,
                                 coords=robot_structure['coords'],
                                 edges=robot_structure['edges'],
                                 pop_num=num,
                                 Material=Material,
                                 num_material=best_robot.shape[0])
    
    eval_population(spring_sim1,k,b,c,True,edge_id,eval=True)
test_robots(1)