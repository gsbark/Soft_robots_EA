import numpy as np 
from Generate_robot import load_robot,create_robot,create_floor_2
from GA_fun import assemble_pop,get_parallel_structure,eval_population,assemble_identical_pop
from simulator import Simulator_v1


def test_robots(num=1):
    file = './saved_robots/gen_500.npy'
    best_robot = np.load(file)
    #robot_structure = create_robot((5,5,5))
    robot_structure = load_robot()
    sim_structure = get_parallel_structure(**robot_structure,population_num=num)

    sim_structure['coords'][:,0] += np.repeat(np.arange(0, num * 5, 5), robot_structure['coords'].shape[0])

    floor_structure = create_floor_2(111,111,111)
    spring_sim1 = Simulator_v1(**sim_structure,**floor_structure,material=best_robot)
    k,b,c,edge_id = assemble_identical_pop(robot=best_robot,
                               coords=robot_structure['coords'],
                               edges=robot_structure['edges'],
                               pop_num=num,
                               num_material=best_robot.shape[0])
    
    eval_population(spring_sim1,k,b,c,True,edge_id,test=True)


test_robots(1)