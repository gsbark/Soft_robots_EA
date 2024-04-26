from GA_utils import run_GA
from utils import*
import shutil
import os 
import taichi as ti 

def main(make_plots:bool=False):
    '''
    Initialize the evolutinary algorithm
    The structure can be either a (nel_x,nel_y,nel_z) structure
    where each element is a cube with 8 vertices and 28 edges.
    Also the structure can be loaded from a gmsh file 
    (Note: Each point is at most connected to 70 other points,
    check if the generated robot has more than that) 
    '''

    shutil.rmtree('./runs', ignore_errors=True)
    shutil.rmtree('./saved_robots', ignore_errors=True)
    os.mkdir('./saved_robots')

    #Initialize taichi 
    ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)
    #Define domain and GA parameters
    #--------------------------------------------------------
    
    domain = (2,2,2)                    # Create a structure 
    #domain = 'robot'                     # Load mesh 
    generations = 2000                   # number of generations
    population_num = 10                  # population number 
    num_experiments = 1                  

    total_history_GA =  np.zeros((generations,population_num,num_experiments),dtype=np.float32)
    
    #Material Database (k_spring, b, c)
    mat_1 = np.array([1000,0,0],dtype=np.float32)
    mat_2 = np.array([20000,0,0],dtype=np.float32)
    mat_3 = np.array([5000,0.1,0],dtype=np.float32)
    mat_4 = np.array([5000,0.1,np.pi],dtype=np.float32)
    mat_5 = np.array([0,0,0],dtype=np.float32)
    Material = np.stack((mat_1,mat_2,mat_3,mat_4,mat_5))
    np.save('Material.npy',Material)
    for i in range(num_experiments):

        history = run_GA(domain=domain,
                         runs=generations,
                         population_num=population_num,
                         save_interval=500,
                         log_interval=10,
                         Material=Material,
                         num_material=20)
        
        total_history_GA[:,:,i] = history

    #Create a Learning curve plot and a dot plot 
    if make_plots==True:

        fit_GA = np.max(total_history_GA,axis=1)
        a_GA = np.std(fit_GA,axis=1)/np.sqrt(fit_GA.shape[1])
        mean_GA = np.mean(fit_GA,axis=1)

        #Create Plots
        x = np.linspace(1,generations,generations)
        p = int(np.log10(generations))
        index_error_bar = (np.append(np.logspace(1,p,p),generations)-1).astype(int)

        Create_plot(x=x,mean_GA=mean_GA,index_error_bar=index_error_bar,a_GA=a_GA)
        dot_plot(x=x,y=fit_GA[:,0],history_GA=total_history_GA[:,:,0])

if __name__ == "__main__":
    main(make_plots=True)
    # cProfile.run("main(make_plots=False)", sort='cumulative')
    





