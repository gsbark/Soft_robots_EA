import matplotlib.pyplot as plt 
import numpy as np 

#Learning curve plot function
def Create_plot(x,mean_GA,index_error_bar,a_GA):

    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6,4))  
    plt.plot(x,mean_GA,label='GA',c='black')
    plt.errorbar(x[index_error_bar], mean_GA[index_error_bar], yerr = a_GA[index_error_bar],capsize=4,ecolor='red',ls = "None")
    plt.xlabel('Generations',fontsize = 12)
    plt.ylabel('Robot speed (m/s)',fontsize = 12)
    plt.xscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize = 12,loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./results/res.pdf')

#Dot plot function
def dot_plot(x,y,history_GA):

    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6,4))  
    population = history_GA.shape[1]
    num = int(np.floor(population/3))
    p = np.log10(history_GA.shape[0])
    l = np.logspace(0,p,num,dtype=int)-1
    plt.plot(x,y,label = 'Best robot ',c='black')
    plt.scatter(np.sort(np.tile(x[l],population)),history_GA[l,:],s=5,c='red',label='population')
    plt.xlabel('Generations',fontsize = 12)
    plt.ylabel('Robot speed (m/s)',fontsize = 12)
    plt.xscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize = 12,loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./results/dot.pdf')

#Visualize structure with material 
def visualize(coords,edges=None,mat=None,edge_id=None):
    fig, ax = plt.subplots(figsize=(6,4), subplot_kw={'projection': '3d'})
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)
    ax.set_zlim(0,3)
    if coords is not None:
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],alpha=1,s=20,c='red')
    colors = ['green','blue','red','yellow','black']*4
    if mat is not None:
        for i in range(mat.shape[0]):
            ax.scatter(mat[i,0],mat[i,1],mat[i,2],alpha=1,s=50,c=colors[i])
    if edges is not None:
        for idx,i in enumerate(edges):
            if edge_id is not None:
                col = colors[edge_id[idx]] 
            else:
                col = 'black'
            x = [coords[i[0],0],coords[i[1],0]]
            y = [coords[i[0],1],coords[i[1],1]]
            z = [coords[i[0],2],coords[i[1],2]]
            line, = ax.plot(x,y,z,c=col)
