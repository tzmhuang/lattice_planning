import numpy as np

class quint_poly():
    opt = False
    y_pts = []
    traj_eval = []
    traj_d_eval = []
    traj_dd_eval = []
    traj_ddd_eval = []
    def __init__(self,d0,d_d0,dd_d0,d1,d_d1,dd_d1,t):
        self.a0 = d0
        self.a1 = d_d0
        self.a2 = dd_d0/2
        self.end = d1
        #print (t)
        T = np.array([[t**3,t**4,t**5],\
                    [3*t**2,4*t**3,5*t**4],\
                    [6*t,12*t**2,20*t**3]])

        b = np.array([[d1-(self.a0 + self.a1*t + self.a2*t**2)],\
                    [d_d1 - (self.a1+self.a2*t*2)],\
                    [dd_d1 - self.a2*2]])

        self.a3, self.a4, self.a5 = np.linalg.solve(T,b)

    def eval(self,t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def first_derivative_eval(self,t):
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def second_derivative_eval(self,t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def third_derivative_eval(self,t):
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2

    def plot(self,ax,col,w):
        line = ax.plot(self.y_pts,self.traj_eval,color = col, lw = w)
        return line

class quad_poly():
    opt = False
    y_pts = []
    traj_eval = []
    traj_d_eval = []
    traj_dd_eval = []
    traj_ddd_eval = []
    def __init__(self,d0,d_d0,dd_d0,d_d1,dd_d1,t):
        self.a0 = d0
        self.a1 = d_d0
        self.a2 = dd_d0/2
        #print (t)
        T = np.array(
            [[3*t**2, 4*t**3],\
            [6*t, 12*t**2]]
            )

        b = np.array([[d_d1-(self.a1 + 2*self.a2*t)],\
                    [dd_d1 - (self.a2*2)]])

        self.a3, self.a4 = np.linalg.solve(T,b)

    def eval(self,t):   #station
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4

    def first_derivative_eval(self,t):  #velocity
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3

    def second_derivative_eval(self,t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2

    def third_derivative_eval(self,t):
        return 6*self.a3 + 24*self.a4*t

    def plot(self,ax,col,w):
        line = ax.plot(self.y_pts,self.traj_d_eval,color = col, lw = w)
        return line


#d0,d_d0,dd_d0,d1,d_d1,dd_d1,t_range,d_range
# fts = frenet_trajectory_sets(3,0,0,0,0,0,[0,3],[-2,2])
# fts.gen_lattice()
# fts.plot_lattice()
# plt.show()

# class cubic_smooth_spline():
#     def __init__(self):
#         self.nodes = []
#         self.node_values = []

#     def solve(self):
#         seg_num = len(self.nodes)
#         A = np.zeros(shape = (seg_num,seg_num))
#         b = np.zeros(shape = (seg_num,1))
#         #adding end_pt constraint
#         A[0][0] = 1
#         b[1] = self.node_values[0]

#         #adding end_pt derivative constraint
#         A[2][1] = 1
#         A[2][2] = 2
#         A[2][3] = 3
#         b[2] = 0
#     pass

class cubic_poly():
    def __init__(self, a0=0, a1=0, a2=0, a3=0):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    def eval(self,x):
        return self.a0 + self.a1*x + self.a2*x**2 + self.a3*x**3
    def first_derivative_eval(self,x):
        return self.a1 + 2*self.a2*x + 3*self.a3*x**2
    def second_derivative_eval(self,x):
        return 2*self.a2 + 6*self.a3*x
    def third_derivative_eval(self):
        return 6*self.a3