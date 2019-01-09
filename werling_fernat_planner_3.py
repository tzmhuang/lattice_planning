import numpy as np
# import matplotlib
# matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

DELTA_T = 1 # delta t in lattice
DELTA_D = 0.5 # delta d in lateral lattice
DELTA_S = 0.5 # delta s in longitudinal lattice
T_RES = 50  #sample points per second
KJ = 0.3
KD = 0.5
KS = 0.5
KT = 0.05
D_T = 1   # Time of planning cycle (s)
P_T = 5     # Preview Time (s)
SIM_TIME = 10 # Time simulation (s)
SIM_STEP = 0.1 #Simulation time step(s)
ROAD_WIDTH = 2
S_MARGIN = 5




class lat_planning():
    def __init__(self,t_range):
        self.t0 = t_range[0]
        self.t1 = t_range[1]
    
    def gen_lat_lattice(self,d0,d_d0,dd_d0,target_path):
        self.lat_path = target_path
        self.lat_lattice = []
        self.min_lat_cost = math.inf
        t_horizon = np.arange(self.t0,self.t1, DELTA_T)

        for t in t_horizon:
            t = t + DELTA_T #avoid t1-t0 = 0
            t_n = t - self.t0 #normalized time
            d_horizon = np.arange(self.lat_path.lower_eval(t),self.lat_path.upper_eval(t),DELTA_D)
            sample_pts = np.arange(0,t_n,1/T_RES)+ self.t0
            for d in d_horizon:
                target_d = self.lat_path.traj.eval(t)
                d_target_d = self.lat_path.traj.first_derivative_eval(t)
                dd_target_d = self.lat_path.traj.second_derivative_eval(t)
                traj = quint_poly(d0,d_d0,dd_d0,d,d_target_d,dd_target_d,self.t0,t)
                traj.t_pts = sample_pts
                traj.traj_eval = [traj.eval(p) for p in sample_pts]
                # traj.traj_d_eval = [traj.first_derivative_eval(p) for p in sample_pts]
                # traj.traj_dd_eval = [traj.second_derivative_eval(p) for p in sample_pts]
                traj.traj_ddd_eval = [traj.third_derivative_eval(p) for p in sample_pts]

                J = sum(np.power(traj.traj_ddd_eval,2))*1/T_RES
                traj.cost = KJ*J + KT*t + KD*abs(d-target_d)**2
                if self.min_lat_cost > traj.cost:
                    self.min_lat_cost = traj.cost
                    self.min_d = d
                    self.min_t = t_n
                    self.lat_opt_traj = traj
                self.lat_lattice.append(traj)
        self.lat_opt_traj.opt = True

    def plot_lat_lattice(self,ax):
        for l in self.lat_lattice:
            line = l.plot(ax,'grey',0.5)
            #self.opt_traj.plot('green',1)



class lon_planning():
    def __init__(self,t_range):
        self.t0 = t_range[0]
        self.t1 = t_range[1]
    
    def gen_lon_lattice(self,s0,d_s0,dd_s0,target_path): #Following/cruising
        self.lon_path = target_path
        self.lon_lattice = []
        self.min_lon_cost = math.inf
        t_horizon = np.arange(self.t0, self.t1, DELTA_T)

        for t in t_horizon:
            t = t+DELTA_T
            t_n = t-self.t0
            s_horizon = np.arange(self.lon_path.lower_eval(t), self.lon_path.upper_eval(t)+0.1*DELTA_S,DELTA_S)
            sample_pts = np.arange(0,t_n,1/T_RES) + self.t0
            for s in s_horizon:
                target_s = self.lon_path.traj.eval(t)
                d_target_s = self.lon_path.traj.first_derivative_eval(t)
                dd_target_s = self.lon_path.traj.second_derivative_eval(t)
                traj = quint_poly(s0,d_s0,dd_s0,s,d_target_s,dd_target_s,self.t0,t)
                traj.t_pts =  sample_pts
                traj.traj_eval = [traj.eval(p) for p in sample_pts]
                # traj.traj_d_eval = [traj.first_derivative_eval(p) for p in sample_pts]
                # traj.traj_dd_eval = [traj.second_derivative_eval(p) for p in sample_pts]
                traj.traj_ddd_eval = [traj.third_derivative_eval(p) for p in sample_pts]

                J  = sum(np.power(traj.traj_ddd_eval,2))*1/T_RES
                traj.cost = KJ*J + KT*t + KS*abs(s-target_s)
                if self.min_lon_cost > traj.cost:
                    self.min_lon_cost = traj.cost
                    self.min_s = s
                    self.min_t = t_n
                    self.lon_opt_traj = traj
                self.lon_lattice.append(traj)
        self.lon_opt_traj.opt = True

    def plot_lon_lattice(self,ax):
        for l in self.lon_lattice:
            l.plot(ax,'grey',0.5)



class quint_poly():
    opt = False
    t_pts = []
    traj_eval = []
    traj_d_eval = []
    traj_dd_eval = []
    traj_ddd_eval = []
    def __init__(self,d0,d_d0,dd_d0,d1,d_d1,dd_d1,t0,t1):
        self.t0 = t0
        self.t1 = t1
        self.a0 = d0
        self.a1 = d_d0
        self.a2 = dd_d0/2
        self.end = d1
        t = t1-t0 #normalize the t to start from 0
        #print (t)
        T = np.array([[t**3,t**4,t**5],\
                    [3*t**2,4*t**3,5*t**4],\
                    [6*t,12*t**2,20*t**3]])

        b = np.array([[d1-(self.a0 + self.a1*t + self.a2*t**2)],\
                    [d_d1 - (self.a1+self.a2*t*2)],\
                    [dd_d1 - self.a2*2]])

        self.a3, self.a4, self.a5 = np.linalg.solve(T,b)

    def eval(self,t):
        t = t-self.t0
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def first_derivative_eval(self,t):
        t = t-self.t0
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def second_derivative_eval(self,t):
        t = t-self.t0
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def third_derivative_eval(self,t):
        t = t-self.t0
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2

    def plot(self,ax,col,w):
        line = ax.plot(self.t_pts,self.traj_eval,color = col, lw = w)
        return line


class quad_poly():
    opt = False
    t_pts = []
    traj_eval = []
    traj_d_eval = []
    traj_dd_eval = []
    traj_ddd_eval = []
    def __init__(self,d0,d_d0,dd_d0,d_d1,dd_d1,t0,t1):
        self.t0 = t0
        self.t1 = t1
        self.a0 = d0
        self.a1 = d_d0
        self.a2 = dd_d0/2
        t = t1-t0 #normalize the t to start from 0
        #print (t)
        T = np.array(
            [[3*t**2, 4*t**3],\
            [6*t, 12*t**2]]
            )

        b = np.array([[d_d1-(self.a1 + self.a2*t)],\
                    [dd_d1 - (self.a2*2)]])

        self.a3, self.a4 = np.linalg.solve(T,b)

    def eval(self,t):
        t = t-self.t0
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4

    def first_derivative_eval(self,t):
        t = t-self.t0
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3

    def second_derivative_eval(self,t):
        t = t-self.t0
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2

    def third_derivative_eval(self,t):
        t = t-self.t0
        return 6*self.a3 + 24*self.a4*t

    def plot(self,ax,col,w):
        line = ax.plot(self.t_pts,self.traj_eval,color = col, lw = w)
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


class ST_path():
    def __init__(self,centre,t_begin,t_end,width):
        self.traj = centre
        self.begin = t_begin
        self.end = t_end
        self.width = width

    def upper_eval(self,t):
        return self.traj.eval(t) + self.width

    def lower_eval(self,t):
        return self.traj.eval(t) - self.width

    def plot(self,ax,res):
        pts = np.arange(self.begin,self.end,res)
        ax.plot([x for x in pts],[self.traj.eval(y) for y in pts],color = 'blue',lw = 0.5)
        ax.plot([x for x in pts],[self.upper_eval(y) for y in pts],color = 'blue',lw = 0.5)
        ax.plot([x for x in pts],[self.lower_eval(y) for y in pts],color = 'blue',lw = 0.5)

class SL_path():
    def __init__(self,centre,t_begin,t_end,width):
        self.traj = centre
        self.begin = t_begin
        self.end = t_end
        self.width = width

    def upper_eval(self,t):
        return self.traj.eval(t) + self.width

    def lower_eval(self,t):
        return self.traj.eval(t) - self.width

    def plot(self,ax,res):
        pts = np.arange(self.begin,self.end,res)
        ax.plot([x for x in pts],[self.traj.eval(y) for y in pts],color = 'blue',lw = 0.5)
        ax.plot([x for x in pts],[self.upper_eval(y) for y in pts],color = 'blue',lw = 0.5)
        ax.plot([x for x in pts],[self.lower_eval(y) for y in pts],color = 'blue',lw = 0.5)

class ref_path():
    def __init__(self,centre,s_begin,s_end,width):
        self.traj = centre                                       #'''place to insert self-drawn or spline curve'''
        self.begin = s_begin
        self.end = s_end
        self.path_pts()
    def global_path_pts(self):      #generate path points with global coordinates
        self.glb_x_pts = np.arange(0,500,0.01)
        self.glb_y_pts = np.array([self.traj.eval(x) for x in self.glb_x_pts]) #'''place to insert self-drawn or spline curve'''
        self.cumulated_s = np.zeros(len(self.glb_x_pts))
        self.heading = np.zeros(len(self.glb_x_pts))
        dx = np.diff(self.glb_x_pts)
        dy = np.diff(self.glb_y_pts)     #'''place to insert self-drawn or spline curve'''
        self.ds = np.sqrt(dx**2 + dy**2)
        for i in range(1,len(self.glb_x_pts)):
            self.cumulated_s[i] = self.ds[i-1] + self.cumulated_s[i-1]
            h = math.atan2(dy[i-1],dx[i-1]) #in radient
            self.heading[i-1] = h
        self.heading[len(self.glb_x_pts)-1] = self.heading[len(self.glb_x_pts)-2]   #last entry always thesame as the second last entry
        dh = np.diff(self.heading)
        self.curvature = dh/self.ds #length = number_of_pts - 1

    def path_pts(self): #Always assume starting from 0,0?
        self.global_path_pts()
        self.s_begin_ind = np.argmin(abs(self.cumulated_s - self.begin))
        self.s_end_ind = np.argmin(abs(self.cumulated_s - self.end))
        self.x_pts = self.glb_x_pts[self.s_begin_ind:self.s_end_ind+1]
        self.y_pts = self.glb_y_pts[self.s_begin_ind:self.s_end_ind+1]


    def frenet_to_cartesian(self,s,d,d_d): #s and l to x and y
        # t-> s(t) -> r(s)=(rx,ry) -> x(r(s),d), y(r(s),d)
        #d_d is dd/dt but we need dd/ds
        s_ind = np.argmin(abs(self.cumulated_s - s))
        d_d = d_d*self.ds[s_ind]
        xr = self.x_pts[s_ind]
        yr = self.y_pts[s_ind]
        theta_r = self.heading[s_ind]
        kappa = self.curvature[s_ind]
        x = xr - d*math.sin(theta_r)
        y = yr + d*math.cos(theta_r)
        theta_x = math.atan2(d_d,(1-d*kappa))+theta_r
        print ('s_ind',s_ind,'s:',self.cumulated_s[s_ind],'path:[', xr, yr,']   path_x: [',x,y, ']  Theta_r:',theta_r*180/math.pi)
        return x,y,theta_x

    def plot(self,ax,res):
        s_pts = np.arange(self.begin,self.end,res)
        inds = [np.argmin(abs(self.cumulated_s - i)) for i in s_pts]
        x_pts = [self.x_pts[i] for i in inds]
        thetas = [self.heading[i] for i in inds]
        ax.plot([x for x in x_pts],[self.traj.eval(x) for x in x_pts],color = 'blue',lw = 0.5)
        ax.plot([self.x_pts[i]-3*math.sin(self.heading[i]) for i in inds],[self.traj.eval(self.x_pts[i])+3*math.cos(self.heading[i]) for i in inds],color = 'blue',lw = 0.5)
        ax.plot([self.x_pts[i]+3*math.sin(self.heading[i]) for i in inds],[self.traj.eval(self.x_pts[i])-3*math.cos(self.heading[i]) for i in inds],color = 'blue',lw = 0.5)


class util():
    def cartesian_rotate(x,y,Theta, counter_clockwise = True): # frame couter clockwise, vector clockwise
        sign = 1 if counter_clockwise else -1
        x_rot = x*math.cos(Theta) + sign*y*math.sin(Theta)
        y_rot = -1*sign*x*math.sin(Theta) + y*math.cos(Theta)
        return x_rot, y_rot

class simulation():
    def __init__(self):
        self.t = 0
        self.d0, self.d_d0, self.dd_d0 = (10,0,0)
        self.s0, self.d_s0, self.dd_s0 = (0,0,0)
        reference_line = cubic_poly(0,2,0,0)                             #'''reference path setting'''
        self.reference_path = ref_path(reference_line,0,100,2*ROAD_WIDTH)            #'''reference path setting'''
        lat_line = cubic_poly(0,0,0,0) # lat_deviation - time plot              '''lateral(d) error to time path setting'''
        self.lat_target_path = SL_path(lat_line,0,SIM_TIME,ROAD_WIDTH)
        lon_line = cubic_poly(0,10,0,0) # station - time plot                   '''longitudinal(s) error to time path setting'''
        self.lon_target_path = ST_path(lon_line,0,SIM_TIME,S_MARGIN)

    def next(self):
        self.t = self.t + SIM_STEP

    def update_pos(self):
        self.d0 = self.lat_fts.lat_opt_traj.eval(self.t)[0]
        self.d_d0 = self.lat_fts.lat_opt_traj.first_derivative_eval(self.t)[0]
        self.dd_d0 = self.lat_fts.lat_opt_traj.second_derivative_eval(self.t)[0]
        self.s0 = self.lon_fts.lon_opt_traj.eval(self.t)[0]
        self.d_s0 = self.lon_fts.lon_opt_traj.first_derivative_eval(self.t)[0]
        self.dd_s0 = self.lon_fts.lon_opt_traj.second_derivative_eval(self.t)[0]

    def plan(self):
        self.lat_fts = lat_planning([self.t, self.t + P_T])
        self.lon_fts = lon_planning([self.t, self.t + P_T])
        self.lat_fts.gen_lat_lattice(self.d0,self.d_d0,self.dd_d0,self.lat_target_path)
        self.lat_fts.plot_lat_lattice(self.ax2[1])
        self.lon_fts.gen_lon_lattice(self.s0,self.d_s0,self.dd_s0,self.lon_target_path)
        self.lon_fts.plot_lon_lattice(self.ax2[0])


    def init_main_plot(self):
        fig, self.ax = plt.subplots(1,1)
        self.ax.set_ylim(min(self.reference_path.y_pts)-10,max(self.reference_path.y_pts)+10)
        self.ax.set_xlim(min(self.reference_path.x_pts)-10,max(self.reference_path.x_pts)+10)

        self.reference_path.plot(self.ax,0.1)
        self.line, = self.ax.plot([],[],color = 'green')
        self.xdata,self.ydata = [],[]
        self.line.set_data(self.xdata,self.ydata)

        self.rect = patches.Rectangle((0,0),width = 1, height = 1, fill = False)
        self.ax.add_patch(self.rect)

    def init_long_lat_plot(self):
        fig2,self.ax2 = plt.subplots(2,1)
        self.lat_target_path.plot(self.ax2[1],0.1)
        self.lon_target_path.plot(self.ax2[0],0.1)
        self.ax2[0].grid(linewidth = 1, ls = '--')
        self.ax2[1].grid(linewidth = 1, ls = '--')
        self.ax2[1].set_ylim(-5,12)
        self.ax2[1].set_xlim(0,SIM_TIME)
        self.line_lon, = self.ax2[0].plot([],[],color = 'green')
        self.line_lat, = self.ax2[1].plot([],[],color = 'green')
        self.lon_xdata,self.lon_ydata = [],[]
        self.lat_xdata,self.lat_ydata = [],[]
        self.line_lon.set_data(self.lon_xdata,self.lon_ydata)
        self.line_lat.set_data(self.lat_xdata,self.lat_ydata)

    def refresh_plots(self):
        self.lon_xdata.append(self.t)
        self.lat_xdata.append(self.t)
        self.lon_ydata.append(self.lon_fts.lon_opt_traj.eval(self.t)[0])
        self.lat_ydata.append(self.lat_fts.lat_opt_traj.eval(self.t)[0])
        self.line_lon.set_data(self.lon_xdata,self.lon_ydata)
        self.line_lat.set_data(self.lat_xdata,self.lat_ydata)
        self.rect.set_visible(False)
        X,Y,Theta = self.reference_path.frenet_to_cartesian(self.s0,self.d0,self.d_d0)
        # print('X:' ,X)
        # ax.scatter(X,Y)
        self.xdata.append(X)
        self.ydata.append(Y)
        self.line.set_data(self.xdata,self.ydata)
        x_rot,y_rot = util.cartesian_rotate(X,Y,Theta)        #coordinate change to plot (X,Y) centered rectangle
        x,y = util.cartesian_rotate(x_rot-2,y_rot-1,Theta,False)
        self.rect = patches.Rectangle(xy = (x,y),width =4 ,height = 2,angle=Theta*180/math.pi,fill = False)
        self.ax.add_patch(self.rect)

    def log(self):
        print ('start:[ t = {}, d0 = {}, d_d0 = {}, dd_d0 = {} ] COST = {}'.format(self.t,self.d0, self.d_d0, self.dd_d0,self.lat_fts.lat_opt_traj.cost))

def spin():
    sim = simulation()
    sim.init_main_plot()
    sim.init_long_lat_plot()
    cycle_count = 0
    while sim.t <= SIM_TIME:
        if cycle_count%int(D_T/SIM_STEP) == 0:
            print ("PLANNED")
            sim.plan()
        sim.refresh_plots()
        sim.next()
        sim.update_pos()
        plt.pause(SIM_STEP)
        sim.log()
        cycle_count += 1
    plt.show()


if __name__ == '__main__':
    spin()
