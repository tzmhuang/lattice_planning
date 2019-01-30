import numpy as np
import matplotlib.pyplot as plt
import math


from polynomial import *

#LS_GRAPH CONFIG
DELTA_S = 2 # delta s in ls graph
DELTA_L = 1	 # delta l in lateral lattice
S_RES = 50	#sample points per meter

#ST_GRAPH CONFIG
DELTA_T = 1 # delta t in st graph
DELTA_DS = 1 # delta ds in cruising lattice
T_RES = 50  #sample points per second


KJ = 0.3
KL = 0.5
KS = 0.5
KDS = 2
KT = 0.05
KS = 0.05


MINIMUM_TURNING_RADIUS = 10
MAXIMUM_ACCELERATION = 4 #m/s-2

class base_lattice(object):
    def __init__(self,var_range):
        self.var0 = var_range[0]
        self.var1 = var_range[1]
    
    def plot(self,ax,lattices,col):
        for l in lattices:
            line = l.plot(ax,col,0.5)


class ls_lattice(base_lattice):
    def __init__(self, s_range):
        super().__init__(s_range)
        self.s0 = self.var0
        self.s1 = self.var1
    
    def gen_ls_lattice(self,l0,d_l0,dd_l0,target_path):
        self.ls_lattice = []
        self.min_ls_cost = math.inf
        s_horizon = np.arange(self.s0,self.s1, DELTA_S) + DELTA_S + MINIMUM_TURNING_RADIUS #normalize time tstart at 0+DELTA_S+3, start planning atleast 3 meters ahead
        for s in s_horizon:
            l_horizon = np.arange(target_path.lower_eval(s),target_path.upper_eval(s) + DELTA_L,DELTA_L)
            sample_pts = np.arange(self.s0,s,1/S_RES)
            for l in l_horizon:
                target_l = target_path.traj.eval(s)
                d_target_l = target_path.traj.first_derivative_eval(s)
                dd_target_l = target_path.traj.second_derivative_eval(s)
                traj = quint_poly(l0,d_l0,dd_l0,l,d_target_l,dd_target_l, s-self.s0)
                traj.y_pts = sample_pts
                traj.traj_eval = [traj.eval(p) for p in sample_pts-self.s0]
                # traj.traj_d_eval = [traj.first_derivative_eval(p) for p in sample_pts]
                # traj.traj_dd_eval = [traj.second_derivative_eval(p) for p in sample_pts]
                traj.traj_ddd_eval = [traj.third_derivative_eval(p) for p in sample_pts-self.s0]

                J = sum(np.power(traj.traj_ddd_eval,2))*1/S_RES
                traj.cost = KJ*J + KS*s + KL*abs(l-target_l)**2
                if self.min_ls_cost > traj.cost:
                    self.min_ls_cost = traj.cost
                    self.ls_min_l = l
                    self.ls_min_s = s
                    self.ls_opt_traj = traj
                self.ls_lattice.append(traj)
        self.ls_opt_traj.opt = True

    def plot(self,ax):
        super().plot(ax, self.ls_lattice,'grey')
        #self.opt_traj.plot('green',1)



class st_lattice(base_lattice):
    def __init__(self, t_range):
        super().__init__(t_range)
        self.t0 = self.var0
        self.t1 = self.var1
    
    def gen_st_lattice(self,s0,d_s0,dd_s0,target_path): #Following/cruising
        self.st_lattice = []
        self.min_st_cost = math.inf
        t_horizon = np.arange(self.t0, self.t1, DELTA_T) + DELTA_T# normalize time to start at 0 + DELTA_T
        for t in t_horizon:
            s_horizon = np.arange(target_path.lower_eval(t), target_path.upper_eval(t) + DELTA_S,DELTA_S)
            sample_pts = np.arange(self.t0, t, 1/T_RES)
            for s in s_horizon:
                target_s = target_path.traj.eval(t)
                d_target_s = target_path.traj.first_derivative_eval(t)
                dd_target_s = target_path.traj.second_derivative_eval(t)
                traj = quint_poly(s0,d_s0,dd_s0,s,d_target_s,dd_target_s, t-self.t0)
                traj.y_pts =  sample_pts
                traj.traj_eval = [traj.eval(p) for p in sample_pts-self.t0]
                # traj.traj_d_eval = [traj.first_derivative_eval(p) for p in sample_pts]
                # traj.traj_dd_eval = [traj.second_derivative_eval(p) for p in sample_pts]
                traj.traj_ddd_eval = [traj.third_derivative_eval(p) for p in sample_pts-self.t0]

                J  = sum(np.power(traj.traj_ddd_eval,2))*1/T_RES
                traj.cost = KJ*J + KT*t + KS*abs(s-target_s)
                if self.min_st_cost > traj.cost:
                    self.min_st_cost = traj.cost
                    self.st_min_s = s
                    self.st_min_t = t
                    self.st_opt_traj = traj
                self.st_lattice.append(traj)
            self.st_opt_traj.opt = True
    
    def gen_cruising_lattice(self,s0,d_s0,dd_s0,target_path): #target_path = dST_path	lattice in the time-ds space
        self.st_lattice = []
        self.min_st_cost = math.inf
        t_horizon = np.arange(self.var0, self.var1, DELTA_T) + DELTA_T + 10/MAXIMUM_ACCELERATION
        for t in t_horizon:
            ds_horizon = np.arange(target_path.lower_eval(t), target_path.upper_eval(t) + DELTA_DS, DELTA_DS)
            sample_pts = np.arange(self.var0,t,1/T_RES)
            for ds in ds_horizon:
                target_ds = target_path.traj.eval(t)    #target speed
                d_target_ds = target_path.traj.first_derivative_eval(t)     #target acceleration
                traj = quad_poly(s0,d_s0,dd_s0,ds,d_target_ds, t - self.t0)
                traj.y_pts = sample_pts
                traj.traj_d_eval = [traj.first_derivative_eval(p) for p in sample_pts-self.t0]
                traj.traj_ddd_eval = [traj.third_derivative_eval(p) for p in sample_pts-self.t0]

                J = sum(np.power(traj.traj_ddd_eval,2))*1/T_RES
                traj.cost = KJ*J + KT*t + KDS*abs(ds - target_ds)**2
                if self.min_st_cost > traj.cost:
                    self.min_st_cost = traj.cost
                    self.min_s = ds
                    self.min_t = t
                    self.st_opt_traj = traj
                self.st_lattice.append(traj)
            self.st_opt_traj.opt = True
    
    def plot(self,ax):
        super().plot(ax, self.st_lattice,'grey')
