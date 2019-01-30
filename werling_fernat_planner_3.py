import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import threading

from lattice import *
from polynomial import *
from util import *

import paths


PLANNING_FREQ = 2   # Time of planning cycle (s)
PREVIEW_TIME = 3     # Preview Time (s)
PREVIEW_STATION = 5
SIM_TIME = 10 # Time simulation (s)
SIM_STEP = 0.1 #Simulation time step(s)
ROAD_WIDTH = 2
S_MARGIN = 5
SPEED_LIMIT = 10

MAX_PLANNING_DST = 100


class simulation():
    def __init__(self):
        self.t = 0
        self.l, self.d_l, self.dd_l = (10,0,0)
        self.s, self.d_s, self.dd_s = (0,0,0)
        reference_line = cubic_poly(0,2,0,0)                             #'''reference path setting'''
        self.reference_path = paths.ref_path(reference_line,0,100,2*ROAD_WIDTH)            #'''reference path setting'''
        ls_line = cubic_poly(0,0,0,0) # ls_deviation - time plot              '''lateral(d) error to time path setting'''
        self.ls_target_path = paths.LS_path(ls_line,0,MAX_PLANNING_DST,ROAD_WIDTH)
        #st_line = cubic_poly(5,0,0,0) # station - time plot                   '''longitudinal(s) error to time path setting'''
        #self.st_target_path = ST_path(st_line,0,SIM_TIME,DS_MARGIN)
        cruise_dST_plt = cubic_poly(10,0,0,0)
        self.dst_target_path = paths.dST_path(target = cruise_dST_plt, t_begin = 0, t_end = SIM_TIME, min = 0, max = SPEED_LIMIT)

    def next(self):
        self.t = self.t + SIM_STEP

    def update_pos(self):
        self.s = self.lon_lattice.st_opt_traj.eval(self.t-self.planning_start_t)[0]                         ###################################
        self.d_s = self.lon_lattice.st_opt_traj.first_derivative_eval(self.t-self.planning_start_t)[0]                       ####################################
        self.dd_s = self.lon_lattice.st_opt_traj.second_derivative_eval(self.t-self.planning_start_t)[0]     ####################################
        #l = lateral_error
        self.l = self.lat_lattice.ls_opt_traj.eval(self.s-self.planning_start_s)[0]
        self.d_l = self.lat_lattice.ls_opt_traj.first_derivative_eval(self.s-self.planning_start_s)[0]
        self.dd_l = self.lat_lattice.ls_opt_traj.second_derivative_eval(self.s-self.planning_start_s)[0]
        #
        #self.X,self.Y,self.Theta = self.reference_path.frenet_to_cartesian(self.s,self.l,self.d_l)
        self.X,self.Y,self.Theta = frenet_to_cartesian(self.reference_path, self.s,self.l,self.d_l)
        print(self.Theta*180/math.pi)

    def plan(self):
        self.planning_start_t = self.t
        self.planning_start_s = self.s
        #lon
        self.lon_lattice = st_lattice([self.t,self.t + PREVIEW_TIME])            ############################
        self.lon_lattice.gen_cruising_lattice(self.s,self.d_s,self.dd_s,self.dst_target_path)
        self.lon_lattice.plot(self.ax2[0])
        #lat
        self.lat_lattice = ls_lattice(np.array([self.s,self.s + PREVIEW_STATION]))         ############################
        self.lat_lattice.gen_ls_lattice(self.l,self.d_l,self.dd_l,self.ls_target_path)
        self.lat_lattice.plot(self.ax2[1])
        #print('*************time:',self.t, self.lon_lattice.st_opt_traj.traj_d_eval,'\n opt_traj:', self.lon_lattice.st_opt_traj.a2)


    def init_main_plot(self):
        fig, self.ax = plt.subplots(1,1)
        self.ax.set_ylim(min(self.reference_path.y_pts)-10,max(self.reference_path.y_pts)+10)   ################33
        self.ax.set_xlim(min(self.reference_path.x_pts)-10,max(self.reference_path.x_pts)+10)   #################

        self.reference_path.plot(self.ax,0.1)
        self.line, = self.ax.plot([],[],color = 'green')
        self.xdata,self.ydata = [],[]
        self.line.set_data(self.xdata,self.ydata)

        self.rect = patches.Rectangle((0,0),width = 1, height = 1, fill = False)
        self.ax.add_patch(self.rect)

    def init_st_ls_plot(self):
        fig2,self.ax2 = plt.subplots(2,1)
        self.ls_target_path.plot(self.ax2[1],0.1)
        self.dst_target_path.plot(self.ax2[0],0.1)
        self.ax2[0].grid(linewidth = 1, ls = '--')
        self.ax2[1].grid(linewidth = 1, ls = '--')
        self.ax2[1].set_ylim(-5,12)
        self.ax2[1].set_xlim(0,SIM_TIME)
        self.line_st, = self.ax2[0].plot([],[],color = 'green',zorder = math.inf)
        self.line_ls, = self.ax2[1].plot([],[],color = 'green',zorder = math.inf)
        self.st_xdata,self.st_ydata = [],[]
        self.ls_xdata,self.ls_ydata = [],[]
        self.line_st.set_data(self.st_xdata,self.st_ydata)
        self.line_ls.set_data(self.ls_xdata,self.ls_ydata)

    def refresh_plots(self):
        self.st_xdata.append(self.t)
        self.st_ydata.append(self.d_s)               ########## SPEED_TIME Graph#############   
        self.ls_xdata.append(self.s)                 #######################             
        self.ls_ydata.append(self.l)
        self.line_st.set_data(self.st_xdata,self.st_ydata)
        self.line_ls.set_data(self.ls_xdata,self.ls_ydata)
        self.rect.set_visible(False)
        # print('X:' ,X)
        # self.ax.scatter(X,Y)
        self.xdata.append(self.X)
        self.ydata.append(self.Y)
        self.line.set_data(self.xdata,self.ydata)
        x_rot,y_rot = cartesian_rotate(self.X,self.Y,self.Theta)        #coordinate change to plot (X,Y) centered rectangle
        x,y = cartesian_rotate(x_rot-2,y_rot-1,self.Theta,False)
        self.rect = patches.Rectangle(xy = (x,y),width =4 ,height = 2,angle=self.Theta*180/math.pi,fill = False)
        self.ax.add_patch(self.rect)

    def log(self):
        print ('start:[ t = {}, l = {}, d_l = {}, dd_l = {} ] COST = {}'.format(self.t,self.l, self.d_l, self.dd_l,self.lat_lattice.ls_opt_traj.cost))

def spin():
    sim = simulation()
    sim.init_main_plot()
    sim.init_st_ls_plot()
    cycle_count = 0
    while sim.t <= SIM_TIME:
        if cycle_count%int((1/PLANNING_FREQ)/SIM_STEP) == 0:
            sim.plan()
            print ("PLANNED")
        sim.next()
        sim.update_pos()
        sim.refresh_plots()
        plt.pause(SIM_STEP)
        #sim.log()
        cycle_count += 1
    plt.show()


if __name__ == '__main__':
    spin()
