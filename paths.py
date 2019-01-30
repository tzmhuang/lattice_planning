import numpy as np
import matplotlib.pyplot as plt
import math

class dST_path():   #speed_time path
    def __init__(self, target, t_begin, t_end, min, max):
        self.traj = target
        self.begin = t_begin
        self.end = t_end
        self.spd_min = min
        self.spd_max = max

    def upper_eval(self,t):
        return self.spd_max

    def lower_eval(self, t):
        return self.spd_min

    def plot(self,ax,res):
        pts = np.arange(self.begin,self.end,res)
        ax.plot([x for x in pts],[self.traj.eval(y) for y in pts],color = 'blue',lw = 0.5)
        ax.plot([x for x in pts],[self.upper_eval(y) for y in pts],color = 'blue',lw = 0.5)
        ax.plot([x for x in pts],[self.lower_eval(y) for y in pts],color = 'blue',lw = 0.5)


class ST_path():    #station_time path
    def __init__(self,target,t_begin,t_end,width):
        self.traj = target
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

class LS_path():    #LatOffset_station path
    def __init__(self,target,t_begin,t_end,width):
        self.traj = target
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
    def __init__(self,target,s_begin,s_end,width):
        self.traj = target                                       #'''place to insert self-drawn or spline curve'''
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
        self.heading[len(self.glb_x_pts)-1] = self.heading[len(self.glb_x_pts)-2]   #last entry always the same as the second to last entry
        dh = np.diff(self.heading)
        self.curvature = dh/self.ds                           #length = number_of_pts - 1

    def path_pts(self): #Always assume starting from 0,0?
        self.global_path_pts()
        self.s_begin_ind = np.argmin(abs(self.cumulated_s - self.begin))
        self.s_end_ind = np.argmin(abs(self.cumulated_s - self.end))
        self.x_pts = self.glb_x_pts[self.s_begin_ind:self.s_end_ind+1]
        self.y_pts = self.glb_y_pts[self.s_begin_ind:self.s_end_ind+1]

    def plot(self,ax,res):
        s_pts = np.arange(self.begin,self.end,res)
        inds = [np.argmin(abs(self.cumulated_s - i)) for i in s_pts]
        x_pts = [self.x_pts[i] for i in inds]
        thetas = [self.heading[i] for i in inds]
        ax.plot([x for x in x_pts],[self.traj.eval(x) for x in x_pts],color = 'blue',lw = 0.5)
        ax.plot([self.x_pts[i]-3*math.sin(self.heading[i]) for i in inds],[self.traj.eval(self.x_pts[i])+3*math.cos(self.heading[i]) for i in inds],color = 'blue',lw = 0.5)
        ax.plot([self.x_pts[i]+3*math.sin(self.heading[i]) for i in inds],[self.traj.eval(self.x_pts[i])-3*math.cos(self.heading[i]) for i in inds],color = 'blue',lw = 0.5)

