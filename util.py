import math
import numpy as np


def cartesian_rotate(x,y,Theta, counter_clockwise = True): # frame couter clockwise, vector clockwise
        sign = 1 if counter_clockwise else -1
        x_rot = x*math.cos(Theta) + sign*y*math.sin(Theta)
        y_rot = -1*sign*x*math.sin(Theta) + y*math.cos(Theta)
        return x_rot, y_rot


def frenet_to_cartesian(reference_path,s,l,d_l): #s and l to x and y
        # t-> s(t) -> r(s)=(rx,ry) -> x(r(s),d), y(r(s),d)
        s_ind = np.argmin(abs(reference_path.cumulated_s - s))
        xr = reference_path.x_pts[s_ind]
        yr = reference_path.y_pts[s_ind]
        theta_r = reference_path.heading[s_ind]
        kappa = reference_path.curvature[s_ind]
        x = xr - l*math.sin(theta_r)
        y = yr + l*math.cos(theta_r)
        theta_x = math.atan2(d_l,(1-l*kappa))+theta_r
        #print ('s_ind',s_ind,'s:',self.cumulated_s[s_ind],'path:[', xr, yr,']   path_x: [',x,y, ']  Theta_r:',theta_r*180/math.pi)
        return x,y,theta_x