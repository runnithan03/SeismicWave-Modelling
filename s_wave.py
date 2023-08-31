import numpy as np

from s_wave_base import *

def snell(theta1, V1, V2):
    '''
    

    Parameters
    ----------
    theta1 : float
        angle of incidence 
    V1 : float
        speed of the wave in the layer 
    V2 : float 
        speed of the wave as it leaves the layer

    Returns
    -------
    theta2 : float
        angle of refraction or reflection {depending on the layer interface interaction}

    '''
    s_theta2 = (V2/V1)*np.sin(theta1) #Snell's Law rearranged for s_theta2
    if s_theta2 > 1:
        theta2 = np.pi/2 #limits the value of theta2 between 0 and pi/2 
    else:    
        theta2 = np.arcsin(s_theta2)
    return theta2
  
class s_wave(s_wave_base):
  """ 
      A class to solve the seismic wave equation in an inhomogeneous environment
  """

  def __init__(self, Xmax, Zmax, Nx, Gamma=0.2, n_abs=30):
    """
      :  param Xmax       : X domain is [0, Xmax]
      :  param Zmax       : Z domain is [0, Zmax]
      :  param Nx         : Node number in X direction
      :  param Gamma      : Maximum damping value
      :  param n_abs      : Width of damping border (an integer)
    """
    super().__init__(Xmax, Zmax, Nx, Gamma, n_abs)
        
  def F(self, t, v_):
      ''' Equation for seismic waves in inhomogeneous  media
        : param t : current time
        : param v_ : current function value (a vector of Shape [Nx, Nz, 4]).
        : return  : right hand side of the equation for v_.
      '''
      # Initial exitation
      e = self.excite(t)
      if len(e) > 0:
        v_[self.excite_pos[0],self.excite_pos[1],:] = e

      eq = np.zeros([self.Nx, self.Nz, 4])
      #v and w are defined in the module s_wave_base 
      v = v_[:,:,0]
      
      #the slice used makes sure the edge nodes are excluded
      #the numpy roll adjusts the nodes that are excluded
      dv_xx = ((np.roll(v,1,0) + np.roll(v,-1,0) - 2*v)/self.dx**2)[1:-1,1:-1]
      dv_zz = ((np.roll(v,1,1) + np.roll(v,-1,1) - 2*v)/self.dz**2)[1:-1,1:-1]
      dv_xz = ((np.roll(v,(1,1),(0,1)) + np.roll(v,(-1,-1),(0,1))\
               - np.roll(v,(1,-1),(0,1)) - np.roll(v,(-1,1),(0,1)))/(4*self.dx*self.dz))[1:-1,1:-1]
      
      #the same idea as above applies except we look at a different layer in the matrix 
      w = v_[:,:,1]
      
      dw_xx = ((np.roll(w,1,0) + np.roll(w,-1,0) - 2*w)/self.dx**2)[1:-1,1:-1]
      dw_zz = ((np.roll(w,1,1) + np.roll(w,-1,1) - 2*w)/self.dz**2)[1:-1,1:-1] 
      dw_xz = ((np.roll(w,(1,1),(0,1)) + np.roll(w,(-1,-1),(0,1))\
               - np.roll(w,(1,-1),(0,1)) - np.roll(w,(-1,1),(0,1)))/(4*self.dx*self.dz))[1:-1,1:-1]

     
      #these are our lambda, mu and rho values we require in the 
      #equations with the edge nodes excluded 
      l = self.lam[1:-1,1:-1]
      m = self.mu[1:-1,1:-1]
      r = self.rho[1:-1,1:-1]
      
      #now we define the eq matrix using all the above variables 
      eq[1:-1,1:-1,0] = v_[1:-1,1:-1,2] #dv_dt
      eq[1:-1,1:-1,1] = v_[1:-1,1:-1,3] #dw_dt 
      
      #dv_dt and dw_dt are defined in the module s_wave_base 
      eq[1:-1,1:-1,2] = ((l+m)*(dv_xx+dw_xz)+m*(dv_xx+dv_zz))/r
      eq[1:-1,1:-1,3] = ((l+m)*(dw_zz+dv_xz)+m*(dw_xx+dw_zz))/r
      return(eq)

  def boundary(self, v_):
    ''' Enforce the boundary conditions. 
        z = 0 : no stress
        other boundaries  damo dv/dt and dw/dt
    '''

    v = v_[:, 1, 0]
    w = v_[:, 1, 1]
    
    #lambda and mu we require in the equations 
    l = self.lam[:,1]
    m = self.mu[:,1]
    
    #the same idea applies from F: as in we exclude the edge nodes
    v_[1:-1, 0, 0] = (v + (np.roll(w, -1) - np.roll(w, 1)) * 0.5)[1:-1]
    v_[1:-1, 0, 1] = (w + 0.5*l/(l + 2*m)*(np.roll(v, -1)-np.roll(v, 1)))[1:-1]
    
    # Absorption on the edges of the domain
    #the dimensions here remain the same
    v_[:,:,2] *= 1-self.Gamma
    v_[:,:,3] *= 1-self.Gamma 
    
  def dist_offset(self, theta, path):
      '''
      Parameters
      ----------
      theta : float  
          incident angle 
      path : the path followed by the wave 
      
      which is a list of type [[l1,w1], [l2,w2],...] where li is the layer index and Wi
      is the type of wave ("P" or "S")
      
      Returns
      -------
      a tuple of xoff and tt where:
          
      xoff : horizontal displacement of the wave at the end of the path 
      tt : the time taken by the wave to reach the end of the path 
      '''
      
      data = self.data 
      
      if path[0][1] == "P": #here we determine the initial speed 
          v_start = data[0][1]
      else:
          v_start = data[0][2]
      
      xoff = 0 
      tt = 0 
      for i in range(len(path)):
          l = path[i][0] #determines what layer we are in 
          W = path[i][1] #determines what type of wave we are looking at 
          D = data[l][0] #thickness of the layer 
          if W == "P":
              v_latest = data[l][1] #same idea as how we determine initial speed 
          else:
              v_latest = data[l][2]
          xoff += D*np.tan(theta)
          tt += D/(v_latest*np.cos(theta))
          theta = snell(theta,v_start,v_latest)
      return xoff,tt

  def angle_and_delay(self, dx, path, err = 0.1):
      '''

      Parameters
      ----------
      dx : float
          the value we want our horizontal distance x to be within the error of -
          i.e. the horizontal distance we are aiming for 
      path : a list of type [[l1,w1], [l2,w2],...]
      
      it represents the path followed by the  wave where li is the layer index and Wi
      is the type of wave ("P" or "S")
      
      err : a float 
          it tells us how we close we want our horizontal distance to be to dx,
          the default is 0.1

      Returns
      -------
      A tuple of theta, x and t 
      theta : float
          the incident angle we require to get the horizontal distance within the error of dx 
      x : float 
          the horizontal distance we should get - this should be within (dx - err, dx + err)
      t : float 
          the time taken for the wave to get to x 

      '''
      #below are the range of possible values for theta 
      theta1 = 0 
      theta2 = np.pi/2 
      x = dx + 1 #an arbitrary value of x which is away from dx by more than the error
        
      while np.abs(x - dx) > err:
          #bisection method 
          theta = (theta2 + theta1)/2 #compute the midpoint of theta1 and theta2 
          x,t = self.dist_offset(theta,path)
          if x-dx < 0:
              theta1 = theta 
          else:
              theta2 = theta 
      return theta, x, t

     
  def eval_detector_diff(self, d1, d2, dn, data_type): 
      '''
      

      Parameters
      ----------
      d1 : a list of the form [[t0,lv0,lw0],[t1,lv1,lw1],...] where lvi and lwi are lists themselves
          d1 is a detector signal
      d2 :  d2 is of the same form as d1
          d2 corresponds to a different detector signal 
      dn : float
          this indexes the necessary lvi/ lwi we need to compute the difference we desire 
      data_type : string 
          this determines the way we will compute the difference and hence obtain each xi 

      Returns
      -------
      result, which is a list of lists of the form [ti,xi]
      ti : time at each corresponding point in the detector 
      xi : difference at each corresponding point in the detector, and this is calculated 
      depending on what data_type is 
      
      '''
      result = []
      
      for i in range(len(d1)):
          ti = d1[i][0] 
          d1_lvi = d1[i][1][dn]
          d2_lvi = d2[i][1][dn]
          
          d1_lwi = d1[i][2][dn]
          d2_lwi = d2[i][2][dn]
      
          v = d2_lvi - d1_lvi
          w = d2_lwi - d1_lwi #v and w displacement difference at time ti
      
          if data_type == "v":
              xi = v
          elif data_type == "w":
              xi = w
          elif data_type == "Mod":
              xi = np.sqrt(v**2 + w**2) #amplitude of the displacement difference at time ti 
          else:
              xi = np.arctan2(w,v) #arctan of the displacement difference at time ti      
          
          result.append([ti, xi])
      
      return result #combination of the time and displacement difference as a list of lists



     
    