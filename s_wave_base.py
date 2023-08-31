import numpy as np
import math
import matplotlib.pyplot as plt
from pde_2d_fast import PDE_2D_fast
  
class s_wave_base(PDE_2D_fast):
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
    # Ensures dz = dx -> update Zmax
    self.dx = Xmax/(Nx-1)
    self.dz = self.dx   # dz is equal to dx
    self.Nx = Nx
    self.Nz = Nz = int(Zmax/self.dz)+1 # So NZ is derived from Zmax and dx
    self.Xmax =  Xmax
    self.Zmax = (Nz-1)*self.dz
    
    # v0[i,j,0] : v : x displacement
    # v0[i,j,1] : w : z displacement
    # v0[i,j,2] : dv/dt 
    # v0[i,j,3] : dw/dt 
    v0 = np.zeros([Nx,self.Nz,4], dtype = np.float64)
    super().__init__(v0, 0, Xmax, 0, Zmax)
    
    self.lam = np.zeros([Nx, Nz], dtype = np.float64) 
    self.mu  = np.zeros([Nx, Nz], dtype = np.float64) 
    self.rho = np.zeros([Nx, Nz], dtype = np.float64) 
    self.detectors = []
    self.excite_pos=[0,0]
    self.excite_f = 100
    self.vmax=0.1
    self.mark_shift = 0
    self.domain =  [0, self.Xmax, self.Zmax, 0]
   
    self.Gamma = np.zeros([Nx,Nz], dtype = np.float64)
    self.Gamma[0:n_abs,:] =\
         np.tile(np.linspace(Gamma,0,n_abs),(Nz,1)).transpose()
    self.Gamma[Nx-n_abs:,:] =\
         np.tile(np.linspace(0,Gamma,n_abs),(Nz,1)).transpose()
    self.Gamma[:,Nz-n_abs:] =\
         np.maximum(self.Gamma[:,Nz-n_abs:],
                    np.tile(np.linspace(0,Gamma,n_abs),(Nx,1)))

    #print("dx=",self.dx)
    #print("dy=",self.dy)
    #print("Nx=",self.Nx)
    #print("Ny=",self.Ny)
    #print("xmax=",self.Xmax)
    #print("zmax=",self.Zmax)
    
  def mu_lam(self, Vp, Vs, rho):
    """ Compute the Lame parameters from Vp and Vs and density
        : param Vp : P-wave speed 
        : param Vps: S-wave speed 
        : param rho : Rock density
        : return : mu and lambda
    """
    return rho*Vs**2, rho*(Vp**2-2*Vs**2)
  
  def detector_index(self, x):
    """ Return index of detector closest to x
        : param x : detector position 
        : return : detector index
    """
    return( round(x/(self.detector_n*self.dx))-1 )
  
  def detector_pos(self, n):
    """ Return position of detector at index n
        : param n : detector index
        : return : detector x position 
    """
    return self.dx*self.detector_n*(n+1)
    
  def set_model(self, src, data, d_detector):
    """ 
      Set the model parameters including the different rock layers.
      Compute the smallest dt value
      : param src : [x,z] position of initial impact
      : param data : list layer properties : 
                     [[thickness1, Vp1, Vs1, rho1, col1], 
                      [thickness2, Vp2, Vs2, rho2, col2], ... ]
      : param d_detector : distance between detectors
    """
    self.data = data
    self.v = np.zeros([self.Nx,self.Nz,4], dtype = np.float64) 
    self.excite_pos = [*self.pos_to_index(src) ]
    print("src =",src , "excite_pos=",self.excite_pos)
    self.detector_n = int(d_detector/self.dx)
    self.number_detectors = int(self.Nx/self.detector_n)
    self.t =0
    self.dt = -1
    self.detectors = []
    # reset for snapshots
    self.l_t = []
    self.l_f = []
    
    # Layers
    depth = 0 # top level of layer
    top_level = 0
    i = 1
    for d in data:
      thickness, Vp, Vs, rho, col = d
      mu, lam = self.mu_lam(Vp, Vs, rho)
      # fill lambda mu and rho all the way to the bottom
      self.lam[:,top_level:] = lam 
      self.mu[:,top_level:] = mu 
      self.rho[:,top_level:] = rho 
      depth += thickness    # top level of next layer
      top_level = int(depth/self.dx) # index level of next layer
      # determine the smallest dt
      layer_dt = 0.25*self.dx*np.sqrt(rho/mu)
      if self.dt < 0 or layer_dt < self.dt:
        self.dt = layer_dt
      # Display the layer vertical crossing time by P and S wave 
      print("Layer {} crossing time = {}s  {}s".format(i,thickness/Vp,
                                                       thickness/Vs))
      i += 1
    print("dt=",self.dt)

  def plot_layers(self, val="rho", fname=""):
    """ Display the layers graphically
        : param val : parameter to plot: "rho", "lam", "mu", "Vp", "Vs", 
                      "Gamma"
        : param fname : output filename in non empty
    """
    if val =="rho":
      im = plt.imshow(self.rho.transpose(), cmap='summer',
                      interpolation='nearest', extent=self.domain )
      plt.title(r'Density (kg/$m^3$)')
    elif val =="lam":
      im = plt.imshow(self.lam.transpose(), cmap='spring',
                      interpolation='nearest', extent=self.domain)
      plt.title(r'$\Lambda$')
    elif val =="mu":
      im = plt.imshow(self.mu.transpose(), cmap='cool',
                      interpolation='nearest', extrent=self.domain)
      plt.title(r'$\mu$')
    elif val =="Vp":
      Vp = np.sqrt((self.lam+2*self.mu)/self.rho).transpose()
      im = plt.imshow(Vp, cmap='autumn', interpolation='nearest',
                      extent=self.domain)
      plt.title(r'$V_p$  (m/s)')
    elif val =="Vs":
      Vs = np.sqrt(self.mu/self.rho).transpose()
      im = plt.imshow(Vs, cmap='winter', interpolation='nearest',
                      extent=self.domain)
      plt.title(r'$V_s$  (m/s)')
    elif val =="Gamma":
      im = plt.imshow(self.Gamma.transpose(), cmap='hot',
                      interpolation='nearest', extent=self.domain)
      plt.title(r'$\Gamma$')
    plt.colorbar(im)
    if fname:
      plt.savefig(fname)
    plt.show()

        
  def F(self, t, v_):
    """ Equation for seismic waves in inhomogeneous  media
      : param t : current time
      : param v_ : current function value (a vector of Shape [Nx, Nz, 4]).
      : return  : right hand side of the equation for v_.
    """
    # Initial exitation
    e = self.excite(t)
    if len(e) > 0:
      v_[self.excite_pos[0],self.excite_pos[1],:] = e

    eq = np.zeros([self.Nx, self.Nz, 4])

    # Scan the entire grid except edge nodes
    for i in range(1, self.Nx-1):
      for j in range(1, self.Nz-1):
    
       dv_xx = (v_[i+1, j, 0] + v_[i-1, j, 0] - 2*v_[i, j, 0])/self.dx**2
       dv_zz = (v_[i, j+1, 0] + v_[i, j-1, 0] - 2*v_[i, j, 0])/self.dz**2
       dv_xz =(v_[i+1, j+1, 0] + v_[i-1, j-1, 0]
              -v_[i+1, j-1, 0] - v_[i-1, j+1, 0])/(4*self.dz*self.dx)
       dw_xx = (v_[i+1, j, 1] + v_[i-1, j, 1] - 2*v_[i, j, 1])/self.dx**2
       dw_zz = (v_[i, j+1, 1] + v_[i, j-1, 1] - 2*v_[i, j, 1])/self.dz**2
       dw_xz = (v_[i+1, j+1, 1] + v_[i-1, j-1, 1]
              -v_[i+1, j-1, 1] - v_[i-1, j+1, 1])/(4*self.dz*self.dx)
    
       eq[i, j, 0] =  v_[i, j, 2]
       eq[i, j, 1] =  v_[i, j, 3]
       eq[i, j, 2] = ((self.lam[i, j]+self.mu[i, j])*(dv_xx+dw_xz)+
                        self.mu[i, j]*(dv_xx+dv_zz))/self.rho[i, j]
    
       eq[i, j, 3] = ((self.lam[i, j]+self.mu[i, j])*(dw_zz+dv_xz)+
                        self.mu[i, j]*(dw_xx+dw_zz) )/self.rho[i, j]
    return(eq) 

  def excite(self, t):
    """ Compute initial exitation.
        : param t : current time
        : return : excitation vector [v,w,dv/dt,dw/dt] 
                   [] when the excitaion is over
    """
    f = self.excite_f
    if t > 1/f:
        return []
    #return  np.array([0,1.0,0,0])
    return  np.array([0,math.sin(2*math.pi*t*f)*math.exp(-t*f),0,0])
  
  def boundary(self, v_):
    """ Enforce the boundary conditions. 
        z = 0 : no stress
        other boundaries  damo dv/dt and dw/dt
    """
    # z = 0 boundary
    for i in range(1, self.Nx-1):
      v_[i, 0, 0] = v_[i, 1, 0] + (v_[i+1, 1, 1] - v_[i-1, 1, 1]) * 0.5
      v_[i ,0, 1] = v_[i, 1, 1] + 0.5*self.lam[i, 1]/\
          (self.lam[i, 1] + 2*self.mu[i, 1])*(v_[i+1, 1, 0] - v_[i-1, 1, 0])

    # Absorption on the edges of the domain
    for i in range(self.Nx):
        for j in range(self.Nz):
            v_[i, j, 2] *= 1-self.Gamma[i,j]
            v_[i, j, 3] *= 1-self.Gamma[i,j]
        
  def extra_data(self):
    """ Save detector values in self.detectors
        Format : [t, v, w] 
    """
    d_v_val = [] # detector of v displacement
    d_w_val = [] # detector of w displacement
    for i in range(self.detector_n, self.Nx, self.detector_n):
      d_v_val.append(self.v[i,0,0])
      d_w_val.append(self.v[i,0,1])
    self.detectors.append([self.t, d_v_val, d_w_val])

    
  def dist_offset(self, theta, path):
      pass

  def angle_and_delay(self, dx, path, err = 0.1):
      pass

  def plot_delay_mark(self, x_d, path, level, tmax, colour):
    """ Add vertical bar and Label on detector signal figure
        : param x_d : detector distance to source 
        : param path : path followed
        : param level : height of vertical bar
        : param colour : colour of vertical bar
    """
    theta, dx, dt = self.angle_and_delay(x_d, path)
    mark = r'$'
    for item in path:
      mark += "{}_{}".format(item[1],item[0])
    mark += r'$'
    
    dt += self.mark_shift  # shift mark
    if dt < tmax:  
      print(mark,": Theta=",theta*180/math.pi," dx=",dx," dt=",dt)
      plt.plot([dt, dt],[0, level], color=colour)
      plt.text(dt, level, mark, color=colour)
        
  def plot_d(self, i, data_type ):  
     """ Plot signal of detector i
         : param i : detector index
         : param datat_type : "v", "w", "Mod", "Phase"
         : return : the largest value plotted 
     """
     t = []
     dval = []
     for item in self.detectors:
         t.append(item[0])
         v, w  = item[1][i], item[2][i]
         if data_type== "v":
            dval.append(v)
            lab = "v"
         elif data_type== "w":
            dval.append(w)
            lab = "w"
         elif data_type== "Mod":
            dval.append(np.sqrt(v**2+w**2))
            lab = "Mod(v,w)"
         elif data_type== "Phase":
            dval.append(math.atan2(w,v))
            lab = "Phase(v,w)"
         else :
           print("Invalid data_type ", data_type)
           exit(5)
     plt.xlabel("t")
     plt.ylabel("D")
     plt.plot(t, dval, label=lab)
     return max(dval)
     
  def eval_detector_diff(self, d1, d2, dn, data_type ):  
      pass

  def plot_detector_diff(self, d1, d2, dn, data_type ):  
     """ Plot d1-d2 signal of detector dn
         : param d1 : first detector
         : param d2 : second detector
         : param dn : detector index
         : param datat_type : "v", "w", "Mod", "Phase" 
         : return : the largest value plotted 
     """
     tdval = self.eval_detector_diff(d1, d2, dn, data_type)
     t, dval = list(zip(*tdval))
     if data_type== "v":
        lab = "v"
     elif data_type== "w":
        lab = "w"
     elif data_type== "Mod":
        lab = "Mod(v,w)"
     elif data_type== "Phase":
        lab = "Phase(v,w)"     
     plt.xlabel("t")
     plt.ylabel("D")
     plt.plot(t, dval, label=lab)
     return max(dval)     

  def extra_snapshot(self, ax):
     """ Draw layer boundaries on the snapshot
     """
     depth = 0
     for d in self.data[:-1]:
       depth += d[0]
       ax.axline([0,depth], [self.Xmax,depth], color='black')
           

