import numpy as np
import pandas as pd
import cclib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy
from scipy.special import laguerre


### Constants

u = 1.660538921e-27
e = 1.60217663e-19
k = 8.9875517923e9
hbar = 1.0546e-34
kb = 1.380649e-23
cm_to_J = 1.986e-23
c_cm_s = 2.997e10
hbar = 1.054571817e-34
bohr_radius = 5.29177210903e-11
p_au_fac = hbar/bohr_radius # factor for atomic units of momentum
p_au_KE_eV_fac = (p_au_fac)**2/(u*e) # factor for converting from atomic units of momentum to KE in eV
cm_to_hartree = 1 / 219474.6 
hartree_to_eV = 27.211396132
u_to_amu = 1 / 5.4857990943e-4
bohr_to_angstrom = 0.529177211


atomno_dict = {'H': 1,
 'He': 2,
 'Li': 3,
 'Be': 4,
 'B': 5,
 'C': 6,
 'N': 7,
 'O': 8,
 'F': 9,
 'Ne': 10,
 'Na': 11,
 'Mg': 12,
 'Al': 13,
 'Si': 14,
 'P': 15,
 'S': 16,
 'Cl': 17,
 'Ar': 18,
 'K': 19,
 'Ca': 20,
 'Sc': 21,
 'Ti': 22,
 'V': 23,
 'Cr': 24,
 'Mn': 25,
 'Fe': 26,
 'Co': 27,
 'Ni': 28,
 'Cu': 29,
 'Zn': 30,
 'Ga': 31,
 'Ge': 32,
 'As': 33,
 'Se': 34,
 'Br': 35,
 'Kr': 36,
 'Rb': 37,
 'Sr': 38,
 'Y': 39,
 'Zr': 40,
 'Nb': 41,
 'Mo': 42,
 'Tc': 43,
 'Ru': 44,
 'Rh': 45,
 'Pd': 46,
 'Ag': 47,
 'Cd': 48,
 'In': 49,
 'Sn': 50,
 'Sb': 51,
 'Te': 52,
 'I': 53}

atomcolor_dict = {'H':'snow',
                  'C':'grey',
                  'O':'red',
                  'N':'blue',
                  'F':'yellow',
                  'I':'magenta'}

mass_dict = {'H': 1.007825,
          'He': 4.002603,
          'Li': 7.016004,
          'Be': 9.012182,
          'B': 11.009305,
          'C': 12.000000,
          'N': 14.003074,
          'O': 15.994915,
          'F': 18.998403,
          'Ne': 19.992440,
          'Na': 22.989770,
          'Mg': 23.985042,
          'Al': 26.981538,
          'Si': 27.976927,
          'P': 30.973762,
          'S': 31.972071,
          'Cl': 34.968853,
          'Ar': 39.962383,
          'K': 38.963707,
          'Ca': 39.962591,
          'Sc': 44.955910,
          'Ti': 47.947947,
          'V': 50.943964,
          'Cr': 51.940512,
          'Mn': 54.938050,
          'Fe': 55.934942,
          'Co': 58.933200,
          'Ni': 57.935348,
          'Cu': 62.929601,
          'Zn': 63.929147,
          'Ga': 68.925581,
          'Ge': 73.921178,
          'As': 74.921596,
          'Se': 79.916522,
          'Br': 78.918338,
          'Kr': 83.911507,
          'Rb': 84.911789,
          'Sr': 87.905614,
          'Y': 88.905848,
          'Zr': 89.904704,
          'Nb': 92.906378,
          'Mo': 97.905408,
          'Tc': 98.907216,
          'Ru': 101.904350,
          'Rh': 102.905504,
          'Pd': 105.903483,
          'Ag': 106.905093,
          'Cd': 113.903358,
          'In': 114.903878,
          'Sn': 119.902197,
          'Sb': 120.903818,
          'Te': 129.906223,
          'I': 126.904468}



# some utility functions

def read_xyz(xyz_file):
    """Read in a simple geometry (.xyz) file

    :param xyz_file: .xyz file containing molecular coordinates

    :return: Geometry object"""

    with open(xyz_file,'r') as geo_data:
        geo_str = geo_data.readlines()

    # Extract element information (elements) and coordinates (geom)
    geo_str_cs = geo_str[2:]
    coordinates = np.zeros((len(geo_str_cs),3))
    element_list = []
    for i in np.arange(len(geo_str_cs)):
        arr = geo_str_cs[i].split()
        element_list.append(arr[0])
        coordinates[i,0] = float(arr[1])
        coordinates[i,1] = float(arr[2])
        coordinates[i,2] = float(arr[3])

    mass_list = []
    for element in element_list:
        try:
            mass_list.append(mass_dict[element])
        except KeyError:
            print(f'ERROR: Mass not found for element {element}!')
            
    
    geom = Geometry(coordinates,np.array(mass_list), element_list=element_list)
    return(geom)


def read_traj_xyz(xyz_file):
    """(Experimental) Read in a trajectory (.xyz) file

    :param xyz_file: .xyz file containing series of molecular coordinates, representing different timesteps in a trajectory

    :return:  (list of Geometry objects, list of timesteps)"""

    # Load geometry file as strings
    with open(xyz_file,'r') as geo_data:
        geo_str = geo_data.readlines()

    # Extract element information (elements) and coordinates (geom)
    n_atoms = int(geo_str[0])
    print(n_atoms)

    geom_list = []
    time_list = []
    for time_point in range(0,2000):

        start_line = (time_point)*(n_atoms+2)+2
        end_line = start_line+n_atoms

        geo_str_cs = geo_str[start_line:end_line]
        # print(len(geo_str_cs))
        if len(geo_str_cs):
            time_string = geo_str[start_line-1].strip()
            time = float(time_string[7:])
            coordinates = np.zeros((len(geo_str_cs),3))
            element_list = []
            for i in np.arange(len(geo_str_cs)):
                arr = geo_str_cs[i].split()
                element_list.append(arr[0])
                coordinates[i,0] = float(arr[1])
                coordinates[i,1] = float(arr[2])
                coordinates[i,2] = float(arr[3])
            mass_list = []
            for element in element_list:
                try:
                    mass_list.append(mass_dict[element])
                except KeyError:
                    print(f'ERROR: Mass not found for element {element}!')

            geom = Geometry(coordinates,np.array(mass_list), element_list=element_list)
            geom_list.append(geom)
            time_list.append(time)
        else:
            break
                    

    print('Finished reading at time point %s' % time_point)
    return(geom_list,time_list)




def read_log(log_file):
    """Read in a log file. Currently just tested for GAMESS(US)

    :param log_file: '.log' file produced by a quantum chemistry software

    :return: Geometry object

    """
    data = cclib.io.ccread(log_file)
    coordinates = data.atomcoords[-1]
    atomnos = data.atomnos
    atommasses = data.atommasses
    omegas = data.vibfreqs
    nmodes = data.vibdisps

    geom = Geometry(coordinates,atommasses,atom_nos=atomnos, omegas=omegas, nmodes=nmodes)
    return(geom)




def random_rotation(r_list):
    """
    Randomly rotate a list of vectors. See https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices

    :param r_list: list of vector arrays.


    :return: list of ranomdly rotated vector arrays.
    """

    
    r_list_rot = []
    rm = rand_rotation_matrix()
    for r in r_list:
        r_list_rot.append(np.dot(rm,r))
    return(r_list_rot)


def rand_rotation_matrix(deflection=1.0, randnums=None):
    
    """
    Creates a random rotation matrix. See http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    and http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    :param deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    :param randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    :return: random rotation matrix
    
    """
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def coulomb_force(r1,r2,q1,q2):
    """Calculate Coulomb force between two charges in SI units.

    :param r1: (3D) position of charge 1 (in m)
    :param r2: (3D) position of charge 2 (in m)
    :param q1: charge 1 (in C)
    :param q2: charge 2 (in C)

    :return: Coulomb force (in N)"""


    q1 = q1
    q2 = q2
    r12 = r1-r2
    r12_mag = np.linalg.norm(r12)
    return(k*q1*q2*r12*(1/r12_mag**3))


def calc_W(P,Q,n=0):
    """Calculate unitless Wigner function.

    :param P: unitless momentum
    :param Q: unitless position
    :param n: vibrational state (0 by default)

    :return: W(P,Q,n)"""
    if n>0:
        Ln = scipy.special.laguerre(0)
        rhosquare = 2.0 * (P**2 + Q**2)
        Ln = scipy.special.laguerre(n)
        W = (-1.0)**n * Ln(rhosquare) * np.exp(-rhosquare / 2.0)
    else:
        W = np.exp(-Q**2)*np.exp(-P**2)
    return(W)

def calc_canonical_partition(omega, T):
    """Calculate vibrational canonical partition function.

    :param omega: vibrational frequency
    :param T: temperature (k)

    :return: Z(omega,T) - canonical partition function"""
    Z = np.exp(-(hbar*omega)/(2*kb*T))/(1-np.exp(-(hbar*omega)/(kb*T)))
    return(Z)

def calc_vib_energy(n,omega):
    """Calculate vibrational energy for given n and frequency.

    :param n: vibrational state
    :param omega: vibrational frequency

    :return: En = hbar * omega * (n+1/2)"""
    En = hbar*omega*(n+1/2)
    return(En)

def calc_Pn(omega_cm, T,n):
    """Calculate probability of populating vibrational state n.

    :param omega_cm: vibrational frequency (in wavenumbers)
    :param T: temperature (in K)
    :param n: vibrational state

    :return: P(n, omega,T) - probability of being in state n"""
    omega_Hz = omega_cm*c_cm_s*2*np.pi
    Z = calc_canonical_partition(omega_Hz,T)
    En = calc_vib_energy(n, omega_Hz)
    if T>0:
        Pn = np.exp(-En/(kb*T))/Z
    # probably don't need this to deal with T=0 but no harm
    else:
        if n==0:
            Pn = 1
        else:
            Pn = 0
    return(Pn)



class Geometry:
    """Class for (equilibrium) molecular geometry and related information.

    :param atom_coords: numpy array of the 3D position of each atom in the structure
    :param atom_masses: numpy array of the atomic mass of each atom in the structure
    :param element_list: list of element of each atom
    :param atom_nos: list of atomic numbers of each atom
    :param atom_labels: list of strings used to label each atom
    :param geom_label: Optional, additional label used to identify this geometry
    :param nmodes: Optional, normal modes
    :param omegs: Optional, vibrational frequencies
    """
    def __init__(self, atom_coords, atom_masses, element_list=[], atom_nos = np.array([]), 
                 atom_labels = np.array([]), geom_label=None, nmodes=np.array([]), omegas=np.array([])):
        self.atom_coords = np.array(atom_coords)
        self.atom_masses = np.array(atom_masses)
        self.natoms = len(atom_coords)
        if atom_nos.any():
            self.atom_nos = atom_nos
            self.get_elements()
        if atom_labels.any():
            self.atom_labels = atom_labels
        if geom_label:
            self.geom_label = geom_label
        if len(element_list):
            self.element_list = element_list
        if nmodes.any():
            self.nmodes = nmodes
            self.check_normal_modes()
        if omegas.any():
            self.omegas = omegas
            
    def get_elements(self):
        """Gets list of elements (stored in self.element_list) from atomic numbers (stored in self.atom_nos)"""
        element_list = []
        for atomno in self.atom_nos:
            if atomno in atomno_dict:
                element_list.append(atomno_dict[atomno])
            else:
                element_list.append('?')
        self.element_list = element_list

    def visualize_2D(self, dim1=0, dim2=1, markersize=30, markeredgewidth=3):
        """Visualizes a 2D projection of the molecular structure

        :param dim1: first dimension (0=x, 1=y etc.) of the 2D histogram
        :param dim2: second dimension (0=x, 1=y etc.) of the 2D histogram
        :param markersize: size of atom marker (default 30)
        :param markeredgewidth: edgewidth of atom marker (default 3)
        """
        dim_list=['x','y','z']
        dim1_range = (np.min(self.atom_coords[:,dim1])-0.5, np.max(self.atom_coords[:,dim1])+0.5)
        dim2_range = (np.min(self.atom_coords[:,dim2])-0.5, np.max(self.atom_coords[:,dim2])+0.5)

        fig,ax=plt.subplots(figsize=(4,4))
        ax.set_xlim(dim1_range[0],dim1_range[1])
        ax.set_ylim(dim2_range[0],dim2_range[1])

        for element,coord in zip(self.element_list,self.atom_coords):
            if element in atomcolor_dict:
                color=atomcolor_dict[element]
            else:
                color='black'
            ax.plot(coord[dim1], coord[dim2], 'o', color=color, markersize=markersize, 
                markeredgewidth=markeredgewidth, markeredgecolor='black')
        ax.set_xlabel(dim_list[dim1] + r" / $\mathrm{\AA}$ ", fontsize=14)
        ax.set_ylabel(dim_list[dim2] + r" / $\mathrm{\AA}$ ", fontsize=14)
        plt.show()

    def find_com(self):
        """Find centre-of-mass of geometry. Stored in self.com"""
        mass_product = np.zeros((3,))
        mass_sum = 0
        for r, mass in zip(self.atom_coords,self.atom_masses):
            mass_product+=r*mass
            mass_sum+=mass
        self.com = mass_product/mass_sum

    def com_geometry(self):
        """Shift geometry to set centre-of-mass to zero.
        New geometry is stored in self.atom_coods_com"""
        self.find_com()
        self.atom_coords_com = self.atom_coords - self.com


    def check_normal_modes(self, threshold=0.05, make_fig=True):
        """Check if normal modes are orthogonal and re-weight them correctly (in self.nmodes_weighted).
        Currently function works for GAMESS format modes, need to expand to other formats."""
    
        results = np.zeros((len(self.nmodes),len(self.nmodes)))
        nmodes2 = []
        
        for mode in self.nmodes:
            norm=0
            for i in range(self.natoms):
                for j in range(3):
                    norm+=mode[i,j]**2 * self.atom_masses[i]
    
            norm=np.sqrt(norm)
            # print(norm)
            
            mode_new = np.zeros_like(mode)
            for i in range(self.natoms):
                # mode_new[i,:] = mode[i,:]*np.sqrt(masses[i]/U_TO_AMU)
                mode_new[i,:] = mode[i,:]*np.sqrt(self.atom_masses[i])/norm
                # mode_new[i,:] = mode[i,:]/ norm / np.sqrt(masses[i]/U_TO_AMU)
                # mode_new[i,:] = mode[i,:]*np.sqrt(masses[i])
                # mode_new[i,:] = mode[i,:]*np.sqrt(masses[i]) / np.sqrt(ANG_TO_BOHR)
            nmodes2.append(mode_new)
        
        for i, mode1 in enumerate(nmodes2):
            for j, mode2 in enumerate(nmodes2):
                dotprod=0
                for k in range(self.natoms):
                    dotprod+=np.dot(mode1[k,:],mode2[k,:])
                results[i,j]=dotprod
    
        for i in range(len(self.nmodes)):
            results[i,i]-=1

        if make_fig:
            fig,ax=plt.subplots(figsize=(4,3))
            map_im = ax.imshow(results, cmap='bwr', vmax=np.max(results), vmin=-np.max(results))
            ax.set_xlabel('mode index', fontsize=14)
            ax.set_ylabel('mode index', fontsize=14)
            ax.set_title('normal mode dot product array, with trace of 1 subtracted', fontsize=14, pad=15)
            plt.colorbar(map_im)
            plt.show()

        print(f"Max absolute value in subtracted dot produce matrix: {np.max(abs(results))}")
    
        if np.max(abs(results))>threshold:
            print('Normal modes are NOT in expected format')
        else:
            print('Normal modes are in expected format')
        self.nmodes_weighted = nmodes2


class CEChannel:
    """Class for CE channel. For now this refers to a charge distribution, but could be extended
    to capture different (molecular) fragments etc.

    :param charges: array of charges
    :param p: probability of this channel occuring in the simulation"""
    def __init__(self, charges, p, label='None'):
        self.charges = np.array(charges)
        self.p = p
        if label:
            self.label=label


class StartingConditions:
    """Class for generating starting conditions for CE simulation.

    :param eq_geometry: (equilibrium) geometry"""
    
    def __init__(self, eq_geometry):
        self.eq_geometry = eq_geometry
        self.multi_channel=False

    def set_channel_list(self, channel_list):
        """Add a list of different channels to the object

        :param channel_list: list of CEChannel objects"""
        channel_p_list = []
        for channel in channel_list:
            channel_p_list.append(channel.p)
        sum_p = np.sum(channel_p_list)
        if abs(sum_p-1)>0.01:
            print(f"Channel probabilities sum to {sum_p}. Will rescale anyway.")
        else:
            self.channel_list=channel_list
            self.channel_p_list=list(np.array(channel_p_list)/sum_p)
            self.multi_channel=True

        # give each channel a unique index
        for i, channel in enumerate(self.channel_list):
            channel.index=i

    def sigma_to_array(self):
        """Convert sigma from single number to 1 element array if needed."""
        # First check if sigma is a single number, convert to single number array if so
        if isinstance(self.sigma, (int, float, complex)) and not isinstance(self.sigma, bool):
            self.sigma = np.array([self.sigma])
            

    def generate_pool(self, n_geoms, method='gaussian',random_rotate=True, sigma=0.1, wigner_sample_max=3, T=0, nmax=5):
        """Main method for generating the pool of starting conditions for simulation.

        :param n_geoms: number of samples in the pool
        :param method: ['gaussian', 'wigner'] method of blurring geometries. If 'gaussian', geometries are just convolved
        by a Gaussian distribution of width sigma. If 'wigner', instead sample from vibrational wigner distribution
        with a specified temeprature. This requires the geometry to include normal modes as appropriate.
        :param sigma: sigma for Gaussian convolution
        :param random_rotate: if True, randomly rotate each sampled geometry
        :param wigner_sample_max: the max (absolute) value of P and Q used for Wigner sampling
        :param T: temperature for wigner sampling
        :param nmax: max vibrational state considered for wigner sampling

        """
        self.method = method
        if self.method=='wigner':
            # If generating a pool of simulations by Wigner sampling, we need
            # the normal modes, their frequencies, and a temperature
            assert self.eq_geometry.nmodes.any()
            assert self.eq_geometry.omegas.any()
            self.T = T
            expected_modes = 3*self.eq_geometry.natoms-6
            expected_modes2 = 3*self.eq_geometry.natoms-5
            self.samp_q_list=[]
            
        self.n_geoms = n_geoms
        self.random_rotate=random_rotate
        self.T=T
        self.wigner_sample_max=wigner_sample_max
        self.nmax=nmax
        
        self.samp_y0_list = []
        self.samp_charges_list = []
        self.samp_masses_list = []
        self.samp_channel_list = []
        self.eq_geometry.com_geometry()

        if self.method=='gaussian':
            # If generating a pool of simulations by Gaussian blurring, we need a sigma
            # Sigma can either by a single number, a (n_atom) length array, or (n_atom,3) dimension 2d array
            self.sigma = sigma
            self.sigma_to_array()

        if self.method=='wigner':
            if T>0:
                # for positive temperature, need to calculate probability of populating excited vib states
                self.n_list = list(range(self.nmax))
                self.Pn_arr = np.zeros((len(self.eq_geometry.nmodes),nmax))
                self.samp_n_list = []
                for n in range(nmax):
                    for i, omega in enumerate(self.eq_geometry.omegas):
                        Pn = calc_Pn(omega,T,n)
                        self.Pn_arr[i,n] = Pn
        
        for i in range(self.n_geoms):
            y0 = np.zeros((self.eq_geometry.natoms*6))
            r_list = []

            if self.multi_channel:
                channel = np.random.choice(self.channel_list,p=self.channel_p_list)
                self.samp_channel_list.append(channel)
                charges = channel.charges*e
                masses = self.eq_geometry.atom_masses*u
            else:
                charges = np.ones(self.eq_geometry.natoms)*e
                masses = self.eq_geometry.atom_masses*u

                

            if self.method=='gaussian':
                for n in range(self.eq_geometry.natoms):
                        if np.shape(self.sigma)[0]==1:
                            blur_sigma=self.sigma
                        elif np.shape(self.sigma)[0]==self.eq_geometry.natoms:
                            blur_sigma=self.sigma[n]
    
                        rs = np.array([(self.eq_geometry.atom_coords_com[n][0]+np.random.normal(loc=0,scale=blur_sigma))*1e-10,
                                       (self.eq_geometry.atom_coords_com[n][1]+np.random.normal(loc=0,scale=blur_sigma))*1e-10,
                                       (self.eq_geometry.atom_coords_com[n][2]+np.random.normal(loc=0,scale=blur_sigma))*1e-10])
                        r_list.append(rs)
                    
            elif self.method=='wigner':
                mode_counter=0
                new_geom = self.eq_geometry.atom_coords_com.copy()
                for nmode, omega in zip(self.eq_geometry.nmodes_weighted, self.eq_geometry.omegas):
                    mode_freq_hartree = omega*cm_to_hartree
                    freq_factor = np.sqrt(mode_freq_hartree)
            
                    wigner_sampled = False

                    # chose vibrational state
                    if T>0:
                        Pn = self.Pn_arr[mode_counter,:].copy()
                        Pn_scaled = Pn/np.sum(Pn)
                        n=np.random.choice(self.n_list,p=Pn_scaled)
                    else:
                        n=0
                    while not wigner_sampled:
                        random_Q = np.random.uniform(-self.wigner_sample_max,self.wigner_sample_max)
                        random_P = np.random.uniform(-self.wigner_sample_max,self.wigner_sample_max)
                        
                        W = calc_W(random_P,random_Q, n=n)
    
                        if W>np.random.uniform(0,1):
                            for i in range(self.eq_geometry.natoms):
                                new_geom[i,:]+=(random_Q/freq_factor)*nmode[i,:]*np.sqrt(1./(self.eq_geometry.atom_masses[i]*u_to_amu))*bohr_to_angstrom

                            if T>0:
                                self.samp_n_list.append(n)
                            self.samp_q_list.append(random_Q)
                            wigner_sampled = True

                for n in range(self.eq_geometry.natoms):
                    r_list.append(new_geom[n,:]*1e-10)
                            
                        
            # NOTE: should probably rework to do everything on the y0 array then rotate this at the end (incl. velocities)
            if self.random_rotate:
                r_list_rot = random_rotation(r_list)
            else:
                r_list_rot = r_list
                        
            for n in range(self.eq_geometry.natoms):
                y0[3*n] = (r_list_rot[n][0])
                y0[3*n+1] = (r_list_rot[n][1])
                y0[3*n+2] = (r_list_rot[n][2])
                
            self.samp_y0_list.append(y0)
            self.samp_charges_list.append(charges)
            self.samp_masses_list.append(masses)

    def visualize_pool_2D(self, dim1=0,dim2=1, vmax=100, nbins=200):
        """Function for visualizing the 2D pool of geometries as a 2D histogram.

        :param dim1: First dimension (0=x, 1=y etc.) of histogram. Default 0
        :param dim2: Second dimension (0=x, 1=y etc.) of histogram. Default 1
        :param vmax: Maximum for colorbar. Default 100
        :param nbins: Number of bins in each dimension for 2D histogram. Default 200
        """
        dim_list=['x','y','z']
        vis_geom_list = []
        for y0 in self.samp_y0_list:
            for n in range(self.eq_geometry.natoms):
                vis_geom_list.append(y0[n*3:n*3+3]*1e10)

        vis_geom_arr =np.vstack(vis_geom_list)
        dim1_range = (np.min(vis_geom_arr[:,dim1])-0.5, np.max(vis_geom_arr[:,dim1])+0.5)
        dim2_range = (np.min(vis_geom_arr[:,dim2])-0.5, np.max(vis_geom_arr[:,dim2])+0.5)

        fig,ax=plt.subplots(figsize=(4,4))
        vmax = self.n_geoms/vmax
        ax.hist2d(vis_geom_arr[:, dim1], vis_geom_arr[:, dim2],
                  bins=nbins, cmap='binary', vmin=0, vmax=vmax)


        ax.set_xlim(dim1_range[0],dim1_range[1])
        ax.set_ylim(dim2_range[0],dim2_range[1])
        ax.set_xlabel(dim_list[dim1] + r" / $\mathrm{\AA}$ ", fontsize=14)
        ax.set_ylabel(dim_list[dim2] + r" / $\mathrm{\AA}$ ", fontsize=14)
        plt.show()

    def wigner_histograms(self):
        """Quick plot of histograms of n and Q"""
        fig,ax=plt.subplots(figsize=(4,3))
        ax.hist(self.samp_n_list, bins=np.arange(self.nmax)-0.5)
        ax.set_xlabel(r'n$_{\mathrm{vib}}$', fontsize=14)
        ax.set_ylabel('no.', fontsize=14)
        plt.show()

        fig,ax=plt.subplots(figsize=(4,3))
        ax.hist(self.samp_q_list, range=(-self.wigner_sample_max,self.wigner_sample_max), bins=int(self.n_geoms/20))
        ax.set_xlabel('Q', fontsize=14)
        ax.set_ylabel('no.', fontsize=14)
        plt.show()
            
class CESim:
    """Class for CE simulation results and methods.
    :param starting_conditions:"""
    def __init__(self, starting_conditions):
        self.starting_conditions=starting_conditions

    def make_timebins(self, t_range_list, n_step_list):
        """Create timebins for simulation.
        
        :param t_range_list: list of tuples of (t1,t2) between which we will populate timestemps
        :param n_step_list: number of timestemps to be populated within each range (t1,t2)
        """
        tsteps_list = []
        for t_range, n_step in zip(t_range_list,n_step_list):
            t_steps = np.linspace(t_range[0],t_range[1], n_step)
            tsteps_list.append(t_steps)

        tsteps_full = np.concatenate(tsteps_list)
        self.set_timebins(tsteps_full)
        
    def set_timebins(self, timebins):
        """Store timebins and related parameters for simulation."""
        self.timebins=timebins
        self.tmin = np.min(timebins)
        self.tmax = np.max(timebins)
        self.n_t_steps = len(self.timebins)

    def store_output(self, solution):
        """Take solution from ODE solver and convert to output (stored in self.output_list)."""
        charges = self.starting_conditions.samp_charges_list[self.sim_counter]
        masses = self.starting_conditions.samp_masses_list[self.sim_counter]
        

        n_atoms = len(charges)
        
        output_array = np.zeros((n_atoms,7))
        for i in range(int(n_atoms)):
            final_v = np.array([solution.y[int((n_atoms+i)*3), self.n_t_steps-1], 
                               solution.y[int((n_atoms+i)*3+1), self.n_t_steps-1],
                               solution.y[int((n_atoms+i)*3+2), self.n_t_steps-1]])

            output_array[i,0:3] = final_v
            output_array[i,3] = charges[i]
            output_array[i,4] = masses[i]
            if len(self.starting_conditions.samp_channel_list):
                output_array[i,5] = self.starting_conditions.samp_channel_list[self.sim_counter].index
            else:
                output_array[i,5]=0
            output_array[i,6] = self.sim_counter

        self.output_list.append(output_array)

    def output_list_to_arr(self):
        """Convert simulation output from list of arrays (self.output_list) 
        to a single array (self.output_arr)"""
        self.output_arr = np.vstack(self.output_list)

    def output_list_to_df(self):
        """Convert simulation output from list of arrays to a Pandas dataframe
        (stored in self.output_df)"""
        try:
            self.output_df = pd.DataFrame(self.output_arr, columns = ['vx_ms','vy_ms','vz_ms','charge_C','mass_kg', 'channel_idx', 'sim_counter'])
        except:
            self.output_list_to_arr()
            self.output_df = pd.DataFrame(self.output_arr, columns = ['vx_ms','vy_ms','vz_ms','charge_C','mass_kg', 'channel_idx', 'sim_counter'])

        self.output_df['charge_e'] = self.output_df['charge_C']/e
        self.output_df['mass_amu'] = self.output_df['mass_kg']/u

        self.output_df['px_SI'] = self.output_df['vx_ms']*self.output_df['mass_kg']
        self.output_df['py_SI'] = self.output_df['vy_ms']*self.output_df['mass_kg']
        self.output_df['pz_SI'] = self.output_df['vz_ms']*self.output_df['mass_kg']

        self.output_df['px_AU'] = self.output_df['px_SI'] / p_au_fac
        self.output_df['py_AU'] = self.output_df['py_SI'] / p_au_fac
        self.output_df['pz_AU'] = self.output_df['pz_SI'] / p_au_fac
        self.output_df['pmag_AU'] = np.sqrt(self.output_df['px_AU']**2+self.output_df['py_AU']**2+self.output_df['pz_AU']**2)

        self.output_df['KE_eV'] = (self.output_df['pmag_AU']**2)/(2*self.output_df['mass_kg']/u)*p_au_KE_eV_fac

        

    def run_sims(self, n_print=100, save_all=False, make_df=True):
        """Simulate CE for each starting condition"""
        self.output_list=[]
        self.save_all=save_all
        if self.save_all:
            self.solution_list = []
        self.sim_counter=0
        for y0 in self.starting_conditions.samp_y0_list:
            solution = solve_ivp(self.newton_equations, [0, self.tmax], y0, t_eval = self.timebins)
            if save_all:
                self.solution_list.append(solution)
            self.store_output(solution)
            if self.sim_counter%n_print==0:
                print(f'On simultion number {self.sim_counter}!')
            self.sim_counter+=1
        if make_df:
            self.output_list_to_df()



    def newton_equations(self,t,y):
        """Setup Newton equations for ODE solver.

        :param t: array of timebins for ODE calculation.
        :param y: the first natoms*3 elements are x,y,z positions of each atom. 
        The next natoms*3 elements are vx,vy,vz of each atom

        :return: dydt array the first natoms*3 are vx,vy,vz positions of each atom (i.e second half of y). 
        Next natoms*3 are ax,ay,az (from F=ma)

        """

        charges = self.starting_conditions.samp_charges_list[self.sim_counter]
        masses = self.starting_conditions.samp_masses_list[self.sim_counter]
        
        dydt = np.zeros((np.shape(y)))
        n_atoms = np.shape(y)[0]/6
        n_atoms = int(n_atoms)
        for n in range(n_atoms):
            ### Get initial velocities for the solution from the input
            dydt[3*n] = y[3*n_atoms + 3*n]
            dydt[3*n+1] = y[3*n_atoms + 3*n+1]
            dydt[3*n+2] = y[3*n_atoms + 3*n+2]
            
            ## Now calculate forces
            force = np.zeros((3,))
            for k in range(n_atoms):
                if k!=n:
                    r1 = np.array([y[3*n], y[3*n+1], y[3*n+2]])
                    r2 = np.array([y[3*k], y[3*k+1], y[3*k+2]])
                    force += coulomb_force(r1,r2, charges[n], charges[k])
                
            dydt[3*n_atoms+3*n:3*n_atoms+3*n+3] = force/masses[n]
    
        return(dydt)
            



            
        
        
            
            
        