import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import random
from tqdm import tqdm
# import solar_system_test as sst
from decimal import Decimal
from scipy.signal import find_peaks

MASS = 0
XCOOR = 1
YCOOR = 2
ZCOOR = 3
XVEL = 4
YVEL = 5
ZVEL = 6

a = [0, 1/2, 1/2, 1]
b = [[0, 0, 0],
     [1/2, 0, 0],
     [0, 1/2, 0],
     [0, 0, 1]]

G = 6.6743e-11
AU = 1.495978707e11

Sun_Mass = 1.9889e30
Earth_Mass = 5.9722e24
Moon_Mass = 7.349e22
Mercury_Mass = 3.2850e23
Venus_Mass = 4.8670e24
Mars_Mass = 0.64169e24
Jupiter_Mass = 1.8990e27
Saturn_Mass = 5.6846e26
Uranus_Mass = 8.6832e25
Neptune_Mass = 1.02413e26
Halley_Mass = 2.2e14

'''
Earth_Distance = 147.095e9
Moon_Distance = 0.3633e9
Mercury_Distance = 46.000e9
Venus_Distance = 107.480e9
Mars_Distance = 206.650e9
Jupiter_Distance = 740.595e9
Saturn_Distance = 1357.554e9
Uranus_Distance = 2732.696e9
Neptune_Distance = 4471.050e9

Earth_Velocity = 30.29e3
Moon_Velocity = 1082
v_Mercury = 58.98e3
v_Venus = 35.26e3
v_Mars = 26.50e3
v_Jupiter = 13.72e3
v_Saturn = 10.14e3
v_Uranus = 7.13e3
v_Neptune = 5.47e3
'''
Sun_Distance = np.array([-1.071744732955535E+09, 8.265357172182539E+08, 1.827580634990131E+07]) #
Mercury_Distance = np.array([3.390662651858164E+09, -6.785161511079584E+10, -6.003201630573049E+09]) #
Venus_Distance = np.array([1.071543418279953E+11, -7.566748993776649E+09, -6.342138557300097E+09]) #
Earth_Distance = np.array([-1.500298801698535E+11, 2.608520190927167E+09, 1.864201840848848E+07]) #
Mars_Distance = np.array([-6.170266366296435E+10, 2.328475976410813E+11, 6.367819636799917E+09]) #
Jupiter_Distance = np.array([5.203174573090805E+11, -5.499326664750028E+11, -9.359433885043085E+09]) #
Saturn_Distance = np.array([8.703453701521429E+11, -1.210327783184330E+12, -1.360577025725293E+10]) #
Uranus_Distance =  np.array([2.265791707408811E+12, 1.898387756505655E+12, -2.230305557915759E+10]) #
Neptune_Distance = np.array([4.412244421913362E+12, -7.454620468766884E+11, -8.633305512793803E+10]) #
Halley_Distance = np.array([-2.972937565492319E+12, 4.072783671762008E+12, -1.489352333183844E+12])
Moon_Distance = np.array([-1.499026167607415E+11, 2.991337822189607E+09, 1.753553103512758E+07]) #

v_Sun = np.array([-1.085082627994050E+01, -1.125456057100236E+01, 3.513715314250373E-01]) #
v_Mercury = np.array([3.882944237115451E+04, 5.648361640922683E+03, -3.100009521428335E+03]) #
v_Venus  = np.array([2.539001605442623E+03, 3.474387448880036E+04, 3.301773297834352E+02]) #
Earth_Velocity = np.array([-8.425020834612558E+02, -2.991321161864315E+04, 7.264813289324934E-01]) #
v_Mars = np.array([-2.253700588660107E+04, -4.080236253694372E+03, 4.676627209476263E+02]) #
v_Jupiter = np.array([9.328973572870076E+03, 9.598019469063308E+03, -2.485335398670885E+02]) #
v_Saturn = np.array([7.304282864831359E+03, 5.617062466050720E+03, -3.888472136263323E+02]) #
v_Uranus =  np.array([-4.423420673270083E+03, 4.902676218257462E+03, 7.558195334851581E+01]) #
v_Neptune = np.array([8.696321070656642E+02, 5.391552914473834E+03, -1.312500173042375E+02]) #
v_Halley = np.array([7.097860487458852E+02, 5.588158316934020E+02, 1.018610223370180E+02])
v_Moon = np.array([-1.766084859545970E+03, -2.963144348203108E+04, 8.632109316705616E+01]) #

rs = AU*1e-4
tz = (365*24*60*60)
ts = (365*24*60*60)/tz
# print(Sun_Distance)
# print(Sun_Distance[0])
# print(Sun_Distance[1])
# print(Sun_Distance[2])
# Sun_x = Sun_Distance[0]/rs
# print(Sun_x)

#G = 1

'''
particle = []
for i in range(3):
    particle.append([100,
                     random.uniform(-4, 4),
                     random.uniform(-4, 4),
                     random.uniform(-8, 8),
                     random.uniform(-8, 8)])
'''
#particle = [[50, 1, np.sqrt(3), -np.sqrt(3), 1], [50, 1, -np.sqrt(3), np.sqrt(3), 1], [50, -2, 0, 0, -2]]
'''
particle = [[Sun_Mass, 0, 0, 0, 0, 0, 0],
            [Earth_Mass, Earth_Distance/rs, 0, 0, 0, Earth_Velocity*(tz/rs), 0], 
            [Moon_Mass, (Earth_Distance + Moon_Distance)/rs, 0, 0, (Earth_Velocity + Moon_Velocity)*(tz/rs)]]
'''
# vi = Earth_Velocity*(tz/rs)
# vi2 = v_Mercury*(tz/rs)
# vi3 = v_Venus*(tz/rs)
# vi4 = v_Mars*(tz/rs)
# vi5 = v_Jupiter*(tz/rs)
# vi6 = v_Saturn*(tz/rs)
# vi7 = v_Uranus*(tz/rs)
# vi8 = v_Neptune*(tz/rs)

# particle = [[Sun_Mass, Sun_Distance[0]/rs, Sun_Distance[1]/rs, Sun_Distance[2]/rs, v_Sun[0]*(tz/rs), v_Sun[1]*(tz/rs), v_Sun[2]*(tz/rs)],
#             [Mercury_Mass, Mercury_Distance[0]/rs, Mercury_Distance[1]/rs, Mercury_Distance[2]/rs, v_Mercury[0]*(tz/rs), v_Mercury[1]*(tz/rs), v_Mercury[2]*(tz/rs)],
#             [Venus_Mass, Venus_Distance[0]/rs, Venus_Distance[1]/rs, Venus_Distance[2]/rs, v_Venus[0]*(tz/rs), v_Venus[1]*(tz/rs), v_Venus[2]*(tz/rs)],
#             [Earth_Mass, Earth_Distance[0]/rs, Earth_Distance[1]/rs, Earth_Distance[2]/rs, Earth_Velocity[0]*(tz/rs), Earth_Velocity[1]*(tz/rs), Earth_Velocity[2]*(tz/rs)],
#             [Mars_Mass, Mars_Distance[0]/rs, Mars_Distance[1]/rs, Mars_Distance[2]/rs, v_Mars[0]*(tz/rs), v_Mars[1]*(tz/rs), v_Mars[2]*(tz/rs)],
#             [Jupiter_Mass, Jupiter_Distance[0]/rs, Jupiter_Distance[1]/rs, Jupiter_Distance[2]/rs, v_Jupiter[0]*(tz/rs), v_Jupiter[1]*(tz/rs), v_Jupiter[2]*(tz/rs)],
#             [Saturn_Mass, Saturn_Distance[0]/rs, Saturn_Distance[1]/rs, Saturn_Distance[2]/rs, v_Saturn[0]*(tz/rs), v_Saturn[1]*(tz/rs), v_Saturn[2]*(tz/rs)],
#             [Uranus_Mass, Uranus_Distance[0]/rs, Uranus_Distance[1]/rs, Uranus_Distance[2]/rs, v_Uranus[0]*(tz/rs), v_Uranus[1]*(tz/rs), v_Uranus[2]*(tz/rs)],
#             [Neptune_Mass, Neptune_Distance[0]/rs, Neptune_Distance[1]/rs, Neptune_Distance[2]/rs, v_Neptune[0]*(tz/rs), v_Neptune[1]*(tz/rs), v_Neptune[2]*(tz/rs)],
#             [Halley_Mass, Halley_Distance[0]/rs, Halley_Distance[1]/rs, Halley_Distance[2]/rs, v_Halley[0]*(tz/rs), v_Halley[1]*(tz/rs), v_Halley[2]*(tz/rs)]]

particle = [[Sun_Mass, Sun_Distance[0]/rs, Sun_Distance[1]/rs, Sun_Distance[2]/rs, v_Sun[0]*(tz/rs), v_Sun[1]*(tz/rs), v_Sun[2]*(tz/rs)],
            [Mercury_Mass, Mercury_Distance[0]/rs, Mercury_Distance[1]/rs, Mercury_Distance[2]/rs, v_Mercury[0]*(tz/rs), v_Mercury[1]*(tz/rs), v_Mercury[2]*(tz/rs)],
            [Venus_Mass, Venus_Distance[0]/rs, Venus_Distance[1]/rs, Venus_Distance[2]/rs, v_Venus[0]*(tz/rs), v_Venus[1]*(tz/rs), v_Venus[2]*(tz/rs)],
            [Earth_Mass, Earth_Distance[0]/rs, Earth_Distance[1]/rs, Earth_Distance[2]/rs, Earth_Velocity[0]*(tz/rs), Earth_Velocity[1]*(tz/rs), Earth_Velocity[2]*(tz/rs)],
            [Moon_Mass, Moon_Distance[0]/rs, Moon_Distance[1]/rs, Moon_Distance[2]/rs, v_Moon[0]*(tz/rs), v_Moon[1]*(tz/rs), v_Moon[2]*(tz/rs)],
            [Mars_Mass, Mars_Distance[0]/rs, Mars_Distance[1]/rs, Mars_Distance[2]/rs, v_Mars[0]*(tz/rs), v_Mars[1]*(tz/rs), v_Mars[2]*(tz/rs)],
            [Jupiter_Mass, Jupiter_Distance[0]/rs, Jupiter_Distance[1]/rs, Jupiter_Distance[2]/rs, v_Jupiter[0]*(tz/rs), v_Jupiter[1]*(tz/rs), v_Jupiter[2]*(tz/rs)],
            [Saturn_Mass, Saturn_Distance[0]/rs, Saturn_Distance[1]/rs, Saturn_Distance[2]/rs, v_Saturn[0]*(tz/rs), v_Saturn[1]*(tz/rs), v_Saturn[2]*(tz/rs)],
            [Uranus_Mass, Uranus_Distance[0]/rs, Uranus_Distance[1]/rs, Uranus_Distance[2]/rs, v_Uranus[0]*(tz/rs), v_Uranus[1]*(tz/rs), v_Uranus[2]*(tz/rs)],
            [Neptune_Mass, Neptune_Distance[0]/rs, Neptune_Distance[1]/rs, Neptune_Distance[2]/rs, v_Neptune[0]*(tz/rs), v_Neptune[1]*(tz/rs), v_Neptune[2]*(tz/rs)]]



# print(vi)

def velocity_x(t, x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2):
    return vx1

def velocity_y(t, x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2):
    return vy1

def velocity_z(t, x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2):
    return vz1

def acceleration_x(t, x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2, m1, m2):
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2 - z1)**2)
    return (-((x1-x2)/r) * ((G*m1*m2)/r**2))*((tz**2)/(rs**3))
    
def acceleration_y(t, x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2, m1, m2):
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2 - z1)**2)
    return (-((y1-y2)/r) * ((G*m1*m2)/r**2))*((tz**2)/(rs**3))

def acceleration_z(t, x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2, m1, m2):
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2 - z1)**2)
    return (-((z1-z2)/r) * ((G*m1*m2)/r**2))*((tz**2)/(rs**3))

def rungekutta_method(particle, t0, t1, h):
    
    N = int((t1 - t0) / h)

    xs = [[] for i in range(len(particle))]
    vxs = [[] for i in range(len(particle))]

    ys = [[] for i in range(len(particle))]
    vys = [[] for i in range(len(particle))]
    
    zs = [[] for i in range(len(particle))]
    vzs = [[] for i in range(len(particle))]

    ts = np.linspace(t0, t1, N)

    kx = [[0 for i in range(len(a)+1)] for i in range(len(particle))]
    ky = [[0 for i in range(len(a)+1)] for i in range(len(particle))]
    kz = [[0 for i in range(len(a)+1)] for i in range(len(particle))]

    kvx = [[0 for i in range(len(a)+1)] for i in range(len(particle))]
    kvy = [[0 for i in range(len(a)+1)] for i in range(len(particle))]
    kvz = [[0 for i in range(len(a)+1)] for i in range(len(particle))]

    minus = [False for i in range(len(particle))]
    observation = [0 for i in range(len(particle))]
    flag = [True for i in range(len(particle))]
    
    for w in tqdm(range(1, N)): 
        t = (t0 + h*(w-1))
        for j in range(len(particle)):
            for k in range(len(particle)):
                if(k!=j):
                    kx[j][1] = velocity_x(t, particle[j][XCOOR], particle[j][XVEL], particle[j][YCOOR], particle[j][YVEL], particle[j][ZCOOR], particle[j][ZVEL],
                                             particle[k][XCOOR], particle[k][XVEL], particle[k][YCOOR], particle[k][YVEL], particle[k][ZCOOR], particle[k][ZVEL])
                    
                    ky[j][1] = velocity_y(t, particle[j][XCOOR], particle[j][XVEL], particle[j][YCOOR], particle[j][YVEL], particle[j][ZCOOR], particle[j][ZVEL],
                                             particle[k][XCOOR], particle[k][XVEL], particle[k][YCOOR], particle[k][YVEL], particle[k][ZCOOR], particle[k][ZVEL])
                    
                    kz[j][1] = velocity_z(t, particle[j][XCOOR], particle[j][XVEL], particle[j][YCOOR], particle[j][YVEL], particle[j][ZCOOR], particle[j][ZVEL],
                                             particle[k][XCOOR], particle[k][XVEL], particle[k][YCOOR], particle[k][YVEL], particle[k][ZCOOR], particle[k][ZVEL])
            
            FxSum = 0
            FySum = 0
            FzSum = 0
           
            for k in range(len(particle)):
                if(k!=j):
                    FxSum = FxSum + acceleration_x(t, particle[j][XCOOR], particle[j][XVEL], particle[j][YCOOR], particle[j][YVEL], particle[j][ZCOOR], particle[j][ZVEL],
                                                      particle[k][XCOOR], particle[k][XVEL], particle[k][YCOOR], particle[k][YVEL], particle[k][ZCOOR], particle[k][ZVEL],
                                                      particle[j][MASS], particle[k][MASS])

                    FySum = FySum + acceleration_y(t, particle[j][XCOOR], particle[j][XVEL], particle[j][YCOOR], particle[j][YVEL], particle[j][ZCOOR], particle[j][ZVEL],
                                                      particle[k][XCOOR], particle[k][XVEL], particle[k][YCOOR], particle[k][YVEL], particle[k][ZCOOR], particle[k][ZVEL],
                                                      particle[j][MASS], particle[k][MASS])

                    FzSum = FzSum + acceleration_z(t, particle[j][XCOOR], particle[j][XVEL], particle[j][YCOOR], particle[j][YVEL], particle[j][ZCOOR], particle[j][ZVEL],
                                                      particle[k][XCOOR], particle[k][XVEL], particle[k][YCOOR], particle[k][YVEL], particle[k][ZCOOR], particle[k][ZVEL],
                                                      particle[j][MASS], particle[k][MASS])
                    
            kvx[j][1] = FxSum/particle[j][MASS]
            kvy[j][1] = FySum/particle[j][MASS]
            kvz[j][1] = FzSum/particle[j][MASS]

        for j in range(1, len(a)):
            cx = [0 for k in range(len(particle))]
            cy = [0 for k in range(len(particle))]
            cz = [0 for k in range(len(particle))]

            cvx = [0 for k in range(len(particle))]
            cvy = [0 for k in range(len(particle))]
            cvz = [0 for k in range(len(particle))]
            
            for l in range(len(particle)):
                for k in range(j):
                    cx[l] = cx[l] + b[j][k]*h*kx[l][k+1]
                    cy[l] = cy[l] + b[j][k]*h*ky[l][k+1]
                    cz[l] = cz[l] + b[j][k]*h*kz[l][k+1]

                    cvx[l] = cvx[l] + b[j][k]*h*kvx[l][k+1]
                    cvy[l] = cvy[l] + b[j][k]*h*kvy[l][k+1]
                    cvz[l] = cvz[l] + b[j][k]*h*kvz[l][k+1]
                    
            for l in range(len(particle)):
                for k in range(len(particle)):
                    if(k!=l):
                        kx[l][j+1] = velocity_x(t, particle[l][XCOOR]+cx[l], particle[l][XVEL]+cvx[l], particle[l][YCOOR]+cy[l], particle[l][YVEL]+cvy[l], particle[l][ZCOOR]+cz[l], particle[l][ZVEL]+cvz[l],
                                                   particle[k][XCOOR]+cx[k], particle[k][XVEL]+cvx[k], particle[k][YCOOR]+cy[k], particle[k][YVEL]+cvy[k], particle[k][ZCOOR]+cz[k], particle[k][ZVEL]+cvz[k])
                        
                        ky[l][j+1] = velocity_y(t, particle[l][XCOOR]+cx[l], particle[l][XVEL]+cvx[l], particle[l][YCOOR]+cy[l], particle[l][YVEL]+cvy[l], particle[l][ZCOOR]+cz[l], particle[l][ZVEL]+cvz[l],
                                                   particle[k][XCOOR]+cx[k], particle[k][XVEL]+cvx[k], particle[k][YCOOR]+cy[k], particle[k][YVEL]+cvy[k], particle[k][ZCOOR]+cz[k], particle[k][ZVEL]+cvz[k])

                        kz[l][j+1] = velocity_z(t, particle[l][XCOOR]+cx[l], particle[l][XVEL]+cvx[l], particle[l][YCOOR]+cy[l], particle[l][YVEL]+cvy[l], particle[l][ZCOOR]+cz[l], particle[l][ZVEL]+cvz[l],
                                                   particle[k][XCOOR]+cx[k], particle[k][XVEL]+cvx[k], particle[k][YCOOR]+cy[k], particle[k][YVEL]+cvy[k], particle[k][ZCOOR]+cz[k], particle[k][ZVEL]+cvz[k])
               
                FxSum = 0
                FySum = 0   
                FzSum = 0

                for k in range(len(particle)):
                    if(k!=l):
                        FxSum = FxSum + acceleration_x(t, particle[l][XCOOR]+cx[l], particle[l][XVEL]+cvx[l], particle[l][YCOOR]+cy[l], particle[l][YVEL]+cvy[l], particle[l][ZCOOR]+cz[l], particle[l][ZVEL]+cvz[l],
                                                          particle[k][XCOOR]+cx[k], particle[k][XVEL]+cvx[k], particle[k][YCOOR]+cy[k], particle[k][YVEL]+cvy[k], particle[k][ZCOOR]+cz[k], particle[k][ZVEL]+cvz[k],
                                                          particle[l][MASS], particle[k][MASS])

                        FySum = FySum + acceleration_y(t, particle[l][XCOOR]+cx[l], particle[l][XVEL]+cvx[l], particle[l][YCOOR]+cy[l], particle[l][YVEL]+cvy[l], particle[l][ZCOOR]+cz[l], particle[l][ZVEL]+cvz[l],
                                                          particle[k][XCOOR]+cx[k], particle[k][XVEL]+cvx[k], particle[k][YCOOR]+cy[k], particle[k][YVEL]+cvy[k], particle[k][ZCOOR]+cz[k], particle[k][ZVEL]+cvz[k],
                                                          particle[l][MASS], particle[k][MASS])
                        
                        FzSum = FzSum + acceleration_z(t, particle[l][XCOOR]+cx[l], particle[l][XVEL]+cvx[l], particle[l][YCOOR]+cy[l], particle[l][YVEL]+cvy[l], particle[l][ZCOOR]+cz[l], particle[l][ZVEL]+cvz[l],
                                                          particle[k][XCOOR]+cx[k], particle[k][XVEL]+cvx[k], particle[k][YCOOR]+cy[k], particle[k][YVEL]+cvy[k], particle[k][ZCOOR]+cz[k], particle[k][ZVEL]+cvz[k],
                                                          particle[l][MASS], particle[k][MASS])
           
                kvx[l][j+1] = FxSum/particle[l][MASS]
                kvy[l][j+1] = FySum/particle[l][MASS]
                kvz[l][j+1] = FzSum/particle[l][MASS]

        xf = []
        yf = []
        zf = []

        vxf = []
        vyf = []
        vzf = []
            
        for i in range(len(particle)):       
            xf.append(particle[i][XCOOR] + h*(kx[i][1] + 2*kx[i][2] + 2*kx[i][3] + kx[i][4])/6)
            yf.append(particle[i][YCOOR] + h*(ky[i][1] + 2*ky[i][2] + 2*ky[i][3] + ky[i][4])/6)
            zf.append(particle[i][ZCOOR] + h*(kz[i][1] + 2*kz[i][2] + 2*kz[i][3] + kz[i][4])/6)

            vxf.append(particle[i][XVEL] + h*(kvx[i][1] + 2*kvx[i][2] + 2*kvx[i][3] + kvx[i][4])/6)
            vyf.append(particle[i][YVEL] + h*(kvy[i][1] + 2*kvy[i][2] + 2*kvy[i][3] + kvy[i][4])/6)
            vzf.append(particle[i][ZVEL] + h*(kvz[i][1] + 2*kvz[i][2] + 2*kvz[i][3] + kvz[i][4])/6)
    
            xs[i].append(xf[i])
            ys[i].append(yf[i])
            zs[i].append(zf[i])

            vxs[i].append(vxf[i])
            vys[i].append(vyf[i])
            vzs[i].append(vzf[i])

            particle[i][XCOOR] = xf[i]
            particle[i][YCOOR] = yf[i]
            particle[i][ZCOOR] = zf[i]

            particle[i][XVEL] = vxf[i]
            particle[i][YVEL] = vyf[i]
            particle[i][ZVEL] = vzf[i]

        day = (w*h*tz)/(24*60*60)
        #prevDay = ((i-1)*h*tz)/(24*60*60)
        #days = (day + prevDay)/2
        
        # for z in range(len(particle)):
        #     if(particle[z][YCOOR]<0):
        #         minus[z] = True
                    
        #     if(minus[z] == True):
        #         if(day>0 and flag[z]==True):
        #             if(yf[z]>0):
        #                 #print(planet, ':', round(days, 3), 'days', end = ' / true value : ')
        #                 #print(planet, ':', day, 'days')
        #                 observation[z] = day
        #                 flag[z] = False

    return np.array(ts), np.array(xs), np.array(ys), np.array(zs), np.array(vxs), np.array(vys), np.array(vzs), observation

def update_view(num, Qs, points_3d, lines_3d, points_xy, lines_xy, points_yz, lines_yz, points_xz, lines_xz, max_x, max_y, max_z, ax):
    Sun_x_offset = Qs[0][num, 0]
    Sun_y_offset = Qs[0][num, 1]
    Sun_z_offset = Qs[0][num, 2]

    for i in range(len(Qs)):
        Qs[i][:, 0] -= Sun_x_offset 
        Qs[i][:, 1] -= Sun_y_offset 
        Qs[i][:, 2] -= Sun_z_offset
        
        # 3d
        points_3d[i].set_offsets(Qs[i][num, :2])
        points_3d[i].set_3d_properties(Qs[i][num, 2], zdir='z')  # specify zdir
        lines_3d[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_3d[i].set_3d_properties(Qs[i][:num, 2], zdir='z')

        # xy plane
        points_xy[i].set_offsets(Qs[i][num, :2])
        points_xy[i].set_3d_properties(zs=-max_z, zdir='z')  # specify zdir
        lines_xy[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_xy[i].set_3d_properties(zs=-max_z, zdir='z')

        # yz plane
        points_yz[i].set_offsets(Qs[i][num, 1:])
        points_yz[i].set_3d_properties(zs=-max_x, zdir='x')  # specify zdir
        lines_yz[i].set_data(Qs[i][:num, 1], Qs[i][:num, 2])
        lines_yz[i].set_3d_properties(zs=-max_x, zdir='x')

        # xz plane
        points_xz[i].set_offsets(Qs[i][num, ::2])
        points_xz[i].set_3d_properties(zs=max_y, zdir='y')  # specify zdir
        lines_xz[i].set_data(Qs[i][:num, 0], Qs[i][:num, 2])
        lines_xz[i].set_3d_properties(zs=max_y, zdir='y')

    # Earth를 중심으로 plot의 범위를 조정
    center_x = 0  # 이제 Earth가 항상 (0,0)에 있기 때문에 중심 좌표는 0,0입니다.
    center_y = 0
    center_z = 0
    window_size = 1.2 # 이 값을 조절하여 원하는 범위를 설정할 수 있습니다.
    
    ax.set_xlim(center_x - window_size, center_x + window_size)
    ax.set_ylim(center_y - window_size, center_y + window_size)
    ax.set_zlim(center_z - window_size, center_z + window_size)



def update_3d_general(num, Qs, points_3d, lines_3d):
    for i in range(len(Qs)):
        points_3d[i].set_offsets(Qs[i][num, :2])
        points_3d[i].set_3d_properties(Qs[i][num, 2], zdir='z')  # specify zdir
        lines_3d[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_3d[i].set_3d_properties(Qs[i][:num, 2], zdir='z')

def update_3d(num, Qs, points_3d, lines_3d, points_xy, lines_xy, points_yz, lines_yz, points_xz, lines_xz, max_x, max_y, max_z):
    for i in range(len(Qs)):
        # 3d
        points_3d[i].set_offsets(Qs[i][num, :2])
        points_3d[i].set_3d_properties(Qs[i][num, 2], zdir='z')  # specify zdir
        lines_3d[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_3d[i].set_3d_properties(Qs[i][:num, 2], zdir='z')

        # xy plane
        points_xy[i].set_offsets(Qs[i][num, :2])
        points_xy[i].set_3d_properties(zs=-max_z, zdir='z')  # specify zdir
        lines_xy[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_xy[i].set_3d_properties(zs=-max_z, zdir='z')

        # yz plane
        points_yz[i].set_offsets(Qs[i][num, 1:])
        points_yz[i].set_3d_properties(zs=-max_x, zdir='x')  # specify zdir
        lines_yz[i].set_data(Qs[i][:num, 1], Qs[i][:num, 2])
        lines_yz[i].set_3d_properties(zs=-max_x, zdir='x')

        # xz plane
        points_xz[i].set_offsets(Qs[i][num, ::2])
        points_xz[i].set_3d_properties(zs=max_y, zdir='y')  # specify zdir
        lines_xz[i].set_data(Qs[i][:num, 0], Qs[i][:num, 2])
        lines_xz[i].set_3d_properties(zs=max_y, zdir='y')


def update_2d(num, Qs, points, lines, plane):
    for i in range(len(Qs)):
        if(plane==0): # xy plane
            points[i].set_offsets(Qs[i][num, :2])
            lines[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])

        elif(plane==1): # yz plane
            points[i].set_offsets(Qs[i][num, 1:])
            lines[i].set_data(Qs[i][:num, 1], Qs[i][:num, 2])

        elif(plane==2): # xz plane
            points[i].set_offsets(Qs[i][num, ::2])
            lines[i].set_data(Qs[i][:num, 0], Qs[i][:num, 2])

if __name__ == '__main__':
    '''
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')   

    ax.set_xlabel("x", size = 10)
    ax.set_ylabel("y", size = 10)
    ax.set_zlabel("z", size = 10)

    ax.view_init(0, 0)
    '''
    fig = plt.figure(figsize=(15, 10))

    initialTime = ts
    finalTime = ts + 1
    interval = 0.00001

    Numbers = int((finalTime - initialTime) / interval)
    
    T = []

    X = [[0 for j in range(Numbers)] for i in range(len(particle))]
    Y = [[0 for j in range(Numbers)] for i in range(len(particle))]
    Z = [[0 for j in range(Numbers)] for i in range(len(particle))]

    realX = [[] for i in range(len(particle))]
    realY = [[] for i in range(len(particle))]
    realZ = [[] for i in range(len(particle))]

    VX = [[] for i in range(len(particle))]
    VY = [[] for i in range(len(particle))]
    VZ = [[] for i in range(len(particle))]
    
    Q = [[] for i in range(len(particle))]

    T, realX, realY, realZ, VX, VY, VZ, obsVal = rungekutta_method(particle, initialTime, initialTime + 2, interval)
    
    np.save('realX.npy', realX)
    np.save('realY.npy', realY)
    np.save('realZ.npy', realZ)
        
    
    # print(obsVal)

    # print(sum(realZ[1])/len(realZ[1]))

    for i in range(len(particle)):
        X[i] = realX[i][::10]
        Y[i] = realY[i][::10]
        Z[i] = realZ[i][::10]

        Q[i] = np.dstack((X[i], Y[i], Z[i]))[0]

    N = len(X[0])
    
    xmax = max([np.abs(X).max(), np.abs(Y).max(), np.abs(Z).max()])
    #ax.set_xlim(-xmax-1, xmax+1)
    #ax.set_ylim(-xmax-1, xmax+1)
    '''
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    point = [[] for i in range(len(particle))]
    line = [[] for i in range(len(particle))]

    for i in range(len(X)):
        point[i] = ax.scatter([], [], [], s=50)
        line[i], = ax.plot([], [], [])    
    '''

    min_x, max_x = -1.2, 1.2
    min_y, max_y = -1.2, 1.2
    min_z, max_z = -1.2, 1.2

    # 3D 플롯 초기화
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    lines_3d = [ax1.plot([], [], [])[0] for _ in Q]
    points_3d = [ax1.scatter([], [], []) for _ in Q]

    lines_3d_xy = [ax1.plot([], [], zs=1.2, zdir='z', color='silver')[0] for _ in Q]
    points_3d_xy = [ax1.scatter([], [], zs=-max_z, zdir='z', color='silver') for _ in Q]
    
    lines_3d_yz = [ax1.plot([], [], zs=1.2, zdir='x', color='silver')[0] for _ in Q]
    points_3d_yz = [ax1.scatter([], [], zs=-max_x, zdir='x', color='silver') for _ in Q]

    lines_3d_xz = [ax1.plot([], [], zs=1.2, zdir='y', color='silver')[0] for _ in Q]
    points_3d_xz = [ax1.scatter([], [], zs=max_y, zdir='y', color='silver') for _ in Q]

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_zlim(min_z, max_z)

    ani1 = animation.FuncAnimation(fig, update_view, N, fargs=(Q, points_3d, lines_3d,
                                                                points_3d_xy, lines_3d_xy,
                                                                points_3d_yz, lines_3d_yz,
                                                                points_3d_xz, lines_3d_xz,
                                                                max_x, max_y, max_z, ax1), interval=1, blit=False)
    
    # XY 평면 정사영 초기화
    ax2 = fig.add_subplot(1, 2, 2)
    lines_xy = [ax2.plot([], [])[0] for _ in Q]
    points_xy = [ax2.scatter([], []) for _ in Q]

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    ax2.set_aspect('equal', 'box')
    ax2.grid()
    ax2.set_title("xy plane (2d view)")

    ani2 = animation.FuncAnimation(fig, update_2d, N, fargs=(Q, points_xy, lines_xy, 0), interval=1, blit=False)

    '''
    # YZ 평면 정사영 초기화
    ax3 = fig.add_subplot(3, 3, 8)
    lines_yz = [ax3.plot([], [])[0] for _ in Q]
    points_yz = [ax3.scatter([], []) for _ in Q]

    ax3.set_xlabel("y")
    ax3.set_ylabel("z")

    ax3.set_xlim(min_y, max_y)
    ax3.set_ylim(min_z, max_z)
    
    ax3.set_aspect('equal', 'box')
    ax3.grid()
    ax3.set_title("yz plane (2d view)")

    ani3 = animation.FuncAnimation(fig, update_2d, N, fargs=(Q, points_yz, lines_yz, 1), interval=1, blit=False)

    # XZ 평면 정사영 초기화
    ax4 = fig.add_subplot(3, 3, 9)
    lines_xz = [ax4.plot([], [])[0] for _ in Q]
    points_xz = [ax4.scatter([], []) for _ in Q]

    ax4.set_xlabel("x")
    ax4.set_ylabel("z")

    ax4.set_xlim(min_x, max_x)
    ax4.set_ylim(min_z, max_z)

    ax4.set_aspect('equal', 'box')
    ax4.grid()
    ax4.set_title("xz plane (2d view)")

    ani4 = animation.FuncAnimation(fig, update_2d, N, fargs=(Q, points_xz, lines_xz, 2), interval=1, blit=False)
    
    # 3D view from xy plane
    ax5 = fig.add_subplot(3, 3, 4, projection='3d')
    lines_3d_view_xy = [ax5.plot([], [], [])[0] for _ in Q]
    points_3d_view_xy = [ax5.scatter([], [], []) for _ in Q]
    
    ax5.view_init(90, 270)
    ax5.set_title("xy plane (3d view)", y=0.95)

    ax5.set_xlabel("x")
    ax5.set_ylabel("y")

    ax5.set_xlim(min_x, max_x)
    ax5.set_ylim(min_y, max_y)
    ax5.set_zlim(min_z, max_z)

    ax5.w_zaxis.set_ticklabels([])

    ani5 = animation.FuncAnimation(fig, update_3d_general, N, fargs=(Q, points_3d_view_xy, lines_3d_view_xy), interval=1, blit=False)
    
    
    # 3D view from yz plane
    ax6 = fig.add_subplot(3, 3, 5, projection='3d')
    lines_3d_view_yz = [ax6.plot([], [], [])[0] for _ in Q]
    points_3d_view_yz = [ax6.scatter([], [], []) for _ in Q]

    ax6.view_init(0, 0)
    ax6.set_title("yz plane (3d view)", y=0.88)

    ax6.set_ylabel("y")
    ax6.set_zlabel("z")

    ax6.set_xlim(min_x, max_x)
    ax6.set_ylim(min_y, max_y)
    ax6.set_zlim(min_z, max_z)

    ax6.w_xaxis.set_ticklabels([])

    ani6 = animation.FuncAnimation(fig, update_3d_general, N, fargs=(Q, points_3d_view_yz, lines_3d_view_yz), interval=1, blit=False)

    # 3D view from xz plane
    ax7 = fig.add_subplot(3, 3, 6, projection='3d')
    lines_3d_view_xz = [ax7.plot([], [], [])[0] for _ in Q]
    points_3d_view_xz = [ax7.scatter([], [], []) for _ in Q]

    ax7.view_init(0, 270)
    ax7.set_title("xz plane (3d view)", y=0.88)

    ax7.set_xlabel("x")
    ax7.set_zlabel("z")

    ax7.set_xlim(min_x, max_x)
    ax7.set_ylim(min_y, max_y)
    ax7.set_zlim(min_z, max_z)

    ax7.w_yaxis.set_ticklabels([])

    ani7 = animation.FuncAnimation(fig, update_3d_general, N, fargs=(Q, points_3d_view_xz, lines_3d_view_xz), interval=1, blit=False)
    '''
    # def combined_update(num, Qs, points_3d, lines_3d, points_3d_xy, lines_3d_xy, points_3d_yz, lines_3d_yz, points_3d_xz, lines_3d_xz, max_x, max_y, max_z,
    #                     points_xy, lines_xy,
    #                     points_yz, lines_yz, 
    #                     points_xz, lines_xz,
    #                     points_3d_view_xy, lines_3d_view_xy,
    #                     points_3d_view_yz, lines_3d_view_yz,
    #                     points_3d_view_xz, lines_3d_view_xz):
        
    #     update_3d(num, Qs, points_3d, lines_3d, 
    #                        points_3d_xy, lines_3d_xy, 
    #                        points_3d_yz, lines_3d_yz, 
    #                        points_3d_xz, lines_3d_xz, 
    #                        max_x, max_y, max_z)
    
    #     update_2d(num, Qs, points_xy, lines_xy, 0)
    #     update_2d(num, Qs, points_yz, lines_yz, 1)
    #     update_2d(num, Qs, points_xz, lines_xz, 2)

    #     update_3d_general(num, Qs, points_3d_view_xy, lines_3d_view_xy)
    #     update_3d_general(num, Qs, points_3d_view_yz, lines_3d_view_yz)
    #     update_3d_general(num, Qs, points_3d_view_xz, lines_3d_view_xz)
        
    # combined = animation.FuncAnimation(fig, combined_update, N, fargs=(Q, points_3d, lines_3d,
    #                                                                       points_3d_xy, lines_3d_xy,
    #                                                                       points_3d_yz, lines_3d_yz,
    #                                                                       points_3d_xz, lines_3d_xz,
    #                                                                       max_x, max_y, max_z,
    #                                                                       points_xy, lines_xy,
    #                                                                       points_yz, lines_yz,
    #                                                                       points_xz, lines_xz,
    #                                                                       points_3d_view_xy, lines_3d_view_xy,
    #                                                                       points_3d_view_yz, lines_3d_view_yz,
    #                                                                       points_3d_view_xz, lines_3d_view_xz), interval=1, blit=False)

    # combined.save('combined.gif', writer='pillow', fps=30)
    # ani1.save('3d.gif', writer='pillow', fps=30)
    # ani2.save('xy.gif', writer='pillow', fps=30)
    plt.show()
