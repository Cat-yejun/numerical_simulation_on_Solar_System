# 🌌 Solar System Orbital Simulation (Runge-Kutta Integration)
 This project simulates the **motion of celestial bodies in the Solar System** using the **Runge--Kutta method (RK4~8)** for numerical integration of Newton's law of gravitation.\
 The simulation visualizes the **trajectories of the Sun, planets, and the Moon** in **3-dimensional space** using Matplotlib.

------------------------------------------------------------------------

## 📜 Theoretical Background
<img width="1860" height="1005" alt="image" src="https://github.com/user-attachments/assets/de21c1ce-4ba0-4ffa-8afd-eca194aea3df" />
<img width="1072" height="620" alt="image" src="https://github.com/user-attachments/assets/6ef44409-6269-4ea9-ba7b-9e8ff71096cb" />

------------------------------------------------------------------------

## ✨ Features

-   **Runge--Kutta Integration (RK4)**
    -   Implements RK4~8th order to integrate positions and velocities under
        Newtonian gravity.\
-   **N-body Simulation**
    -   Models Sun, planets (Mercury → Neptune), and the Moon with
        initial conditions.\
-   **Realistic Initial Conditions**
    -   Positions and velocities are initialized based on astronomical
        data. (Data Source: NASA JPL)\
-   **Matplotlib Animations**
    -   Renders real-time or saved animations of 3D trajectories.\
    -   Supports multiple views: 3D orbit, XY-plane, YZ-plane, XZ-plane
        projections.\
-   **Data Export**
    -   Saves computed positions into `.npy` files (`realX.npy`,
        `realY.npy`, `realZ.npy`) for reuse in other visualization
        pipelines (e.g., OpenGL ray marching project).

------------------------------------------------------------------------

## ⚙️ Requirements

-   Python 3.8+
-   Libraries:
    -   numpy\
    -   matplotlib\
    -   tqdm\
    -   scipy

------------------------------------------------------------------------

## 🚀 Usage

Run the simulation:

``` bash
python N_body_many_planets.py
```

This will:\
1. Compute trajectories of Sun, Moon, and planets using RK4~8.\
2. Save trajectory data as `.npy` files.\
3. Open a Matplotlib window showing **animations of 3D traejectories**

------------------------------------------------------------------------

## 🖼️ Example Output

### Using 4th~8th order Runge-Kutta method
![3d](https://github.com/user-attachments/assets/ebe80174-5c4f-4296-9e90-a927cefcc5fa)

### Geocentric view of the Solar system
![geocentric_theory](https://github.com/user-attachments/assets/881b4172-8218-46a5-9835-4c342eb48789)

