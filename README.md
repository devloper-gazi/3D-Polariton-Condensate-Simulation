# 3D-GPE-Simulation 🚀

## Description 🌟

A Python-based 3D Gross–Pitaevskii Equation solver employing the Split-Step Fourier Method. It simulates the time evolution of polariton condensates within a photonic-crystal-like periodic potential and visualises the density modulation on a central slice.

---

## Contents 📂

- **`gpe_3d_simulation.py`**  
  Main simulation script: sets up grid, initial wavefunction, periodic potential, time-evolution loop and plotting.

- **`README.md`**  
  This document: mathematical model, numerical method, default parameters, step-by-step code walkthrough, flow chart, run instructions, file structure.

- **`LICENSE`**  
  MIT License.

---

## 1. Default Parameters ⚙️

- **Spatial grid**  
  - Number of points: `Nₓ = Nᵧ = N𝓏 = 64`  
  - Physical extent: `Lₓ = Lᵧ = L𝓏 = 10.0` → Δx = Δy = Δz = 10.0 ⁄ 64 ≈ 0.15625  

- **Time step**  
  - Δt = 0.0002  

- **Interaction coefficient**  
  - g = 1.0  

- **Periodic potential**  
  - Depth: V₀ = 10.0  
  - Period: a = 2.0  

- **Number of steps**  
  - `steps = 1000`

---

## 2. Mathematical Model 📐

We solve the conservative 3D Gross–Pitaevskii Equation (ℏ = 1, m* = 1):

i ∂ψ(x,y,z,t) ∕ ∂t = [ –½ ∇² + V(x,y,z) + g |ψ|² ] ψ

where

∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²

### 2.1 Periodic Potential 📏

V(x,y,z) = V₀ [ cos²(πx⁄a) + cos²(πy⁄a) + cos²(πz⁄a) ]


- V₀ controls well depth.  
- a controls lattice period (stripe spacing every a units).

---

## 3. Numerical Method: Split-Step Fourier ⚛️

Discretise domain [–Lₓ/2, +Lₓ/2] × [–Lᵧ/2, +Lᵧ/2] × [–L𝓏/2, +L𝓏/2] into Nₓ × Nᵧ × N𝓏 grid. Each Δt step:

1. **Nonlinear + Potential (half step)**
ψ ← ψ × exp[ –i (V + g |ψ|²) × (Δt/2) ]


2. **Kinetic (half step in k-space)**  
ψ_k = 𝔽₃ᴰ[ψ]
ψ_k ← ψ_k × exp[ –i K² × (Δt/2) ]
ψ ← 𝔽₃ᴰ⁻¹[ψ_k]

3. **Nonlinear + Potential (half step)**  

ψ ← ψ × exp[ –i (V + g |ψ|²) × (Δt/2) ]

4. **Normalisation**  
ψ ← ψ ÷ sqrt( ∑ |ψ|² Δx Δy Δz )


---

## 4. Code Walkthrough 📝

```python
import numpy as np
import matplotlib.pyplot as plt

# 1) Parameters & 3D Grid Definition
Nx = Ny = Nz = 64       # cubic grid
Lx = Ly = Lz = 10.0     # domain [–5, +5] each axis
dx = Lx / Nx            # Δx ≈ 0.15625
dy = Ly / Ny            # Δy
dz = Lz / Nz            # Δz

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(-Lz/2, Lz/2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

dt    = 0.0002   # time step
g     = 1.0      # nonlinearity
V0    = 10.0     # potential depth
a     = 2.0      # lattice period
steps = 1000     # iterations

# 2) 3D Periodic Potential
Vx = np.cos(np.pi * X / a)**2
Vy = np.cos(np.pi * Y / a)**2
Vz = np.cos(np.pi * Z / a)**2
V  = V0 * (Vx + Vy + Vz)

# 3) Initial Wavefunction (3D Gaussian)
sigma = 1.0
psi   = np.exp(-(X**2 + Y**2 + Z**2) / (2*sigma**2)) \
     .astype(np.complex128)
psi  /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy * dz)

# 4) Fourier-Space Parameters
kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
kz = 2*np.pi*np.fft.fftfreq(Nz, d=dz)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# 5) Time Evolution Loop
for _ in range(steps):
 # a) Nonlinear + Potential half-step
 psi *= np.exp(-1j*(V + g*np.abs(psi)**2)*dt/2)
 # b) Kinetic half-step in Fourier space
 psi_k = np.fft.fftn(psi)
 psi_k *= np.exp(-1j*K2*dt/2)
 psi   = np.fft.ifftn(psi_k)
 # c) Nonlinear + Potential half-step
 psi *= np.exp(-1j*(V + g*np.abs(psi)**2)*dt/2)
 # d) Renormalise
 norm = np.sqrt(np.sum(np.abs(psi)**2)*dx*dy*dz)
 psi  /= norm

# 6) Compute Density & Plot central slice
density   = np.abs(psi)**2
mid_index = Nz // 2
slice_d   = density[:, :, mid_index]

plt.figure(figsize=(6,5))
plt.imshow(slice_d.T,
        extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
        origin='lower',
        cmap='inferno')
plt.colorbar(label='Density |ψ(x,y,0)|²')
plt.title("3D GPE: Density on z=0 Plane")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
```
## 5. Flow Chart 🔄
1. Initialisation → Define grid, parameters & potential

2. Initial Wavefunction → 3D Gaussian, normalised

3. Precompute K-space → fftfreq & K²

4. Loop (steps)

  ● Nonlinear + V (Δt/2)

  ● FFT → kinetic phase → iFFT

  ● Nonlinear + V (Δt/2)

  ● Renormalise

5. Density Extraction → |ψ|², central slice

6. Plot → heatmap of slice

## 6. How to Run 💻
* Requirements
(Python 3.8+, NumPy, Matplotlib)
```bash
pip install numpy matplotlib
```
* Clone & Execute
```bash
git clone https://github.com/your-username/3D-GPE-Simulation.git
cd 3D-GPE-Simulation
python gpe_3d_simulation.py
```
Edit the top of gpe_3d_simulation.py to tweak grid size, Δt, V₀, g or steps.

## 7. File Structure 🗂️
```bash
├── gpe_3d_simulation.py  # Main simulation code
├── README.md             # This document
└── LICENSE               # MIT License
```
# Unicode Quick-Reference
  ● ubscripts: ₓ, ᵧ, 𝓏
  ●  Superscripts: ²
  ● Greek: π, Δ, σ
  ● Fraction slash: ⁄
  ●Approximation: ≈
  
  Happy simulating! 🚀
