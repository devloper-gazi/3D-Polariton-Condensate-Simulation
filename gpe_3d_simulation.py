import numpy as np
import matplotlib.pyplot as plt

# ================================
# 3D Gross–Pitaevskii Simulation
# ================================

# 1) Parametreler ve 3D Grid Tanımı
Nx = Ny = Nz = 64       # Her eksendeki nokta sayısı (kübik ızgara)
Lx = Ly = Lz = 10.0     # Fiziksel boyutlar [-L/2, L/2] her eksende
dx = Lx / Nx            # Uzay adımı (x)
dy = Ly / Ny            # Uzay adımı (y)
dz = Lz / Nz            # Uzay adımı (z)

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(-Lz/2, Lz/2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

dt = 0.0002             # Zaman adımı
g = 1.0                 # Etkileşim katsayısı (örneğin GaAs için)
V0 = 10.0               # Potansiyel derinliği
a = 2.0                 # Potansiyel periyodu
steps = 1000            # Zaman evrimi adım sayısı (deneme için küçük tutuldu)

# 2) 3D Periyodik Potansiyel
Vx = np.cos(np.pi * X / a)**2
Vy = np.cos(np.pi * Y / a)**2
Vz = np.cos(np.pi * Z / a)**2
V = V0 * (Vx + Vy + Vz)  # V(x,y,z) = V0 [cos^2(pi x/a) + cos^2(pi y/a) + cos^2(pi z/a)]

# 3) Başlangıç Dalga Fonksiyonu: 3D Gaussian
sigma = 1.0
psi = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2)).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy * dz)

# 4) Fourier Uzayı Parametreleri (3D)
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# 5) Zaman Evrimi: Split-Step Fourier Yöntemi (3D)
for _ in range(steps):
    # 5a) Nonlinear + Potansiyel Adımı (yarı adım)
    psi *= np.exp(-1j * (V + g * np.abs(psi)**2) * dt / 2)
    
    # 5b) Kinetik Adım (3D Fourier Uzayında, yarı adım)
    psi_k = np.fft.fftn(psi)
    psi_k *= np.exp(-1j * K2 * dt / 2)
    psi = np.fft.ifftn(psi_k)
    
    # 5c) Nonlinear + Potansiyel Adımı (yarı adım tekrar)
    psi *= np.exp(-1j * (V + g * np.abs(psi)**2) * dt / 2)
    
    # 5d) Normalize Etme (Norm Koruması)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy * dz)
    psi /= norm

# 6) Yoğunluğu Hesapla
density = np.abs(psi)**2

# 7) Bir kesit görüntüsü: z = 0 düzlemindeki yoğunluk
mid_index = Nz // 2
slice_density = density[:, :, mid_index]

plt.figure(figsize=(6, 5))
plt.imshow(slice_density.T,
           extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
           origin='lower',
           cmap='inferno')
plt.colorbar(label='Yoğunluk $|\psi(x,y,0)|^2$')
plt.title("3D GPE: z=0 Düzleminde Yoğunluk Modülasyonu")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
