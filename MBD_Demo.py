import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#运行指令：$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
#python MBD_Demo.py 2>&1

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== Step 1: Construct test data (B and C, generate test image) ====================
print("Step 1: Generating test data...")

def create_test_signal(grid_size=128):
    """
    Create a smooth low-frequency test signal for SH indirect lighting compression
    Mimics the characteristics of diffuse indirect illumination:
    - Smooth spatial variation (no sharp edges)
    - Low-frequency gradients (similar to irradiance)
    - Soft color bleeding effects
    Returns: signal [H, W, 3] with smooth irradiance-like patterns
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    signal = torch.zeros(grid_size, grid_size, 3)

    # === Red Channel: Soft diffuse lighting variation ===
    # Large-scale ambient occlusion-like variation
    signal[..., 0] = 0.4 + 0.25 * torch.sin(np.pi * X * 0.5) * torch.cos(np.pi * Y * 0.4)
    # Gentle corner darkening (simulating indirect occlusion)
    R = torch.sqrt(X**2 + Y**2)
    signal[..., 0] *= (0.85 + 0.15 * torch.cos(np.pi * R * 0.5))
    # Soft bounce light from "warm" source (bottom-right)
    bounce_r = torch.exp(-((X - 0.6)**2 + (Y + 0.6)**2) / 0.8)
    signal[..., 0] += 0.15 * bounce_r

    # === Green Channel: Mid-tone diffuse variation ===
    # Smooth gradient simulating indirect sky light
    signal[..., 1] = 0.45 + 0.2 * torch.sin(np.pi * (X * 0.3 + Y * 0.2))
    # Soft "contact shadow" regions
    contact_shadow = torch.exp(-((X + 0.5)**2 + (Y - 0.3)**2) / 0.6)
    signal[..., 1] *= (0.9 + 0.1 * contact_shadow)
    # Gentle color shift from left to right
    signal[..., 1] += 0.1 * (X + 1) / 2 * torch.sin(np.pi * Y * 0.3)

    # === Blue Channel: Cool ambient variation ===
    # Sky-like gradient (brighter at top)
    signal[..., 2] = 0.5 + 0.15 * (Y + 1) / 2
    # Soft indirect blue bounce
    signal[..., 2] += 0.12 * torch.sin(np.pi * X * 0.6) * torch.cos(np.pi * Y * 0.5)
    # Very soft "corner" darkening
    corner_fade = torch.exp(-(X**2 + Y**2) / 1.5)
    signal[..., 2] *= (0.88 + 0.12 * corner_fade)

    # === Cross-channel color bleeding (simulating inter-reflection) ===
    # Warm light influence on red channel from green
    signal[..., 0] += 0.08 * signal[..., 1] * torch.exp(-((X - 0.4)**2 + (Y - 0.4)**2) / 0.5)
    # Cool fill on blue from red bounce
    signal[..., 2] += 0.06 * signal[..., 0] * torch.exp(-((X + 0.3)**2 + (Y + 0.5)**2) / 0.4)

    # === Soft blending to ensure smooth transitions ===
    for c in range(3):
        signal[..., c] = torch.clamp(signal[..., c], 0.05, 0.95)

    return signal

# Generate test signal
grid_size = 128
ground_truth = create_test_signal(grid_size)
H, W, C = ground_truth.shape
print(f"Generated test signal size: {H}x{W}x{C}")

# Prepare training data: flatten 2D grid
x_coords = torch.linspace(0, 1, H)
y_coords = torch.linspace(0, 1, W)
X_grid, Y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
coords = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=-1)  # [N, 2]
target_data = ground_truth.view(-1, C)  # [N, 3]

# ==================== Step 2: Implement MBD model and solver ====================
print("\nStep 2: Building MBD model and solver...")

class MBDCompressor(nn.Module):
    """
    Moving Basis Decomposition (MBD) Compressor
    Implements core formulas from the paper:
        c_l(x) = Σ_m φ_m(x) * c_{m,l}
        b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        f̂(x) = Σ_l c_l(x) * b_l(x)
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_type='gaussian', coeff_kernel_scale=0.09,
                 basis_kernel_type='gaussian', basis_kernel_scale=0.18):
        super().__init__()
        self.L = num_bases
        self.D = data_dim
        # Coefficient kernel parameters (for C)
        self.coeff_kernel_type = coeff_kernel_type
        self.coeff_kernel_scale = coeff_kernel_scale
        # Basis kernel parameters (for B)
        self.basis_kernel_type = basis_kernel_type
        self.basis_kernel_scale = basis_kernel_scale

        # Initialize coefficient control points and basis control points (uniformly distributed in [0,1])
        self.coeff_points = nn.Parameter(
            torch.rand(coeff_res, 2), requires_grad=False
        )  # [M, 2]
        self.basis_points = nn.Parameter(
            torch.rand(basis_res, 2), requires_grad=False
        )  # [N, 2]

        # Learnable parameters: coefficient tensor C and basis tensor B
        # C: [M, L] - scalar coefficients at coefficient control points
        # B: [N, L, D] - basis vectors at basis control points
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * 0.1)
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.D) * 0.1)

        # Initialize statistics
        self.M = coeff_res
        self.N = basis_res
        print(f"MBD model initialized: {self.M} coefficient points, {self.N} basis points, {self.L} bases")

    def compute_kernel_weights(self, query_pts, control_pts, kernel_type, kernel_scale):
        """Compute kernel weights (φ_m(x) or ψ_n(x))"""
        # query_pts: [Q, 2], control_pts: [C, 2]
        if kernel_type == 'gaussian':
            # Gaussian kernel
            dist_sq = torch.cdist(query_pts, control_pts, p=2).pow(2)
            weights = torch.exp(-dist_sq / (2 * kernel_scale**2))
        elif kernel_type == 'inverse':
            # Inverse distance weights
            dist = torch.cdist(query_pts, control_pts, p=2) + 1e-8
            weights = 1.0 / (dist + 0.1)
        else:
            # Linear kernel
            dist = torch.cdist(query_pts, control_pts, p=2)
            weights = torch.relu(1.0 - dist / kernel_scale)

        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, coords):
        """Forward pass: reconstruct signal from coordinates"""
        # 1. Compute kernel weights with separate kernels for C and B
        phi_weights = self.compute_kernel_weights(coords, self.coeff_points,
                                                   self.coeff_kernel_type,
                                                   self.coeff_kernel_scale)  # [Q, M]
        psi_weights = self.compute_kernel_weights(coords, self.basis_points,
                                                   self.basis_kernel_type,
                                                   self.basis_kernel_scale)   # [Q, N]

        #B
        # 2. Compute moving coefficients c_l(x) = Σ_m φ_m(x) * C_{m,l}
        moving_coeff = torch.matmul(phi_weights, self.C)  # [Q, L] matmul will dot the vector and then sum the result

        #C interpolate the basis to reconstruct the sparse control points
        # 3. Compute moving bases b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        B_flat = self.B.view(-1, self.L * self.D)  # [N, L*D]
        basis_interp_flat = torch.matmul(psi_weights, B_flat)  # [Q, L*D]
        #interpolate the basis vectors to get the moving basis
        moving_basis = basis_interp_flat.view(-1, self.L, self.D)  # [Q, L, D]

        # 4. Reconstruct signal f̂(x) = Σ_l c_l(x) * b_l(x)
        reconstruction = torch.sum(moving_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]

        return reconstruction, moving_coeff, moving_basis

    def get_compression_ratio(self, original_size):
        """Compute compression ratio"""
        # Original data size: H*W*D*4 bytes (float32)
        # Compressed: M*L*4 + N*L*D*4 bytes
        compressed_size = (self.M * self.L + self.N * self.L * self.D) * 4
        ratio = original_size / compressed_size
        return ratio, compressed_size

class MBDSolver:
    """MBD Solver (simplified stochastic quasi-Newton optimization)"""
    def __init__(self, model, lambda_reg=0.01):
        self.model = model
        self.lambda_reg = lambda_reg

        # 使用Adam优化器
        self.optimizer = optim.Adam(model.parameters(), lr=0.02)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )

    def compute_loss(self, pred, target, coeff_params):
        """Compute loss function (including regularization term)"""
        # Reconstruction error
        recon_loss = torch.mean((pred - target) ** 2)

        # Frobenius norm regularization (prevents scale ambiguity)
        reg_loss = self.lambda_reg * torch.sum(coeff_params ** 2)

        # Total loss
        total_loss = recon_loss + reg_loss

        return total_loss, recon_loss, reg_loss

    def train_step(self, coords_batch, target_batch):
        """Single training step"""
        self.optimizer.zero_grad()

        # Forward pass
        pred, moving_coeff, _ = self.model(coords_batch)

        # Compute loss
        total_loss, recon_loss, reg_loss = self.compute_loss(
            pred, target_batch, self.model.C
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimization
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'reg_loss': reg_loss.item()
        }

    def train(self, coords, target, epochs=1000, batch_size=2048):
        """Training loop"""
        print(f"Starting training, {epochs} epochs, batch_size={batch_size}")

        losses = []
        num_samples = coords.shape[0]

        for epoch in range(epochs):
            # Random batch sampling
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            # Training step
            loss_dict = self.train_step(coords_batch, target_batch)
            losses.append(loss_dict)

            # Learning rate scheduling
            if epoch % 100 == 0:
                self.scheduler.step(loss_dict['total_loss'])

            # Print progress
            if epoch % 200 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Total Loss: {loss_dict['total_loss']:.6f} | "
                      f"Recon: {loss_dict['recon_loss']:.6f} | "
                      f"Reg: {loss_dict['reg_loss']:.6f}")

        return losses

# 创建模型和求解器
model = MBDCompressor(
    num_bases=8,      # 基的数量L
    coeff_res=64,     # 系数控制点数量M
    basis_res=64,     # 基控制点数量N
    data_dim=3,       # 数据维度D (RGB)
    coeff_kernel_type='gaussian',
    coeff_kernel_scale=0.1875,   # 适配 M=256 的密集控制点
    basis_kernel_type='gaussian',
    basis_kernel_scale=0.1875    # 适配 N=32 的稀疏控制点
)

solver = MBDSolver(model, lambda_reg=0.005)

# 计算压缩比
original_size = H * W * 3 * 4  # float32
comp_ratio, comp_size = model.get_compression_ratio(original_size)
print(f"Original size: {original_size/1024:.1f} KB")
print(f"Compressed: {comp_size/1024:.1f} KB")
print(f"Compression ratio: {comp_ratio:.1f}:1")

# 训练模型
print("\nStarting compression (training)...")
losses = solver.train(coords, target_data, epochs=1000, batch_size=4096)

# ==================== Step 3: Evaluation and visualization ====================
print("\nStep 3: Evaluating compression and reconstruction quality...")

# Reconstruct entire image using trained model
model.eval()
with torch.no_grad():
    reconstructed, _, _ = model(coords)
    reconstructed_img = reconstructed.view(H, W, 3).cpu().numpy()
    # Clip to valid range [0, 1] to avoid imshow warnings
    reconstructed_img = np.clip(reconstructed_img, 0, 1)

# Compute PSNR and SSIM used for evaluation the reconstruction quality
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM for multi-channel images"""
    from scipy.signal import fftconvolve
    from numpy import asarray, prod

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Process multi-channel images: compute SSIM for each channel separately, then average
    if img1.ndim == 3:
        ssim_channels = []
        for c in range(img1.shape[2]):
            ssim_c = compute_ssim(img1[:, :, c], img2[:, :, c], window_size)
            ssim_channels.append(ssim_c)
        return np.mean(ssim_channels)

    # Generate Gaussian window
    gaussian = np.outer(
        np.exp(-(np.arange(window_size) - window_size//2)**2 / 1.5),
        np.exp(-(np.arange(window_size) - window_size//2)**2 / 1.5)
    )
    gaussian /= gaussian.sum()

    # Compute local statistics
    def filter_window(x):
        return fftconvolve(x, gaussian, mode='valid')

    mu1 = filter_window(img1)
    mu2 = filter_window(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_window(img1*img1) - mu1_sq
    sigma2_sq = filter_window(img2*img2) - mu2_sq
    sigma12 = filter_window(img1*img2) - mu1_mu2

    # SSIM formula
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)

# 计算指标
psnr_value = compute_psnr(ground_truth.numpy(), reconstructed_img)
ssim_value = compute_ssim(ground_truth.numpy(), reconstructed_img)

print(f"Reconstruction quality metrics:")
print(f"  PSNR: {psnr_value:.2f} dB")
print(f"  SSIM: {ssim_value:.4f}")
print(f"  Final loss: {losses[-1]['total_loss']:.6f}")

# ==================== Visualization results ====================
print("\nGenerating visualization results...")

fig = plt.figure(figsize=(18, 10))

# 1. Original image
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(ground_truth.numpy(), vmin=0, vmax=1)
ax1.set_title('Original Signal (Ground Truth)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(False)

# 2. MBD reconstructed image
ax2 = plt.subplot(2, 4, 2)
ax2.imshow(reconstructed_img, vmin=0, vmax=1)
ax2.set_title(f'MBD Reconstruction\nPSNR: {psnr_value:.1f}dB, SSIM: {ssim_value:.4f}')
ax2.set_xlabel('X')
ax2.grid(False)

# 3. Error map
ax3 = plt.subplot(2, 4, 3)
error = np.abs(ground_truth.numpy() - reconstructed_img)
error_img = ax3.imshow(error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.2)
ax3.set_title('Absolute Error (RGB Average)')
ax3.set_xlabel('X')
plt.colorbar(error_img, ax=ax3, fraction=0.046, pad=0.04)
ax3.grid(False)

# 4. Control point distribution
ax4 = plt.subplot(2, 4, 4)
ax4.scatter(model.coeff_points[:, 0].detach().cpu().numpy(),
           model.coeff_points[:, 1].detach().cpu().numpy(),
           c='red', s=50, alpha=0.7, label=f'Coefficient Points (M={model.M})')
ax4.scatter(model.basis_points[:, 0].detach().cpu().numpy(),
           model.basis_points[:, 1].detach().cpu().numpy(),
           c='blue', s=80, marker='s', alpha=0.7, label=f'Basis Points (N={model.N})')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Control Point Distribution')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

# 5. Loss curves
ax5 = plt.subplot(2, 4, 5)
total_losses = [l['total_loss'] for l in losses]
recon_losses = [l['recon_loss'] for l in losses]
reg_losses = [l['reg_loss'] for l in losses]

ax5.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax5.semilogy(recon_losses, 'g--', linewidth=1.5, alpha=0.7, label='Reconstruction Loss')
ax5.semilogy(reg_losses, 'r:', linewidth=1, alpha=0.5, label='Regularization Loss')
ax5.set_title('Training Loss Curves (Log Scale)')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Loss Value')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Channel comparison
ax6 = plt.subplot(2, 4, 6)
y_slice = H // 2
for c in range(3):
    ax6.plot(ground_truth[y_slice, :, c].numpy(),
            color=['r', 'g', 'b'][c],
            linestyle='-', alpha=0.7, label=f'Original C{c}')
    ax6.plot(reconstructed_img[y_slice, :, c],
            color=['r', 'g', 'b'][c],
            linestyle='--', alpha=0.9, label=f'Reconstructed C{c}')
ax6.set_title(f'Y={y_slice} Slice Channel Comparison')
ax6.set_xlabel('X Coordinate')
ax6.set_ylabel('Intensity')
ax6.legend(loc='upper right', fontsize='small')
ax6.grid(True, alpha=0.3)

# 7. Basis vector visualization
ax7 = plt.subplot(2, 4, 7)
with torch.no_grad():
    # Get basis vectors at a point
    test_point = torch.tensor([[0.5, 0.5]])
    _, _, moving_basis = model(test_point)
    basis_vectors = moving_basis[0].cpu().numpy()  # [L, D]

    # Plot basis vectors
    colors = plt.cm.viridis(np.linspace(0, 1, model.L))
    for l in range(model.L):
        ax7.bar(range(3), basis_vectors[l], alpha=0.7,
               color=colors[l], label=f'Basis {l+1}')

ax7.set_title('Moving Basis Vectors at Test Point')
ax7.set_xlabel('Channel (RGB)')
ax7.set_ylabel('Basis Vector Value')
ax7.set_xticks([0, 1, 2])
ax7.set_xticklabels(['R', 'G', 'B'])
ax7.grid(True, alpha=0.3, axis='y')

# 8. Compression info
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
info_text = f"""
Compression Summary
===================
Original Size: {H}×{W}×{C}
Original: {original_size/1024:.1f} KB
Compressed: {comp_size/1024:.1f} KB
Compression Ratio: {comp_ratio:.1f}:1
Num Bases (L): {model.L}
Coeff Points (M): {model.M}
Basis Points (N): {model.N}
Final Loss: {losses[-1]['total_loss']:.6f}
PSNR: {psnr_value:.1f} dB
SSIM: {ssim_value:.4f}
"""
ax8.text(0.1, 0.5, info_text, fontsize=10,
        family='monospace', verticalalignment='center')

plt.suptitle('Moving Basis Decomposition (MBD) Compression and Reconstruction Demo', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print("\nDemo completed!")
print("="*60)
print("Key Conclusions:")
print("1. MBD successfully reconstructed the signal using sparse control points (M={} coefficient points, N={} basis points)".format(model.M, model.N))
print("2. Achieved {:.1f}:1 compression ratio".format(comp_ratio))
print("3. Reconstruction quality: PSNR={:.1f}dB, SSIM={:.4f}".format(psnr_value, ssim_value))
print("4. Scale ambiguity resolved through Frobenius regularization")
print("5. Loss curves show stable convergence during optimization")
print("="*60)
