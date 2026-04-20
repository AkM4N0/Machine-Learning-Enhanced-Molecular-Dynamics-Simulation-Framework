import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import yaml
from scipy.spatial.distance import cdist


def read_lammps_data(filename):
    """Read LAMMPS data file with debug information"""
    print(f"Reading LAMMPS data file: {filename}")
    
    atoms = []
    box = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    reading_atoms = False
    for i, line in enumerate(lines):
        if 'atoms' in line:
            n_atoms = int(line.split()[0])
        elif 'xlo xhi' in line:
            box.append([float(x) for x in line.split()[:2]])
        elif 'ylo yhi' in line:
            box.append([float(x) for x in line.split()[:2]])
        elif 'zlo zhi' in line:
            box.append([float(x) for x in line.split()[:2]])
        elif 'Atoms' in line:
            reading_atoms = True
            atoms_start = i + 2
            continue
            
        if reading_atoms and i >= atoms_start and len(atoms) < n_atoms:
            data = line.split()
            if len(data) >= 5:  # atom-ID atom-type x y z
                atoms.append([float(data[2]), float(data[3]), float(data[4])])
                
    # Debug: Print file contents summary
    print(f"Read {len(atoms)} atoms")
    print(f"Box dimensions: {box}")
    atoms_array = np.array(atoms)
    if len(atoms) > 0:
        print(f"Position ranges: min={atoms_array.min(axis=0)}, max={atoms_array.max(axis=0)}")
    
    return np.array(atoms), np.array(box)


def identify_clusters(positions, cutoff_distance=3.0, min_atoms=10):
    """
    Identify nanoparticle clusters using DBSCAN, with fallback to manual split.
    """
    print(f"Clustering parameters: cutoff={cutoff_distance}, min_atoms={min_atoms}")
    print(f"Input positions shape: {positions.shape}")

    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected Nx3 positions array, got shape {positions.shape}")

    n = len(positions)

    # === Case 1: cutoff = 0 → manual ===
    if cutoff_distance <= 0.0:
        print("[INFO] Cutoff <= 0.0, skipping DBSCAN. Assigning two clusters manually.")
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        return labels

    # === Case 2: use DBSCAN ===
    clustering = DBSCAN(
        eps=cutoff_distance,
        min_samples=min_atoms,
        metric='euclidean'
    ).fit(positions)

    labels = clustering.labels_
    unique_clusters = set(labels)
    n_clusters = len(unique_clusters) - (1 if -1 in labels else 0)

    print(f"[INFO] DBSCAN found {n_clusters} clusters.")
    print(f"Cluster labels: {np.unique(labels, return_counts=True)}")

    # === Case 3: fallback if not exactly 2 clusters ===
    if n_clusters != 2:
        print(f"[WARN] DBSCAN failed to find 2 clusters (found {n_clusters}). Fallback to manual split.")
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        return labels

    return labels

def enforce_right_handed(matrix):
    """Ensure matrix is orthogonal and right-handed"""
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def calculate_com(positions):
    return np.mean(positions, axis=0)

def calculate_principal_axes(positions):

    centered_pos = positions - calculate_com(positions)
    pca = PCA(n_components=3)
    pca.fit(centered_pos)
    axes = pca.components_.T

    axes = enforce_right_handed(axes)

    return axes, pca.explained_variance_


def calculate_particle_distance(pos0, pos1, box=None):
    # pos0, pos1: (N0,3), (N1,3)
    pos0 = np.asarray(pos0, float)
    pos1 = np.asarray(pos1, float)
    if box is None or np.size(box) != 6:
        from scipy.spatial.distance import cdist
        return float(cdist(pos0, pos1).min())
    box = np.asarray(box, float)
    L = box[:,1] - box[:,0]
    # 对 pos1 平移到 pos0 的最邻近映像（广播实现）
    # 对每个维度，尝试偏移 -L, 0, +L 的组合并取最小
    shifts = np.array([[dx,dy,dz] for dx in (-L[0],0,L[0])
                               for dy in (-L[1],0,L[1])
                               for dz in (-L[2],0,L[2])], float)  # (27,3)
    dmin = np.inf
    for s in shifts:
        dmin = min(dmin, float(((pos0[:,None,:] - (pos1[None,:,:] + s))**2).sum(axis=2).min()**0.5))
    return dmin


# def calculate_axis_angles(axes1, axes2):
#     """Calculate angles between corresponding principal axes of two particles"""
#     angles = []
#     for ax1, ax2 in zip(axes1, axes2):
#         # Calculate angle between axes in degrees
#         dot_product = np.dot(ax1, ax2)
#         # Ensure dot product is in valid range for arccos
#         dot_product = np.clip(dot_product, -1.0, 1.0)
#         angle = np.arccos(abs(dot_product)) * 180.0 / np.pi
#         angles.append(min(angle, 180.0 - angle))  # Take acute angle
#     return angles

def calculate_axis_angles(axes1, axes2):
    angles = []
    for i in range(3):
        a1 = axes1[:, i] / np.linalg.norm(axes1[:, i])
        a2 = axes2[:, i] / np.linalg.norm(axes2[:, i])
        if np.dot(a1, a2) < 0:
            a2 = -a2
        cosv = np.clip(np.dot(a1, a2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosv)))
    return angles

def compute_radius_and_height(
    positions,
    q_low=0.5, q_high=99.5,
    r_quantile=0.99,
    min_height=1e-3,
    refine_midsection=True,
    # 兼容旧代码：
    reference=None, reference_blend=0.0,
    **kwargs,   # 兜底，避免以后还有别的关键字打进来
):
    P = np.asarray(positions, float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        return 0.0, max(min_height, 0.0)

    # 1) PCA 主轴（右手系）
    com = P.mean(axis=0)
    X = P - com
    axes = PCA(n_components=3).fit(X).components_.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    a = axes[:, 0] / (np.linalg.norm(axes[:, 0]) + 1e-15)

    # 2) 端到端长度 L（宽分位，避免“吃帽”）
    proj = X @ a
    qlo, qhi = np.percentile(proj, [q_low, q_high])
    L = float(qhi - qlo)

    # 3) 垂距 → 半径（中段 + 高分位）
    dperp = np.linalg.norm(np.cross(X, a), axis=1)
    R0 = float(np.quantile(dperp, 0.75))
    if refine_midsection and R0 > 0:
        mid = (proj > qlo + R0) & (proj < qhi - R0)
        d_mid = dperp[mid] if np.any(mid) else dperp
    else:
        d_mid = dperp
    R = float(np.quantile(d_mid, r_quantile))

    # 4) h = L - 2R
    h = max(min_height, L - 2.0 * R)

    # 5) 兼容旧接口：若你真的想“轻微拉回参考值”，这里保留了原逻辑
    if reference and 0.0 < float(reference_blend) <= 1.0:
        R_ref = float(reference.get("radius", R))
        H_ref = float(reference.get("height", h))
        blend = float(reference_blend)
        R = (1 - blend) * R + blend * R_ref
        h = (1 - blend) * h + blend * H_ref

    return R, h

def unwrap_cluster_framewise(P, box):
    """
    以团内 COM 为参考，按帧把该团原子拉回同一镜像（不依赖上一帧）。
    P: (N,3)，box: [(xlo,xhi),(ylo,yhi),(zlo,zhi)]
    返回：同形状数组
    """
    P = np.asarray(P, float).copy()
    Lx, Ly, Lz = [hi - lo for lo, hi in box]
    Ls = np.array([Lx, Ly, Lz], float)
    com = P.mean(axis=0)
    for i in range(P.shape[0]):
        d = P[i] - com
        # 将 d 映射到 (-L/2, L/2]，再还原到以 COM 为中心的最近镜像
        shift = -np.floor((d / Ls) + 0.5) * Ls
        P[i] += shift
    return P


def main():

    # === Load cutoff_distance from config file ===
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    cutoff = config['simulation'].get('cutoff_distance', 3.0)

    # === Read structure file ===
    positions, box = read_lammps_data('structure.data')
    n_atoms = len(positions)

    # === Perform clustering only once ===
    if cutoff <= 0.0:
        print("[INFO] cutoff_distance <= 0.0, manually splitting atoms into two clusters.")
        cluster_labels = np.zeros(n_atoms, dtype=int)
        cluster_labels[n_atoms // 2:] = 1
    else:
        cluster_labels = identify_clusters(positions, cutoff_distance=cutoff)

    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Filter out noise

    # === Precompute atom indices for each cluster ===
    cluster_atom_indices = {}
    for cluster_id in unique_clusters:
        cluster_atom_indices[cluster_id] = np.where(cluster_labels == cluster_id)[0] + 1  # LAMMPS atom IDs start from 1

    particles = {}
    csv_rows = []

    # === Write analysis results ===
    with open('nanoparticle_analysis.txt', 'w') as f:
        f.write('Nanoparticle Analysis Results\n')
        f.write('============================\n\n')

        # --- Individual particle analysis ---
        f.write('Individual Particle Analysis\n')
        f.write('-----------------------------\n\n')

        for cluster_id in unique_clusters:
            cluster_positions = positions[cluster_labels == cluster_id]
            com = calculate_com(cluster_positions)
            principal_axes, variances = calculate_principal_axes(cluster_positions)
            radius, height = compute_radius_and_height(cluster_positions)
            r_h_ratio = radius / height if height != 0 else 0.0

            particles[cluster_id] = {
                'com': com,
                'axes': principal_axes,
                'n_atoms': len(cluster_positions),
                'positions': cluster_positions,
                'atom_indices': cluster_atom_indices[cluster_id]
            }

            f.write(f'Nanoparticle {cluster_id}\n')
            f.write(f'Number of atoms: {len(cluster_positions)}\n')
            f.write(f'Center of Mass: {com[0]:.4f} {com[1]:.4f} {com[2]:.4f}\n')
            f.write('Principal axes:\n')

            row_data = {
                'timestep': 0,
                f'particle{cluster_id}_n_atoms': len(cluster_positions),
                f'particle{cluster_id}_com_x': f"{com[0]:.4f}",
                f'particle{cluster_id}_com_y': f"{com[1]:.4f}",
                f'particle{cluster_id}_com_z': f"{com[2]:.4f}",
                f'particle{cluster_id}_radius': f"{radius:.4f}",
                f'particle{cluster_id}_height': f"{height:.4f}",
                f'particle{cluster_id}_r_over_h': f"{r_h_ratio:.4f}"
            }
            csv_rows.append(row_data)

            for i, (axis, var) in enumerate(zip(principal_axes, variances)):
                f.write(f'  Axis {i+1}: {axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f} (variance: {var:.4f})\n')
            f.write('\n')

        # --- Pairwise particle analysis ---
        f.write('\nParticle-Pair Analysis\n')
        f.write('-----------------------\n\n')

        for i in unique_clusters:
            for j in unique_clusters:
                if i >= j:
                    continue
                distance = calculate_particle_distance(particles[i]['com'], particles[j]['com'])
                angles = calculate_axis_angles(particles[i]['axes'], particles[j]['axes'])

                f.write(f'Particle pair {i}-{j}\n')
                f.write(f'Center-to-center distance: {distance:.4f}\n')
                f.write('Relative orientation angles:\n')
                for ax_idx, angle in enumerate(angles):
                    f.write(f'  Axis {ax_idx+1}: {angle:.2f} degrees\n')
                f.write('\n')

    # === Write CSV output ===
    if csv_rows:
        import csv
        csv_file = "nanoparticle_analysis.csv"
        keys = sorted(csv_rows[0].keys())
        with open(csv_file, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[INFO] CSV saved to {csv_file}")


if __name__ == '__main__':
    main()
