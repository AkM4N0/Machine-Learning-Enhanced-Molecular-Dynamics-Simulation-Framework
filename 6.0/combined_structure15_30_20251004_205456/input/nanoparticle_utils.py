import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess
import tempfile
import os

def identify_nanoparticles(lmp, cutoff_distance):
    """Identify nanoparticles based on distance criterion"""
    # Get atom positions
    natoms = lmp.get_natoms()
    x = lmp.gather_atoms("x", 1, 3)
    
    # Convert to numpy array
    positions = np.array(x).reshape(natoms, 3)
    
    # Find clusters using distance criterion
    clusters = []
    visited = set()
    
    def find_neighbors(atom_id):
        neighbors = set()
        atom_pos = positions[atom_id]
        
        for i in range(natoms):
            if i != atom_id and i not in visited:
                dist = np.linalg.norm(positions[i] - atom_pos)
                if dist < cutoff_distance:
                    neighbors.add(i)
        return neighbors
    
    # Find clusters
    for i in range(natoms):
        if i not in visited:
            cluster = set([i])
            to_visit = find_neighbors(i)
            
            while to_visit:
                current = to_visit.pop()
                if current not in visited:
                    cluster.add(current)
                    visited.add(current)
                    to_visit.update(find_neighbors(current))
            
            clusters.append(sorted(list(cluster)))
    
    # Create LAMMPS groups
    for i, cluster in enumerate(clusters, 1):
        group_str = " ".join(str(atom_id + 1) for atom_id in cluster)
        lmp.command(f"group group{i} id {group_str}")
    
    return clusters


def calculate_uint_from_structure(structure_path, lmp_path="lmp"):
    """
    Calculate interaction energy (U_int) by loading structure, deleting each group,
    and measuring system PE with only one group present.

    Parameters:
        structure_path (str): Path to .data structure file.
        lmp_path (str): LAMMPS executable path (default: 'lmp')
    Returns:
        float: Computed U_int = U_total - U_nano0 - U_nano1
    """

    def write_input(data_file, group_to_keep):
        return f"""
        units metal
        atom_style atomic
        read_data {data_file}
        group nano0 type 1
        group nano1 type 2
        group keep_group union {group_to_keep}
        delete_atoms group all
        read_data {data_file} add yes
        group keep_group delete
        pair_style eam
        pair_coeff * * Au_u3.eam
        compute pe_all all pe
        run 0
        variable pe equal pe
        print ${{pe}}
        """

    with tempfile.TemporaryDirectory() as tmpdir:
        base_input = os.path.join(tmpdir, "input.in")
        base_output = os.path.join(tmpdir, "log.lammps")

        # Run full system
        full_input = f"""
        units metal
        atom_style atomic
        read_data {structure_path}
        pair_style eam
        pair_coeff * * Au_u3.eam
        compute pe_all all pe
        run 0
        variable pe equal pe
        print ${{pe}}
        """
        full_input_path = os.path.join(tmpdir, "full.in")
        with open(full_input_path, "w") as f:
            f.write(full_input)

        full_pe = float(subprocess.check_output([lmp_path, "-in", full_input_path]).decode().splitlines()[-1])

        # Run nano0 only
        nano0_input_path = os.path.join(tmpdir, "nano0.in")
        with open(nano0_input_path, "w") as f:
            f.write(write_input(structure_path, "nano0"))

        nano0_pe = float(subprocess.check_output([lmp_path, "-in", nano0_input_path]).decode().splitlines()[-1])

        # Run nano1 only
        nano1_input_path = os.path.join(tmpdir, "nano1.in")
        with open(nano1_input_path, "w") as f:
            f.write(write_input(structure_path, "nano1"))

        nano1_pe = float(subprocess.check_output([lmp_path, "-in", nano1_input_path]).decode().splitlines()[-1])

    u_int = full_pe - nano0_pe - nano1_pe
    return u_int


def rotate_around_axis(positions, axis, angle, center=None):
    """
    Rotate positions around a given axis by a specific angle.

    Args:
        positions (ndarray): Nx3 atomic positions.
        axis (array-like): 3D vector for axis of rotation.
        angle (float): angle in radians.
        center (array-like): center of rotation (default: origin).

    Returns:
        ndarray: rotated positions
    """
    if center is None:
        center = np.zeros(3)

    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    centered = positions - center
    rotated = centered @ R.T
    return rotated + center

def write_rotation_info(step, particles, rotation_writer,
                        previous_rotations, cumulative_angles,
                        enable_rotation=True):
    """
    Write real rotation angle (in degrees) using quaternion rotation difference.
    Assumes axes are 3x3 orthonormal matrices (principal components).
    """
    row = [step]

    for pid in [0, 1]:
        if not enable_rotation or pid not in particles:
            row.extend(["0.0000", "0.0000", "0.0000"])
            continue

        curr_axes = particles[pid]['axes']
        curr_rot = R.from_matrix(curr_axes)

        if pid not in previous_rotations:
            previous_rotations[pid] = curr_rot
            cumulative_angles[pid] = 0.0
            angle = 0.0
            axis = np.array([0.0, 0.0, 0.0])
        else:
            prev_rot = previous_rotations[pid]
            delta_rot = curr_rot * prev_rot.inv()
            angle = delta_rot.magnitude() * 180 / np.pi  # radians → degrees
            axis = delta_rot.as_rotvec()
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
            else:
                axis = np.array([0.0, 0.0, 0.0])

            cumulative_angles[pid] += angle
            previous_rotations[pid] = curr_rot

        row.append(f"{angle:.4f}")  # rotation angle
        row.extend([f"{x:.4f}" for x in axis])  # axis x, y, z
        row.append(f"{cumulative_angles[pid]:.4f}")  # cumulative angle

    rotation_writer.writerow(row)
    return previous_rotations, cumulative_angles



def quaternion_rotation(positions, quaternion, center=None):
    """
    Rotate positions using quaternion representation.
    
    Args:
        positions: Nx3 array of atomic positions
        quaternion: [w, x, y, z] quaternion
        center: point to rotate around (if None, rotate around origin)
    """
    if center is None:
        center = np.zeros(3)
    
    # Normalize quaternion
    quaternion = np.array(quaternion)
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    
    # Rotation matrix from quaternion
    rot_matrix = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    
    # Center the positions
    centered_pos = positions - center
    
    # Apply rotation
    rotated = np.dot(centered_pos, rot_matrix.T)
    
    # Move back to original center
    return rotated + center

def create_rotation_quaternion(axis, angle):
    """Create a quaternion for rotation around an axis by given angle."""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    half_angle = angle / 2
    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)
    
    return np.array([w, x, y, z])

def validate_angles(angles):
    """Validate rotation angles are within expected ranges"""
    for angle in angles:
        if not -360 <= angle <= 360:
            raise ValueError(f"Rotation angle {angle} outside valid range [-360, 360]")

def euler_to_matrix(angles, sequence='xyz'):
    """Convert Euler angles to rotation matrix using proper sequence"""
    validate_angles(angles)
    rot = R.from_euler(sequence, angles, degrees=True)
    return rot.as_matrix()

def rotate_coordinates(coords, angles, center, sequence='xyz'):
    """Rotate coordinates around center point using Euler angles"""
    # Convert to numpy array if not already
    coords = np.array(coords)
    center = np.array(center)
    
    # Center coordinates
    centered = coords - center
    
    # Get rotation matrix
    rot_matrix = euler_to_matrix(angles, sequence)
    
    # Apply rotation
    rotated = np.dot(centered, rot_matrix.T)
    
    # Move back to original position
    final = rotated + center
    
    return final

def get_center_of_mass(coords):
    """Calculate center of mass of coordinates"""
    return np.mean(coords, axis=0)

def validate_rotated_structure(original, rotated, angles, tolerance=1e-6):
    """Validate that rotation was applied correctly"""
    # Check shape preservation
    if original.shape != rotated.shape:
        raise ValueError("Rotation changed structure shape")
        
    # Check distance preservation
    orig_dist = np.linalg.norm(original - np.mean(original, axis=0), axis=1)
    rot_dist = np.linalg.norm(rotated - np.mean(rotated, axis=0), axis=1)
    if not np.allclose(orig_dist, rot_dist, atol=tolerance):
        raise ValueError("Rotation did not preserve distances")

