import os
import sys
import argparse
import yaml
import logging
from lammps import lammps
import numpy as np
from datetime import datetime
import csv
import shutil
import pandas as pd
from scipy.spatial.distance import cdist
from analyze_nanoparticles import (read_lammps_data, identify_clusters,
                                   calculate_com, calculate_principal_axes,
                                   calculate_particle_distance, calculate_axis_angles, compute_radius_and_height,
                                   enforce_right_handed)
from nanoparticle_utils import (rotate_around_axis, quaternion_rotation,
                                create_rotation_quaternion)
from scipy.spatial.transform import Rotation as R
from nanoparticle_utils import write_rotation_info, calculate_uint_from_structure

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_structure_name(structure_file):
    """Extract base name from structure file without extension"""
    return os.path.splitext(os.path.basename(structure_file))[0]


def setup_working_directory(config_file):
    """Create organized directory structure and copy input files"""
    # Load config to get input file names
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Create main directory with structure name and timestamp
    structure_name = get_structure_name(config['simulation']['structure_file'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(os.getcwd(), f"{structure_name}_{timestamp}")

    # Create subdirectories
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    dump_structure_dir = os.path.join(base_dir, "dump_structure")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dump_structure_dir, exist_ok=True)

    # Copy input files to input directory
    input_files = [
        config_file,
        config['simulation']['structure_file'],
        config['simulation']['potential_file'],
        'main.py',
        'analyze_nanoparticles.py',
        'nanoparticle_utils.py',
        'config.yml'
    ]

    for file in input_files:
        if os.path.exists(file):
            shutil.copy2(file, input_dir)
            logging.info(f"Copied {file} to input directory")
        else:
            logging.warning(f"Input file not found: {file}")

    # Setup logging to both console and file
    log_file = os.path.join(output_dir, 'simulation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    return base_dir, input_dir, output_dir, dump_structure_dir, config


def get_atom_positions(lmp):
    """Extract atom positions from LAMMPS"""
    natoms = lmp.get_natoms()
    positions = np.array(lmp.gather_atoms("x", 1, 3))
    return positions.reshape((natoms, 3))


class NanoparticleSimulation:
    def __init__(self, structure_file, potential_file, cutoff_distance, input_dir, output_dir, dump_structure_dir):
        self.structure_file = os.path.join(input_dir, os.path.basename(structure_file))
        self.potential_file = os.path.join(input_dir, os.path.basename(potential_file))
        self.cutoff_distance = cutoff_distance
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dump_structure_dir = dump_structure_dir
        self.log_file = os.path.join(output_dir, 'lammps.log')
        self.lmp = lammps(cmdargs=["-log", self.log_file])
        self.analysis_data = []  # Store analysis results for CSV output
        self.cluster_axes = {}  # Store principal axes for each cluster
        self.previous_axes = {}
        self.previous_rotations = {}
        self.cumulative_angles = {}

    def setup_simulation(self):
        try:
            # Initialize LAMMPS
            self.lmp.command("units metal")
            self.lmp.command("dimension 3")
            self.lmp.command("boundary p p p")
            self.lmp.command("atom_style atomic")
            self.lmp.command("neighbor 2.0 bin")
            self.lmp.command("neigh_modify delay 0 every 1 check yes")

            # Read structure file
            self.lmp.command(f"read_data {self.structure_file}")

            # Setup simpler LJ potential for testing
            self.lmp.command("pair_style lj/cut 10.0")
            self.lmp.command("pair_coeff * * 0.1681 2.315")  # Parameters for Au

            # Set mass for Au
            self.lmp.command("mass * 196.97")  # Mass of Au in g/mol

            # Set cutoff distance for neighbor list
            self.lmp.command(f"neighbor {self.cutoff_distance / 2} bin")
            self.lmp.command("neigh_modify delay 0 every 1 check yes")

            # Print system info for debugging
            self.lmp.command("variable natoms equal count(all)")
            self.lmp.command("variable ntypes equal atoms")
            self.lmp.command("print 'Number of atoms: ${natoms}'")
            self.lmp.command("print 'Number of atom types: ${ntypes}'")

            # Minimize system before dynamics
            self.lmp.command("minimize 1.0e-3 1.0e-3 1000 10000")

            self.lmp.command("compute vel_all all property/atom vx vy vz")

            # # Get atom positions
            # positions = get_atom_positions(self.lmp)
            #
            # # Identify clusters using DBSCAN
            # cluster_labels = identify_clusters(positions, self.cutoff_distance)
            # unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])

            # Get atom positions and perform DBSCAN once
            positions = get_atom_positions(self.lmp)
            cluster_labels = identify_clusters(positions, self.cutoff_distance)
            self.static_cluster_labels = cluster_labels  # 存储静态聚类结果
            unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])

            # 创建 group，仅一次
            self.cluster_atom_indices = {}
            for cluster_id in unique_clusters:
                atom_indices = np.where(cluster_labels == cluster_id)[0] + 1  # LAMMPS从1开始
                self.cluster_atom_indices[cluster_id] = atom_indices
                group_str = " ".join(map(str, atom_indices))
                self.lmp.command(f"group nano{cluster_id} id {group_str}")
                logging.info(f"[Static] Created group nano{cluster_id} with {len(atom_indices)} atoms")

            # Setup all computes first
            self.lmp.command("compute pe_all all pe/atom")
            self.lmp.command("compute ke_all all ke/atom")

            # Create LAMMPS groups and their computes
            for cluster_id in unique_clusters:
                # Create group
                atom_indices = np.where(cluster_labels == cluster_id)[0] + 1
                indices_str = ' '.join(map(str, atom_indices))
                group_name = f"nano{cluster_id}"
                self.lmp.command(f"group {group_name} id {indices_str}")
                logging.info(f"Created group {group_name} with {len(atom_indices)} atoms")

                # Create computes for this group with reduce sum
                self.lmp.command(f"compute pe_{group_name}_scalar {group_name} reduce sum c_pe_all")
                self.lmp.command(f"compute ke_{group_name}_scalar {group_name} reduce sum c_ke_all")

            self.lmp.command("compute u_int nano0 group/group nano1")

            # Setup trajectory outputs based on config
            dump_freq = self.config['simulation']['dump_frequency']
            dump_formats = self.config['simulation']['dump_formats']

            for format in dump_formats:
                output_file = os.path.join(self.output_dir,
                                           self.config['output']['trajectory_formats'][format])

                if format == 'lammpstrj':
                    self.lmp.command(f"dump traj_{format} all custom {dump_freq} {output_file} "
                                     "id type x y z vx vy vz c_pe_all")
                    self.lmp.command(f"dump_modify traj_{format} sort id")

                elif format == 'xyz':
                    self.lmp.command(f"dump traj_{format} all xyz {dump_freq} {output_file}")

                elif format == 'dcd':
                    self.lmp.command(f"dump traj_{format} all dcd {dump_freq} {output_file}")

                logging.info(f"Added trajectory output in {format} format")

            # Continue with existing dump commands for individual nanoparticles
            for cluster_id in range(2):  # For first two clusters (0 and 1)
                group_name = f"nano{cluster_id}"
                base_name = f"particle{cluster_id}"

                # Structure file
                data_file = os.path.join(self.dump_structure_dir, f"{base_name}.data")
                self.lmp.command(
                    f"dump struct_{cluster_id} {group_name} custom {dump_freq} {data_file} id type x y z c_pe_all")
                self.lmp.command(f"dump_modify struct_{cluster_id} sort id")

            # Setup energy file output
            energy_file = os.path.join(self.output_dir, 'energies.txt')
            energy_csv = os.path.join(self.output_dir, 'energies.csv')
            energy_header = ("Step Time PE KE TotalE Nano0_PE Nano1_PE "
                             "Nano0_KE Nano1_KE Temp Press U_int")

            # Write headers to CSV file
            with open(energy_csv, 'w') as f:
                f.write(f"{energy_header}\n")

            dump_freq = self.config['simulation']['analysis_frequency']

            # Setup for TXT output
            self.lmp.command(f"fix energy_txt all print {dump_freq} "
                             f"\"$(step) $(time) $(pe) $(ke) $(etotal) "
                             f"$(c_pe_nano0_scalar) $(c_pe_nano1_scalar) "
                             f"$(c_ke_nano0_scalar) $(c_ke_nano1_scalar) "
                             f"$(temp) $(press) $(c_u_int)\" "
                             f"file {energy_file} title \"{energy_header}\"")

            # Setup for CSV output
            self.lmp.command(f"fix energy_csv all print {dump_freq} "
                             "\"$(step) $(time) $(pe) $(ke) $(etotal) "
                             "$(c_pe_nano0_scalar) $(c_pe_nano1_scalar) "
                             "$(c_ke_nano0_scalar) $(c_ke_nano1_scalar) "
                             "$(temp) $(press) $(c_u_int)\" "
                             f"file {energy_csv} screen no append yes")

            # Setup thermo output
            self.lmp.command("thermo 100")
            self.lmp.command("thermo_style custom step time temp press pe ke etotal "
                             "c_pe_nano0_scalar c_pe_nano1_scalar "
                             "c_ke_nano0_scalar c_ke_nano1_scalar")

            # Initialize velocities at lower temperature
            self.lmp.command("velocity all create 100.0 12345 dist gaussian")

            logging.info("Simulation setup completed successfully")
            logging.info(f"LAMMPS log file: {self.log_file}")
        except Exception as e:
            logging.error(f"Error during simulation setup: {str(e)}")
            raise

    def analyze_nanoparticles(self, timestep, temp_structure_file):
        """Analyze nanoparticle properties from structure file"""
        # Read the temporary structure file
        positions, box = read_lammps_data(temp_structure_file)

        # Identify clusters
        cluster_labels = identify_clusters(positions, self.cutoff_distance)
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])

        # Store particle properties
        particles = {}
        analysis_results = {'timestep': timestep}

        # Analyze individual particles
        for cluster_id in unique_clusters:
            cluster_positions = positions[cluster_labels == cluster_id]

            # Calculate properties
            com = calculate_com(cluster_positions)
            principal_axes, variances = calculate_principal_axes(cluster_positions)

            # Store particle data
            particles[cluster_id] = {
                'com': com,
                'axes': principal_axes,
                'n_atoms': len(cluster_positions),
                'positions': cluster_positions,
                'atom_ids': np.where(cluster_labels == cluster_id)[0] + 1
            }

            reference = {
                # force_cutoff = self.config['simulation']['force_cutoff']
                "radius": self.config['reference']['radius'],
                "height": self.config['reference']['height']
            }
            radius, height = compute_radius_and_height(positions, reference=reference)
            r_h_ratio = radius / height if height != 0 else 0.0

            # Store geometry results
            analysis_results.update({
                f'particle{cluster_id}_com_x': f"{com[0]:.4f}",
                f'particle{cluster_id}_com_y': f"{com[1]:.4f}",
                f'particle{cluster_id}_com_z': f"{com[2]:.4f}",
                f'particle{cluster_id}_n_atoms': len(cluster_positions),
                f'particle{cluster_id}_radius': f"{radius:.4f}",
                f'particle{cluster_id}_height': f"{height:.4f}",
                f'particle{cluster_id}_r_over_h': f"{r_h_ratio:.4f}"
            })

            # === Add 3-axis rotation angle tracking using quaternions ===

            if cluster_id in self.previous_axes:
                prev_axes = self.previous_axes[cluster_id]

                # === 修复 prev_axes ===
                q_prev, _ = np.linalg.qr(prev_axes)
                if np.linalg.det(q_prev) < 0:
                    q_prev[:, -1] *= -1
                prev_axes = q_prev

                # === 修复 principal_axes ===
                q_curr, _ = np.linalg.qr(principal_axes)
                if np.linalg.det(q_curr) < 0:
                    q_curr[:, -1] *= -1
                principal_axes = q_curr

                # Align axes direction
                for i in range(3):
                    if np.dot(prev_axes[i], principal_axes[i]) < 0:
                        principal_axes[i] *= -1

                # 继续构造旋转
                prev_axes = enforce_right_handed(prev_axes)
                principal_axes = enforce_right_handed(principal_axes)

                prev_rot = R.from_matrix(prev_axes)
                curr_rot = R.from_matrix(principal_axes)

                delta_rot = curr_rot * prev_rot.inv()
                delta_matrix = delta_rot.as_matrix()

                # Extract axis-wise angular difference
                identity = np.eye(3)
                for axis_idx, unit_axis in enumerate(identity):
                    rotated_vec = delta_matrix @ unit_axis
                    angle_rad = np.arccos(np.clip(np.dot(unit_axis, rotated_vec), -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)
                    analysis_results[f'particle{cluster_id}_axis_angle{axis_idx + 1}'] = f"{angle_deg:.4f}"
            else:
                for i in range(1, 4):
                    analysis_results[f'particle{cluster_id}_axis_angle{i}'] = "0.0000"

            self.previous_axes[cluster_id] = principal_axes

            # Dump individual particle structure
            particle_file = os.path.join(
                self.dump_structure_dir,
                f'particle{cluster_id}_step{timestep}.data'
            )
            self.write_particle_structure(particle_file, cluster_positions, box)

            # Store axes information
            self.cluster_axes[cluster_id] = {
                'axes': principal_axes,
                'variances': variances
            }

        # Analyze particle pairs
        force_cutoff = self.config['simulation']['force_cutoff']
        should_apply_force = True

        for i in unique_clusters:
            for j in unique_clusters:
                if i >= j:
                    continue

                distance = calculate_particle_distance(
                    particles[i]['com'],
                    particles[j]['com']
                )

                ref_length = compute_radius_and_height(particles[i]['positions'])[0]
                dr_ratio = distance / ref_length if ref_length != 0 else 0.0

                angles = calculate_axis_angles(
                    particles[i]['axes'],
                    particles[j]['axes']
                )

                if distance < force_cutoff:
                    should_apply_force = False
                    logging.info(f"Particles too close (distance={distance:.4f} A), forces disabled")

                pair_key = f'pair_{i}_{j}'
                analysis_results.update({
                    f'{pair_key}_distance': f"{distance:.4f}",
                    f'{pair_key}_angle1': f"{angles[0]:.4f}",
                    f'{pair_key}_angle2': f"{angles[1]:.4f}",
                    f'{pair_key}_angle3': f"{angles[2]:.4f}",
                    f'{pair_key}_d_over_r': f"{dr_ratio:.4f}"
                })

        analysis_results['forces_active'] = should_apply_force
        self.analysis_data.append(analysis_results)
        return particles, should_apply_force

    def write_particle_structure(self, filename, positions, box):
        """Write individual particle structure to LAMMPS data file"""
        with open(filename, 'w') as f:
            f.write("LAMMPS data file for single nanoparticle\n\n")
            f.write(f"{len(positions)} atoms\n")
            f.write("1 atom types\n\n")

            # Write box bounds with formatting
            for low, high in box:
                f.write(f"{low:.4f} {high:.4f} xlo xhi\n")

            f.write("\nAtoms\n\n")
            for i, pos in enumerate(positions, 1):
                f.write(f"{i} {i} 1 {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")

    def update_atom_positions(self, atom_indices, new_positions):
        """
        Update positions of specific atoms using scatter_atoms.

        Args:
            atom_indices: array of atom indices (0-based)
            new_positions: Nx3 array of new coordinates
        """
        for atom_id, pos in zip(atom_indices, new_positions):
            self.lmp.command(f"set atom {atom_id} x {pos[0]} y {pos[1]} z {pos[2]}")

    # Add safe_unfix method
    def safe_unfix(self, fix_id):
        try:
            self.lmp.command(f"unfix {fix_id}")
        except Exception as e:
            logging.warning(f"Skip unfix {fix_id}: {str(e)}")

    def run_simulation(self, timestep=0.001, steps=10000):
        try:
            self.lmp.command(f"timestep {timestep}")
            self.lmp.command("minimize 1.0e-3 1.0e-3 1000 10000")
            self.lmp.command("fix 1 all nvt temp 300.0 300.0 100.0")
            self.lmp.command("velocity all set 0.0 0.0 0.0")

            if self.config['simulation'].get('enable_rigid', False):
                self.lmp.command("fix rigid0 nano0 rigid single")
                self.lmp.command("fix rigid1 nano1 rigid single")
                logging.info("Enabled rigid body definition for nano0 and nano1")

            cutoff = float(self.config['simulation']['cutoff_distance'])  # 截断距离
            total_steps = int(self.config['simulation']['total_steps'])  # 总步数
            ref_height = float(self.config['reference']['height'])
            ref_radius = float(self.config['reference']['radius'])  # 每个粒子的半径
            force_scale = float(self.config['reference']['force_scale'])
            mass_per_atom = float(self.config['simulation'].get('atom_mass'))
            analysis_freq = self.config['simulation']['analysis_frequency']
            total_time = total_steps * timestep
            d_target = cutoff + 2 * ref_radius  # 目标质心距离 = 边缘3 + 两个半长度
            amu_to_ev_ps2_per_A2 = 1.036427e-4  # amu → eV·ps²/Å²
            v_target = 1

            self.previous_axes = {}
            self.cumulative_rotations = {}

            rotation_path = os.path.join(self.output_dir, 'rotation_analysis_refined.csv')
            rotation_file = open(rotation_path, 'w', newline='')
            rotation_writer = csv.writer(rotation_file)
            rotation_writer.writerow([
                "Timestep",
                "particle0_angle_deg", "particle0_axis_x", "particle0_axis_y", "particle0_axis_z", "particle0_cum_deg",
                "particle1_angle_deg", "particle1_axis_x", "particle1_axis_y", "particle1_axis_z", "particle1_cum_deg"
            ])

            for step in range(0, steps + 1, analysis_freq):
                temp_structure = os.path.join(self.output_dir, f'temp_structure_{step}.data')
                self.lmp.command(f"write_data {temp_structure}")

                particles, should_apply_force = self.analyze_nanoparticles(step, temp_structure)
                apply_rotation = True
                if len(particles) < 2:
                    logging.warning(f"Only {len(particles)} cluster(s) at step {step}, skipping force/torque.")
                    should_apply_force = False
                    apply_rotation = False

                # === write refined rotation info ===
                self.previous_rotations, self.cumulative_angles = write_rotation_info(
                    step=step,
                    particles=particles,
                    rotation_writer=rotation_writer,
                    previous_rotations=self.previous_rotations,
                    cumulative_angles=self.cumulative_angles,
                    enable_rotation=self.config['simulation'].get('enable_rotation', True)
                )

                # === APPLY FOCUS ===
                positions0 = np.array(particles[0]['positions'])  # shape: (N0, 3)
                positions1 = np.array(particles[1]['positions'])  # shape: (N1, 3)

                dists = cdist(positions0, positions1)
                edge_dist = np.min(dists)

                mass0 = particles[0]['n_atoms'] * mass_per_atom * amu_to_ev_ps2_per_A2
                mass1 = particles[1]['n_atoms'] * mass_per_atom * amu_to_ev_ps2_per_A2

                if step == 0 and should_apply_force:
                    a_target = edge_dist / total_time ** 2
                    f0 = a_target * mass0 * force_scale
                    f1 = a_target * mass1 * force_scale
                    self.lmp.command(f"fix 2 nano0 addforce {f0:.6f} 0.0 0.0")
                    self.lmp.command(f"fix 3 nano1 addforce {-f1:.6f} 0.0 0.0")
                    logging.info(f"[Accelerate] Step {step} | a = {a_target:.2e} | f0 = {f0:.3e} | f1 = {-f1:.3e}")

                # === Apply torque ===
                if self.config['simulation'].get('enable_rotation', True) and apply_rotation:
                    rot_cfg = self.config.get('rotation_parameters', {})
                    torque_scale = float(rot_cfg.get('torque_scale'))

                    # --- Particle 0 ---
                    p0_cfg = rot_cfg.get('particle0', {})
                    axis0 = np.array(p0_cfg.get('axis', [1.0, 0.0, 0.0]))
                    torque0 = p0_cfg.get('base_torque')
                    if p0_cfg.get('scale_by_mass', False):
                        mass0 = particles[0]['n_atoms'] * mass_per_atom
                        mass_ref = mass_per_atom  # 你可以设成平均质量、单位质量等参考值
                        torque0 *= torque_scale * (mass0 / mass_ref) * 0.001

                    self.safe_unfix("4")
                    tx0, ty0, tz0 = (axis0 * torque0).tolist()
                    self.lmp.command(f"fix 4 nano0 addtorque {tx0:.6f} {ty0:.6f} {tz0:.6f}")
                    logging.info(f"[Apply Torque] Particle 0 torque: ({tx0:.4f}, {ty0:.4f}, {tz0:.4f})")

                    # --- Particle 1 ---
                    p1_cfg = rot_cfg.get('particle1', {})
                    axis1 = np.array(p1_cfg.get('axis', [0.0, 0.0, 1.0]))
                    torque1 = p1_cfg.get('base_torque')
                    if p1_cfg.get('scale_by_mass', False):
                        mass1 = particles[1]['n_atoms'] * mass_per_atom
                        mass_ref = mass_per_atom
                        torque1 *= torque_scale * (mass1 / mass_ref) * 0.001

                    self.safe_unfix("5")
                    tx1, ty1, tz1 = (axis1 * torque1).tolist()
                    self.lmp.command(f"fix 5 nano1 addtorque {tx1:.6f} {ty1:.6f} {tz1:.6f}")
                    logging.info(f"[Apply Torque] Particle 1 torque: ({tx1:.4f}, {ty1:.4f}, {tz1:.4f})")
                else:
                    self.safe_unfix("4")
                    self.safe_unfix("5")
                    self.lmp.command("fix 4 nano0 addtorque 0.0 0.0 0.0")
                    self.lmp.command("fix 5 nano1 addtorque 0.0 0.0 0.0")

                os.remove(temp_structure)
                self.lmp.command(f"run {analysis_freq}")

                # === 检查是否接近接触，终止模拟 ===
                positions0 = np.array(particles[0]['positions'])  # shape: (N0, 3)
                positions1 = np.array(particles[1]['positions'])  # shape: (N1, 3)

                dists = cdist(positions0, positions1)
                min_dist = np.min(dists)

                if min_dist < cutoff:
                    logging.info(f"[BREAK] Step {step} | Min atomic distance = {min_dist:.3f} < {cutoff} (contact)")
                    break

            self.write_analysis_results()

            for fix_id in ["1", "2", "3", "4", "5"]:
                self.safe_unfix(fix_id)
            rotation_file.close()

        except Exception as e:
            logging.error(f"Error during simulation: {str(e)}")
            raise

    def write_analysis_results(self):
        """Write analysis results to both TXT and CSV formats"""
        # Format floating point numbers in DataFrame
        df = pd.DataFrame(self.analysis_data)
        for column in df.select_dtypes(include=['float64']).columns:
            df[column] = df[column].apply(lambda x: f"{float(x):.4f}")

        # Write to CSV
        csv_file = os.path.join(self.output_dir, 'nanoparticle_analysis.csv')
        df.to_csv(csv_file, index=False)

        # Write to TXT with formatting
        txt_file = os.path.join(self.output_dir, 'nanoparticle_analysis.txt')
        with open(txt_file, 'w') as f:
            f.write('Nanoparticle Analysis Results\n')
            f.write('============================\n\n')

            for data in self.analysis_data:
                f.write(f"Timestep: {data['timestep']}\n")
                f.write('-' * 40 + '\n')

                # Write particle properties with formatting for floating point values
                for key, value in data.items():
                    if key != 'timestep':
                        if isinstance(value, (float, np.float64)):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                f.write('\n')

    def convert_energy_file(self):
        """Convert energy.txt to energy.csv with formatted floating point numbers"""
        try:
            txt_file = os.path.join(self.output_dir, 'energies.txt')
            csv_file = os.path.join(self.output_dir, 'energies.csv')

            # Read the txt file
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            # Get header and data
            header = lines[0].strip()
            data = [line.strip().split() for line in lines[1:] if line.strip()]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=header.split())

            # Format all numeric columns to 4 decimal places
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                if col != 'Step':  # Keep Step as integer
                    df[col] = df[col].astype(float).round(4)

            # Save to CSV
            df.to_csv(csv_file, index=False, float_format='%.4f')
            logging.info(f"Energy data converted and saved to {csv_file}")

        except Exception as e:
            logging.error(f"Error converting energy file: {str(e)}")

    def generate_nn_input_csv(self, output_dir):
        try:
            analysis_path = os.path.join(output_dir, "nanoparticle_analysis.csv")
            energy_path = os.path.join(output_dir, "energies.csv")
            nn_output_path = os.path.join(output_dir, "nn_input_data.csv")

            # Load CSVs
            df_analysis = pd.read_csv(analysis_path)
            df_energy = pd.read_csv(energy_path)
            df_energy = df_energy.drop_duplicates(subset="Step", keep="first")

            # Rename Step column for merge
            df_energy = df_energy.rename(columns={"Step": "timestep"})

            # Ensure timestep columns are integers
            df_analysis['timestep'] = df_analysis['timestep'].astype(int)
            df_energy['timestep'] = df_energy['timestep'].astype(int)

            # Merge on timestep
            df_merged = pd.merge(df_analysis, df_energy, on="timestep", how="inner")

            # Sort by distance before computing derivative
            df_merged = df_merged.sort_values("pair_0_1_distance").reset_index(drop=True)

            # Compute interaction force: F = -dU/dr
            r = df_merged["pair_0_1_distance"].values
            U = df_merged["U_int"].values
            F = -np.gradient(U, r)
            df_merged["F_interaction"] = F

            # Select and rename columns
            selected_columns = df_merged[[
                "timestep",
                "pair_0_1_distance",
                "particle0_radius",
                "particle0_height",
                "particle0_r_over_h",
                "pair_0_1_angle1",
                "pair_0_1_angle2",
                "pair_0_1_angle3",
                "PE",
                "Nano0_PE",
                "Nano1_PE",
                "U_int",
                "F_interaction"
            ]].rename(columns={
                "pair_0_1_distance": "d",
                "particle0_radius": "r0",
                "particle0_height": "h0",
                "particle0_r_over_h": "r0_over_h0",
            })

            # Save to CSV
            selected_columns.to_csv(nn_output_path, index=False, float_format="%.6f")
            print(f"[OK] NN input CSV written to: {nn_output_path}")

        except Exception as e:
            print(f"[ERROR] Failed to generate NN input CSV: {str(e)}")

    def cleanup(self):
        try:
            # Undump trajectory files
            for format in self.config['simulation']['dump_formats']:
                self.lmp.command(f"undump traj_{format}")

            # Unfix energy outputs for both files
            self.lmp.command("unfix energy_txt")
            self.lmp.command("unfix energy_csv")

            # Convert energy file before closing
            self.convert_energy_file()

            self.write_analysis_results()
            self.generate_nn_input_csv(output_dir=self.output_dir)
            self.lmp.close()

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            raise


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_structure(structure_path):
    coords = []
    with open(structure_path) as f:
        read_coords = False
        for line in f:
            if 'Atoms' in line:
                read_coords = True
                next(f)  # Skip header line
                continue
            if read_coords and line.strip():
                parts = line.strip().split()
                if len(parts) >= 4:  # Valid coordinate line
                    coords.append([float(x) for x in parts[2:5]])
    return np.array(coords)


def main():
    parser = argparse.ArgumentParser(description="Nanoparticle Simulation")
    parser.add_argument("--config", default="config.yml",
                        help="Path to configuration file")
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        logging.info("Please ensure the config file exists in the current directory")
        sys.exit(1)

    try:
        # Set up directory structure and copy files
        base_dir, input_dir, output_dir, dump_structure_dir, config = setup_working_directory(args.config)
        logging.info(f"Created directory structure in: {base_dir}")

        sim = NanoparticleSimulation(
            structure_file=config['simulation']['structure_file'],
            potential_file=config['simulation']['potential_file'],
            cutoff_distance=config['simulation']['cutoff_distance'],
            input_dir=input_dir,
            output_dir=output_dir,
            dump_structure_dir=dump_structure_dir
        )
        # Store config for force control access
        sim.config = config

        sim.setup_simulation()
        sim.run_simulation(
            timestep=config['simulation']['timestep'],
            steps=config['simulation']['total_steps'],
        )
        sim.cleanup()

        logging.info(f"Simulation completed. Results saved in: {output_dir}")

    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
