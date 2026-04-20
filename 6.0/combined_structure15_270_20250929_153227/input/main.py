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
                                   calculate_particle_distance, calculate_axis_angles, compute_radius_and_height)
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
            self.lmp.command("change_box all x scale 2 y scale 2 z scale 2")

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
            self.lmp.command("compute pe_nano0 nano0 reduce sum c_pe_all")  # 组势能（global scalar）
            self.lmp.command("compute pe_nano1 nano1 reduce sum c_pe_all")  # 组势能（global scalar）
            # --- 组内动能：ke 可直接对任意组 ---
            self.lmp.command("compute ke_nano0 nano0 ke")
            self.lmp.command("compute ke_nano1 nano1 ke")

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
            self.lmp.command(
                'variable Fint equal "sqrt(c_u_int[1]*c_u_int[1] + c_u_int[2]*c_u_int[2] + c_u_int[3]*c_u_int[3])"')

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
                             "Nano0_KE Nano1_KE Temp Press U_int U_int_Fx U_int_Fy U_int_Fz U_int_Fmag")

            # Write headers to CSV file
            with open(energy_csv, 'w') as f:
                f.write(f"{energy_header}\n")

            dump_freq = self.config['simulation']['analysis_frequency']

            # Setup for TXT output
            self.lmp.command(
                f"fix energy_txt all print {dump_freq} "
                "\"$(step) $(time) $(pe) $(ke) $(etotal) "
                "$(c_pe_nano0_scalar) $(c_pe_nano1_scalar) "
                "$(c_ke_nano0_scalar) $(c_ke_nano1_scalar) "
                "$(temp) $(press) "
                "$(c_u_int:%.6g) $(c_u_int[1]:%.6g) $(c_u_int[2]:%.6g) $(c_u_int[3]:%.6g) $(v_Fint)\" "
                f"file {energy_file} title \"{energy_header}\""
            )

            # Setup for CSV output
            self.lmp.command(
                f"fix energy_csv all print {dump_freq} "
                "\"$(step) $(time) $(pe) $(ke) $(etotal) "
                "$(c_pe_nano0_scalar) $(c_pe_nano1_scalar) "
                "$(c_ke_nano0_scalar) $(c_ke_nano1_scalar) "
                "$(temp) $(press) "
                "$(c_u_int) $(c_u_int[1]) $(c_u_int[2]) $(c_u_int[3]) $(v_Fint)\" "
                f"file {energy_csv} screen no append yes"
            )

            # Setup thermo output
            self.lmp.command("thermo 100")
            self.lmp.command(
                "thermo_style custom step time temp press pe ke etotal "
                "c_pe_nano0 c_pe_nano1 c_ke_nano0 c_ke_nano1 "
                "c_u_int c_u_int[1] c_u_int[2] c_u_int[3] v_Fint"
            )

            # Initialize velocities at lower temperature
            self.lmp.command("velocity all create 100.0 12345 dist gaussian")

            logging.info("Simulation setup completed successfully")
            logging.info(f"LAMMPS log file: {self.log_file}")
        except Exception as e:
            logging.error(f"Error during simulation setup: {str(e)}")
            raise

    def analyze_nanoparticles(self, timestep, temp_structure_file):
        """Analyze nanoparticle properties from structure file (quaternion-only)"""

        # ===== 内部工具：计算 & 记录相对四元数（仅此一套角度体系） =====
        def _compute_and_record_qrel(i, j, particles, edge_distance, dr_ratio, analysis_results):
            """
            计算粒子 i → j 的相对四元数 q_rel，并做符号连续化修正，然后写入 analysis_results。
            输出字段：
              {pair}_distance, {pair}_d_over_r,
              {pair}_qrel_x/y/z/w   # SciPy格式 (x,y,z,w), 范数≈1
            """
            pair_key = f'pair_{i}_{j}'

            # 当前两粒子姿态矩阵 → 四元数（SciPy: 默认 scalar-last → (x,y,z,w)）
            Ri = R.from_matrix(particles[i]['axes'])
            Rj = R.from_matrix(particles[j]['axes'])
            qi = Ri.as_quat()  # (x,y,z,w)
            qj = Rj.as_quat()  # (x,y,z,w)

            # 相对旋转：把 i 旋转到 j 的朝向（乘法次序很重要）
            q_rel = R.from_quat(qj) * R.from_quat(qi).inv()
            q_xyzw = q_rel.as_quat()  # (x,y,z,w)

            # 半球/时间连续性修正：避免 q/−q 跳变
            if not hasattr(self, "prev_qrel"):
                self.prev_qrel = {}
            prev_key = f'{pair_key}_qrel'

            if prev_key in self.prev_qrel:
                if float(np.dot(q_xyzw, self.prev_qrel[prev_key])) < 0.0:
                    q_xyzw = -q_xyzw
            else:
                # 首帧：统一 w>=0（与 SciPy canonical 逻辑一致）
                if q_xyzw[3] < 0.0:
                    q_xyzw = -q_xyzw

            self.prev_qrel[prev_key] = q_xyzw
            qx, qy, qz, qw = map(float, q_xyzw)

            # 仅保留 q_rel（以及距离特征）
            analysis_results.update({
                f'{pair_key}_distance': f"{edge_distance:.4f}",
                f'{pair_key}_d_over_r': f"{dr_ratio:.4f}",
                f'{pair_key}_qrel_x': f"{qx:.6f}",
                f'{pair_key}_qrel_y': f"{qy:.6f}",
                f'{pair_key}_qrel_z': f"{qz:.6f}",
                f'{pair_key}_qrel_w': f"{qw:.6f}",
            })

        # ===== 1) 读取坐标/盒 =====
        positions, box = read_lammps_data(temp_structure_file)

        # ===== 2) 聚类并稳定化为 0/1（按 COM_x 左→右）=====
        cluster_labels = identify_clusters(positions, self.cutoff_distance)
        valid = cluster_labels >= 0
        unique_clusters = np.unique(cluster_labels[valid])

        com_x_list = []
        for cid in unique_clusters:
            pos_c = positions[cluster_labels == cid]
            com_c = calculate_com(pos_c)
            com_x_list.append((cid, com_c[0], com_c))
        order = [cid for cid, _, _ in sorted(com_x_list, key=lambda t: t[1])]
        id_map = {old: new for new, old in enumerate(order)}

        new_labels = np.full_like(cluster_labels, -1)
        for old in id_map:
            new_labels[cluster_labels == old] = id_map[old]
        cluster_labels = new_labels
        unique_clusters = np.array(sorted(id_map.values()))

        particles = {}
        analysis_results = {'timestep': timestep}
        if not hasattr(self, "previous_axes"):
            self.previous_axes = {}

        # ===== 3) 单体分析：COM、主轴、r/h、自旋增量（仅作调试，不导出 angle1/2/3）=====
        for cluster_id in unique_clusters:
            cluster_positions = positions[cluster_labels == cluster_id]

            # (a) COM 与 PCA 主轴（右手系）
            com = calculate_com(cluster_positions)
            principal_axes, variances = calculate_principal_axes(cluster_positions)  # (3,3) 列为轴
            qmat, _ = np.linalg.qr(principal_axes)
            if np.linalg.det(qmat) < 0:
                qmat[:, -1] *= -1
            principal_axes = qmat

            # (b) 与上一帧对齐符号，避免逐帧±翻转
            if cluster_id in self.previous_axes:
                prev_axes = self.previous_axes[cluster_id]
                qp, _ = np.linalg.qr(prev_axes)
                if np.linalg.det(qp) < 0:
                    qp[:, -1] *= -1
                prev_axes = qp
                aligned = principal_axes.copy()
                for i_ax in range(3):
                    if np.dot(prev_axes[:, i_ax], aligned[:, i_ax]) < 0:
                        aligned[:, i_ax] *= -1.0
                if np.linalg.det(aligned) < 0:
                    aligned[:, 2] *= -1.0
                principal_axes = aligned

            # (c) 保存粒子数据
            particles[cluster_id] = {
                'com': com,
                'axes': principal_axes,  # 3x3，列为轴
                'n_atoms': len(cluster_positions),
                'positions': cluster_positions,  # (N,3)
                'atom_ids': np.where(cluster_labels == cluster_id)[0] + 1
            }

            # (d) r/h
            reference = {
                "radius": self.config['reference']['radius'],
                "height": self.config['reference']['height']
            }
            radius, height = compute_radius_and_height(cluster_positions, reference=reference)
            r_h_ratio = radius / height if height != 0 else 0.0

            # (f) 更新上一帧主轴
            self.previous_axes[cluster_id] = principal_axes

            # (g) 写入单体几何
            analysis_results.update({
                f'particle{cluster_id}_com_x': f"{com[0]:.4f}",
                f'particle{cluster_id}_com_y': f"{com[1]:.4f}",
                f'particle{cluster_id}_com_z': f"{com[2]:.4f}",
                f'particle{cluster_id}_n_atoms': len(cluster_positions),
                f'particle{cluster_id}_radius': f"{radius:.4f}",
                f'particle{cluster_id}_height': f"{height:.4f}",
                f'particle{cluster_id}_r_over_h': f"{r_h_ratio:.4f}",
            })

            # (h) 可选：dump 单体结构
            particle_file = os.path.join(self.dump_structure_dir, f'particle{cluster_id}_step{timestep}.data')
            self.write_particle_structure(particle_file, cluster_positions, box)

            # (i) 存 axes/var
            self.cluster_axes[cluster_id] = {'axes': principal_axes, 'variances': variances}

        # ===== 4) 分子对分析：最小原子距 + 相对四元数（唯一角度体系）=====
        force_cutoff = self.config['simulation']['force_cutoff']
        should_apply_force = True

        for i in unique_clusters:
            for j in unique_clusters:
                if i >= j:
                    continue

                # (a) 最小原子-原子距离（如有 PBC 可在 calculate_particle_distance 内部做最小镜像）
                pos_i = particles[i]['positions']
                pos_j = particles[j]['positions']
                edge_distance = calculate_particle_distance(pos_i, pos_j)

                # (b) 归一化距离（用 i 粒子半径）
                r_ref = compute_radius_and_height(pos_i)[0]
                dr_ratio = edge_distance / r_ref if r_ref != 0 else 0.0

                # (c) 仅记录 q_rel（以及距离特征），不再输出 angle1/2/3/轴角/欧拉角
                _compute_and_record_qrel(i, j, particles, edge_distance, dr_ratio, analysis_results)

                # (d) 近距停力
                if edge_distance < force_cutoff:
                    should_apply_force = False
                    logging.info(f"Particles too close (edge_distance={edge_distance:.4f} A), forces disabled")

                # (e) 可选：TXT 行日志
                if hasattr(self, "txt_logger") and self.txt_logger:
                    pair_key = f'pair_{i}_{j}'
                    qx = analysis_results[f'{pair_key}_qrel_x']
                    qy = analysis_results[f'{pair_key}_qrel_y']
                    qz = analysis_results[f'{pair_key}_qrel_z']
                    qw = analysis_results[f'{pair_key}_qrel_w']
                    self.txt_logger.write(
                        f"t={timestep} {pair_key} | dist={edge_distance:.4f} d/r={dr_ratio:.4f} | "
                        f"qrel(xyzw)=({qx},{qy},{qz},{qw})\n"
                    )

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
                # 定义刚体
                self.lmp.command("fix rigid0 nano0 rigid single")
                self.lmp.command("fix rigid1 nano1 rigid single")
                logging.info("Enabled rigid body definition for nano0 and nano1")

                # Langevin 控温
                self.lmp.command("fix thermostat all langevin 300.0 300.0 100.0 48279")
                self.lmp.command("fix_modify thermostat temp thermo_temp")
                logging.info("Enabled rigid body definition for nano0 and nano1")

            cutoff = float(self.config['simulation']['cutoff_distance'])  # 截断距离
            total_steps = int(self.config['simulation']['total_steps'])  # 总步数
            force_scale = float(self.config['reference']['force_scale'])
            mass_per_atom = float(self.config['simulation'].get('atom_mass'))
            analysis_freq = self.config['simulation']['analysis_frequency']
            total_time = total_steps * timestep
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
        """
        读取 nanoparticle_analysis.csv 与 energies.csv，
        仅保留四元数 q_rel（xyzw）+ 距离/能量等需要的列，去除所有 angle 相关列，
        写出 nn_input_data.csv 供机器学习使用。
        """
        import os, re
        import pandas as pd
        import numpy as np

        try:
            analysis_path = os.path.join(output_dir, "nanoparticle_analysis.csv")
            energy_path = os.path.join(output_dir, "energies.csv")
            nn_output_path = os.path.join(output_dir, "nn_input_data.csv")

            # 1) 读取并按 timestep 对齐
            df_a = pd.read_csv(analysis_path)
            df_e = pd.read_csv(energy_path)

            if "timestep" not in df_a.columns:
                raise ValueError("nanoparticle_analysis.csv 缺少列 'timestep'。")
            if "Step" in df_e.columns and "timestep" not in df_e.columns:
                df_e = df_e.drop_duplicates("Step").rename(columns={"Step": "timestep"})

            if "timestep" not in df_e.columns:
                raise ValueError("energies.csv 缺少列 'timestep'（或 'Step'）。")

            df_a["timestep"] = df_a["timestep"].astype(int)
            df_e["timestep"] = df_e["timestep"].astype(int)

            df = (pd.merge(df_a, df_e, on="timestep", how="inner")
                  .sort_values("timestep").reset_index(drop=True))

            # 2) 识别所有 pair 的四元数前缀（pair_i_j）
            pair_prefixes = sorted({
                re.match(r'(pair_\d+_\d+)_qrel_[xyzw]$', c).group(1)
                for c in df.columns
                if re.match(r'(pair_\d+_\d+)_qrel_[xyzw]$', c)
            })
            if not pair_prefixes:
                raise ValueError("未在 nanoparticle_analysis.csv 中找到相对四元数列 (pair_*_qrel_x/y/z/w)。")

            # 选一个“主 pair”（用于生成通用 d 列）
            primary_pair = 'pair_0_1' if 'pair_0_1' in pair_prefixes else pair_prefixes[0]

            # 3) 组装要输出的列（不包含任何 angle 相关列）
            keep_cols = ["timestep"]

            # 距离：保留主 pair 的 d，同时保留所有 pair 的 distance 与 d_over_r（若存在）
            if f"{primary_pair}_distance" in df.columns:
                df["d"] = df[f"{primary_pair}_distance"].astype(float)
                keep_cols.append("d")

            for p in pair_prefixes:
                # 四元数 xyzw
                for comp in ["x", "y", "z", "w"]:
                    col = f"{p}_qrel_{comp}"
                    if col in df.columns:
                        keep_cols.append(col)
                # 配套距离
                dist_col = f"{p}_distance"
                if dist_col in df.columns:
                    keep_cols.append(dist_col)
                dor_col = f"{p}_d_over_r"
                if dor_col in df.columns:
                    keep_cols.append(dor_col)

            # 粒子几何（保留一些常用几何特征；可按需增减）
            mapping = {
                "particle0_radius": "r0",
                "particle0_height": "h0",
                "particle0_r_over_h": "r0_over_h0",
            }
            for src, dst in mapping.items():
                if src in df.columns:
                    df[dst] = df[src].astype(float)
                    keep_cols.append(dst)

            # 能量/力（若存在就带上）
            energy_cols = ["PE", "Nano0_PE", "Nano1_PE",
                           "U_int", "U_int_Fx", "U_int_Fy", "U_int_Fz", "U_int_Fmag"]
            for c in energy_cols:
                if c in df.columns:
                    keep_cols.append(c)

            # —— 关键：排除 angle 类列（我们只通过“选择需要的列”来实现，不去选取 angle ——）
            # 下面这些都不加入 keep_cols：pair_*_angle1/2/3、particle*_axis_angle*、
            # 以及 axis–angle（*_axis_* / *_theta_deg）与 Euler（*_roll_deg/_pitch_deg/_yaw_deg）等。

            # 去重并确保都存在
            keep_cols = [c for i, c in enumerate(keep_cols) if c in df.columns and c not in keep_cols[:i]]

            out = df[keep_cols].copy()

            # 4) 写表（不会携带任何 angle 列）
            out.to_csv(nn_output_path, index=False, float_format="%.6f")
            print(
                f"[OK] NN input CSV written to: {nn_output_path}  (pairs: {', '.join(pair_prefixes)}; primary={primary_pair})")

        except Exception as e:
            print(f"[ERROR] Failed to generate NN input CSV: {e}")

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
