import numpy as np
from ase import io, Atoms
from ase.geometry import cell_to_cellpar
import os
import shutil
import subprocess
import random
import datetime

class Cluster:
    def __init__(self, atoms, name):
        self.atoms = atoms
        self.name = name
        self.positions = atoms.get_positions()
        self.symbols = atoms.get_chemical_symbols()
        self.masses = atoms.get_masses()
        self.com = self.calculate_center_of_mass()

    def calculate_center_of_mass(self):
        """Calculate center of mass for the cluster"""
        total_mass = np.sum(self.masses)
        com = np.dot(self.masses, self.positions) / total_mass
        return com

    def translate(self, vector):
        """Translate all atoms in the cluster by a vector"""
        self.positions += vector
        self.atoms.set_positions(self.positions)
        self.com = self.calculate_center_of_mass()
    
    def rotate(self, angles):
        """Rotate cluster by given angles (in degrees) around x, y, z axes"""
        from scipy.spatial.transform import Rotation
        
        # Convert angles to radians
        angles_rad = np.radians(angles)
        
        # Create rotation object
        rotation = Rotation.from_euler('xyz', angles_rad)
        
        # Apply rotation around center of mass
        positions_centered = self.positions - self.com
        rotated_positions = rotation.apply(positions_centered)
        new_positions = rotated_positions + self.com
        
        # Update positions
        self.positions = new_positions
        self.atoms.set_positions(new_positions)

def load_particle(file_path):
    """Load a single particle from a CIF file and return as a Cluster object"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    atoms = io.read(file_path)
    name = os.path.splitext(os.path.basename(file_path))[0]
    return Cluster(atoms, name)

def combine_clusters(cluster1, cluster2, distance, rotation1=(0,0,0), rotation2=(0,0,0)):
    """Combine two clusters with specified distance and rotations"""
    # Apply rotations
    cluster1.rotate(rotation1)
    cluster2.rotate(rotation2)
    
    # Calculate initial positions relative to cluster1's COM
    com1 = cluster1.com
    com2 = cluster2.com
    
    # Calculate required translation to achieve desired separation
    translation = np.array([distance, 0, 0])
    
    # Create new positions for both clusters
    positions1 = cluster1.positions.copy()
    positions2 = cluster2.positions.copy() + translation
    
    # Combine positions and symbols
    all_positions = np.vstack((positions1, positions2))
    all_symbols = cluster1.symbols + cluster2.symbols
    
    # Create combined structure
    combined = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        pbc=[True, True, True]
    )
    
    return combined

def create_unit_cell(structure, cell_parameters):
    """Create a unit cell with specified parameters"""
    cell = np.array([
        [cell_parameters[0], 0, 0],
        [0, cell_parameters[1], 0],
        [0, 0, cell_parameters[2]]
    ])
    structure.set_cell(cell, scale_atoms=False)
    return structure

def center_structure_in_cell(structure):
    """Center the structure in the middle of the unit cell"""
    # Calculate current center of mass
    positions = structure.get_positions()
    masses = structure.get_masses()
    com = np.dot(masses, positions) / np.sum(masses)
    
    # Calculate cell center
    cell_center = structure.get_cell().diagonal() / 2
    
    # Calculate translation vector
    translation = cell_center - com
    
    # Translate structure
    structure.translate(translation)
    
    return structure

def wrap_positions(structure):
    """Wrap atomic positions to ensure they're within the unit cell"""
    cell = structure.get_cell()
    positions = structure.get_positions()
    
    # Wrap positions into the unit cell
    for i in range(len(positions)):
        for j in range(3):
            while positions[i,j] < 0:
                positions[i,j] += cell[j,j]
            while positions[i,j] >= cell[j,j]:
                positions[i,j] -= cell[j,j]
    
    structure.set_positions(positions)
    return structure

def convert_cif_to_pdb(cif_file, pdb_file):
    """Convert CIF to PDB using Open Babel"""
    try:
        cmd = f"obabel {cif_file} -O {pdb_file}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"Successfully converted CIF to PDB: {pdb_file}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not convert to PDB: {e}")
        print("Skipping PDB conversion...")

def create_lammps_data(structure, output_file):
    """Create LAMMPS data file manually"""
    atoms = structure
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Create atom type mapping
    unique_symbols = list(set(symbols))
    atom_types = {sym: i+1 for i, sym in enumerate(unique_symbols)}
    
    with open(output_file, 'w') as f:
        f.write("# LAMMPS data file from ASE\n\n")
        
        # Write header
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(atom_types)} atom types\n\n")
        
        # Write box dimensions
        f.write(f"0.0 {cell[0][0]} xlo xhi\n")
        f.write(f"0.0 {cell[1][1]} ylo yhi\n")
        f.write(f"0.0 {cell[2][2]} zlo zhi\n\n")
        
        # Write masses
        f.write("Masses\n\n")
        for sym, type_id in atom_types.items():
            mass = atoms.get_masses()[symbols.index(sym)]
            f.write(f"{type_id} {mass}\n")
        f.write("\n")
        
        # Write atom positions
        f.write("Atoms # atomic\n\n")
        for i, (pos, sym) in enumerate(zip(positions, symbols), 1):
            type_id = atom_types[sym]
            f.write(f"{i} {type_id} {pos[0]} {pos[1]} {pos[2]}\n")
    
    print(f"Successfully created LAMMPS data file: {output_file}")

def save_structure(structure, output_dir, base_name):
    """Save the structure in CIF, PDB, and LAMMPS formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Center the structure in the unit cell
    structure = center_structure_in_cell(structure)
    
    # Wrap positions to ensure they're within the unit cell
    structure = wrap_positions(structure)
    
    # Define output files
    cif_file = os.path.join(output_dir, f"{base_name}.cif")
    pdb_file = os.path.join(output_dir, f"{base_name}.pdb")
    lammps_file = os.path.join(output_dir, f"{base_name}.data")
    
    # Save CIF file
    io.write(cif_file, structure, format='cif')
    print(f"Structure saved as CIF: {cif_file}")
    
    # Convert CIF to PDB using Open Babel
    convert_cif_to_pdb(cif_file, pdb_file)
    
    # Create LAMMPS data file
    create_lammps_data(structure, lammps_file)
    
    return cif_file

def save_results_to_txt(output_dir, cluster1, cluster2, combined_structure):
    """Save cluster information to a text file"""
    txt_file = os.path.join(output_dir, "results.txt")
    with open(txt_file, "w") as f:
        f.write(f"Results for Combined Structure:\n\n")
        
        f.write(f"Cluster 1 ({cluster1.name}):\n")
        f.write(f"Number of atoms: {len(cluster1.atoms)}\n")
        f.write(f"Center of Mass: {cluster1.com}\n\n")
        
        f.write(f"Cluster 2 ({cluster2.name}):\n")
        f.write(f"Number of atoms: {len(cluster2.atoms)}\n")
        f.write(f"Center of Mass: {cluster2.com}\n\n")
        
        f.write(f"Unit Cell Parameters:\n")
        f.write(f"a, b, c = {combined_structure.cell.diagonal()}\n")
        
        # Add final center of mass information
        positions = combined_structure.get_positions()
        masses = combined_structure.get_masses()
        final_com = np.dot(masses, positions) / np.sum(masses)
        f.write(f"\nFinal Center of Mass: {final_com}\n")
    
    print(f"Results saved to: {txt_file}")

def copy_files_to_output(output_dir, input_files):
    """Copy input files and script to output directory"""
    for file in input_files:
        if os.path.isfile(file):
            dest = os.path.join(output_dir, os.path.basename(file))
            shutil.copy2(file, dest)
            print(f"Copied {file} to {dest}")
        else:
            print(f"Warning: Could not find file {file}")

def get_rotation_input(particle_num):
    """Get rotation input from user"""
    print(f"\nRotation settings for particle {particle_num}:")
    print("Options:")
    print("1. Enter rotation angles manually")
    print("2. Use random rotation")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        print(f"Random rotation will be applied to particle {particle_num}")
        return 'random'
    else:
        print("\nEnter rotation angles in degrees:")
        print("Rotations will be applied in order: X, Y, Z")
        try:
            rot_x = float(input("Enter X-axis rotation (0-360 degrees): "))
            rot_y = float(input("Enter Y-axis rotation (0-360 degrees): "))
            rot_z = float(input("Enter Z-axis rotation (0-360 degrees): "))
            
            # Normalize angles to 0-360 range
            rot_x = rot_x % 360
            rot_y = rot_y % 360
            rot_z = rot_z % 360
            
            print(f"\nSelected rotation for particle {particle_num}:")
            print(f"X: {rot_x}°, Y: {rot_y}°, Z: {rot_z}°")
            
            return (rot_x, rot_y, rot_z)
        except ValueError:
            print("\nInvalid input. Using default rotation (0,0,0)")
            return (0.0, 0.0, 0.0)

def copy_scripts_to_output(output_dir):
    """Copy both script files to output directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_files = [
        os.path.join(script_dir, "nanoparticle_pair_builder.py"),
        os.path.join(script_dir, "nanoparticle_pair_builder_gui.py")
    ]
    
    for script_file in script_files:
        if os.path.exists(script_file):
            dest = os.path.join(output_dir, os.path.basename(script_file))
            shutil.copy2(script_file, dest)
            print(f"Copied {script_file} to {dest}")

class ProcessingError(Exception):
    """Custom error for processing failures"""
    pass

def process_with_gui(particles, output_dir, separation, cell_parameters, progress_callback=None):
    """Process particles with GUI support"""
    try:
        if len(particles) < 2:
            raise ProcessingError("At least two particles are required")
            
        clusters = []
        
        # Load and process each particle
        for i, particle_info in enumerate(particles):
            if progress_callback:
                progress_callback(f"Loading particle {i+1} from {particle_info['path']}...")
            
            cluster = load_particle(particle_info['path'])
            
            # Apply rotation
            if progress_callback:
                progress_callback(f"Applying rotation to particle {i+1}...")
                
            if particle_info['rotation'] == 'random':
                rotation = (
                    random.uniform(0, 360),
                    random.uniform(0, 360),
                    random.uniform(0, 360)
                )
            else:
                rotation = particle_info['rotation']
            
            cluster.rotate(rotation)
            clusters.append(cluster)
        
        # Combine clusters
        if progress_callback:
            progress_callback("Combining particles...")
        combined_structure = combine_clusters(clusters[0], clusters[1], separation)
        
        # Create unit cell
        if progress_callback:
            progress_callback("Creating unit cell...")
        combined_structure = create_unit_cell(combined_structure, cell_parameters)
        
        # Save results
        if progress_callback:
            progress_callback("Saving structure...")
        
        new_cif = save_structure(combined_structure, output_dir, "combined_structure")
        save_results_to_txt(output_dir, clusters[0], clusters[1], combined_structure)
        
        if progress_callback:
            progress_callback(f"All files saved to: {output_dir}")
        
        return new_cif
        
    except Exception as e:
        raise ProcessingError(f"Processing failed: {str(e)}")

def main():
    """Command line interface version"""
    try:
        # Get input files and parameters
        particle1_file = input("Enter the path to the first CIF file: ")
        particle2_file = input("Enter the path to the second CIF file: ")
        distance = float(input("Enter the distance between clusters (Å): "))
        
        # Get rotation parameters
        rotation1 = get_rotation_input(1)
        rotation2 = get_rotation_input(2)
        
        # Unit cell parameters
        print("\nUnit cell parameters:")
        a = float(input("Enter the unit cell parameter 'a' (Å): "))
        b = float(input("Enter the unit cell parameter 'b' (Å): "))
        c = float(input("Enter the unit cell parameter 'c' (Å): "))
        cell_parameters = [a, b, c]

        # Create output directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name1 = os.path.splitext(os.path.basename(particle1_file))[0]
        name2 = os.path.splitext(os.path.basename(particle2_file))[0]
        output_dir = f"{name1}_{name2}_pair_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Load particles as clusters
        cluster1 = load_particle(particle1_file)
        cluster2 = load_particle(particle2_file)

        # Apply rotations
        if rotation1 == 'random':
            rotation1 = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        if rotation2 == 'random':
            rotation2 = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))

        # Combine clusters with rotations
        combined_structure = combine_clusters(cluster1, cluster2, distance, rotation1, rotation2)
        
        # Create unit cell
        combined_structure = create_unit_cell(combined_structure, cell_parameters)
        
        # Center structure in unit cell and save
        new_cif = save_structure(combined_structure, output_dir, "combined_structure")
        
        # Save analysis results
        save_results_to_txt(output_dir, cluster1, cluster2, combined_structure)

        # Copy input files and scripts
        input_files = [particle1_file, particle2_file]
        copy_files_to_output(output_dir, input_files)
        copy_scripts_to_output(output_dir)

        print("\nProcess completed successfully!")
        print(f"Output files saved to: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
