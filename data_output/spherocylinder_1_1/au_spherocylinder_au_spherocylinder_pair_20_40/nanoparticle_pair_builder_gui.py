import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import datetime
import shutil
from nanoparticle_pair_builder import process_with_gui, ProcessingError

class ParticleBuilderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nanoparticle Pair Builder")
        self.root.geometry("800x600")
        
        self.particle_rows = []
        self.default_settings = {
            'separation': 50.0,
            'cell_x': 300.0,
            'cell_y': 300.0,
            'cell_z': 300.0
        }
        
        # Create main container
        self.create_main_container()
        
        # Add first particle row
        self.add_particle_row()

    def create_main_container(self):
        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollable frame for particles
        self.scroll_frame = ttk.Frame(self.main_frame)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # Particle section
        self.particle_frame = ttk.LabelFrame(self.scroll_frame, text="Particle Input", padding="5")
        self.particle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.particle_frame, text="Add Particle", 
                  command=self.add_particle_row).pack(pady=5)
        
        # Parameters section
        self.create_parameters_section()
        
        # Buttons section
        self.create_buttons_section()
        
        # Progress section
        self.create_progress_section()

    def create_parameters_section(self):
        param_frame = ttk.LabelFrame(self.scroll_frame, text="Parameters", padding="5")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Separation
        sep_frame = ttk.Frame(param_frame)
        sep_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sep_frame, text="Particle Separation (Å):").pack(side=tk.LEFT)
        self.separation_var = tk.DoubleVar(value=self.default_settings['separation'])
        ttk.Entry(sep_frame, textvariable=self.separation_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Cell parameters
        cell_frame = ttk.Frame(param_frame)
        cell_frame.pack(fill=tk.X, pady=2)
        ttk.Label(cell_frame, text="Cell Parameters (Å):").pack(side=tk.LEFT)
        
        self.cell_x = tk.DoubleVar(value=self.default_settings['cell_x'])
        self.cell_y = tk.DoubleVar(value=self.default_settings['cell_y'])
        self.cell_z = tk.DoubleVar(value=self.default_settings['cell_z'])
        
        ttk.Label(cell_frame, text="X:").pack(side=tk.LEFT, padx=(5,0))
        ttk.Entry(cell_frame, textvariable=self.cell_x, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(cell_frame, text="Y:").pack(side=tk.LEFT, padx=(5,0))
        ttk.Entry(cell_frame, textvariable=self.cell_y, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(cell_frame, text="Z:").pack(side=tk.LEFT, padx=(5,0))
        ttk.Entry(cell_frame, textvariable=self.cell_z, width=8).pack(side=tk.LEFT, padx=2)

    def create_buttons_section(self):
        button_frame = ttk.Frame(self.scroll_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Run", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=5)

    def create_progress_section(self):
        progress_frame = ttk.LabelFrame(self.scroll_frame, text="Progress", padding="5")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.progress_text = scrolledtext.ScrolledText(progress_frame, height=10)
        self.progress_text.pack(fill=tk.BOTH, expand=True)

    def add_particle_row(self):
        row_frame = ttk.Frame(self.particle_frame)
        row_frame.pack(fill=tk.X, pady=2)
        
        # File selection
        path_var = tk.StringVar()
        ttk.Label(row_frame, text=f"Particle {len(self.particle_rows) + 1}:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row_frame, textvariable=path_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(row_frame, text="Browse", 
                  command=lambda v=path_var: self.browse_file(v)).pack(side=tk.LEFT, padx=5)
        
        # Rotation controls
        rot_frame = ttk.Frame(row_frame)
        rot_frame.pack(side=tk.LEFT, padx=5)
        
        rot_x = tk.DoubleVar(value=0.0)
        rot_y = tk.DoubleVar(value=0.0)
        rot_z = tk.DoubleVar(value=0.0)
        random_rot = tk.BooleanVar(value=False)
        
        ttk.Label(rot_frame, text="Rot:").pack(side=tk.LEFT)
        ttk.Entry(rot_frame, textvariable=rot_x, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Entry(rot_frame, textvariable=rot_y, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Entry(rot_frame, textvariable=rot_z, width=5).pack(side=tk.LEFT, padx=1)
        
        ttk.Checkbutton(rot_frame, text="Random", 
                       variable=random_rot).pack(side=tk.LEFT, padx=5)
        
        # Number of particles
        num_particles = tk.IntVar(value=1)
        ttk.Label(row_frame, text="Count:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row_frame, textvariable=num_particles, width=5).pack(side=tk.LEFT)
        
        # Delete button
        ttk.Button(row_frame, text="X", 
                  command=lambda: self.delete_particle_row(row_frame)).pack(side=tk.LEFT, padx=5)
        
        self.particle_rows.append({
            'frame': row_frame,
            'path': path_var,
            'rot_x': rot_x,
            'rot_y': rot_y,
            'rot_z': rot_z,
            'random_rot': random_rot,
            'num_particles': num_particles
        })

    def delete_particle_row(self, row_frame):
        row_frame.destroy()
        self.particle_rows = [row for row in self.particle_rows if row['frame'] != row_frame]

    def browse_file(self, path_var):
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select CIF file",
            filetypes=(("CIF files", "*.cif"), ("all files", "*.*"))
        )
        if filename:
            path_var.set(filename)

    def run_simulation(self):
        try:
            self.progress_text.delete(1.0, tk.END)
            self.log("Starting simulation...")
            
            # Gather input data
            particles = []
            for row in self.particle_rows:
                if not row['frame'].winfo_exists():  # Skip deleted rows
                    continue
                    
                if not os.path.exists(row['path'].get()):
                    raise ValueError(f"File not found: {row['path'].get()}")
                
                rotation = (
                    row['rot_x'].get(),
                    row['rot_y'].get(),
                    row['rot_z'].get()
                ) if not row['random_rot'].get() else 'random'
                
                particles.append({
                    'path': row['path'].get(),
                    'rotation': rotation,
                    'count': row['num_particles'].get()
                })
            
            # Create output directory with both timestamp and CIF names
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            names = [os.path.splitext(os.path.basename(p['path']))[0] for p in particles]
            output_dir = f"{'_'.join(names)}_pair_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process particles
            cell_parameters = [self.cell_x.get(), self.cell_y.get(), self.cell_z.get()]
            
            result = process_with_gui(
                particles,
                output_dir,
                self.separation_var.get(),
                cell_parameters,
                self.log
            )
            
            # Copy input files and scripts
            input_files = [row['path'].get() for row in self.particle_rows 
                         if row['frame'].winfo_exists()]
            for file in input_files:
                if os.path.exists(file):
                    dest = os.path.join(output_dir, os.path.basename(file))
                    shutil.copy2(file, dest)
                    self.log(f"Copied {file} to {dest}")
            
            self.copy_scripts_to_output(output_dir)
            
            self.log(f"\nFiles saved to: {os.path.abspath(output_dir)}")
            messagebox.showinfo("Success", "Simulation completed successfully!")
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def copy_scripts_to_output(self, output_dir):
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
                self.log(f"Copied {script_file} to {dest}")

    def save_settings(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            settings = {
                'particles': [{
                    'path': row['path'].get(),
                    'rot_x': row['rot_x'].get(),
                    'rot_y': row['rot_y'].get(),
                    'rot_z': row['rot_z'].get(),
                    'random_rot': row['random_rot'].get(),
                    'num_particles': row['num_particles'].get()
                } for row in self.particle_rows if row['frame'].winfo_exists()],
                'separation': self.separation_var.get(),
                'cell_x': self.cell_x.get(),
                'cell_y': self.cell_y.get(),
                'cell_z': self.cell_z.get()
            }
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            self.log(f"Settings saved to: {filename}")

    def load_settings(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r') as f:
                settings = json.load(f)
            
            # Clear existing rows
            for row in self.particle_rows:
                if row['frame'].winfo_exists():
                    row['frame'].destroy()
            self.particle_rows.clear()
            
            # Load particle rows
            for particle in settings['particles']:
                self.add_particle_row()
                row = self.particle_rows[-1]
                row['path'].set(particle['path'])
                row['rot_x'].set(particle['rot_x'])
                row['rot_y'].set(particle['rot_y'])
                row['rot_z'].set(particle['rot_z'])
                row['random_rot'].set(particle['random_rot'])
                row['num_particles'].set(particle['num_particles'])
            
            # Load other settings
            self.separation_var.set(settings['separation'])
            self.cell_x.set(settings['cell_x'])
            self.cell_y.set(settings['cell_y'])
            self.cell_z.set(settings['cell_z'])
            
            self.log(f"Settings loaded from: {filename}")

    def show_help(self):
        help_text = """
Nanoparticle Pair Builder Help

1. Add particles using the 'Add Particle' button
2. For each particle:
   - Select CIF file using 'Browse'
   - Set rotation angles (X, Y, Z) or enable random rotation
   - Specify number of particles
3. Set separation distance and cell parameters
4. Click 'Run' to start the simulation
5. Use 'Save/Load Settings' to store configurations

Output files will be created in a timestamped directory.
"""
        messagebox.showinfo("Help", help_text)

    def log(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = ParticleBuilderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
