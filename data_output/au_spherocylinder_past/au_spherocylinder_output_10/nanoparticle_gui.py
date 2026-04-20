import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from typing import Dict, Any, Optional
import nanoparticle_builder as builder
import time
import shutil

class NanoparticleBuilderGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Nanoparticle Builder")
        self.master.geometry("800x600")
        
        # Variables
        self.cif_file = tk.StringVar()
        self.scale_x = tk.StringVar(value="30")
        self.scale_y = tk.StringVar(value="30")
        self.scale_z = tk.StringVar(value="30")
        self.selected_shape = tk.StringVar()
        self.param_entries = {}  # Store parameter entry widgets
        self.current_params = {}  # Store current parameter values
        
        # Add log file handling
        self.log_file = None
        self._initialize_log_file()
        
        self.timestamp = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding=5)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="CIF File:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.cif_file, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        # Scaling factors section
        scale_frame = ttk.LabelFrame(main_frame, text="Scaling Factors", padding=5)
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for i, (var, label) in enumerate([
            (self.scale_x, "X:"),
            (self.scale_y, "Y:"),
            (self.scale_z, "Z:")
        ]):
            ttk.Label(scale_frame, text=label).grid(row=0, column=i*2, padx=5)
            ttk.Entry(scale_frame, textvariable=var, width=10).grid(row=0, column=i*2+1, padx=5)
        
        # Shape selection section
        shape_frame = ttk.LabelFrame(main_frame, text="Shape Selection", padding=5)
        shape_frame.pack(fill=tk.X, padx=5, pady=5)
        
        shapes = list(builder.SHAPE_PARAMS.keys())
        shape_combo = ttk.Combobox(shape_frame, textvariable=self.selected_shape, values=shapes)
        shape_combo.pack(side=tk.LEFT, padx=5)
        shape_combo.bind('<<ComboboxSelected>>', self.on_shape_selected)
        
        # Parameters section
        self.params_frame = ttk.LabelFrame(main_frame, text="Shape Parameters", padding=5)
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons section
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Run", command=self.run_job).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=5)
        
        # Status section
        self.status_text = tk.Text(main_frame, height=5, width=60)
        self.status_text.pack(fill=tk.X, padx=5, pady=5)

    def _initialize_log_file(self):
        """Initialize log file for recording job information"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file = open(f"nanoparticle_builder_{timestamp}.log", "w")
            self.log_file.write("Nanoparticle Builder Job Log\n")
            self.log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self.log_file.flush()
        except Exception as e:
            print(f"Error initializing log file: {e}")

    def browse_file(self):
        """Open file browser for CIF file selection"""
        filename = filedialog.askopenfilename(
            title="Select CIF File",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if filename:
            self.cif_file.set(filename)
            self.log_status(f"Selected file: {filename}")

    def on_shape_selected(self, event=None):
        """Handle shape selection and create parameter entry fields"""
        shape = self.selected_shape.get()
        
        # Clear existing parameter entries
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()
        
        if shape in builder.SHAPE_PARAMS:
            info = builder.SHAPE_PARAMS[shape]
            row = 0
            
            # Create coordinate entries with default values
            ttk.Label(self.params_frame, text="Center:").grid(row=row, column=0, padx=5, pady=2, sticky='e')
            coord_frame = ttk.Frame(self.params_frame)
            coord_frame.grid(row=row, column=1, columnspan=2, sticky='w')
            
            self.param_entries['center_x'] = ttk.Entry(coord_frame, width=6)
            self.param_entries['center_y'] = ttk.Entry(coord_frame, width=6)
            self.param_entries['center_z'] = ttk.Entry(coord_frame, width=6)
            
            # Set default center coordinates to 40
            self.param_entries['center_x'].insert(0, "40")
            self.param_entries['center_y'].insert(0, "40")
            self.param_entries['center_z'].insert(0, "40")
            
            ttk.Label(coord_frame, text="x:").pack(side=tk.LEFT, padx=2)
            self.param_entries['center_x'].pack(side=tk.LEFT, padx=2)
            ttk.Label(coord_frame, text="y:").pack(side=tk.LEFT, padx=2)
            self.param_entries['center_y'].pack(side=tk.LEFT, padx=2)
            ttk.Label(coord_frame, text="z:").pack(side=tk.LEFT, padx=2)
            self.param_entries['center_z'].pack(side=tk.LEFT, padx=2)
            
            row += 1
            
            # Shape-specific parameter configurations
            SHAPE_CONFIGS = {
                'sphere': [('radius', 'Radius:', '20.0')],
                'cube': [('side_length', 'Side Length:', '40.0')],
                'cylinder': [
                    ('radius', 'Radius:', '20.0'),
                    ('height', 'Height:', '50.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'cone': [
                    ('radius', 'Base Radius:', '20.0'),
                    ('height', 'Height:', '40.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'cuboid': {
                    'dimensions': [
                        ('dim_x', 'x:', '30.0'),
                        ('dim_y', 'y:', '40.0'),
                        ('dim_z', 'z:', '50.0')
                    ]
                },
                'ellipsoid': {
                    'radii': [
                        ('radius_x', 'x:', '20.0'),
                        ('radius_y', 'y:', '30.0'),
                        ('radius_z', 'z:', '40.0')
                    ]
                },
                'hexagonal_prism': [
                    ('side_length', 'Side Length:', '20.0'),
                    ('height', 'Height:', '60.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'nanotube': [
                    ('inner_radius', 'Inner Radius:', '15.0'),
                    ('outer_radius', 'Outer Radius:', '20.0'),
                    ('height', 'Height:', '50.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'pyramid': [
                    ('base_length', 'Base Length:', '40.0'),
                    ('height', 'Height:', '30.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'spherical_shell': [
                    ('inner_radius', 'Inner Radius:', '15.0'),
                    ('outer_radius', 'Outer Radius:', '20.0')
                ],
                'torus': [
                    ('major_radius', 'Major Radius:', '30.0'),
                    ('minor_radius', 'Minor Radius:', '10.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'helix': [
                    ('radius', 'Radius:', '20.0'),
                    ('pitch', 'Pitch:', '10.0'),
                    ('turns', 'Turns:', '3.0'),
                    ('thickness', 'Thickness:', '5.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'square_pyramid': [
                    ('base_length', 'Base Length:', '40.0'),
                    ('height', 'Height:', '30.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'tapered_cylinder': [
                    ('base_radius', 'Base Radius:', '25.0'),
                    ('top_radius', 'Top Radius:', '15.0'),
                    ('height', 'Height:', '50.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'triangular_prism': [
                    ('side_length', 'Side Length:', '30.0'),
                    ('height', 'Height:', '50.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'octahedron': [('edge_length', 'Edge Length:', '30.0')],
                'truncated_octahedron': [
                    ('edge_length', 'Edge Length:', '30.0'),
                    ('truncation_factor', 'Truncation (0-0.5):', '0.3')
                ],
                'icosahedron': [('radius', 'Radius:', '25.0')],
                'dodecahedron': [('radius', 'Radius:', '25.0')],
                'tetrahedron': [('edge_length', 'Edge Length:', '35.0')],
                'bipyramid': [
                    ('base_width', 'Base Width:', '30.0'),
                    ('height', 'Height:', '50.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'curved_cylinder': [
                    ('radius', 'Radius:', '20.0'),
                    ('bend_radius', 'Bend Radius:', '50.0'),
                    ('angle', 'Angle (degrees):', '90.0'),
                    ('thickness', 'Thickness:', '10.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'capsule': [
                    ('radius', 'Radius:', '15.0'),
                    ('length', 'Length:', '40.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'spherocylinder': [
                    ('radius', 'Radius:', '15.0'),
                    ('cylinder_length', 'Cylinder Length:', '40.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'double_cone': [
                    ('base_radius', 'Base Radius:', '20.0'),
                    ('height', 'Height:', '60.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'star_prism': [
                    ('outer_radius', 'Outer Radius:', '25.0'),
                    ('inner_radius', 'Inner Radius:', '15.0'),
                    ('height', 'Height:', '40.0'),
                    ('points', 'Points:', '5'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'nanoshell': [
                    ('inner_radius', 'Inner Radius:', '15.0'),
                    ('outer_radius', 'Outer Radius:', '20.0'),
                    ('shell_thickness', 'Shell Thickness:', '2.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
                'nanocage': [
                    ('cage_size', 'Cage Size:', '30.0'),
                    ('wall_thickness', 'Wall Thickness:', '3.0'),
                    ('pore_size', 'Pore Size:', '8.0'),
                    ('corner_radius', 'Corner Radius:', '2.0'),
                    ('axis', 'Axis (x/y/z):', 'z')
                ],
            }

            if shape in SHAPE_CONFIGS:
                config = SHAPE_CONFIGS[shape]
                if isinstance(config, dict):
                    # Handle grouped parameters (like dimensions or radii)
                    for group_name, params in config.items():
                        group_frame = ttk.Frame(self.params_frame)
                        group_frame.grid(row=row, column=1, columnspan=2, sticky='w')
                        ttk.Label(self.params_frame, text=f"{group_name.title()}:").grid(
                            row=row, column=0, padx=5, pady=2, sticky='e')
                        
                        for param_name, label_text, default in params:
                            ttk.Label(group_frame, text=label_text).pack(side=tk.LEFT, padx=2)
                            entry = ttk.Entry(group_frame, width=8)
                            entry.insert(0, default)
                            entry.pack(side=tk.LEFT, padx=2)
                            self.param_entries[param_name] = entry
                        row += 1
                else:
                    # Handle simple parameter lists
                    for param_name, label_text, default in config:
                        ttk.Label(self.params_frame, text=label_text).grid(
                            row=row, column=0, padx=5, pady=2, sticky='e')
                        entry = ttk.Entry(self.params_frame, width=20)
                        entry.insert(0, default)
                        entry.grid(row=row, column=1, padx=5, pady=2, sticky='w')
                        self.param_entries[param_name] = entry
                        row += 1

    def load_example(self, shape: str):
        """Load example parameters for the selected shape"""
        if shape in builder.SHAPE_PARAMS:
            example = builder.SHAPE_PARAMS[shape]['example']
            try:
                # Parse example string
                params = example.strip('()').split(',')
                
                # Handle center coordinates
                center_str = params[0].strip('[]')
                center_coords = [x.strip() for x in center_str.split(',')]
                self.param_entries['center_x'].delete(0, tk.END)
                self.param_entries['center_y'].delete(0, tk.END)
                self.param_entries['center_z'].delete(0, tk.END)
                self.param_entries['center_x'].insert(0, center_coords[0])
                self.param_entries['center_y'].insert(0, center_coords[1])
                self.param_entries['center_z'].insert(0, center_coords[2])
                
                # Handle remaining parameters
                remaining_params = params[1:]
                param_keys = [key for key in self.param_entries.keys() 
                             if not key.startswith('center_')]
                
                for key, value in zip(param_keys, remaining_params):
                    entry = self.param_entries[key]
                    entry.delete(0, tk.END)
                    entry.insert(0, value.strip().strip("'\""))
                    
                self.log_status(f"Loaded example parameters for {shape}")
            except Exception as e:
                self.log_status(f"Error loading example: {str(e)}")

    def parse_and_fill_example(self, example: str):
        """Parse example string and fill parameter entries"""
        try:
            # Remove outer parentheses and split by commas
            params = example.strip('()').split(',')
            
            # Fill entries with corresponding values
            for (key, entry), value in zip(self.param_entries.items(), params):
                entry.delete(0, tk.END)
                entry.insert(0, value.strip())
                
        except Exception as e:
            self.log_status(f"Error loading example: {str(e)}")

    def run_job(self):
        """Run the nanoparticle building job"""
        try:
            if not self.validate_inputs():
                return

            # Generate timestamp at job start
            self.timestamp = builder.get_timestamp()

            # Collect parameters into a dictionary
            shape_params = self.collect_parameters()
            params = {
                'cif_file': self.cif_file.get(),
                'scaling_factors': [
                    int(self.scale_x.get()),
                    int(self.scale_y.get()),
                    int(self.scale_z.get())
                ],
                'shape': self.selected_shape.get(),
                'shape_params': shape_params
            }

            # Log job parameters
            self._log_job_parameters(params)

            self.log_status("\nProcessing structure...")
            output_dir = builder.main(params, timestamp=self.timestamp)
            
            if output_dir:
                # Verify output files
                base_name = os.path.splitext(os.path.basename(params['cif_file']))[0]
                shape_name = params['shape']
                
                expected_files = {
                    'supercell': f"{base_name}_supercell.cif",
                    'shape': f"{base_name}_{shape_name}.cif",
                    'lammps': f"{base_name}_{shape_name}_lmp.data",
                    'pdb': f"{base_name}_{shape_name}.pdb"
                }
                
                # Check each file
                for file_type, filename in expected_files.items():
                    full_path = os.path.join(output_dir, filename)
                    if (os.path.exists(full_path) and 
                            os.path.getsize(full_path) > 0):
                        self.log_status(f"Generated {file_type} file: {filename}")
                    else:
                        self.log_status(f"Error: Failed to generate {filename}")
                
                # Move log file to output directory if it exists
                if self.log_file:
                    try:
                        # Get current log file path
                        log_path = self.log_file.name
                        self.log_file.close()
                        
                        # Move to output directory
                        new_log_path = os.path.join(output_dir, os.path.basename(log_path))
                        shutil.move(log_path, new_log_path)
                        
                        # Reopen log file at new location
                        self.log_file = open(new_log_path, 'a')
                        self.log_status(f"Log file moved to: {new_log_path}")
                    except Exception as e:
                        self.log_status(f"Error moving log file: {str(e)}")
                
                self.log_status("\nJob completed!")
            else:
                self.log_status("Error: Job failed to complete")
                
        except Exception as e:
            self.log_status(f"Error running job: {str(e)}")
            messagebox.showerror("Error", str(e))

    def _log_job_parameters(self, params: Dict[str, Any]):
        """Log job parameters to file"""
        if self.log_file:
            try:
                self.log_file.write(f"\nNew Job - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write(f"Input File: {params['cif_file']}\n")
                self.log_file.write(f"Shape: {params['shape']}\n")
                self.log_file.write(f"Scaling Factors: {params['scaling_factors']}\n")
                self.log_file.write("Shape Parameters:\n")
                for key, value in params['shape_params'].items():
                    self.log_file.write(f"  {key}: {value}\n")
                self.log_file.write("\n")
                self.log_file.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}")

    def save_settings(self):
        """Save current settings to a JSON file"""
        try:
            settings = {
                'cif_file': self.cif_file.get(),
                'scaling': {
                    'x': self.scale_x.get(),
                    'y': self.scale_y.get(),
                    'z': self.scale_z.get()
                },
                'shape': self.selected_shape.get(),
                'parameters': {
                    key: entry.get()
                    for key, entry in self.param_entries.items()
                }
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=2)
                self.log_status(f"Settings saved to {filename}")
                
        except Exception as e:
            self.log_status(f"Error saving settings: {str(e)}")
            messagebox.showerror("Error", str(e))

    def show_help(self):
        """Show help information"""
        help_text = """
Nanoparticle Builder Help:

1. File Selection:
   - Click 'Browse' to select a CIF input file
   - Supports .cif format files

2. Scaling Factors:
   - Enter integer values for x, y, z scaling
   - Must be positive numbers

3. Shape Selection:
   - Choose a shape from the dropdown menu
   - Each shape has specific required parameters

4. Parameters:
   - Fill in required parameters for the selected shape
   - Click '?' next to each parameter for specific help
   - 'Use Example' loads example parameters

5. Buttons:
   - Run: Execute the particle building job
   - Save Settings: Save current configuration
   - Help: Show this help message

For more information, check the documentation.
        """
        messagebox.showinfo("Help", help_text)

    def show_param_help(self, shape: str, param: str):
        """Show help for specific parameter"""
        if shape in builder.SHAPE_PARAMS:
            help_text = builder.SHAPE_PARAMS[shape]['help']
            messagebox.showinfo(f"{shape} - {param}", help_text)

    def validate_inputs(self) -> bool:
        """Validate all inputs before running"""
        if not self.cif_file.get():
            messagebox.showerror("Error", "Please select a CIF file")
            return False
            
        try:
            for scale in [self.scale_x.get(), self.scale_y.get(), self.scale_z.get()]:
                if not scale.isdigit() or int(scale) < 1:
                    messagebox.showerror("Error", "Scaling factors must be positive integers")
                    return False
        except ValueError:
            messagebox.showerror("Error", "Invalid scaling factors")
            return False
            
        if not self.selected_shape.get():
            messagebox.showerror("Error", "Please select a shape")
            return False
            
        # Validate parameters based on shape
        for key, entry in self.param_entries.items():
            if not entry.get().strip():
                messagebox.showerror("Error", f"Please fill in {key}")
                return False
                
        return True

    def collect_parameters(self) -> Dict[str, Any]:
        """Collect and validate parameters for all shapes"""
        try:
            params = {}
            shape = self.selected_shape.get()
            
            # Collect center coordinates (common for all shapes)
            try:
                center = [
                    float(self.param_entries['center_x'].get().strip()),
                    float(self.param_entries['center_y'].get().strip()),
                    float(self.param_entries['center_z'].get().strip())
                ]
                params['center'] = center
            except ValueError:
                raise ValueError("Center coordinates must be numeric values")

            # Add shape-specific parameter collection methods
            def _collect_sphere_params(self) -> Dict[str, Any]:
                return {'radius': self._validate_numeric(self.param_entries['radius'].get(), 'Radius')}

            def _collect_cube_params(self) -> Dict[str, Any]:
                return {'side_length': self._validate_numeric(self.param_entries['side_length'].get(), 'Side Length')}

            def _collect_cylinder_params(self) -> Dict[str, Any]:
                return {
                    'radius': self._validate_numeric(self.param_entries['radius'].get(), 'Radius'),
                    'height': self._validate_numeric(self.param_entries['height'].get(), 'Height'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_cone_params(self) -> Dict[str, Any]:
                return {
                    'radius': self._validate_numeric(self.param_entries['radius'].get(), 'Base Radius'),
                    'height': self._validate_numeric(self.param_entries['height'].get(), 'Height'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_cuboid_params(self) -> Dict[str, Any]:
                return {
                    'dimensions': [
                        self._validate_numeric(self.param_entries['dim_x'].get(), 'X dimension'),
                        self._validate_numeric(self.param_entries['dim_y'].get(), 'Y dimension'),
                        self._validate_numeric(self.param_entries['dim_z'].get(), 'Z dimension')
                    ]
                }

            def _collect_ellipsoid_params(self) -> Dict[str, Any]:
                return {
                    'radii': [
                        self._validate_numeric(self.param_entries['radius_x'].get(), 'X radius'),
                        self._validate_numeric(self.param_entries['radius_y'].get(), 'Y radius'),
                        self._validate_numeric(self.param_entries['radius_z'].get(), 'Z radius')
                    ]
                }

            def _collect_hexagonal_prism_params(self) -> Dict[str, Any]:
                return {
                    'side_length': self._validate_numeric(self.param_entries['side_length'].get(), 'Side Length'),
                    'height': self._validate_numeric(self.param_entries['height'].get(), 'Height'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_nanotube_params(self) -> Dict[str, Any]:
                return {
                    'inner_radius': self._validate_numeric(self.param_entries['inner_radius'].get(), 'Inner Radius'),
                    'outer_radius': self._validate_numeric(self.param_entries['outer_radius'].get(), 'Outer Radius'),
                    'height': self._validate_numeric(self.param_entries['height'].get(), 'Height'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_pyramid_params(self) -> Dict[str, Any]:
                return {
                    'base_length': self._validate_numeric(self.param_entries['base_length'].get(), 'Base Length'),
                    'height': self._validate_numeric(self.param_entries['height'].get(), 'Height'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_capsule_params(self) -> Dict[str, Any]:
                return {
                    'radius': self._validate_numeric(
                        self.param_entries['radius'].get(), 'Radius'),
                    'length': self._validate_numeric(
                        self.param_entries['length'].get(), 'Length'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_double_cone_params(self) -> Dict[str, Any]:
                return {
                    'base_radius': self._validate_numeric(
                        self.param_entries['base_radius'].get(), 'Base Radius'),
                    'height': self._validate_numeric(
                        self.param_entries['height'].get(), 'Height'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_star_prism_params(self) -> Dict[str, Any]:
                return {
                    'outer_radius': self._validate_numeric(
                        self.param_entries['outer_radius'].get(), 'Outer Radius'),
                    'inner_radius': self._validate_numeric(
                        self.param_entries['inner_radius'].get(), 'Inner Radius'),
                    'height': self._validate_numeric(
                        self.param_entries['height'].get(), 'Height'),
                    'points': int(self._validate_numeric(
                        self.param_entries['points'].get(), 'Points', min_value=3)),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_nanoshell_params(self) -> Dict[str, Any]:
                return {
                    'inner_radius': self._validate_numeric(
                        self.param_entries['inner_radius'].get(), 'Inner Radius'),
                    'outer_radius': self._validate_numeric(
                        self.param_entries['outer_radius'].get(), 'Outer Radius'),
                    'shell_thickness': self._validate_numeric(
                        self.param_entries['shell_thickness'].get(), 'Shell Thickness'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_nanocage_params(self) -> Dict[str, Any]:
                return {
                    'cage_size': self._validate_numeric(
                        self.param_entries['cage_size'].get(), 'Cage Size'),
                    'wall_thickness': self._validate_numeric(
                        self.param_entries['wall_thickness'].get(), 'Wall Thickness'),
                    'pore_size': self._validate_numeric(
                        self.param_entries['pore_size'].get(), 'Pore Size'),
                    'corner_radius': self._validate_numeric(
                        self.param_entries['corner_radius'].get(), 'Corner Radius'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            def _collect_spherocylinder_params(self) -> Dict[str, Any]:
                return {
                    'radius': self._validate_numeric(
                        self.param_entries['radius'].get(), 'Radius'),
                    'length': self._validate_numeric(
                        self.param_entries['length'].get(), 'Length'),
                    'axis': self._validate_axis(self.param_entries['axis'].get())
                }

            # Add the methods to the instance
            for method in [_collect_sphere_params, _collect_cube_params, _collect_cylinder_params,
                          _collect_cone_params, _collect_cuboid_params, _collect_ellipsoid_params,
                          _collect_hexagonal_prism_params, _collect_nanotube_params, _collect_pyramid_params,
                          _collect_capsule_params, _collect_double_cone_params, _collect_star_prism_params,
                          _collect_nanoshell_params, _collect_nanocage_params, _collect_spherocylinder_params]:
                setattr(self.__class__, method.__name__, method)

            # Use appropriate collector based on shape
            collector = f"_collect_{shape}_params"
            if hasattr(self, collector):
                shape_params = getattr(self, collector)()
                params.update(shape_params)
            else:
                raise ValueError(f"No parameter collector defined for shape: {shape}")

            self.log_status(f"Collected parameters: {params}")
            return params

        except Exception as e:
            self.log_status(f"Error parsing parameters: {str(e)}")
            raise

    def _validate_numeric(self, value: str, param_name: str, min_value: float = 0) -> float:
        """Validate numeric parameter"""
        try:
            num = float(value.strip())
            if num <= min_value:
                raise ValueError(f"{param_name} must be greater than {min_value}")
            return num
        except ValueError:
            raise ValueError(f"{param_name} must be a valid number")

    def _validate_axis(self, axis: str) -> str:
        """Validate axis parameter"""
        axis = axis.strip().lower()
        if axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        return axis

    # Add parameter collection methods for each shape
    def _collect_sphere_params(self) -> Dict[str, Any]:
        return {
            'radius': self._validate_numeric(
                self.param_entries['radius'].get(), 
                'Radius'
            )
        }

    def _collect_torus_params(self) -> Dict[str, Any]:
        return {
            'major_radius': self._validate_numeric(
                self.param_entries['major_radius'].get(), 
                'Major Radius'
            ),
            'minor_radius': self._validate_numeric(
                self.param_entries['minor_radius'].get(), 
                'Minor Radius'
            ),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def _collect_spherical_shell_params(self) -> Dict[str, Any]:
        return {
            'inner_radius': self._validate_numeric(
                self.param_entries['inner_radius'].get(), 'Inner Radius'),
            'outer_radius': self._validate_numeric(
                self.param_entries['outer_radius'].get(), 'Outer Radius')
        }
        
    def _collect_square_pyramid_params(self) -> Dict[str, Any]:
        return {
            'base_length': self._validate_numeric(
                self.param_entries['base_length'].get(), 'Base Length'),
            'height': self._validate_numeric(
                self.param_entries['height'].get(), 'Height'),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }
        
    def _collect_truncated_octahedron_params(self) -> Dict[str, Any]:
        trunc = self._validate_numeric(
            self.param_entries['truncation_factor'].get(), 
            'Truncation Factor')
        if not 0 <= trunc <= 0.5:
            raise ValueError("Truncation factor must be between 0 and 0.5")
        return {
            'edge_length': self._validate_numeric(
                self.param_entries['edge_length'].get(), 'Edge Length'),
            'truncation_factor': trunc
        }
        
    def _collect_curved_cylinder_params(self) -> Dict[str, Any]:
        return {
            'radius': self._validate_numeric(
                self.param_entries['radius'].get(), 'Radius'),
            'bend_radius': self._validate_numeric(
                self.param_entries['bend_radius'].get(), 'Bend Radius'),
            'angle': self._validate_numeric(
                self.param_entries['angle'].get(), 'Angle'),
            'thickness': self._validate_numeric(
                self.param_entries['thickness'].get(), 'Thickness'),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def _collect_tapered_cylinder_params(self) -> Dict[str, Any]:
        return {
            'base_radius': self._validate_numeric(
                self.param_entries['base_radius'].get(), 'Base Radius'),
            'top_radius': self._validate_numeric(
                self.param_entries['top_radius'].get(), 'Top Radius'),
            'height': self._validate_numeric(
                self.param_entries['height'].get(), 'Height'),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def _collect_triangular_prism_params(self) -> Dict[str, Any]:
        return {
            'side_length': self._validate_numeric(
                self.param_entries['side_length'].get(), 'Side Length'),
            'height': self._validate_numeric(
                self.param_entries['height'].get(), 'Height'),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def _collect_octahedron_params(self) -> Dict[str, Any]:
        return {
            'edge_length': self._validate_numeric(
                self.param_entries['edge_length'].get(), 'Edge Length')
        }

    def _collect_icosahedron_params(self) -> Dict[str, Any]:
        return {
            'radius': self._validate_numeric(
                self.param_entries['radius'].get(), 'Radius')
        }

    def _collect_dodecahedron_params(self) -> Dict[str, Any]:
        return {
            'radius': self._validate_numeric(
                self.param_entries['radius'].get(), 'Radius')
        }

    def _collect_tetrahedron_params(self) -> Dict[str, Any]:
        return {
            'edge_length': self._validate_numeric(
                self.param_entries['edge_length'].get(), 'Edge Length')
        }

    def _collect_bipyramid_params(self) -> Dict[str, Any]:
        return {
            'base_width': self._validate_numeric(
                self.param_entries['base_width'].get(), 'Base Width'),
            'height': self._validate_numeric(
                self.param_entries['height'].get(), 'Height'),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def _collect_star_prism_params(self) -> Dict[str, Any]:
        return {
            'outer_radius': self._validate_numeric(
                self.param_entries['outer_radius'].get(), 'Outer Radius'),
            'inner_radius': self._validate_numeric(
                self.param_entries['inner_radius'].get(), 'Inner Radius'),
            'height': self._validate_numeric(
                self.param_entries['height'].get(), 'Height'),
            'points': int(self._validate_numeric(
                self.param_entries['points'].get(), 'Points', min_value=3)),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def _collect_helix_params(self) -> Dict[str, Any]:
        return {
            'radius': self._validate_numeric(
                self.param_entries['radius'].get(), 'Radius'),
            'pitch': self._validate_numeric(
                self.param_entries['pitch'].get(), 'Pitch'),
            'turns': self._validate_numeric(
                self.param_entries['turns'].get(), 'Turns'),
            'thickness': self._validate_numeric(
                self.param_entries['thickness'].get(), 'Thickness'),
            'axis': self._validate_axis(self.param_entries['axis'].get())
        }

    def log_status(self, message: str):
        """Log message to status text widget"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)

    def cleanup(self):
        """Cleanup resources"""
        if self.log_file:
            try:
                self.log_file.write(f"\nSession ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def main():
    root = tk.Tk()
    app = NanoparticleBuilderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
