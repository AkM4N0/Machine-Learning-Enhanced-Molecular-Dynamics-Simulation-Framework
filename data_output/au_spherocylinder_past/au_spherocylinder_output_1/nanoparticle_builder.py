import os
import shutil
import numpy as np
from pymatgen.core import Structure
from ase.io import read, write
import time

SHAPE_PARAMS = {
    "sphere": {
        "description": "A perfect sphere defined by center and radius",
        "params": "(center: [x,y,z], radius)",
        "example": "([0,0,0], 20)",
        "help": "center: coordinates of sphere center\nradius: sphere radius in Angstroms"
    },
    "cube": {
        "description": "A cube defined by center and side length",
        "params": "(center: [x,y,z], side_length)",
        "example": "([0,0,0], 40)",
        "help": "center: coordinates of cube center\nside_length: length of cube side in Angstroms"
    },
    "cylinder": {
        "description": "A cylinder defined by base center, radius, height, and axis",
        "params": "(base_center: [x,y,z], radius, height, axis)",
        "example": "([0,0,0], 20, 50, 'z')",
        "help": "base_center: coordinates of cylinder base\nradius: cylinder radius\nheight: cylinder height\naxis: orientation axis ('x', 'y', or 'z')"
    },
    "cone": {
        "description": "A cone defined by base center, base radius, height, and axis",
        "params": "(base_center: [x,y,z], radius, height, axis)",
        "example": "([0,0,0], 25, 40, 'z')",
        "help": "base_center: coordinates of cone base\nradius: base radius\nheight: cone height\naxis: orientation axis"
    },
    "cuboid": {
        "description": "A rectangular cuboid defined by center and dimensions",
        "params": "(center: [x,y,z], dimensions: [dx,dy,dz])",
        "example": "([0,0,0], [30,40,50])",
        "help": "center: coordinates of cuboid center\ndimensions: length, width, height in Angstroms"
    },
    "ellipsoid": {
        "description": "An ellipsoid defined by center and three radii",
        "params": "(center: [x,y,z], radii: [rx,ry,rz])",
        "example": "([0,0,0], [20,30,40])",
        "help": "center: coordinates of ellipsoid center\nradii: semi-axes lengths in x,y,z directions"
    },
    "hexagonal_prism": {
        "description": "A hexagonal prism defined by base center, side length, height, and axis",
        "params": "(base_center: [x,y,z], side_length, height, axis)",
        "example": "([0,0,0], 20, 60, 'z')",
        "help": "base_center: coordinates of base center\nside_length: length of hexagon side\nheight: prism height\naxis: orientation axis"
    },
    "nanotube": {
        "description": "A nanotube defined by base center, inner/outer radii, height, and axis",
        "params": "(base_center: [x,y,z], inner_radius, outer_radius, height, axis)",
        "example": "([0,0,0], 15, 20, 50, 'z')",
        "help": "base_center: coordinates of tube base\ninner_radius: internal radius\nouter_radius: external radius\nheight: tube length\naxis: orientation axis"
    },
    "pyramid": {
        "description": "A pyramid defined by base center, base length, height, and axis",
        "params": "(base_center: [x,y,z], base_length, height, axis)",
        "example": "([0,0,0], 40, 30, 'z')",
        "help": "base_center: coordinates of pyramid base\nbase_length: length of base side\nheight: pyramid height\naxis: orientation axis"
    },
    "sphere": {
        "description": "A perfect sphere defined by center and radius",
        "params": "(center: [x,y,z], radius)",
        "example": "([0,0,0], 20)",
        "help": "center: coordinates of sphere center\nradius: sphere radius in Angstroms"
    },
    "cube": {
        "description": "A cube defined by center and side length",
        "params": "(center: [x,y,z], side_length)",
        "example": "([0,0,0], 40)",
        "help": "center: coordinates of cube center\nside_length: length of cube side in Angstroms"
    },
    "cylinder": {
        "description": "A cylinder defined by base center, radius, height, and axis",
        "params": "(base_center: [x,y,z], radius, height, axis)",
        "example": "([0,0,0], 20, 50, 'z')",
        "help": "base_center: coordinates of cylinder base\nradius: cylinder radius\nheight: cylinder height\naxis: orientation axis ('x', 'y', or 'z')"
    },
    "cone": {
        "description": "A cone defined by base center, base radius, height, and axis",
        "params": "(base_center: [x,y,z], radius, height, axis)",
        "example": "([0,0,0], 25, 40, 'z')",
        "help": "base_center: coordinates of cone base\nradius: base radius\nheight: cone height\naxis: orientation axis"
    },
    "cuboid": {
        "description": "A rectangular cuboid defined by center and dimensions",
        "params": "(center: [x,y,z], dimensions: [dx,dy,dz])",
        "example": "([0,0,0], [30,40,50])",
        "help": "center: coordinates of cuboid center\ndimensions: length, width, height in Angstroms"
    },
    "ellipsoid": {
        "description": "An ellipsoid defined by center and three radii",
        "params": "(center: [x,y,z], radii: [rx,ry,rz])",
        "example": "([0,0,0], [20,30,40])",
        "help": "center: coordinates of ellipsoid center\nradii: semi-axes lengths in x,y,z directions"
    },
    "hexagonal_prism": {
        "description": "A hexagonal prism defined by base center, side length, height, and axis",
        "params": "(base_center: [x,y,z], side_length, height, axis)",
        "example": "([0,0,0], 20, 60, 'z')",
        "help": "base_center: coordinates of base center\nside_length: length of hexagon side\nheight: prism height\naxis: orientation axis"
    },
    "nanotube": {
        "description": "A nanotube defined by base center, inner/outer radii, height, and axis",
        "params": "(base_center: [x,y,z], inner_radius, outer_radius, height, axis)",
        "example": "([0,0,0], 15, 20, 50, 'z')",
        "help": "base_center: coordinates of tube base\ninner_radius: internal radius\nouter_radius: external radius\nheight: tube length\naxis: orientation axis"
    },
    "pyramid": {
        "description": "A pyramid defined by base center, base length, height, and axis",
        "params": "(base_center: [x,y,z], base_length, height, axis)",
        "example": "([0,0,0], 40, 30, 'z')",
        "help": "base_center: coordinates of pyramid base\nbase_length: length of base side\nheight: pyramid height\naxis: orientation axis"
    },
    "spherical_shell": {
        "description": "A spherical shell defined by center and inner/outer radii",
        "params": "(center: [x,y,z], inner_radius, outer_radius)",
        "example": "([0,0,0], 15, 20)",
        "help": "center: coordinates of shell center\ninner_radius: internal radius\nouter_radius: external radius"
    },
    "square_pyramid": {
        "description": "A square pyramid defined by base center, base length, height, and axis",
        "params": "(base_center: [x,y,z], base_length, height, axis)",
        "example": "([0,0,0], 40, 30, 'z')",
        "help": "base_center: coordinates of pyramid base\nbase_length: length of base edge\nheight: pyramid height\naxis: orientation axis"
    },
    "tapered_cylinder": {
        "description": "A tapered cylinder with different top and bottom radii",
        "params": "(base_center: [x,y,z], base_radius, top_radius, height, axis)",
        "example": "([0,0,0], 25, 15, 50, 'z')",
        "help": "base_center: coordinates of cylinder base\nbase_radius: bottom radius\ntop_radius: top radius\nheight: cylinder height\naxis: orientation axis"
    },
    "triangular_prism": {
        "description": "A triangular prism defined by base center, side length, height, and axis",
        "params": "(base_center: [x,y,z], side_length, height, axis)",
        "example": "([0,0,0], 30, 50, 'z')",
        "help": "base_center: coordinates of prism base\nside_length: length of triangle side\nheight: prism height\naxis: orientation axis"
    },
    "octahedron": {
        "description": "An octahedral particle defined by center and edge length",
        "params": "(center: [x,y,z], edge_length)",
        "example": "([0,0,0], 30)",
        "help": "center: coordinates of octahedron center\nedge_length: length of octahedron edge"
    },
    "truncated_octahedron": {
        "description": "A truncated octahedron with center and edge length",
        "params": "(center: [x,y,z], edge_length, truncation_factor)",
        "example": "([0,0,0], 30, 0.3)",
        "help": "center: coordinates of center\nedge_length: original edge length\ntruncation_factor: amount to truncate (0-0.5)"
    },
    "icosahedron": {
        "description": "An icosahedral particle with 20 triangular faces",
        "params": "(center: [x,y,z], radius)",
        "example": "([0,0,0], 25)",
        "help": "center: coordinates of center\nradius: radius of circumscribed sphere"
    },
    "dodecahedron": {
        "description": "A dodecahedral particle with 12 pentagonal faces",
        "params": "(center: [x,y,z], radius)",
        "example": "([0,0,0], 25)",
        "help": "center: coordinates of center\nradius: radius of circumscribed sphere"
    },
    "tetrahedron": {
        "description": "A tetrahedral particle with 4 triangular faces",
        "params": "(center: [x,y,z], edge_length)",
        "example": "([0,0,0], 35)",
        "help": "center: coordinates of center\nedge_length: length of tetrahedron edge"
    },
    "bipyramid": {
        "description": "A bipyramidal particle with two pyramid bases",
        "params": "(center: [x,y,z], base_width, height, axis)",
        "example": "([0,0,0], 30, 50, 'z')",
        "help": "center: coordinates of center\nbase_width: width of middle section\nheight: total height\naxis: orientation axis"
    },
    "torus": {
        "description": "A toroidal particle (donut shape)",
        "params": "(center: [x,y,z], major_radius, minor_radius, axis)",
        "example": "([0,0,0], 30, 10, 'z')",
        "help": "center: coordinates of center\nmajor_radius: radius from center to torus center\nminor_radius: radius of torus tube\naxis: orientation axis"
    },
    "capsule": {
        "description": "A cylindrical particle with hemispherical caps",
        "params": "(center: [x,y,z], radius, length, axis)",
        "example": "([0,0,0], 15, 40, 'z')",
        "help": "center: coordinates of center\nradius: radius of cylinder and caps\nlength: length of cylindrical section\naxis: orientation axis"
    },
    "spherocylinder": {
        "description": "A cylindrical particle with hemispherical caps (rod-like shape)",
        "params": "(center: [x,y,z], radius, cylinder_length, axis)",
        "example": "([0,0,0], 15, 40, 'z')",
        "help": "center: coordinates of center\nradius: radius of cylinder and caps\ncylinder_length: length of cylindrical section only\naxis: orientation axis"
    },
    "double_cone": {
        "description": "Two cones joined at their bases",
        "params": "(center: [x,y,z], base_radius, height, axis)",
        "example": "([0,0,0], 20, 60, 'z')",
        "help": "center: coordinates of center\nbase_radius: radius at widest point\nheight: total height\naxis: orientation axis"
    },
    "star_prism": {
        "description": "A star-shaped prismatic particle",
        "params": "(center: [x,y,z], outer_radius, inner_radius, height, points, axis)",
        "example": "([0,0,0], 25, 15, 40, 5, 'z')",
        "help": "center: coordinates of center\nouter_radius: radius to points\ninner_radius: radius to valleys\nheight: prism height\npoints: number of star points\naxis: orientation axis"
    },
    "helix": {
        "description": "A helical or spiral particle",
        "params": "(center: [x,y,z], radius, pitch, turns, thickness, axis)",
        "example": "([0,0,0], 20, 10, 3, 5, 'z')",
        "help": "center: coordinates of center\nradius: helix radius\npitch: vertical distance per turn\nturns: number of complete turns\nthickness: thickness of helix tube\naxis: orientation axis"
    },
    "curved_cylinder": {
        "description": "A cylinder bent into an arc",
        "params": "(center: [x,y,z], radius, bend_radius, angle, thickness, axis)",
        "example": "([0,0,0], 20, 50, 90, 10, 'z')",
        "help": "center: coordinates of center\nradius: cylinder radius\nbend_radius: radius of curvature\nangle: angle of arc in degrees\nthickness: cylinder thickness\naxis: orientation axis"
    },
    "nanoshell": {
        "description": "A hollow shell with controllable thickness",
        "params": "(center: [x,y,z], inner_radius, outer_radius, shell_thickness, axis)",
        "example": "([0,0,0], 15, 20, 2, 'z')",
        "help": "center: coordinates of center\ninner_radius: internal radius\nouter_radius: external radius\nshell_thickness: thickness of shell wall\naxis: orientation axis"
    },
    "nanocage": {
        "description": "A cubic cage with pores and rounded corners",
        "params": "(center: [x,y,z], cage_size, wall_thickness, pore_size, corner_radius, axis)",
        "example": "([0,0,0], 30, 3, 8, 2, 'z')",
        "help": "center: coordinates of center\ncage_size: size of cage\nwall_thickness: thickness of walls\npore_size: size of pores\ncorner_radius: radius for rounded corners\naxis: orientation axis"
    }

}

def show_shape_help(shape: str) -> None:
    """Display help information for a specific shape"""
    if shape in SHAPE_PARAMS:
        info = SHAPE_PARAMS[shape]
        print(f"\n{shape.upper()} Parameters:")
        print(f"Description: {info['description']}")
        print(f"Parameter format: {info['params']}")
        print(f"Example: {info['example']}")
        print("Details:")
        print(info['help'])
    else:
        print(f"No help available for shape: {shape}")

def parse_parameters(param_str: str, shape: str) -> tuple:
    """Parse parameter string into appropriate format - with flexible parsing for all shapes"""
    try:
        # Support multiple input formats
        if isinstance(param_str, (tuple, list)):
            return tuple(param_str)
            
        # Convert string format to tuple
        def parse_numeric(val):
            """Parse numeric value from string"""
            try:
                return float(val.strip())
            except ValueError:
                return val.strip().strip("'\"").lower()

        def parse_array(arr_str):
            """Parse array string [x,y,z] into list of floats"""
            return [float(x.strip()) for x in arr_str.strip('[]').split(',')]

        # Clean input string
        if isinstance(param_str, str):
            param_str = param_str.strip()
            if param_str.startswith('(') and param_str.endswith(')'):
                param_str = param_str[1:-1]

            # Split parameters handling nested brackets
            parts = []
            bracket_count = 0
            current_part = ''
            
            for char in param_str:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == ',' and bracket_count == 0:
                    parts.append(current_part.strip())
                    current_part = ''
                    continue
                current_part += char
            parts.append(current_part.strip())

            # Parse each part based on type
            parsed_parts = []
            for part in parts:
                part = part.strip()
                if part.startswith('[') and part.endswith(']'):
                    # Handle coordinate arrays
                    parsed_parts.append(parse_array(part))
                elif part.replace('.', '').replace('-', '').isdigit():
                    # Handle numeric values
                    parsed_parts.append(float(part))
                else:
                    # Handle string values (like axis)
                    parsed_parts.append(part.strip().strip("'\"").lower())

            return tuple(parsed_parts)

        # Handle dictionary input
        if isinstance(param_str, dict):
            # Standardized shape parameter configurations
            SHAPE_CONFIGS = {
                'sphere': ('center', 'radius'),
                'cube': ('center', 'side_length'),
                'cylinder': ('center', 'radius', 'height', 'axis'),
                'cone': ('center', 'radius', 'height', 'axis'),
                'cuboid': ('center', 'dimensions'),
                'ellipsoid': ('center', 'radii'),
                'hexagonal_prism': ('center', 'side_length', 'height', 'axis'),
                'nanotube': ('center', 'inner_radius', 'outer_radius', 'height', 'axis'),
                'pyramid': ('center', 'base_length', 'height', 'axis'),
                'spherical_shell': ('center', 'inner_radius', 'outer_radius'),
                'square_pyramid': ('center', 'base_length', 'height', 'axis'),
                'tapered_cylinder': ('center', 'base_radius', 'top_radius', 'height', 'axis'),
                'triangular_prism': ('center', 'side_length', 'height', 'axis'),
                'octahedron': ('center', 'edge_length'),
                'truncated_octahedron': ('center', 'edge_length', 'truncation_factor'),
                'icosahedron': ('center', 'radius'),
                'dodecahedron': ('center', 'radius'),
                'tetrahedron': ('center', 'edge_length'),
                'bipyramid': ('center', 'base_width', 'height', 'axis'),
                'torus': ('center', 'major_radius', 'minor_radius', 'axis'),
                'capsule': ('center', 'radius', 'length', 'axis'),
                'double_cone': ('center', 'base_radius', 'height', 'axis'),
                'star_prism': ('center', 'outer_radius', 'inner_radius', 'height', 'points', 'axis'),
                'helix': ('center', 'radius', 'pitch', 'turns', 'thickness', 'axis'),
                'curved_cylinder': ('center', 'radius', 'bend_radius', 'angle', 'thickness', 'axis'),
                'nanoshell': ('center', 'inner_radius', 'outer_radius', 'shell_thickness', 'axis'),
                'nanocage': ('center', 'cage_size', 'wall_thickness', 'pore_size', 'corner_radius', 'axis'),
                'spherocylinder': ('center', 'radius', 'cylinder_length', 'axis')

            }

            if shape in SHAPE_CONFIGS:
                # Get required parameters for the shape
                required_params = SHAPE_CONFIGS[shape]
                parsed_values = []
                
                # Handle center coordinates consistently
                center = param_str.get('center', [0,0,0])
                if not isinstance(center, list):
                    center = parse_array(str(center))
                parsed_values.append(center)
                
                # Handle remaining parameters
                for param in required_params[1:]:
                    value = param_str.get(param)
                    if value is None:
                        raise ValueError(f"Missing required parameter: {param}")
                    if isinstance(value, (list, tuple)):
                        parsed_values.append([float(x) for x in value])
                    else:
                        parsed_values.append(parse_numeric(str(value)))
                
                return tuple(parsed_values)

        # If all parsing attempts fail, use example parameters
        print(f"\nUnable to parse parameters, using example parameters")
        return eval(SHAPE_PARAMS[shape]["example"])
            
    except Exception as e:
        print(f"\nError parsing parameters: {str(e)}")
        print(f"For {shape}, format should be: {SHAPE_PARAMS[shape]['params']}")
        print(f"Example: {SHAPE_PARAMS[shape]['example']}")
        print("\nUsing example parameters instead.")
        return eval(SHAPE_PARAMS[shape]["example"])

def get_shape_parameters(shape: str) -> tuple:
    """Get shape parameters from user with guidance"""
    print("\nEnter parameters for", shape)
    show_shape_help(shape)
    print("\nYou can:")
    print("1. Use the example parameters")
    print("2. Enter custom parameters")
    choice = input("Enter 1 or 2: ")
    
    try:
        if choice == "1":
            params_str = SHAPE_PARAMS[shape]["example"]
            print(f"Using example parameters: {params_str}")
            return eval(params_str)
        else:
            print(f"\nEnter parameters in this format: {SHAPE_PARAMS[shape]['params']}")
            print("Examples:")
            print(f"- Full format: {SHAPE_PARAMS[shape]['params']}")
            print(f"- Simplified: {SHAPE_PARAMS[shape]['example']}")
            print("- Center coordinates can be entered as: [x,y,z] or [x, y, z]")
            print("- Separate values with commas")
            params_str = input("Parameters: ")
            return parse_parameters(params_str, shape)
    except Exception as e:
        print(f"Error parsing parameters: {str(e)}")
        print("Using example parameters instead.")
        return eval(SHAPE_PARAMS[shape]["example"])

def create_directory(file_path, shape_name):
    """Create a new directory with automatic numbering, including shape name."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_dir = f"{base_name}_{shape_name}_output"
    
    # Find existing directories with the same base name
    existing_dirs = [d for d in os.listdir('')
                    if os.path.isdir(d) and d.startswith(base_dir + "_")]
    
    if not existing_dirs:
        # No existing directories, create first one with _1
        new_dir = f"{base_dir}_1"
    else:
        # Find highest existing number and increment
        numbers = []
        for dir_name in existing_dirs:
            try:
                num = int(dir_name.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue
        
        # Create new directory with next number
        next_num = max(numbers) + 1 if numbers else 1
        new_dir = f"{base_dir}_{next_num}"
    
    # Create the directory
    full_path = os.path.join(os.getcwd(), new_dir)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created output directory: {new_dir}")
    return full_path

def load_structure(file_path):
    """Load the structure from a CIF file."""
    return Structure.from_file(file_path)

def scale_structure(structure, scaling_factors):
    """Scale the structure along x, y, z dimensions."""
    structure.make_supercell(scaling_factors)
    return structure

def cut_shape(structure, shape, params):
    """Cut the structure into a specific shape."""
    new_sites = []
    for site in structure:
        x, y, z = site.coords  # Cartesian coordinates
        if shape == "cone":
            base_center, radius, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z" and 0 <= dz <= height:
                if np.sqrt(dx**2 + dy**2) <= radius * (1 - dz / height):
                    new_sites.append(site)
        elif shape == "cube":
            center, side_length = params
            half_side = side_length / 2
            if all(abs(coord - c) <= half_side for coord, c in zip([x, y, z], center)):
                new_sites.append(site)
        elif shape == "cuboid":
            center, dimensions = params
            half_dims = [dim / 2 for dim in dimensions]
            if all(abs(coord - c) <= h for coord, c, h in zip([x, y, z], center, half_dims)):
                new_sites.append(site)
        elif shape == "cylinder":
            base_center, radius, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z":
                if np.sqrt(dx**2 + dy**2) <= radius and 0 <= dz <= height:
                    new_sites.append(site)
        elif shape == "ellipsoid":
            center, radii = params
            dx, dy, dz = [x - center[0], y - center[1], z - center[2]]
            if (dx / radii[0])**2 + (dy / radii[1])**2 + (dz / radii[2])**2 <= 1:
                new_sites.append(site)
        elif shape == "hexagonal_prism":
            base_center, side_length, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z":
                if dz < 0 or dz > height:
                    continue
                if abs(dx) > side_length or abs(dy) > side_length:
                    continue
                if abs(dx) + abs(dy) > side_length:
                    continue
                new_sites.append(site)
        elif shape == "nanotube":
            base_center, inner_radius, outer_radius, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z" and 0 <= dz <= height:
                distance = np.sqrt(dx**2 + dy**2)
                if inner_radius <= distance <= outer_radius:
                    new_sites.append(site)
        elif shape == "pyramid":
            base_center, base_length, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z" and 0 <= dz <= height:
                if abs(dx) <= (base_length / 2) * (1 - dz / height) and abs(dy) <= (base_length / 2) * (1 - dz / height):
                    new_sites.append(site)
        elif shape == "sphere":
            center, radius = params
            if np.linalg.norm(np.array([x, y, z]) - np.array(center)) <= radius:
                new_sites.append(site)
        elif shape == "spherical_shell":
            center, inner_radius, outer_radius = params
            distance = np.linalg.norm(np.array([x, y, z]) - np.array(center))
            if inner_radius <= distance <= outer_radius:
                new_sites.append(site)
        elif shape == "square_pyramid":
            base_center, base_length, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z" and 0 <= dz <= height:
                if abs(dx) <= (base_length / 2) * (1 - dz / height) and abs(dy) <= (base_length / 2) * (1 - dz / height):
                    new_sites.append(site)
        elif shape == "tapered_cylinder":
            base_center, base_radius, top_radius, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z" and 0 <= dz <= height:
                radius_at_z = base_radius + (top_radius - base_radius) * (dz / height)
                if np.sqrt(dx**2 + dy**2) <= radius_at_z:
                    new_sites.append(site)
        elif shape == "triangular_prism":
            base_center, side_length, height, axis = params
            dx, dy, dz = [x - base_center[0], y - base_center[1], z - base_center[2]]
            if axis == "z" and 0 <= dz <= height:
                if abs(dx) + abs(dy) <= side_length * (1 - dz / height):
                    new_sites.append(site)
        elif shape == "octahedron":
            center, edge_length = params
            dx, dy, dz = abs(x - center[0]), abs(y - center[1]), abs(z - center[2])
            if dx + dy + dz <= edge_length:
                new_sites.append(site)
                
        elif shape == "truncated_octahedron":
            center, edge_length, trunc = params
            dx, dy, dz = abs(x - center[0]), abs(y - center[1]), abs(z - center[2])
            if (dx + dy + dz <= edge_length and
                dx <= edge_length * (1 - trunc) and
                dy <= edge_length * (1 - trunc) and
                dz <= edge_length * (1 - trunc)):
                new_sites.append(site)
                
        elif shape == "icosahedron":
            center, radius = params
            phi = (1 + np.sqrt(5)) / 2  # golden ratio
            coords = np.array([x - center[0], y - center[1], z - center[2]])
            if all(abs(np.dot(coords, vertex)) <= radius 
                  for vertex in [[0,1,phi], [0,-1,phi], [0,1,-phi], [0,-1,-phi],
                               [1,phi,0], [-1,phi,0], [1,-phi,0], [-1,-phi,0],
                               [phi,0,1], [-phi,0,1], [phi,0,-1], [-phi,0,-1]]):
                new_sites.append(site)
                
        elif shape == "dodecahedron":
            center, radius = params
            phi = (1 + np.sqrt(5)) / 2  # golden ratio
            coords = np.array([x - center[0], y - center[1], z - center[2]])
            # Check if point is inside dodecahedron using plane equations
            planes = [
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
                [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
                [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
            ]
            if all(abs(np.dot(coords, plane)) <= radius for plane in planes):
                new_sites.append(site)

        elif shape == "tetrahedron":
            center, edge_length = params
            # Transform to tetrahedron coordinates
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            h = edge_length / np.sqrt(6)
            planes = [
                [0, 0, 1],  # bottom
                [np.sqrt(3)/3, 0, -1/3],  # front
                [-np.sqrt(3)/6, 1/2, -1/3],  # right
                [-np.sqrt(3)/6, -1/2, -1/3]  # left
            ]
            if all(np.dot([dx, dy, dz], plane) <= h for plane in planes):
                new_sites.append(site)

        elif shape == "capsule":
            center, radius, length, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                xy_dist = np.sqrt(dx**2 + dy**2)
                if xy_dist <= radius:  # Within cylinder radius
                    half_length = length / 2
                    if -half_length <= dz <= half_length:  # Cylindrical section
                        new_sites.append(site)
                    elif dz < -half_length and np.sqrt(xy_dist**2 + (dz + half_length)**2) <= radius:  # Bottom cap
                        new_sites.append(site)
                    elif dz > half_length and np.sqrt(xy_dist**2 + (dz - half_length)**2) <= radius:  # Top cap
                        new_sites.append(site)

        elif shape == "torus":
            center, major_radius, minor_radius, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                xy_dist = np.sqrt(dx**2 + dy**2)
                dist_to_ring = abs(xy_dist - major_radius)
                if dist_to_ring**2 + dz**2 <= minor_radius**2:
                    new_sites.append(site)

        elif shape == "helix":
            center, radius, pitch, turns, thickness, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                # Parameter equations for helix
                height = pitch * turns
                if 0 <= dz <= height:
                    t = (dz / height) * 2 * np.pi * turns
                    helix_x = radius * np.cos(t)
                    helix_y = radius * np.sin(t)
                    dist_to_center = np.sqrt((dx - helix_x)**2 + (dy - helix_y)**2)
                    if dist_to_center <= thickness:
                        new_sites.append(site)

        elif shape == "star_prism":
            center, outer_radius, inner_radius, height, points, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z' and 0 <= dz <= height:
                angle = np.arctan2(dy, dx)
                r = np.sqrt(dx**2 + dy**2)
                # Check if point is within star pattern
                theta = angle * points / (2 * np.pi)
                theta = theta - np.floor(theta)  # Normalize to [0,1]
                r_max = inner_radius + (outer_radius - inner_radius) * abs(2 * theta - 1)
                if r <= r_max:
                    new_sites.append(site)

        elif shape == "double_cone":
            center, base_radius, height, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                xy_dist = np.sqrt(dx**2 + dy**2)
                half_height = height / 2
                if -half_height <= dz <= half_height:
                    radius_at_z = base_radius * (1 - abs(dz) / half_height)
                    if xy_dist <= radius_at_z:
                        new_sites.append(site)

        elif shape == "curved_cylinder":
            center, radius, bend_radius, angle_deg, thickness, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                angle = np.radians(angle_deg)
                # Convert to polar coordinates
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                # Check if point is within the curved cylinder
                arc_length = bend_radius * angle
                if 0 <= dz <= thickness:
                    local_r = abs(r - bend_radius)
                    if local_r <= radius and theta <= angle:
                        new_sites.append(site)

        elif shape == "bipyramid":
            center, base_width, height, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                xy_dist = np.sqrt(dx**2 + dy**2)
                half_height = height / 2
                half_width = base_width / 2
                if -half_height <= dz <= half_height:
                    max_radius = half_width * (1 - abs(dz) / half_height)
                    if xy_dist <= max_radius:
                        new_sites.append(site)

        elif shape == "nanoshell":
            center, inner_radius, outer_radius, shell_thickness, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if inner_radius <= r <= outer_radius:
                # Create shell walls
                if abs(r - inner_radius) <= shell_thickness or abs(r - outer_radius) <= shell_thickness:
                    new_sites.append(site)

        elif shape == "nanocage":
            center, cage_size, wall_thickness, pore_size, corner_radius, axis = params
            dx, dy, dz = abs(x - center[0]), abs(y - center[1]), abs(z - center[2])
            half_size = cage_size / 2
            
            # Check if point is within cage walls
            in_wall_x = dx >= half_size - wall_thickness and dx <= half_size
            in_wall_y = dy >= half_size - wall_thickness and dy <= half_size
            in_wall_z = dz >= half_size - wall_thickness and dz <= half_size
            
            # Check if point is in corner region
            in_corner = (dx > half_size - corner_radius and 
                        dy > half_size - corner_radius and 
                        dz > half_size - corner_radius)
            
            # Check if point is in pore
            not_in_pore = not (dx < pore_size/2 and dy < pore_size/2) and \
                        not (dy < pore_size/2 and dz < pore_size/2) and \
                        not (dx < pore_size/2 and dz < pore_size/2)
            
            if ((in_wall_x or in_wall_y or in_wall_z) and not_in_pore) or \
            (in_corner and np.sqrt(dx**2 + dy**2 + dz**2) <= half_size):
                new_sites.append(site)
                
        elif shape == "spherocylinder":
            center, radius, cylinder_length, axis = params
            dx, dy, dz = x - center[0], y - center[1], z - center[2]
            if axis == 'z':
                xy_dist = np.sqrt(dx**2 + dy**2)
                if xy_dist <= radius:  # Within cylinder radius
                    half_length = cylinder_length / 2
                    if -half_length <= dz <= half_length:  # Cylindrical section
                        new_sites.append(site)
                    elif dz < -half_length and np.sqrt(xy_dist**2 + (dz + half_length)**2) <= radius:  # Bottom cap
                        new_sites.append(site)
                    elif dz > half_length and np.sqrt(xy_dist**2 + (dz - half_length)**2) <= radius:  # Top cap
                        new_sites.append(site)

        else:
            raise ValueError(f"Shape '{shape}' is not implemented.")
    if not new_sites:
        raise ValueError("The resulting structure is empty. No atoms fit within the specified shape.")
    return Structure(
        structure.lattice,
        [site.species for site in new_sites],
        [site.coords for site in new_sites],
        coords_are_cartesian=True
    )

def convert_to_lammps_and_pdb(cif_file, output_dir):
    """
    Convert the generated CIF file to LAMMPS and PDB formats using ASE with optimizations
    for large systems.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file names
    lammps_file = os.path.join(output_dir, os.path.basename(cif_file).replace(".cif", ".data"))
    pdb_file = os.path.join(output_dir, os.path.basename(cif_file).replace(".cif", ".pdb"))

    try:
        # Read the CIF file with optimized settings
        print("Reading structure file (this may take a while for large systems)...")
        structure = read(cif_file, format='cif', parallel=True)  # Enable parallel reading
        n_atoms = len(structure)
        print(f"Successfully read {n_atoms} atoms from CIF file")

        # For very large systems, write LAMMPS format directly
        if n_atoms > 100000:
            print("Large system detected, using direct LAMMPS writer...")
            write(lammps_file, structure, format="lammps-data", 
                  direct=True,        # Use direct writing mode
                  wrap=True,          # Wrap atoms into cell
                  velocities=False,   # Skip velocities for faster writing
                  units="metal")      # Use metal units
        else:
            # Regular LAMMPS writing for smaller systems
            write(lammps_file, structure, format="lammps-data")
        print(f"LAMMPS file saved to: {lammps_file}")

        # Optional PDB conversion for visualization
        if n_atoms <= 1000000:  # Skip PDB for extremely large systems
            write(pdb_file, structure, format="proteindatabank")
            print(f"PDB file saved to: {pdb_file}")
        else:
            print("System too large for PDB conversion, skipping...")

    except MemoryError:
        print("Memory error: System too large to process at once.")
        print("Try reducing the system size or using a different conversion method.")
    except Exception as e:
        print(f"Error converting CIF file: {e}")

def get_timestamp():
    """Generate a consistent timestamp format for both GUI and command line"""
    return time.strftime("%Y%m%d_%H%M%S")

def create_log_file(output_dir, timestamp=None):
    """Create and initialize log file with given or new timestamp"""
    if timestamp is None:
        timestamp = get_timestamp()
    
    log_path = os.path.join(output_dir, f"nanoparticle_builder_{timestamp}.log")
    log_file = open(log_path, 'w')
    log_file.write(f"Nanoparticle Builder Job Log\n")
    log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return log_file, log_path

def main(params=None, timestamp=None):
    """Main function that handles both GUI and command-line usage"""
    try:
        if params is None:
            # Command line mode
            file_path = input("Enter the path to the CIF file: ")
            scaling_factors = [
                int(input("Enter scaling factor along x: ")),
                int(input("Enter scaling factor along y: ")),
                int(input("Enter scaling factor along z: ")),
            ]

            print("\nAvailable shapes:")
            for i, shape in enumerate(SHAPE_PARAMS.keys(), 1):
                print(f"{i}. {shape}")
                print(f"   {SHAPE_PARAMS[shape]['description']}")

            shape_option = int(input("\nEnter the number corresponding to the shape: "))
            shape = list(SHAPE_PARAMS.keys())[shape_option - 1]
            shape_params = get_shape_parameters(shape)

            params = {
                'cif_file': file_path,
                'scaling_factors': scaling_factors,
                'shape': shape,
                'shape_params': shape_params
            }

        # Process the structure
        output_dir = create_directory(params['cif_file'], params['shape'])
        print(f"Working directory: {output_dir}")
        
        # Initialize log file with provided or new timestamp
        log_file, log_path = create_log_file(output_dir, timestamp)
        
        try:
            # Log job parameters
            log_file.write("Job Parameters:\n")
            log_file.write(f"Input File: {params['cif_file']}\n")
            log_file.write(f"Shape: {params['shape']}\n")
            log_file.write(f"Scaling Factors: {params['scaling_factors']}\n")
            log_file.write(f"Shape Parameters: {params['shape_params']}\n\n")
            
            # Copy input files with absolute paths
            shutil.copy(params['cif_file'], output_dir)
            log_file.write(f"Copied input file: {params['cif_file']}\n")
            
            # Copy the script files
            script_files = [
                os.path.abspath(__file__),  # Main script
                os.path.join(os.path.dirname(__file__), 'nanoparticle_gui.py')  # GUI script
            ]
            
            for script in script_files:
                if os.path.exists(script):
                    shutil.copy(script, output_dir)
                    log_file.write(f"Copied script: {script}\n")

            # Load and process structure
            structure = load_structure(params['cif_file'])
            supercell_structure = scale_structure(structure, params['scaling_factors'])
            
            # Save files with absolute paths
            supercell_file = os.path.join(output_dir, 
                                         f"{os.path.splitext(os.path.basename(params['cif_file']))[0]}_supercell.cif")
            supercell_structure.to(filename=supercell_file)
            print(f"Supercell structure saved to {supercell_file}")
            log_file.write(f"Supercell structure saved to {supercell_file}\n")

            # Parse parameters appropriately
            shape_params = parse_parameters(params['shape_params'], params['shape'])
            
            # Cut shape
            try:
                shaped_structure = cut_shape(supercell_structure, params['shape'], shape_params)
            except ValueError as e:
                print(e)
                log_file.write(f"Error: {e}\n")
                return

            # Save shaped structure with absolute paths
            shaped_file = os.path.join(output_dir, 
                                      f"{os.path.splitext(os.path.basename(params['cif_file']))[0]}_{params['shape']}.cif")
            shaped_structure.to(filename=shaped_file)
            print(f"{params['shape'].capitalize()} structure saved to {shaped_file}")
            log_file.write(f"{params['shape'].capitalize()} structure saved to {shaped_file}\n")

            # Convert formats and log results
            convert_to_lammps_and_pdb(shaped_file, output_dir)
            log_file.write(f"Converted to LAMMPS and PDB formats\n")
            
            # Log completion
            log_file.write(f"\nJob completed successfully: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Log file saved to: {log_path}")
            
            return output_dir
            
        except Exception as e:
            if log_file:
                log_file.write(f"\nError occurred: {str(e)}\n")
            raise
        
        finally:
            if log_file:
                log_file.close()
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        if log_file:
            try:
                log_file.write(f"\nFatal error: {str(e)}\n")
                log_file.close()
            except:
                pass
        raise

if __name__ == "__main__":
    main()