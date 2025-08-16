#!/usr/bin/env python3
"""
Script to rotate an OBJ file by a specified angle around a specified axis.
This permanently modifies the vertex positions in the OBJ file.

Usage: python3 rotate_obj.py input.obj output.obj --axis x --angle -90
"""

import numpy as np
import argparse
import sys

def rotation_matrix_x(angle_degrees):
    """Create rotation matrix for rotation around X-axis."""
    angle = np.radians(angle_degrees)
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rotation_matrix_y(angle_degrees):
    """Create rotation matrix for rotation around Y-axis."""
    angle = np.radians(angle_degrees)
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotation_matrix_z(angle_degrees):
    """Create rotation matrix for rotation around Z-axis."""
    angle = np.radians(angle_degrees)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def rotate_obj_file(input_file, output_file, axis='x', angle=-90):
    """
    Rotate all vertices in an OBJ file.
    
    Args:
        input_file: Path to input OBJ file
        output_file: Path to output OBJ file
        axis: Rotation axis ('x', 'y', or 'z')
        angle: Rotation angle in degrees
    """
    
    # Select appropriate rotation matrix
    if axis.lower() == 'x':
        rot_matrix = rotation_matrix_x(angle)
    elif axis.lower() == 'y':
        rot_matrix = rotation_matrix_y(angle)
    elif axis.lower() == 'z':
        rot_matrix = rotation_matrix_z(angle)
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
    
    print(f"Rotating {input_file} by {angle}° around {axis.upper()}-axis...")
    print(f"Rotation matrix:\n{rot_matrix}")
    
    # Read and process the OBJ file
    output_lines = []
    vertex_count = 0
    normal_count = 0
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('v '):  # Vertex position
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertex = np.array([x, y, z])
                rotated = rot_matrix @ vertex
                
                # Preserve any additional vertex data (like color)
                extra = ' '.join(parts[4:]) if len(parts) > 4 else ''
                if extra:
                    output_lines.append(f"v {rotated[0]:.6f} {rotated[1]:.6f} {rotated[2]:.6f} {extra}\n")
                else:
                    output_lines.append(f"v {rotated[0]:.6f} {rotated[1]:.6f} {rotated[2]:.6f}\n")
                vertex_count += 1
        
        elif line.startswith('vn '):  # Vertex normal
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                normal = np.array([x, y, z])
                rotated = rot_matrix @ normal
                # Normalize the normal vector
                rotated = rotated / np.linalg.norm(rotated)
                output_lines.append(f"vn {rotated[0]:.6f} {rotated[1]:.6f} {rotated[2]:.6f}\n")
                normal_count += 1
        
        else:
            # Copy all other lines as-is (faces, texture coords, materials, etc.)
            output_lines.append(line + '\n')
    
    # Write the output file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"✓ Rotated {vertex_count} vertices and {normal_count} normals")
    print(f"✓ Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Rotate an OBJ file around a specified axis')
    parser.add_argument('input', help='Input OBJ file')
    parser.add_argument('output', help='Output OBJ file')
    parser.add_argument('--axis', default='x', choices=['x', 'y', 'z'],
                        help='Axis to rotate around (default: x)')
    parser.add_argument('--angle', type=float, default=-90,
                        help='Rotation angle in degrees (default: -90)')
    
    args = parser.parse_args()
    
    try:
        rotate_obj_file(args.input, args.output, args.axis, args.angle)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
