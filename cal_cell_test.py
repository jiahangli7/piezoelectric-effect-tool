import numpy as np

# define mass
M_A = 207  # atom A mass
M_B = 48   # atom B mass
M_O = 16   # atom O mass

M_alpha = M_A + M_B + M_O
M_beta = 2 * M_O
M_uc = M_alpha + M_beta

def read_poscar(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # Extract lattice vectors
    lattice_vectors = np.array([
        list(map(float, content[2].split())),
        list(map(float, content[3].split())),
        list(map(float, content[4].split()))
    ])
    
    # Extract the elements and their counts
    elements = content[5].split()
    counts = list(map(int, content[6].split()))
    
    # Find the start of the atomic coordinates
    coord_type = content[7].strip()
    start_index = 8
    
    # Extract atomic coordinates (assume they are in fractional coordinates)
    total_atoms = sum(counts)
    atomic_coords = np.array([list(map(float, content[start_index + i].split()[:3])) for i in range(total_atoms)])
    
    # Extract coordinates for Sr, Ti, and O atoms
    sr_fractional_coords = atomic_coords[:counts[0]]
    ti_fractional_coords = atomic_coords[counts[0]:counts[0] + counts[1]]
    o_fractional_coords = atomic_coords[counts[0] + counts[1]:]
    
    return lattice_vectors, sr_fractional_coords, ti_fractional_coords, o_fractional_coords

def fractional_to_cartesian(lattice_vectors, fractional_coords):
    # Convert fractional coordinates to Cartesian coordinates
    return np.dot(fractional_coords, lattice_vectors)

def find_nearest_atoms_by_x(coords_origin, coords_now):
    matched_indices = []
    for i in range(len(coords_origin)):
        min_distance = float('inf')
        min_j = -1
        for j in range(len(coords_now)):
            if j not in matched_indices:
                # Compare x-coordinates only
                distance = abs(coords_origin[i][0] - coords_now[j][0])
                if distance < min_distance:
                    min_distance = distance
                    min_j = j
        matched_indices.append(min_j)
    return matched_indices


def find_nearest_atoms_by_x_and_y(coords_origin, coords_now, y_lattice_length):
    matched_indices = []
    for i in range(len(coords_origin)):
        min_distance = float('inf')
        min_j = -1
        
        for j in range(len(coords_now)):
            if j in matched_indices:
                continue
            
            # Check if the x-coordinate is within the ±0.5 Å range
            if abs(coords_origin[i][0] - coords_now[j][0]) > 1.0:
                continue
            
            # Further filter by y-coordinate
            if (coords_origin[i][1] < y_lattice_length / 2 and coords_now[j][1] >= y_lattice_length / 2) or \
               (coords_origin[i][1] >= y_lattice_length / 2 and coords_now[j][1] < y_lattice_length / 2):
                continue
            
            # Calculate the x-coordinate distance
            distance = abs(coords_origin[i][0] - coords_now[j][0])
            
            if distance < min_distance:
                min_distance = distance
                min_j = j
        
        if min_j != -1:
            matched_indices.append(min_j)
        else:
            matched_indices.append(-1)  # If no match is found, mark it as -1
    
    return matched_indices

def calculate_y_displacement_with_periodicity(coords_origin, coords_now, matched_indices, y_lattice_length):
    y_displacements = []
    matched_displacements = []

    # 计算匹配到的O原子的y方向位移，并记录匹配到的原子信息
    for i, j in enumerate(matched_indices):
        if j != -1:  # 有匹配
            displacement = coords_now[j][1] - coords_origin[i][1]
            # 应用周期性边界条件
            if abs(displacement) > y_lattice_length / 2:
                displacement = displacement - np.sign(displacement) * y_lattice_length
            y_displacements.append(displacement)
            matched_displacements.append((coords_origin[i][0], displacement))  # 保存x坐标和位移
        else:
            y_displacements.append(None)  # 暂时标记为None

    # 处理没有匹配到的O原子，继承x方向上最近的匹配O原子的y方向位移
    for i in range(len(y_displacements)):
        if y_displacements[i] is None:
            origin_x = coords_origin[i][0]
            # 找到x方向上最近的匹配到的O原子
            closest_displacement = min(matched_displacements, key=lambda x: abs(x[0] - origin_x))[1]
            y_displacements[i] = closest_displacement

    return np.array(y_displacements)

#def calculate_y_displacement_with_periodicity(coords_origin, coords_now, matched_indices, y_lattice_length):
    #y_displacements = []
    #for i, j in enumerate(matched_indices):
        #displacement = coords_now[j][1] - coords_origin[i][1]
        
        # Apply periodic boundary condition
        #if abs(displacement) > y_lattice_length / 2:
            #displacement = displacement - np.sign(displacement) * y_lattice_length
        
        #y_displacements.append(displacement)
    
    #return np.array(y_displacements)

def find_alpha_planes(sr_coords, ti_coords, o_coords, y_displacements_sr, y_displacements_ti, y_displacements_o):
    alpha_planes = []
    used_o_indices = set()

    for i in range(len(ti_coords)):
        # find Sr and O near Ti
        sr_diffs = np.abs(sr_coords[:, 0] - ti_coords[i, 0])
        o_diffs = np.abs(o_coords[:, 0] - ti_coords[i, 0])
        
        nearest_sr_index = np.argmin(sr_diffs)
        nearest_o_index = np.argmin(o_diffs)

        if nearest_o_index in used_o_indices:
            continue  # jump if this O atom is already used in α plane
        
        used_o_indices.add(nearest_o_index)

        x1_alpha = (sr_coords[nearest_sr_index][0] * M_A + ti_coords[i][0] * M_B + o_coords[nearest_o_index][0] * M_O) / (M_A + M_B + M_O)
        u3_alpha = (y_displacements_sr[nearest_sr_index] * M_A + y_displacements_ti[i] * M_B + y_displacements_o[nearest_o_index] * M_O) / (M_A + M_B + M_O)
        #print("Sr_x Ti_x O_x:",sr_coords[nearest_sr_index][0], ti_coords[i][0], o_coords[nearest_o_index][0])
        #print("Sr Ti O:",y_displacements_sr[nearest_sr_index], y_displacements_ti[i], y_displacements_o[nearest_o_index])

        alpha_planes.append({
            'x1': x1_alpha,
            'u3': u3_alpha
        })
    
    return alpha_planes, used_o_indices

def find_beta_planes(o_coords, y_displacements_o, used_o_indices):
    beta_planes = []
    available_o_indices = [i for i in range(len(o_coords)) if i not in used_o_indices]
    available_o_coords = o_coords[available_o_indices]
    
    used_o_in_beta = set()

    for i, o_coord in enumerate(available_o_coords):
        if i in used_o_in_beta:
            continue  # jump if this O atom is already used in β plane
        
        # find the nearest another O atom
        min_distance = float('inf')
        min_j = -1
        for j in range(len(available_o_coords)):
            if i != j and j not in used_o_in_beta:
                distance = abs(o_coord[0] - available_o_coords[j][0])
                if distance < min_distance:
                    min_distance = distance
                    min_j = j
        
        if min_j != -1:
            # calculate β plane's x1 and u3
            x1_beta = (o_coord[0] + available_o_coords[min_j][0]) / 2
            u3_beta = (y_displacements_o[available_o_indices[i]] + y_displacements_o[available_o_indices[min_j]]) / 2
            print("O1_x and O2_x:", o_coord[0], available_o_coords[min_j][0])
            #print("O1 and O2:", y_displacements_o[available_o_indices[i]], y_displacements_o[available_o_indices[min_j]])

            beta_planes.append({
                'x1': x1_beta,
                'u3': u3_beta
            })
            used_o_in_beta.add(i)
            used_o_in_beta.add(min_j)
    
    return beta_planes


def find_nearest_planes(planes, target_x): 
    nearest_smaller_plane = None
    nearest_larger_plane = None

    for plane in planes:
        x1 = plane['x1']
        if x1 < target_x:
            if nearest_smaller_plane is None or x1 > nearest_smaller_plane['x1']:
                nearest_smaller_plane = plane
        elif x1 > target_x:
            if nearest_larger_plane is None or x1 < nearest_larger_plane['x1']:
                nearest_larger_plane = plane

    return nearest_smaller_plane, nearest_larger_plane

def calculate_cell_positions_and_displacements(alpha_planes, beta_planes):
    cell_results = []

    # calculate α cell elements
    for plane in alpha_planes:
        nearest_smaller_beta, nearest_larger_beta = find_nearest_planes(beta_planes, plane['x1'])

        if nearest_smaller_beta and nearest_larger_beta:
            x1_alpha_cell = (
                (plane['x1'] * M_alpha) +
                (((nearest_larger_beta['x1'] - plane['x1']) * nearest_smaller_beta['x1'] +
                (plane['x1'] - nearest_smaller_beta['x1']) * nearest_larger_beta['x1']
            ) / (nearest_larger_beta['x1'] - nearest_smaller_beta['x1'])) * M_beta) / M_uc

            u3_alpha_cell = (
                (plane['u3'] * M_alpha) +
                (((nearest_larger_beta['x1'] - plane['x1']) * nearest_smaller_beta['u3'] +
                (plane['x1'] - nearest_smaller_beta['x1']) * nearest_larger_beta['u3']
            ) / (nearest_larger_beta['x1'] - nearest_smaller_beta['x1'])) * M_beta) / M_uc

            cell_results.append({'x1': x1_alpha_cell, 'u3': u3_alpha_cell, 'type': 'alpha'})

    # calculate β cell elements
    for plane in beta_planes:
        nearest_smaller_alpha, nearest_larger_alpha = find_nearest_planes(alpha_planes, plane['x1'])

        if nearest_smaller_alpha and nearest_larger_alpha:
            x1_beta_cell = (
                (plane['x1'] * M_beta) +
                (((nearest_larger_alpha['x1'] - plane['x1']) * nearest_smaller_alpha['x1'] +
                (plane['x1'] - nearest_smaller_alpha['x1']) * nearest_larger_alpha['x1']
            ) / (nearest_larger_alpha['x1'] - nearest_smaller_alpha['x1'])) * M_alpha) / M_uc

            u3_beta_cell = (
                (plane['u3'] * M_beta) +
                (((nearest_larger_alpha['x1'] - plane['x1']) * nearest_smaller_alpha['u3'] +
                (plane['x1'] - nearest_smaller_alpha['x1']) * nearest_larger_alpha['u3']
            ) / (nearest_larger_alpha['x1'] - nearest_smaller_alpha['x1'])) * M_alpha) / M_uc

            cell_results.append({'x1': x1_beta_cell, 'u3': u3_beta_cell, 'type': 'beta'})

    # sort by x1 values
    cell_results.sort(key=lambda x: x['x1'])

    return cell_results

def calculate_shear_strain(cell_results):
    shear_strains = []

    # calculate α cell's strain
    for i, cell in enumerate(cell_results):
        if cell['type'] == 'alpha':
            if i > 0 and i < len(cell_results) - 1:
                prev_beta = cell_results[i - 1]
                next_beta = cell_results[i + 1]
                if prev_beta['type'] == 'beta' and next_beta['type'] == 'beta':
                    e5_alpha = (next_beta['u3'] - prev_beta['u3']) / (next_beta['x1'] - prev_beta['x1'])
                    shear_strains.append({'x1': cell['x1'], 'e5': e5_alpha})

    # calculate β cell's strain
    for i, cell in enumerate(cell_results):
        if cell['type'] == 'beta':
            if i > 0 and i < len(cell_results) - 1:
                prev_alpha = cell_results[i - 1]
                next_alpha = cell_results[i + 1]
                if prev_alpha['type'] == 'alpha' and next_alpha['type'] == 'alpha':
                    e5_beta = (next_alpha['u3'] - prev_alpha['u3']) / (next_alpha['x1'] - prev_alpha['x1'])
                    shear_strains.append({'x1': cell['x1'], 'e5': e5_beta})

    # sort by x1 values
    shear_strains.sort(key=lambda x: x['x1'])

    return shear_strains

# Now running the main process with the provided POSCAR files

file_path_origin = './POSCAR_origin.vasp'
file_path_now = './POSCAR_now.vasp'

lattice_vectors_origin, sr_fractional_coords_origin, ti_fractional_coords_origin, o_fractional_coords_origin = read_poscar(file_path_origin)
lattice_vectors_now, sr_fractional_coords_now, ti_fractional_coords_now, o_fractional_coords_now = read_poscar(file_path_now)

# Convert fractional coordinates to Cartesian coordinates
sr_cartesian_coords_origin = fractional_to_cartesian(lattice_vectors_origin, sr_fractional_coords_origin)
ti_cartesian_coords_origin = fractional_to_cartesian(lattice_vectors_origin, ti_fractional_coords_origin)
o_cartesian_coords_origin = fractional_to_cartesian(lattice_vectors_origin, o_fractional_coords_origin)

sr_cartesian_coords_now = fractional_to_cartesian(lattice_vectors_now, sr_fractional_coords_now)
ti_cartesian_coords_now = fractional_to_cartesian(lattice_vectors_now, ti_fractional_coords_now)
o_cartesian_coords_now = fractional_to_cartesian(lattice_vectors_now, o_fractional_coords_now)

# Find nearest matching atoms by x-coordinate and calculate y displacements
matched_indices_sr = find_nearest_atoms_by_x(sr_cartesian_coords_origin, sr_cartesian_coords_now)
matched_indices_ti = find_nearest_atoms_by_x(ti_cartesian_coords_origin, ti_cartesian_coords_now)

y_lattice_length = lattice_vectors_origin[1][1]
matched_indices_o = find_nearest_atoms_by_x_and_y(o_cartesian_coords_origin, o_cartesian_coords_now, y_lattice_length)
#y_displacements_o = calculate_y_displacement_with_periodicity(o_cartesian_coords_origin, o_cartesian_coords_now, matched_indices_o, lattice_vectors_origin[1][1])
#matched_indices_o = find_nearest_atoms_by_x(o_cartesian_coords_origin, o_cartesian_coords_now)
#y_lattice_length = lattice_vectors_origin[1][1]

y_displacements_sr = calculate_y_displacement_with_periodicity(sr_cartesian_coords_origin, sr_cartesian_coords_now, matched_indices_sr, y_lattice_length)
y_displacements_ti = calculate_y_displacement_with_periodicity(ti_cartesian_coords_origin, ti_cartesian_coords_now, matched_indices_ti, y_lattice_length)
y_displacements_o = calculate_y_displacement_with_periodicity(o_cartesian_coords_origin, o_cartesian_coords_now, matched_indices_o, y_lattice_length)

sr_data = [(sr_cartesian_coords_origin[i][0], y_displacements_sr[i]) for i in range(len(y_displacements_sr))]
ti_data = [(ti_cartesian_coords_origin[i][0], y_displacements_ti[i]) for i in range(len(y_displacements_ti))]
o_data = [(o_cartesian_coords_origin[i][0], y_displacements_o[i]) for i in range(len(y_displacements_o))]

sr_data.sort(key=lambda x: x[0])
ti_data.sort(key=lambda x: x[0])
o_data.sort(key=lambda x: x[0])

with open('Sr_displacements.txt', 'w', encoding='utf-8') as f_sr:
    for x, y_disp in sr_data:
        f_sr.write(f"{x:.6f} {y_disp:.6f}\n")

with open('Ti_displacements.txt', 'w', encoding='utf-8') as f_ti:
    for x, y_disp in ti_data:
        f_ti.write(f"{x:.6f} {y_disp:.6f}\n")

with open('O_displacements.txt', 'w', encoding='utf-8') as f_o:
    for x, y_disp in o_data:
        f_o.write(f"{x:.6f} {y_disp:.6f}\n")

print("Results have been written to Sr_displacements.txt, Ti_displacements.txt, and O_displacements.txt")


# Find α planes and used O indices
alpha_planes, used_o_indices = find_alpha_planes(sr_cartesian_coords_origin, ti_cartesian_coords_origin, o_cartesian_coords_origin, y_displacements_sr, y_displacements_ti, y_displacements_o)

# Find β planes
beta_planes = find_beta_planes(o_cartesian_coords_origin, y_displacements_o, used_o_indices)

# Combine α and β planes
all_planes = [(f'α{i+1}', plane) for i, plane in enumerate(alpha_planes)]
all_planes += [(f'β{i+1}', plane) for i, plane in enumerate(beta_planes)]

all_planes.sort(key=lambda plane: plane[1]['x1'])

output_plane_path = './sorted_planes.txt'
with open(output_plane_path, 'w', encoding='utf-8') as f:
    for label, plane in all_planes:
        f.write(f"{label}: {plane['x1']:.6f} {plane['u3']:.6f}\n")

# Calculate cell positions and displacements
cell_results = calculate_cell_positions_and_displacements(alpha_planes, beta_planes)

# Output results
output_cell_path = './alpha_beta_cells_positions.txt'
with open(output_cell_path, 'w', encoding='utf-8') as f:
    for cell in cell_results:
        f.write(f"{cell['x1']:.6f} {cell['u3']:.6f}\n")

output_cell_path

shear_strains = calculate_shear_strain(cell_results)

output_strain_path = './shear_strains.txt'
with open(output_strain_path, 'w', encoding='utf-8') as f:
    for strain in shear_strains:
        f.write(f"{strain['x1']:.6f} {strain['e5']:.6f}\n")

print(f"Shear strains have been written to {output_strain_path}")