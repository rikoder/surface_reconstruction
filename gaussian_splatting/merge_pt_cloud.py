import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def find_common_points_within_radius(pc1, pc2, radius=0.05):
    points1 = np.asarray(pc1.points)
    points2 = np.asarray(pc2.points)
    
    tree1 = cKDTree(points1)
    common_points = []
    
    for i, point in enumerate(points2):
        neighbors = tree1.query_ball_point(point, radius)
        if neighbors:
            common_points.append(i)  # Store the index of the common point in pc2
    
    return common_points

def prune_inconsistent_points(pc, common_points):
    points = np.asarray(pc.points)
    
    # Use a mask to filter out points that are too close to any common point
    mask = np.ones(len(points), dtype=bool)
    mask[common_points] = False  # Mark common points as False (to exclude them)
    
    # Create a new point cloud excluding inconsistent points
    pruned_points = points[mask]
    pruned_colors = np.asarray(pc.colors)[mask]  # Keep the colors of the pruned points
    
    pruned_pc = o3d.geometry.PointCloud()
    pruned_pc.points = o3d.utility.Vector3dVector(pruned_points)
    pruned_pc.colors = o3d.utility.Vector3dVector(pruned_colors)  # Keep the original colors
    
    return pruned_pc

def merge_point_clouds(pc1, pc2):
    points1 = np.asarray(pc1.points)
    points2 = np.asarray(pc2.points)
    
    colors1 = np.asarray(pc1.colors)
    colors2 = np.asarray(pc2.colors)

    combined_points = np.vstack((points1, points2))
    combined_colors = np.vstack((colors1, colors2))  # Combine the colors

    # Create the merged point cloud
    merged_pc = o3d.geometry.PointCloud()
    merged_pc.points = o3d.utility.Vector3dVector(combined_points)
    merged_pc.colors = o3d.utility.Vector3dVector(combined_colors)  # Assign the merged colors
    
    return merged_pc

def compute_normals(pc, radius=0.05):
    # Estimate normals using a k-nearest neighbors approach
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    # Optionally, orient the normals to be consistent
    pc.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))  # You can modify this to suit your needs
    
    return pc

def save_point_cloud(pc, file_path):
    o3d.io.write_point_cloud(file_path, pc)

# Load point clouds
point_cloud1 = load_point_cloud('/home/rikhilgupta/Desktop/Data/mipnerf360/bonsai/12_views/dense/fused.ply')
point_cloud2 = load_point_cloud('/home/rikhilgupta/Desktop/Data/mipnerf360/bonsai/24_views/dense/fused.ply')

# Find common points within a radius
common_points = find_common_points_within_radius(point_cloud1, point_cloud2, radius=0.05)

# Prune inconsistent points from point_cloud2 (remove common points)
pruned_point_cloud2 = prune_inconsistent_points(point_cloud2, common_points)

# Merge the point clouds, preserving colors
merged_point_cloud = merge_point_clouds(point_cloud1, pruned_point_cloud2)

# Compute normals for the merged point cloud
merged_point_cloud = compute_normals(merged_point_cloud)

# Save the common points
common_points_pc = o3d.geometry.PointCloud()
common_points_pc.points = o3d.utility.Vector3dVector(np.asarray(point_cloud2.points)[common_points])
save_point_cloud(common_points_pc, 'common_points.ply')

# Save the farthest points (pruned points)
farthest_points_pc = o3d.geometry.PointCloud()
farthest_points_pc.points = o3d.utility.Vector3dVector(np.asarray(pruned_point_cloud2.points))
farthest_points_pc.colors = o3d.utility.Vector3dVector(np.asarray(pruned_point_cloud2.colors))  # Keep colors
save_point_cloud(farthest_points_pc, 'farthest_points.ply')

# Save the final merged point cloud with original colors and computed normals
save_point_cloud(merged_point_cloud, 'merged_point_cloud_with_normals_bonsai.ply')

print("Point clouds processed, normals computed, and saved with original colors.")
