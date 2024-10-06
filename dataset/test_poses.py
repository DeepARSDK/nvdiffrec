import trimesh
import pandas as pd
import nvdiffrast.torch as dr
import torch
import numpy as np
import imageio


# Load mesh from an OBJ file
def load_mesh(obj_file_path):
    mesh = trimesh.load(obj_file_path)
    vertices = mesh.geometry["Mesh"].vertices
    faces = mesh.geometry["Mesh"].faces
    return vertices, faces


# Load camera poses from a CSV file
def load_camera_poses(csv_file_path):
    df = pd.read_csv(csv_file_path)
    poses = []

    for index, row in df.iterrows():
        translation = np.array([row["TranslationX"], row["TranslationY"], row["TranslationZ"]], dtype=np.float32)
        rotation = np.array(
            [
                [row["Rotation00"], row["Rotation01"], row["Rotation02"]],
                [row["Rotation10"], row["Rotation11"], row["Rotation12"]],
                [row["Rotation20"], row["Rotation21"], row["Rotation22"]],
            ],
            dtype=np.float32,
        )
        poses.append((translation, rotation))

    return poses


# Create view matrix from translation and rotation
def create_view_matrix(translation, rotation):
    view_matrix = torch.eye(4)
    view_matrix[:3, :3] = torch.tensor(rotation).t()
    view_matrix[:3, 3] = -torch.matmul(torch.tensor(rotation).t(), torch.tensor(translation))
    return view_matrix


# Create perspective projection matrix
def create_perspective_matrix(fov, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov) / 2)
    proj_matrix = torch.tensor(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
    )
    return proj_matrix


# Render function
def render(vertices, faces, view_matrix, proj_matrix, image_size=(256, 256)):
    ctx = dr.RasterizeGLContext()

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int32)

    vertices_homogeneous = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim=1)
    mvp_matrix = torch.matmul(proj_matrix, view_matrix)
    vertices_screen = torch.matmul(mvp_matrix, vertices_homogeneous.t()).t()

    vertices_screen = vertices_screen.contiguous().cuda()
    faces = faces.contiguous().cuda()
    rast_out, _ = dr.rasterize(ctx, vertices_screen.unsqueeze(0), faces, image_size)
    color = torch.ones(vertices_screen.shape[0], 3).contiguous().cuda()
    interpolated, _ = dr.interpolate(color.unsqueeze(0), rast_out, faces)

    image = dr.antialias(interpolated, rast_out, vertices_screen.unsqueeze(0), faces)
    return image[0].cpu().numpy()


# Main function to test the rendering and camera pose
def main():
    glb_file_path = "../data/flexi/model.glb"
    csv_file_path = "../data/flexi/camera_positions.csv"
    output_image_path = 'rendered_image.png'

    vertices, faces = load_mesh(glb_file_path)
    camera_poses = load_camera_poses(csv_file_path)

    # Using the first camera pose for demonstration
    translation, rotation = camera_poses[0]
    view_matrix = create_view_matrix(translation, rotation)
    print(view_matrix)
    proj_matrix = create_perspective_matrix(fov=45.0, aspect=1.0, near=0.1, far=100.0)

    image = render(vertices, faces, view_matrix, proj_matrix)
    image = (image * 255).astype(np.uint8)

    # Save the rendered image
    imageio.imwrite(output_image_path, image)
    print(f"Rendered image saved to {output_image_path}")


if __name__ == "__main__":
    main()
