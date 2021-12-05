import taichi as ti
import open3d as o3d
import numpy as np
import copy

# ======= Load meshes ========
cloth_mesh = o3d.io.read_triangle_mesh("./test2.obj")
cloth_mesh.remove_duplicated_vertices()
cloth_mesh.compute_adjacency_list()
links = [np.array(list(v)) for v in cloth_mesh.adjacency_list]
links_length = [len(v) for v in cloth_mesh.adjacency_list]
links_length_np = np.array(links_length)
faces = np.asarray(cloth_mesh.triangles)
vertices = np.asarray(cloth_mesh.vertices)
copied_vertices = copy.deepcopy(vertices)
vertices[:,0] += 0.4
# ============================

ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu)

N = vertices.shape[0]
gravity = 0.2
stiffness = 1000
damping = 2.0
dt = 5e-4

ball_radius = 0.1
ball_center = ti.Vector.field(3, float, (1, ))

x = ti.Vector.field(3, float, N)
v = ti.Vector.field(3, float, N)
links_field = ti.field(ti.int32, shape=(N, links_length_np.max()))
links_length = ti.field(ti.int32, N)
x_original = ti.Vector.field(3, float, N)
unit_length = 1.0

num_triangles = faces.shape[0]
# Each triangle contains indicies for its vertices
triangles = ti.Vector.field(3, dtype = ti.i32, shape=num_triangles)
# TODO: The rule of index may be problematic
indices = ti.field(int, shape=num_triangles * 3)

def initialize(vertices, faces):
    x.from_numpy(vertices)
    x_original.from_numpy(vertices)
    v.from_numpy(np.zeros_like(vertices))
    links_length.from_numpy(links_length_np)
    links_np = np.zeros((N, links_length_np.max()),dtype=np.int32)
    for i, link in enumerate(links):
        links_np[i,:len(link)] = link
    links_field.from_numpy(links_np)
    ball_center[0] = ti.Vector([0.5, -0.5, -0.0])
    triangles.from_numpy(faces)
    indices.from_numpy(faces.flatten())

# Semi-implicit integration
@ti.kernel
def step_as_triangle():
    # TODO: Right now the if an edge belongs to two triangle the force will be computed twice which is wrong
    for i in range(num_triangles):
        force_1 = ti.Vector([0.0, 0.0, 0.0])
        force_2 = ti.Vector([0.0, 0.0, 0.0])
        force_3 = ti.Vector([0.0, 0.0, 0.0])
        relative_pos_1 = x[triangles[i][1]] - x[triangles[i][0]]
        relative_pos_2 = x[triangles[i][2]] - x[triangles[i][1]]
        relative_pos_3 = x[triangles[i][0]] - x[triangles[i][2]]
        current_length_1 = relative_pos_1.norm()
        current_length_2 = relative_pos_2.norm()
        current_length_3 = relative_pos_3.norm()

        rest_pos_1 = x_original[triangles[i][1]] - x_original[triangles[i][0]]
        rest_pos_2 = x_original[triangles[i][2]] - x_original[triangles[i][1]]
        rest_pos_3 = x_original[triangles[i][0]] - x_original[triangles[i][2]]
        rest_length_1 = rest_pos_1.norm()
        rest_length_2 = rest_pos_2.norm()
        rest_length_3 = rest_pos_3.norm()
        if rest_length_1 != 0:
            force_1 += (stiffness * (unit_length/rest_length_1)) * relative_pos_1.normalized() * (
                current_length_1 - rest_length_1)
            force_2 -= (stiffness * (unit_length/rest_length_1)) * relative_pos_1.normalized() * (
                current_length_1 - rest_length_1)
        if rest_length_2 != 0:
            force_2 += (stiffness * (unit_length/rest_length_2)) * relative_pos_2.normalized() * (
                current_length_2 - rest_length_2)
            force_3 -= (stiffness * (unit_length/rest_length_2)) * relative_pos_2.normalized() * (
                current_length_2 - rest_length_2)
        if rest_length_3 != 0:
            force_3 += (stiffness * (unit_length/rest_length_3)) * relative_pos_3.normalized() * (
                current_length_3 - rest_length_3)
            force_1 -= (stiffness * (unit_length/rest_length_3)) * relative_pos_3.normalized() * (
                current_length_3 - rest_length_3)

        v[triangles[i][0]] += force_1 * dt
        v[triangles[i][1]] += force_2 * dt
        v[triangles[i][2]] += force_3 * dt

    # Add gravity
    for i in range(N):
        v[i].y -= gravity * dt
    # May be need to add friction model is required
    # May be just assume part of particles that are
    # attached to the gripper and have velocity directly controlled by gripper
    for i in range(N):
        v[i] *= ti.exp(-damping * dt)
        if (x[i] - ball_center[0]).norm() <= ball_radius:
            v[i] = ti.Vector([0.0, 0.0, 0.0])
        x[i] += dt * v[i]

@ti.kernel
def step_as_vertices():
    for i in range(N):
        v[i].y -= gravity * dt

    for i in range(N):
        # Step using adjacency list no duplication
        for neighbor in range(links_length[i]):
            relative_pos = x[links_field[i, neighbor]] - x[i]
            current_length = relative_pos.norm()
            rest_pos = x_original[links_field[i, neighbor]] - x_original[i]
            rest_length = rest_pos.norm()
            v[i] += (stiffness * unit_length/rest_length) * relative_pos.normalized() * (current_length - rest_length) * dt
    
    for i in range(N):
        v[i] *= ti.exp(-damping * dt)
        if (x[i] - ball_center[0]).norm() <= ball_radius:
            v[i] = ti.Vector([0.0, 0.0, 0.0])
        x[i] += dt * v[i]
    
initialize(vertices, faces)


window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

import time
last_t = time.time()
cnt = 0

while cnt < 1000:
    for i in range(50):
        step_as_vertices()
    
    cnt += 1
    
    camera.position(0.5, -0.5, 2)
    camera.lookat(0.5, -0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    #scene.particles(x, radius = 0.01, color=(0.5, 0.5, 0.5))
    scene.mesh(x, indices=indices, color = (0.5, 0.5, 0.5), two_sided=True)
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()
    
print(f"Frame Rate: {1000/(time.time()-last_t)}")