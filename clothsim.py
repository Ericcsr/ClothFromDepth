from numpy.core.arrayprint import dtype_is_implied
from numpy.lib.utils import _set_function_name
import taichi as ti
import open3d as o3d
import numpy as np

# TODO: Taichi Visualization need the field size to be actual particle number which limit flexibility
MAX_VERTICES = 509
MAX_FACES = 924
# If a vertex connect more than 10 verticies then, the mesh construction has some problem
MAX_NEIGHBOR = 10
MAX_BODIES = 1
ELASTICITY = 0.4

# Define Ball Types
BALL_STICK = 1
BALL_SLIDE = 2

@ti.data_oriented
class ClothSim:
    def __init__(
        self,
        init_mesh = None,
        gravity=0.2,
        stiffness = 1000,
        damping = 2.0,
        unit_length = 1.0,
        dt = 5e-4,
        n_substeps = 50):
        self.gravity = gravity
        self.stiffness = stiffness
        self.damping = damping
        self.dt = dt
        # Initialize with conservative estimation, since mesh may be updated 
        # afterward
        self.N = 0
        self.num_triangles = 0
        self.unit_length = unit_length
        self.elasticity = ELASTICITY
        self.num_substeps = n_substeps

        self.x = ti.Vector.field(3, float, MAX_VERTICES)
        self.x_rest = ti.Vector.field(3, float, MAX_VERTICES)
        self.v = ti.Vector.field(3, float, MAX_VERTICES)
        self.adj_list = ti.field(ti.int32, shape=(MAX_VERTICES, MAX_NEIGHBOR))
        self.adj_num = ti.field(ti.int32, MAX_VERTICES)
        self.triangles = ti.Vector.field(3, ti.int32, MAX_FACES)
        self.indices = ti.field(int, shape = MAX_FACES*3)

        # Here we assume all the rigid body are balls
        self.num_objects = 0
        self.balls_x = ti.Vector.field(3, float, MAX_BODIES) # Should be ti.Vector center of balls centers
        self.balls_x_rest = ti.Vector.field(3, float, MAX_BODIES)
        self.balls_x_inc = ti.Vector.field(3, float, MAX_BODIES)
        self.balls_v = ti.Vector.field(3, float, MAX_BODIES) # Should be ti.Vector velocity of balls centers
        self.balls_v_inc = ti.Vector.field(3, float, MAX_BODIES)
        self.balls_r = ti.field(float, MAX_BODIES) # Should be float radius of ball
        self.balls_type = ti.field(ti.int32, MAX_BODIES)
        self.render_initialized = False
        if init_mesh != None:
            self.load_mesh(init_mesh)

    # This function cannot be called within gradient tape
    def load_mesh(self, mesh):
        '''Load mesh from open3d
        Parameters:
        mesh : open3d triangular mesh
        Return:
        success flag : bool 
        '''
        self.cloth_mesh = mesh
        self.cloth_mesh.remove_duplicated_vertices()
        self.cloth_mesh.compute_adjacency_list()
        links = [np.array(list(v)) for v in self.cloth_mesh.adjacency_list]
        faces = np.asarray(self.cloth_mesh.triangles)
        self.num_triangles = len(faces)
        assert(self.num_triangles <= MAX_FACES)
        print(self.num_triangles)
        # Extrapolate
        faces = np.vstack([faces, np.zeros((MAX_FACES-self.num_triangles, 3))])
        self.triangles.from_numpy(faces)
        self.indices.from_numpy(faces.flatten())

        vertices = np.asarray(self.cloth_mesh.vertices)
        vertices[:,0] += 0.4
        self.N = len(vertices)
        assert(self.N <= MAX_VERTICES)
        print(self.N)
        # Extrapolate
        vertices = np.vstack([vertices, np.zeros((MAX_VERTICES - self.N, 3))])
        self.x.from_numpy(vertices)
        self.x_rest.from_numpy(vertices)
        self.v.from_numpy(np.zeros_like(vertices))
        
        links_np = np.zeros((MAX_VERTICES, MAX_NEIGHBOR))
        links_len = np.array([len(link) for link in links])
        assert(np.max(links_len) <= MAX_NEIGHBOR)
        links_len = np.hstack([links_len, np.zeros(MAX_VERTICES-self.N)])
        for i, link in enumerate(links):
            links_np[i, :len(link)] = link
        self.adj_list.from_numpy(links_np)
        self.adj_num.from_numpy(links_len)
        return True

    def initialize_renderer(self):
        self.window = ti.ui.Window("Cloth", (800, 800), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()
        self.render_initialized = True

    def load_objects(self,centers, radius, collision_type):
        '''Load rigid bodies that can interact with clothes
        Parameters:
        centers : np.ndarray(n_objects, 3)
        radius : np.ndarray(n_objects)
        collision_type : np.ndarray(n_objects)
        Return:
        success_flag : bool
        '''
        assert(len(centers) == len(radius))
        self.num_objects = len(centers)
        n_objects = len(centers)
        centers = np.vstack([centers, np.zeros((MAX_BODIES - n_objects, 3))])
        self.balls_x.from_numpy(centers)
        self.balls_x_rest.from_numpy(centers)
        self.balls_v.from_numpy(np.zeros_like(centers))
        self.balls_x_inc.from_numpy(np.zeros_like(centers))
        self.balls_v_inc.from_numpy(np.zeros_like(centers))
        radius = np.hstack([radius, np.zeros(MAX_BODIES - n_objects)])
        self.balls_r.from_numpy(radius)
        collision_type = np.hstack([collision_type, np.zeros(MAX_BODIES - n_objects)])
        self.balls_type.from_numpy(collision_type)
        return True
        
    @ti.func
    def collide_pair(self,i, j):
        imp = ti.Vector([0.0, 0.0, 0.0])
        x_inc_contrib = ti.Vector([0.0, 0.0, 0.0])
        if i!=j:
            dist = (self.balls_x[i] + self.dt * self.balls_v[i])\
                   - (self.balls_x[j] + self.dt * self.balls_v[j])
            dist_norm = dist.norm()
            mini_gap = self.balls_r[i] + self.balls_r[j]
            if dist_norm < mini_gap:
                dir = dist.normalized()
                rela_v = self.balls_v[i] - self.balls_v[j]
                projected_v = dir.dot(rela_v)
                if projected_v < 0:
                    imp = -(1 + self.elasticity) * 0.5 * projected_v * dir
                    toi = (dist_norm - mini_gap) / min(-1e-3, projected_v)
                    x_inc_contrib = min(toi - self.dt, 0) * imp

        self.balls_x_inc[i] += x_inc_contrib
        self.balls_v_inc[i] += imp

    @ti.func
    def collide(self):
        for i in range(self.num_objects):
            self.balls_v_inc[i] = ti.Vector([0.0, 0.0, 0.0])
            self.balls_x_inc[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.num_objects):
            for j in range(i):
                self.collide_pair(i, j)
        for i in range(self.num_objects):
            for j in range(i+1, self.num_objects):
                self.collide_pair(i, j)
    
    @ti.kernel
    def substep(self):
        '''
        Substep for physical object update
        
        '''
        for i in range(self.N):
            self.v[i].y -= self.gravity * self.dt
        
        for i in range(self.N):
            for neighbor in range(self.adj_num[i]):
                relative_pos = self.x[self.adj_list[i, neighbor]] - self.x[i]
                current_length = relative_pos.norm()
                rest_pos = self.x_rest[self.adj_list[i, neighbor]] - self.x_rest[i]
                rest_length = rest_pos.norm()
                self.v[i] += (self.stiffness * self.unit_length / rest_length)\
                             * relative_pos.normalized() * (current_length - rest_length) * self.dt

        for i in range(self.N):
            self.v[i] *= ti.exp(-self.damping * self.dt)
            for j in ti.static(range(self.num_objects)):
                r = self.x[i] - self.balls_x[j]
                if r.norm() <= self.balls_r[j]:
                    if self.balls_type[j] == BALL_STICK:
                        self.v[i] = ti.Vector([0.0, 0.0, 0.0])
                    else:
                        proj_v = self.v[i].dot(r.normalized())
                        norm_v = proj_v * r.normalized()
                        tang_v = self.v[i] - norm_v
                        if proj_v <=0:
                            norm_ball_v = self.balls_v[j].dot(r.normalized()) * r.normalized()
                            self.v[i] = tang_v + norm_ball_v
            self.x[i] += self.dt * self.v[i]
                
            
        # Update the position of balls need to handle collision
        # Can follow the idea of billard in difftaichi
        self.collide()
        for i in range(self.num_objects):
            self.balls_v[i] = self.balls_v[i] + self.balls_v_inc[i]
            self.balls_x[i] = self.balls_x[i] + self.dt * self.balls_v[i] + self.balls_x_inc[i]

    def step(self,action=None):
        """step by action
        action : np.ndarray (3 * n_objects)
        """
        if action == None:
            action = np.zeros(self.num_objects * 3)
        assert(len(action)//3 == self.num_objects)
        for i in range(self.num_objects):
            self.balls_v[i] = action[3*i:3*(i+1)]

        for _ in range(self.num_substeps):
            self.substep()

    def render(self):
        if not self.render_initialized:
            print("Render Not initialized")
            exit(-1)
        self.camera.position(0.5, -0.5, 2)
        self.camera.lookat(0.5, -0.5, 0)
        self.scene.set_camera(self.camera)
        self.scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
        self.scene.mesh(self.x,
                        indices = self.indices,
                        color=(0.5, 0.5, 0.5),
                        two_sided=True)
        self.scene.particles(self.balls_x, radius = 0.1, color=(0.5, 0, 0))
        self.canvas.scene(self.scene)
        self.window.show()

    @ti.kernel
    def reset(self):
        for i in range(self.N):
            self.x[i] = self.x_rest[i]
            self.v[i] = ti.Vector([0.0, 0.0, 0.0])

        for i in range(self.num_objects):
            self.balls_x[i] = self.balls_x_rest[i]
            self.balls_v[i] = ti.Vector([0.0, 0.0, 0.0])
    
    def change_all_type(self, collision_type):
        collision_type = np.hstack([collision_type, np.zeros(MAX_BODIES - self.num_objects)])
        self.balls_type.from_numpy(collision_type)

    def change_type_index(self,index, collision_type):
        assert(index < self.num_objects and (collision_type==1 or collision_type==2))
        self.balls_type[index] = collision_type
    
if __name__ == "__main__":
    ti.init(arch = ti.cpu)
    clothmesh = o3d.io.read_triangle_mesh("./test2.obj")
    sim = ClothSim(init_mesh=clothmesh)
    sim.initialize_renderer()
    centers = np.array([[0.5, -0.5, 0.0]])
    radius = np.array([0.1])
    type = np.array([BALL_SLIDE])
    sim.load_objects(centers, radius, type)
    for s in range(1000):
        sim.step()
        sim.render()

