import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# [1] 환경 설정
# ==============================

DOF = 4
joint_labels = [f"joint_{i+1}" for i in range(DOF)]
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
link_lengths = [0.3, 0.3, 0.2, 0.1]

np.random.seed(42)

def forward_kinematics_3d(joint_angles):
    x, y, z = 0, 0, 0
    theta_y, theta_z = 0, 0
    coords = [(x, y, z)]
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        if i % 2 == 0:
            theta_y += angle
        else:
            theta_z += angle
        dx = length * np.cos(theta_y) * np.cos(theta_z)
        dy = length * np.sin(theta_y)
        dz = length * np.cos(theta_y) * np.sin(theta_z)
        x += dx
        y += dy
        z += dz
        coords.append((x, y, z))
    return np.array(coords)

def random_reachable_target(radius=0.9, z_min=-0.1, z_max=0.5):
    while True:
        point = np.random.uniform(low=[0.0, -radius, z_min], high=[radius, radius, z_max])
        if np.linalg.norm(point) <= radius:
            return point

def fitness_function(joint_angles, target):
    end_effector = forward_kinematics_3d(joint_angles)[-1]
    return np.linalg.norm(end_effector - target)

# ==============================
# [2] HPSO + Sliding Window 구조
# ==============================

class Particle:
    def __init__(self, label, bounds):
        self.label = label
        self.position = np.array([np.random.uniform(*bounds)])
        self.velocity = np.random.uniform(-0.1, 0.1, size=1)
        self.best_position = self.position.copy()
        self.best_score = None

class HierarchicalSlidingPSO:
    def __init__(self, joint_labels, particles_per_joint, bounds, max_iter, window_size=10, sliding_step=5, time_horizon=200):
        self.joint_labels = joint_labels
        self.bounds = bounds
        self.max_iter = max_iter
        self.dim = len(joint_labels)
        self.time_horizon = time_horizon
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.particles_per_joint = particles_per_joint

        self.particle_groups = {
            label: [Particle(label, bounds[i]) for _ in range(particles_per_joint)]
            for i, label in enumerate(joint_labels)
        }
        self.gbest = np.zeros(self.dim)
        self.gbest_score = float('inf')

    def optimize(self, target):
        gbest_history = []
        prev_best_by_joint = {}

        for t_start in range(0, self.time_horizon, self.sliding_step):
            t_end = t_start + self.window_size

            if prev_best_by_joint:
                for j, label in enumerate(self.joint_labels):
                    for i in range(min(5, self.particles_per_joint)):
                        p = self.particle_groups[label][i]
                        p.position = prev_best_by_joint[label].copy()
                        p.best_position = prev_best_by_joint[label].copy()
                        p.best_score = None
                        p.velocity = np.random.uniform(-0.05, 0.05, size=1)

            for iteration in range(self.max_iter):
                for i in range(self.particles_per_joint):
                    candidate = np.array([
                        self.particle_groups[label][i].position[0]
                        for label in self.joint_labels
                    ])
                    score = fitness_function(candidate, target)

                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest = candidate.copy()

                    for j, label in enumerate(self.joint_labels):
                        p = self.particle_groups[label][i]
                        if p.best_score is None or score < p.best_score:
                            p.best_score = score
                            p.best_position = np.array([candidate[j]])

                gbest_history.append(self.gbest.copy())

                w, c1, c2 = 0.5, 1.5, 1.5
                for j, label in enumerate(self.joint_labels):
                    for p in self.particle_groups[label]:
                        r1, r2 = np.random.rand(), np.random.rand()
                        p.velocity = (
                            w * p.velocity
                            + c1 * r1 * (p.best_position - p.position)
                            + c2 * r2 * (self.gbest[j] - p.position)
                        )
                        p.position += p.velocity
                        low, high = self.bounds[j]
                        p.position = np.clip(p.position, low, high)

            prev_best_by_joint = {
                label: min(
                    self.particle_groups[label], key=lambda p: p.best_score if p.best_score is not None else float('inf')).best_position.copy()
                for j, label in enumerate(self.joint_labels)
            }

        return self.gbest, self.gbest_score, gbest_history

# ==============================
# [3] 실행 및 시각화
# ==============================

target = random_reachable_target()
hpso_sw = HierarchicalSlidingPSO(joint_labels, particles_per_joint=30, bounds=joint_bounds, max_iter=100)
best_angles, best_error, gbest_history = hpso_sw.optimize(target)

print("최적 관절 각도 (rad):", best_angles)
print("최종 오차:", best_error)
print("목표 위치:", target)

errors = [fitness_function(angles, target) for angles in gbest_history]
plt.figure()
plt.plot(range(1, 201), errors[:200])
plt.title("HPSO + Sliding Window Convergence")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.grid(True)
plt.show()

def animate_arm_3d(gbest_history, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.grid(True)
    line, = ax.plot([], [], [], '-o', lw=2)
    ax.scatter(target[0], target[1], target[2], color='red', label='Target')
    ax.legend()

    def update(frame):
        angles = gbest_history[frame]
        coords = forward_kinematics_3d(angles)
        line.set_data(coords[:, 0], coords[:, 1])
        line.set_3d_properties(coords[:, 2])
        ax.set_title(f"Frame {frame + 1}/200 (1 frame = 100ms)")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True, repeat=False)
    plt.show()

animate_arm_3d(gbest_history, target)
