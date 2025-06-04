import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# [1] Initialization: 환경 및 입자 초기화
# ==========================================

DOF = 4
joint_labels = [f"joint_{i+1}" for i in range(DOF)]
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
gbest_history = []

# 로봇팔 최대 작업반경 내에서 타깃 생성
def random_reachable_target(radius=0.9):
    while True:
        point = np.random.uniform(-radius, radius, 3)
        if np.linalg.norm(point) <= radius:
            return point

target_position = random_reachable_target()

# 순기구학 (3D)
def forward_kinematics_3d(joint_angles, link_lengths=[0.3, 0.3, 0.2, 0.1]):
    coords = [(0, 0, 0)]
    x, y, z = 0, 0, 0
    theta_y, theta_z = 0, 0
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        if i % 2 == 0:
            theta_y += angle  # YZ 평면 회전
        else:
            theta_z += angle  # XZ 평면 회전
        dx = length * np.cos(theta_y) * np.cos(theta_z)
        dy = length * np.sin(theta_y)
        dz = length * np.cos(theta_y) * np.sin(theta_z)
        x += dx
        y += dy
        z += dz
        coords.append((x, y, z))
    return np.array(coords)

# 목적 함수: 종단위치와 타깃 거리
def fitness_function(joint_angles):
    end_effector = forward_kinematics_3d(joint_angles)[-1]
    return np.linalg.norm(end_effector - target_position)

# 입자 클래스
class Particle:
    def __init__(self, label, bounds):
        self.label = label
        self.position = np.array([np.random.uniform(*bounds)])
        self.velocity = np.zeros(1)
        self.best_position = self.position.copy()
        self.best_score = None

# HPSO 클래스
class HPSO:
    def __init__(self, joint_labels, particles_per_joint, bounds, max_iter):
        self.joint_labels = joint_labels
        self.bounds = bounds
        self.max_iter = max_iter
        self.dim = len(joint_labels)
        self.particle_groups = {
            label: [Particle(label, bounds[i]) for _ in range(particles_per_joint)]
            for i, label in enumerate(joint_labels)
        }
        self.gbest = np.zeros(self.dim)
        self.gbest_score = float('inf')

    def optimize(self):
        for iteration in range(self.max_iter):

            # ==========================================
            # [2] Calculation: 현재 위치 기반 평가
            # ==========================================
            num_particles = len(next(iter(self.particle_groups.values())))
            for i in range(num_particles):
                candidate = np.array([
                    self.particle_groups[label][i].position[0]
                    for label in self.joint_labels
                ])
                score = fitness_function(candidate)

                # ==========================================
                # [4] Evaluation: 개인/전역 최적 업데이트
                # ==========================================
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = candidate.copy()
                for j, label in enumerate(self.joint_labels):
                    p = self.particle_groups[label][i]
                    if p.best_score is None or score < p.best_score:
                        p.best_score = score
                        p.best_position = np.array([candidate[j]])

            gbest_history.append(self.gbest.copy())

            # ==========================================
            # [3] Position Update: 속도 및 위치 갱신
            # ==========================================
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

        return self.gbest, self.gbest_score

# ==========================================
# 최적화 수행
# ==========================================
hpso = HPSO(joint_labels=joint_labels, particles_per_joint=30, bounds=joint_bounds, max_iter=100)
best_angles, best_score = hpso.optimize()

print("최적 관절 각도 (radian):")
for label, angle in zip(joint_labels, best_angles):
    print(f"{label}: {angle:.4f}")
print("최종 위치 오차:", best_score)
print("목표 위치 (target):", target_position)

# ==========================================
# 결과 시각화 (3D 애니메이션)
# ==========================================
def animate_arm_3d(gbest_history, target, link_lengths=[0.3, 0.3, 0.2, 0.1]):
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
        coords = forward_kinematics_3d(angles, link_lengths)
        line.set_data(coords[:, 0], coords[:, 1])
        line.set_3d_properties(coords[:, 2])
        ax.set_title(f'Iteration {frame + 1}')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(gbest_history),
                                  interval=300, blit=True, repeat=False)
    plt.show()

animate_arm_3d(gbest_history, target_position)
