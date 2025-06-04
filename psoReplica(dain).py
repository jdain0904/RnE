import numpy as np
import matplotlib.pyplot as plt

# === 기본 설정 ===
DOF = 4  # 4자유도
joint_labels = [f"joint_{i+1}" for i in range(DOF)]
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
target_position = np.array([0.5, 0.5])

# === 순기구학 ===
def forward_kinematics(joint_angles, link_lengths=[0.3, 0.3, 0.2, 0.1]):
    x, y, theta = 0, 0, 0
    for angle, length in zip(joint_angles, link_lengths):
        theta += angle
        x += length * np.cos(theta)
        y += length * np.sin(theta)
    return np.array([x, y])

# === 목적 함수 ===
def fitness_function(joint_angles):
    end_effector = forward_kinematics(joint_angles)
    return np.linalg.norm(end_effector - target_position)

# === Particle 정의 ===
class Particle:
    def __init__(self, label, bounds):
        self.label = label  # 관절 이름 (예: 'joint_1')
        self.position = np.array([np.random.uniform(*bounds)])
        self.velocity = np.zeros(1)
        self.best_position = self.position.copy()
        self.best_score = None

# === Hierarchical PSO 클래스 정의 ===
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
        for _ in range(self.max_iter):
            num_particles = len(next(iter(self.particle_groups.values())))

            for i in range(num_particles):
                candidate = np.array([
                    self.particle_groups[label][i].position[0]
                    for label in self.joint_labels
                ])
                score = fitness_function(candidate)

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = candidate.copy()

                for j, label in enumerate(self.joint_labels):
                    p = self.particle_groups[label][i]
                    if p.best_score is None or score < p.best_score:
                        p.best_score = score
                        p.best_position = np.array([candidate[j]])

            # PSO 업데이트
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

# === 최적화 실행 ===
hpso = HPSO(joint_labels=joint_labels, particles_per_joint=30, bounds=joint_bounds, max_iter=100)
best_angles, best_score = hpso.optimize()

print("최적 관절 각도 (radian):")
for label, angle in zip(joint_labels, best_angles):
    print(f"{label}: {angle:.4f}")
print("최종 위치 오차:", best_score)

# === 결과 시각화 ===
def plot_arm(joint_angles, target, link_lengths=[0.3, 0.3, 0.2, 0.1]):
    x, y, theta = 0, 0, 0
    X, Y = [0], [0]
    for angle, length in zip(joint_angles, link_lengths):
        theta += angle
        x += length * np.cos(theta)
        y += length * np.sin(theta)
        X.append(x)
        Y.append(y)
    plt.plot(X, Y, '-o', label='Robot Arm')
    plt.plot(target[0], target[1], 'rx', label='Target')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Hierarchical PSO - 4DOF Robot Arm')
    plt.show()

plot_arm(best_angles, target_position)
