
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math

# ==========================================
# [1] 초기 설정
# ==========================================

DOF = 4
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
link_lengths = [0.3, 0.3, 0.2, 0.1]

np.random.seed(42)

# 로봇팔 말단(손부분) 위치계산 : 원래는 dh파라미터 이용해야 하나, 근사적 계산 위해 단순 삼각함수 이용
def forward_kinematics_3d(joint_angles):
    coords = [(0, 0, 0)]
    x, y, z = 0, 0, 0
    theta_y, theta_z = 0, 0
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

# ==========================================
# [2] PSO
# ==========================================

# PSO 파라미터
DIMENSIONS = DOF            # 관절 개수 (4개)
POPULATION = 20             # 파티클 개수
V_MAX = 0.1                 # 최대 속도
PERSONAL_C = 2.0            # 개인 계수
SOCIAL_C = 2.0              # 사회 계수
CONVERGENCE = 0.001         # 수렴 기준
MAX_ITER = 50              # 최대 반복 횟수

# 목표 위치 (전역 변수로 설정)
TARGET_POSITION = None

# 파티클 클래스
class Particle():
    def __init__(self, joint_angles, velocity):
        self.pos = joint_angles.copy()              # 현재 관절 각도
        self.velocity = velocity.copy()             # 속도
        self.best_pos = self.pos.copy()             # 개인 최적 위치
        self.pos_error = self.calculate_error()     # 현재 오차
        self.best_error = self.pos_error            # 개인 최적 오차
    
    def calculate_error(self):
        """현재 관절 각도에서 목표점까지의 거리 계산"""
        end_pos = forward_kinematics_3d(self.pos)[-1]
        return np.linalg.norm(end_pos - TARGET_POSITION)
    
    def update_position(self):
        """위치 업데이트 및 경계 처리"""
        self.pos += self.velocity
        
        # 관절 각도 경계 처리
        for i in range(DIMENSIONS):
            if self.pos[i] > joint_bounds[i][1]:
                self.pos[i] = joint_bounds[i][1]
                self.velocity[i] *= -0.5  # 경계에서 속도 반전 및 감쇠
            elif self.pos[i] < joint_bounds[i][0]:
                self.pos[i] = joint_bounds[i][0]
                self.velocity[i] *= -0.5
        
        # 오차 재계산
        self.pos_error = self.calculate_error()
        
        # 개인 베스트 업데이트
        if self.pos_error < self.best_error:
            self.best_pos = self.pos.copy()
            self.best_error = self.pos_error

# 스웜 클래스
class Swarm():
    def __init__(self, pop, v_max):
        self.particles = []                     # 파티클 리스트
        self.best_pos = None                    # 글로벌 베스트 위치
        self.best_error = math.inf              # 글로벌 베스트 오차
        
        # 파티클 초기화
        for _ in range(pop):
            # 랜덤 관절 각도 초기화
            joint_angles = np.array([
                np.random.uniform(joint_bounds[i][0], joint_bounds[i][1]) 
                for i in range(DIMENSIONS)
            ])
            # 랜덤 속도 초기화
            velocity = np.random.uniform(-v_max, v_max, DIMENSIONS)
            
            particle = Particle(joint_angles, velocity)
            self.particles.append(particle)
            
            # 글로벌 베스트 업데이트
            if particle.pos_error < self.best_error:
                self.best_pos = particle.pos.copy()
                self.best_error = particle.pos_error

def hpso_inverse_kinematics(target, pop_size=20, max_iter=200, w=0.5, c1=1.5, c2=1.5, hc=0.5, levels=3):
    gbest_history = []

    # 계층별 파티클 초기화
    particles_per_level = pop_size // levels
    hierarchical_particles = []

    for level in range(levels):
        level_particles = []
        for (low, high) in joint_bounds:
            exploration_factor = 1.0 - (level / levels) * 0.5
            joint_pos = np.random.uniform(low * exploration_factor, high * exploration_factor, particles_per_level)
            particle = {
                "position": joint_pos,
                "velocity": np.zeros(particles_per_level),
                "pbest_pos": joint_pos.copy(),
                "pbest_val": np.full(particles_per_level, np.inf),
                "parent": None,
                "children": []
            }
            level_particles.append(particle)
        hierarchical_particles.append(level_particles)

    # 계층 연결
    for level in range(levels - 1):
        for j in range(DOF):
            parent = hierarchical_particles[level][j]
            child = hierarchical_particles[level + 1][j]
            parent["children"].append(child)
            child["parent"] = parent

    def evaluate(config):
        end = forward_kinematics_3d(config)[-1]
        return np.linalg.norm(end - target)

    for iter in range(max_iter):
        best_angles = None
        best_error = np.inf

        for level in range(levels):
            for i in range(particles_per_level):
                config = np.array([joint["position"][i] for joint in hierarchical_particles[level]])
                error = evaluate(config)

                for j, joint in enumerate(hierarchical_particles[level]):
                    if error < joint["pbest_val"][i]:
                        joint["pbest_val"][i] = error
                        joint["pbest_pos"][i] = joint["position"][i]

                if error < best_error:
                    best_error = error
                    best_angles = config

        gbest_history.append(best_angles.copy())

        # 속도 및 위치 업데이트
        for level in range(levels):
            for j in range(DOF):
                joint = hierarchical_particles[level][j]
                r1, r2, r3 = np.random.rand(particles_per_level), np.random.rand(particles_per_level), np.random.rand(particles_per_level)
                pbest_term = c1 * r1 * (joint["pbest_pos"] - joint["position"])
                gbest_term = c2 * r2 * (best_angles[j] - joint["position"])

                hierarchy_term = np.zeros(particles_per_level)
                if joint["parent"] is not None:
                    parent_best = joint["parent"]["pbest_pos"]
                    hierarchy_term += hc * r3 * (parent_best - joint["position"])
                if joint["children"]:
                    children_avg = np.mean([child["pbest_pos"] for child in joint["children"]], axis=0)
                    hierarchy_term += hc * r3 * (children_avg - joint["position"]) * 0.5

                joint["velocity"] = w * joint["velocity"] + pbest_term + gbest_term + hierarchy_term
                joint["position"] += joint["velocity"]
                joint["position"] = np.clip(joint["position"], joint_bounds[j][0], joint_bounds[j][1])

    final_angles = gbest_history[-1]
    final_error = evaluate(final_angles)
    return final_angles, final_error, gbest_history
# ==========================================
# [3] 실행 및 결과 준비 + 수렴곡선 시각화
# ==========================================

target = random_reachable_target()
print(f"목표 위치: {target}")

pso_angles, pso_error, gbest_frames = hpso_inverse_kinematics(target)

errors = [np.linalg.norm(forward_kinematics_3d(angles)[-1] - target) for angles in gbest_frames]

plt.figure()
plt.plot(errors, marker='o')
plt.title('HPSO Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Error (Distance to Target)')
plt.grid(True)
plt.show()

pso_coords_seq = [forward_kinematics_3d(angles) for angles in gbest_frames]
max_frames = len(gbest_frames)

# ==========================================
# [4] 3D 애니메이션 시각화
# ==========================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_title("HPSO-based 4DOF Robot Arm Optimization")
target_point = ax.scatter(*target, color='red', label='Target')
line_pso, = ax.plot([], [], [], '-o', color='green', label='HPSO')
frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init():
    line_pso.set_data([], [])
    line_pso.set_3d_properties([])
    frame_text.set_text('')
    return line_pso, frame_text

def update(frame):
    if frame < len(pso_coords_seq):
        pso_coords = pso_coords_seq[frame]
        line_pso.set_data(pso_coords[:, 0], pso_coords[:, 1])
        line_pso.set_3d_properties(pso_coords[:, 2])
    frame_text.set_text(f"Frame {frame + 1}/{max_frames}")
    return line_pso, frame_text

ani = FuncAnimation(fig, update, frames=max_frames, init_func=init,
                    interval=100, blit=False, repeat=False)

plt.legend()
plt.show()

rel_error = pso_error / np.linalg.norm(target)
print(f"상대 오차율: {rel_error:.3%}")
print(f"최종 관절 각도: {pso_angles}")
print(f"최종 끝점 위치: {forward_kinematics_3d(pso_angles)[-1]}")
