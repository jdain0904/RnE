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
# [2] HPSO (Hierarchical PSO)
# ==========================================

# HPSO 파라미터
DIMENSIONS = DOF            # 관절 개수 (4개)
POPULATION = 20             # 파티클 개수
V_MAX = 0.1                 # 최대 속도
PERSONAL_C = 2.0            # 개인 계수
SOCIAL_C = 2.0              # 사회 계수
HIERARCHY_C = 1.5           # 계층 계수 (새로 추가)
CONVERGENCE = 0.001         # 수렴 기준
MAX_ITER = 200              # 최대 반복 횟수
HIERARCHY_LEVELS = 3        # 계층 레벨 수

# 목표 위치 (전역 변수로 설정)
TARGET_POSITION = None

# HPSO 파티클 클래스
class HierarchicalParticle():
    def __init__(self, joint_angles, velocity, level=0):
        self.pos = joint_angles.copy()              # 현재 관절 각도
        self.velocity = velocity.copy()             # 속도
        self.level = level                          # 계층 레벨
        self.best_pos = self.pos.copy()             # 개인 최적 위치
        self.pos_error = self.calculate_error()     # 현재 오차
        self.best_error = self.pos_error            # 개인 최적 오차
        self.parent = None                          # 상위 계층 파티클
        self.children = []                          # 하위 계층 파티클들
    
    def calculate_error(self):
        """현재 관절 각도에서 목표점까지의 거리 계산"""
        end_pos = forward_kinematics_3d(self.pos)[-1]
        return np.linalg.norm(end_pos - TARGET_POSITION)
    
    def update_position(self, hierarchy_influence=None):
        """위치 업데이트 및 경계 처리 (계층적 영향 포함)"""
        # 계층적 영향이 있다면 속도에 반영
        if hierarchy_influence is not None:
            self.velocity += hierarchy_influence
        
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
    
    def add_child(self, child_particle):
        """하위 계층 파티클 추가"""
        self.children.append(child_particle)
        child_particle.parent = self

# HPSO 스웜 클래스
class HierarchicalSwarm():
    def __init__(self, pop, v_max, hierarchy_levels):
        self.particles = []                     # 파티클 리스트
        self.hierarchy_levels = hierarchy_levels # 계층 레벨 수
        self.level_particles = [[] for _ in range(hierarchy_levels)]  # 레벨별 파티클 분류
        self.best_pos = None                    # 글로벌 베스트 위치
        self.best_error = math.inf              # 글로벌 베스트 오차
        
        # 계층별 파티클 수 계산
        particles_per_level = pop // hierarchy_levels
        remaining_particles = pop % hierarchy_levels
        
        # 각 계층별로 파티클 생성
        for level in range(hierarchy_levels):
            level_pop = particles_per_level + (1 if level < remaining_particles else 0)
            
            for _ in range(level_pop):
                # 계층별로 다른 탐색 범위 설정 (상위 계층일수록 넓은 탐색)
                exploration_factor = 1.0 - (level / hierarchy_levels) * 0.5
                
                # 랜덤 관절 각도 초기화
                joint_angles = np.array([
                    np.random.uniform(
                        joint_bounds[i][0] * exploration_factor, 
                        joint_bounds[i][1] * exploration_factor
                    ) for i in range(DIMENSIONS)
                ])
                
                # 랜덤 속도 초기화 (계층별로 다른 속도 범위)
                velocity_range = v_max * (0.5 + 0.5 * exploration_factor)
                velocity = np.random.uniform(-velocity_range, velocity_range, DIMENSIONS)
                
                particle = HierarchicalParticle(joint_angles, velocity, level)
                self.particles.append(particle)
                self.level_particles[level].append(particle)
                
                # 글로벌 베스트 업데이트
                if particle.pos_error < self.best_error:
                    self.best_pos = particle.pos.copy()
                    self.best_error = particle.pos_error
        
        # 계층 구조 설정 (부모-자식 관계)
        self._setup_hierarchy()
    
    def _setup_hierarchy(self):
        """계층 구조 설정"""
        for level in range(self.hierarchy_levels - 1):
            parent_particles = self.level_particles[level]
            child_particles = self.level_particles[level + 1]
            
            # 각 상위 계층 파티클에 하위 계층 파티클들을 균등하게 할당
            children_per_parent = len(child_particles) // len(parent_particles)
            remaining_children = len(child_particles) % len(parent_particles)
            
            child_idx = 0
            for i, parent in enumerate(parent_particles):
                num_children = children_per_parent + (1 if i < remaining_children else 0)
                for _ in range(num_children):
                    if child_idx < len(child_particles):
                        parent.add_child(child_particles[child_idx])
                        child_idx += 1

def hpso_inverse_kinematics(target):
    """HPSO를 이용한 역운동학 해결"""
    global TARGET_POSITION
    TARGET_POSITION = target
    
    # 스웜 초기화
    swarm = HierarchicalSwarm(POPULATION, V_MAX, HIERARCHY_LEVELS)
    
    # 관성 가중치 초기화
    inertia_weight = 0.5 + (np.random.rand() / 2)
    
    # 최적화 과정 기록
    gbest_frames = []
    
    curr_iter = 0
    while curr_iter < MAX_ITER:
        
        # 각 계층별로 업데이트 (상위 계층부터)
        for level in range(HIERARCHY_LEVELS):
            level_particles = swarm.level_particles[level]
            
            for particle in level_particles:
                
                # 각 차원별로 속도 업데이트
                for i in range(DIMENSIONS):
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    r3 = np.random.uniform(0, 1)
                    
                    # 기본 PSO 속도 업데이트 공식
                    personal_coefficient = PERSONAL_C * r1 * (particle.best_pos[i] - particle.pos[i])
                    social_coefficient = SOCIAL_C * r2 * (swarm.best_pos[i] - particle.pos[i])
                    
                    # 계층적 영향 추가
                    hierarchy_coefficient = 0
                    if particle.parent is not None:
                        # 부모 파티클의 영향
                        hierarchy_coefficient += HIERARCHY_C * r3 * (particle.parent.best_pos[i] - particle.pos[i])
                    
                    # 자식 파티클들의 영향 (평균)
                    if particle.children:
                        children_avg = np.mean([child.best_pos[i] for child in particle.children])
                        hierarchy_coefficient += HIERARCHY_C * r3 * (children_avg - particle.pos[i]) * 0.5
                    
                    new_velocity = (inertia_weight * particle.velocity[i] + 
                                  personal_coefficient + social_coefficient + hierarchy_coefficient)
                    
                    # 속도 제한
                    if new_velocity > V_MAX:
                        particle.velocity[i] = V_MAX
                    elif new_velocity < -V_MAX:
                        particle.velocity[i] = -V_MAX
                    else:
                        particle.velocity[i] = new_velocity
                
                # 위치 업데이트
                particle.update_position()
                
                # 글로벌 베스트 업데이트
                if particle.pos_error < swarm.best_error:
                    swarm.best_pos = particle.pos.copy()
                    swarm.best_error = particle.pos_error
        
        # 현재 최적해 기록
        gbest_frames.append(swarm.best_pos.copy())
        
        # 수렴 확인
        if swarm.best_error < CONVERGENCE:
            print(f"HPSO가 {curr_iter + 1}번째 반복에서 수렴했습니다.")
            break
            
        curr_iter += 1
        
        # 관성 가중치 점진적 감소
        inertia_weight = 0.9 * inertia_weight
    
    print(f"HPSO 최적화 완료: {curr_iter + 1}회 반복, 최종 오차: {swarm.best_error:.6f}")
    
    return swarm.best_pos, swarm.best_error, gbest_frames

# ==========================================
# [3] 실행 및 결과 준비 + 수렴곡선 시각화
# ==========================================

target = random_reachable_target()
print(f"목표 위치: {target}")

hpso_angles, hpso_error, gbest_frames = hpso_inverse_kinematics(target)

errors = [np.linalg.norm(forward_kinematics_3d(angles)[-1] - target) for angles in gbest_frames]

plt.figure()
plt.plot(errors, marker='o')
plt.title('HPSO Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Error (Distance to Target)')
plt.grid(True)
plt.show()

hpso_coords_seq = [forward_kinematics_3d(angles) for angles in gbest_frames]
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
line_hpso, = ax.plot([], [], [], '-o', color='blue', label='HPSO')
frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init():
    line_hpso.set_data([], [])
    line_hpso.set_3d_properties([])
    frame_text.set_text('')
    return line_hpso, frame_text

def update(frame):
    if frame < len(hpso_coords_seq):
        hpso_coords = hpso_coords_seq[frame]
        line_hpso.set_data(hpso_coords[:, 0], hpso_coords[:, 1])
        line_hpso.set_3d_properties(hpso_coords[:, 2])
    frame_text.set_text(f"Frame {frame + 1}/{max_frames}")
    return line_hpso, frame_text

ani = FuncAnimation(fig, update, frames=max_frames, init_func=init,
                    interval=100, blit=False, repeat=False)

plt.legend()
plt.show()

rel_error = hpso_error / np.linalg.norm(target)
print(f"상대 오차율: {rel_error:.3%}")
print(f"최종 관절 각도: {hpso_angles}")
print(f"최종 끝점 위치: {forward_kinematics_3d(hpso_angles)[-1]}")

# 계층별 성능 분석 (추가 정보)
print(f"사용된 계층 레벨 수: {HIERARCHY_LEVELS}")
print(f"총 파티클 수: {POPULATION}")
print(f"계층별 파티클 분포: {[len(level) for level in [[] for _ in range(HIERARCHY_LEVELS)]]}")
