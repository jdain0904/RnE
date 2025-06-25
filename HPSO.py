import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DOF = 4
link_lengths = [0.3, 0.3, 0.2, 0.1]
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
np.random.seed(42)

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

def hierarchical_pso(target, pop_size=20, max_iter=50, w=0.5, c1=1.5, c2=1.5):
    """
    계층화된 PSO (Hierarchical PSO)
    - Level 1 (상위): 전체 관절 각도 벡터 최적화
    - Level 2 (하위): 각 관절별 세밀한 조정
    """
    
    # === Level 1: 상위 레벨 PSO (Master PSO) ===
    # 전체 관절 각도 벡터를 최적화하는 메인 스웜
    master_swarm = []
    for _ in range(pop_size):
        position = np.array([np.random.uniform(*joint_bounds[i]) for i in range(DOF)])
        velocity = np.zeros(DOF)
        particle = {
            'position': position.copy(),
            'velocity': velocity.copy(),
            'pbest_position': position.copy(),
            'pbest_fitness': np.inf,
            'sub_swarms': []  # 각 입자가 자신의 서브 스웜을 가짐
        }
        master_swarm.append(particle)
    
    # 전역 최적해
    gbest_position = master_swarm[0]['position'].copy()
    gbest_fitness = np.inf
    
    # 최적화 과정 기록
    optimization_history = []
    
    for iteration in range(max_iter):
        # === Level 1 평가 및 Level 2 서브 PSO 실행 ===
        for master_particle in master_swarm:
            
            # Level 2: 서브 PSO - 각 관절별 세밀한 최적화
            refined_position = level2_sub_pso(
                master_particle['position'], target, 
                sub_pop_size=8, sub_max_iter=10
            )
            
            # 세밀하게 조정된 위치로 적합도 계산
            end_effector = forward_kinematics_3d(refined_position)[-1]
            fitness = np.linalg.norm(end_effector - target)
            
            # 개인 최적해 업데이트
            if fitness < master_particle['pbest_fitness']:
                master_particle['pbest_fitness'] = fitness
                master_particle['pbest_position'] = refined_position.copy()
            
            # 전역 최적해 업데이트
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = refined_position.copy()
            
            # 최적화 과정 기록
            optimization_history.append(refined_position.copy())
        
        # === Level 1: Master PSO 업데이트 ===
        for master_particle in master_swarm:
            r1, r2 = np.random.rand(2)
            
            # 속도 업데이트
            cognitive = c1 * r1 * (master_particle['pbest_position'] - master_particle['position'])
            social = c2 * r2 * (gbest_position - master_particle['position'])
            master_particle['velocity'] = w * master_particle['velocity'] + cognitive + social
            
            # 위치 업데이트
            master_particle['position'] += master_particle['velocity']
            
            # 경계 조건 적용
            for i in range(DOF):
                master_particle['position'][i] = np.clip(
                    master_particle['position'][i], *joint_bounds[i]
                )
        
        # 수렴 확인
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best fitness = {gbest_fitness:.8f}")
        
        # 조기 종료 조건
        if gbest_fitness < 1e-6:
            print(f"목표 정확도 달성! (Iteration {iteration})")
            break
    
    return gbest_position, gbest_fitness, optimization_history

def level2_sub_pso(master_position, target, sub_pop_size=8, sub_max_iter=10):
    """
    Level 2: 서브 PSO - 각 관절별 세밀한 최적화
    마스터 위치를 기반으로 각 관절을 독립적으로 미세조정
    """
    refined_position = master_position.copy()
    
    # 각 관절에 대해 서브 PSO 실행
    for joint_idx in range(DOF):
        # 현재 관절에 대한 서브 스웜 초기화
        sub_swarm = []
        noise_range = 0.2  # 마스터 위치 주변의 탐색 범위
        
        for _ in range(sub_pop_size):
            # 마스터 각도 주변에서 초기화
            pos = master_position[joint_idx] + np.random.uniform(-noise_range, noise_range)
            pos = np.clip(pos, *joint_bounds[joint_idx])
            
            particle = {
                'position': pos,
                'velocity': 0.0,
                'pbest_position': pos,
                'pbest_fitness': np.inf
            }
            sub_swarm.append(particle)
        
        # 서브 스웜의 전역 최적해
        sub_gbest_position = sub_swarm[0]['position']
        sub_gbest_fitness = np.inf
        
        # 서브 PSO 반복
        for sub_iter in range(sub_max_iter):
            # 각 서브 입자 평가
            for sub_particle in sub_swarm:
                # 테스트 각도 생성 (현재 관절만 변경)
                test_angles = refined_position.copy()
                test_angles[joint_idx] = sub_particle['position']
                
                # 적합도 계산
                end_effector = forward_kinematics_3d(test_angles)[-1]
                fitness = np.linalg.norm(end_effector - target)
                
                # 개인 최적해 업데이트
                if fitness < sub_particle['pbest_fitness']:
                    sub_particle['pbest_fitness'] = fitness
                    sub_particle['pbest_position'] = sub_particle['position']
                
                # 서브 전역 최적해 업데이트
                if fitness < sub_gbest_fitness:
                    sub_gbest_fitness = fitness
                    sub_gbest_position = sub_particle['position']
            
            # 서브 입자 업데이트
            for sub_particle in sub_swarm:
                r1, r2 = np.random.rand(2)
                
                cognitive = 1.5 * r1 * (sub_particle['pbest_position'] - sub_particle['position'])
                social = 1.5 * r2 * (sub_gbest_position - sub_particle['position'])
                sub_particle['velocity'] = 0.5 * sub_particle['velocity'] + cognitive + social
                
                sub_particle['position'] += sub_particle['velocity']
                sub_particle['position'] = np.clip(sub_particle['position'], *joint_bounds[joint_idx])
        
        # 현재 관절을 최적화된 값으로 업데이트
        refined_position[joint_idx] = sub_gbest_position
    
    return refined_position

# 실행 및 시각화
def run_hierarchical_pso():
    target = random_reachable_target()
    print(f"목표 위치: {target}")
    
    # 계층화된 PSO 실행
    final_angles, final_error, history = hierarchical_pso(target)
    
    print(f"\n=== 결과 ===")
    print(f"최종 관절 각도: {final_angles}")
    print(f"최종 오차: {final_error:.10f} m")
    
    # 최종 결과
    final_coords = forward_kinematics_3d(final_angles)
    
    # 시각화
    fig = plt.figure(figsize=(18, 5))
    
    # 1. 최종 팔 구성 (3D) + 엔드이펙터 궤적
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 엔드이펙터 궤적 그리기 (회색)
    trajectory_coords = np.array([forward_kinematics_3d(angles)[-1] for angles in history])
    ax1.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], trajectory_coords[:, 2], 
             '--', color='gray', alpha=0.5, linewidth=1, label='End-effector Path')
    
    # 최종 로봇 팔 구성
    ax1.plot(final_coords[:, 0], final_coords[:, 1], final_coords[:, 2], 
             '-o', linewidth=3, markersize=6, color='blue', label="Robot Arm")
    ax1.scatter(*target, color='red', s=100, label="Target", marker='*')
    ax1.scatter(*final_coords[-1], color='green', s=80, label="End Effector", marker='o')
    
    ax1.set_title("Final Configuration with Trajectory\n(Hierarchical PSO)", fontsize=14, fontweight='bold')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 2. 수렴 곡선
    ax2 = fig.add_subplot(132)
    target_distance_from_origin = np.linalg.norm(target)
    errors = [np.linalg.norm(forward_kinematics_3d(angles)[-1] - target) / target_distance_from_origin for angles in history]
    ax2.plot(errors, 'b-', linewidth=2, alpha=0.7)
    ax2.set_title('Convergence Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative Error (End-effector to Target / Origin to Target)')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 관절 각도 진화
    ax3 = fig.add_subplot(133)
    history_array = np.array(history)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(DOF):
        ax3.plot(history_array[:, i], color=colors[i], linewidth=2, 
                label=f'Joint {i+1}', alpha=0.8)
    ax3.set_title('Joint Angles Evolution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Joint Angle (rad)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return final_angles, final_error, history



# 실행
if __name__ == "__main__":
    final_angles, final_error, history = run_hierarchical_pso()
