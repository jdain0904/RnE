import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# [1] 초기 설정
# ==========================================

DOF = 4
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
link_lengths = [0.3, 0.3, 0.2, 0.1]

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

def visualize_3d_only(target, current_best_angles, history, iteration_count, current_error, total_evaluations):
    """
    3D 로봇팔 시각화만 출력하는 함수 (HPSO와 동일한 형식)
    """
    # 현재까지의 최적 구성
    current_coords = forward_kinematics_3d(current_best_angles)
    
    # 3D 로봇팔 시각화만 출력
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 엔드이펙터 궤적 그리기 (회색) - 데이터가 있을 때만
    if len(history) > 0:
        trajectory_coords = np.array([forward_kinematics_3d(angles)[-1] for angles in history])
        ax.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], trajectory_coords[:, 2], 
                 '--', color='gray', alpha=0.5, linewidth=1, label='End-effector Path')
    
    # 현재 로봇 팔 구성
    ax.plot(current_coords[:, 0], current_coords[:, 1], current_coords[:, 2], 
             '-o', linewidth=4, markersize=8, color='blue', label="Robot Arm")
    ax.scatter(*target, color='red', s=150, label="Target", marker='*')
    ax.scatter(*current_coords[-1], color='green', s=120, label="End Effector", marker='o')
    
    ax.set_title(f"Robot Arm at Iteration {iteration_count}\nError: {current_error:.8f} | Total Evaluations: {total_evaluations}", 
                fontsize=16, fontweight='bold')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_full_results(target, current_best_angles, history, fitness_history, iteration_count, current_error, total_evaluations):
    """
    200회 완료 후 전체 결과 시각화 (3D + 수렴곡선 + 관절각도) - HPSO와 동일한 형식
    """
    # 현재까지의 최적 구성
    current_coords = forward_kinematics_3d(current_best_angles)
    
    # 전체 시각화
    fig = plt.figure(figsize=(18, 5))
    
    # 1. 3D 로봇팔 + 궤적
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 엔드이펙터 궤적 그리기
    trajectory_coords = np.array([forward_kinematics_3d(angles)[-1] for angles in history])
    ax1.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], trajectory_coords[:, 2], 
             '--', color='gray', alpha=0.5, linewidth=1, label='End-effector Path')
    
    # 최종 로봇 팔 구성
    ax1.plot(current_coords[:, 0], current_coords[:, 1], current_coords[:, 2], 
             '-o', linewidth=3, markersize=6, color='blue', label="Final Robot Arm")
    ax1.scatter(*target, color='red', s=100, label="Target", marker='*')
    ax1.scatter(*current_coords[-1], color='green', s=80, label="End Effector", marker='o')
    
    ax1.set_title(f"Final Configuration\n({iteration_count} Iterations, {total_evaluations} Evaluations)", fontsize=14, fontweight='bold')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 2. 수렴 곡선 (iteration 기준)
    ax2 = fig.add_subplot(132)
    target_distance_from_origin = np.linalg.norm(target)
    relative_errors = [error / target_distance_from_origin for error in fitness_history]
    iterations_x = list(range(1, len(relative_errors) + 1))
    
    ax2.plot(iterations_x, relative_errors, 'b-', linewidth=2, alpha=0.7)
    ax2.set_title(f'Convergence Curve ({len(relative_errors)} Iterations)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Relative Error')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xlim([1, len(relative_errors)])
    
    # 3. 관절 각도 진화 (iteration 기준)
    ax3 = fig.add_subplot(133)
    history_array = np.array(history)
    colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(DOF):
        ax3.plot(iterations_x, history_array[:, i], color=colors[i], linewidth=2, 
                label=f'Joint {i+1}', alpha=0.8)
    ax3.set_title(f'Joint Angles Evolution ({len(history)} Iterations)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Joint Angle (rad)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, len(history)])
    
    plt.tight_layout()
    plt.show()

# ==========================================
# [2] 표준 PSO 알고리즘 (HPSO 연결용)
# ==========================================

def standard_pso(target, pop_size=30, max_iterations=150, w=0.5, c1=1.5, c2=1.5):
    """
    표준 PSO 알고리즘 - 우리 HPSO와 연결되는 구조
    각 입자가 전체 관절각 벡터를 가짐 (HPSO의 마스터 입자와 동일한 구조)
    """
    
    # === 입자 스웜 초기화 ===
    swarm = []
    for i in range(pop_size):
        position = np.array([np.random.uniform(*joint_bounds[j]) for j in range(DOF)])
        velocity = np.zeros(DOF)
        
        particle = {
            'id': i,
            'position': position.copy(),
            'velocity': velocity.copy(),
            'pbest_position': position.copy(),
            'pbest_fitness': np.inf
        }
        swarm.append(particle)
    
    # 전역 최적해
    gbest_position = swarm[0]['position'].copy()
    gbest_fitness = np.inf
    
    # 최적화 과정 기록
    optimization_history = []
    fitness_history = []
    total_evaluations = 0  # 총 입자 평가 횟수
    
    print("=== 표준 PSO 시작 ===")
    print(f"입자 개수: {pop_size}")
    print(f"총 iteration 횟수: {max_iterations}")
    
    # 초기 상태 (0회 iteration) 시각화
    print(f"\n--- 초기 상태 (0회 iteration) ---")
    visualize_3d_only(target, gbest_position, optimization_history, 0, gbest_fitness, total_evaluations)
    
    # 150회 iteration 수행
    for iteration in range(1, max_iterations + 1):
        iteration_evaluations = 0  # 이번 iteration의 평가 횟수
        
        # === 단계 1: 모든 입자 평가 ===
        for particle in swarm:
            # 적합도 계산
            end_effector = forward_kinematics_3d(particle['position'])[-1]
            fitness = np.linalg.norm(end_effector - target)
            iteration_evaluations += 1
            total_evaluations += 1
            
            # 개인 최적해 업데이트
            if fitness < particle['pbest_fitness']:
                particle['pbest_fitness'] = fitness
                particle['pbest_position'] = particle['position'].copy()
            
            # 전역 최적해 업데이트
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = particle['position'].copy()
        
        # 현재 iteration의 최적해 기록
        optimization_history.append(gbest_position.copy())
        fitness_history.append(gbest_fitness)
        
        # === 단계 2: PSO 업데이트 ===
        for particle in swarm:
            r1, r2 = np.random.rand(2)
            
            # 속도 업데이트 (표준 PSO)
            cognitive = c1 * r1 * (particle['pbest_position'] - particle['position'])
            social = c2 * r2 * (gbest_position - particle['position'])
            particle['velocity'] = w * particle['velocity'] + cognitive + social
            
            # 위치 업데이트
            particle['position'] += particle['velocity']
            
            # 경계 조건 적용
            for i in range(DOF):
                particle['position'][i] = np.clip(
                    particle['position'][i], *joint_bounds[i]
                )
        
        # 50회마다 시각화 (50, 100, 150)
        if iteration % 50 == 0:
            print(f"\n--- {iteration}회 iteration 완료 ---")
            print(f"이번 iteration 평가 횟수: {iteration_evaluations}")
            print(f"총 평가 횟수: {total_evaluations}")
            print(f"현재 최적 적합도: {gbest_fitness:.8f}")
            visualize_3d_only(target, gbest_position, optimization_history, iteration, gbest_fitness, total_evaluations)
    
    print(f"\n최종 통계: {iteration} iterations, {total_evaluations} total evaluations")
    return gbest_position, gbest_fitness, optimization_history, fitness_history, total_evaluations

# ==========================================
# [3] 실행 함수
# ==========================================

def run_standard_pso():
    target = random_reachable_target()
    print(f"목표 위치: {target}")
    
    # 표준 PSO 실행 (200회 iteration)
    final_angles, final_error, history, fitness_history, total_evaluations = standard_pso(target, max_iterations=200)
    
    print(f"\n=== 최종 결과 ===")
    print(f"최종 관절 각도: {final_angles}")
    print(f"최종 오차: {final_error:.10f} m")
    print(f"총 iteration 횟수: {len(history)}")
    print(f"총 입자 평가 횟수: {total_evaluations}")
    
    # 최종 전체 결과 시각화 (5번째 출력)
    print(f"\n--- 최종 전체 결과 시각화 ---")
    visualize_full_results(target, final_angles, history, fitness_history, len(history), final_error, total_evaluations)
    
    return final_angles, final_error, history, fitness_history, total_evaluations

# 실행
if __name__ == "__main__":
    final_angles, final_error, history, fitness_history, total_evaluations = run_standard_pso()
    
    # 최종 통계
    target = random_reachable_target()  # 동일한 시드로 같은 타겟 생성
    print(f"\n=== 최종 통계 ===")
    print(f"총 iteration 횟수: {len(history)}")
    print(f"총 입자 평가 횟수: {total_evaluations}")
    print(f"평가/iteration 비율: {total_evaluations/len(history):.1f}")
    print(f"상대 오차 (목표거리 대비): {final_error/np.linalg.norm(target)*100:.6f}%")
    print(f"최종 엔드이펙터 위치: {forward_kinematics_3d(final_angles)[-1]}")
    print(f"목표 위치: {target}")
    print(f"효율성: {len(history)}회 iteration으로 {total_evaluations}회 입자 평가 완료")
    
    print(f"\n=== PSO → HPSO 발전 과정 ===")
    print(f"현재 PSO: 단일 스웜 {30}개 입자")
    print(f"→ HPSO로 발전: 마스터 스웜 {15}개 + 각각 서브 스웜 {8}개")
    print(f"구조적 연결점: 둘 다 각 입자가 전체 관절각 벡터를 가짐")
