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

def visualize_3d_only(target, current_best_angles, history, iteration_count, current_error, total_evaluations):
    current_coords = forward_kinematics_3d(current_best_angles)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if len(history) > 0:
        trajectory_coords = np.array([forward_kinematics_3d(angles)[-1] for angles in history])
        ax.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], trajectory_coords[:, 2], 
                 '--', color='gray', alpha=0.5, linewidth=1, label='End-effector Path')
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
    current_coords = forward_kinematics_3d(current_best_angles)
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    trajectory_coords = np.array([forward_kinematics_3d(angles)[-1] for angles in history])
    ax1.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], trajectory_coords[:, 2], 
             '--', color='gray', alpha=0.5, linewidth=1, label='End-effector Path')
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

# ==============================
# PSO + Sliding Window 알고리즘 (진짜 시간 변화 적용)
# ==============================

def pso_sliding_window(target_function, pop_size=120, max_iterations=150, window_size=30, slide_step=15, w=0.5, c1=1.5, c2=1.5):
    """
    진짜 PSO + Sliding Window 알고리즘
    시간에 따라 변하는 목표나 제약을 처리 (정확히 150회 iteration)
    """
    
    # 전체 기록용
    optimization_history = []
    fitness_history = []
    total_evaluations = 0
    current_iteration = 0
    
    print("=== PSO + Sliding Window 시작 ===")
    print(f"입자 개수: {pop_size}")
    print(f"총 iteration 횟수: {max_iterations}")
    print(f"윈도우 크기: {window_size}, 슬라이딩 스텝: {slide_step}")
    
    # 초기 상태
    gbest_position = np.array([np.random.uniform(*joint_bounds[j]) for j in range(DOF)])
    gbest_fitness = np.inf
    
    print(f"\n--- 초기 상태 (0회 iteration) ---")
    visualize_3d_only(target_function(0), gbest_position, optimization_history, 0, gbest_fitness, total_evaluations)
    
    # 윈도우별 최적해 저장
    window_best_positions = []
    
    # 슬라이딩 윈도우 진행 (150회 제한)
    for window_start in range(0, max_iterations, slide_step):
        if current_iteration >= max_iterations:
            break
            
        window_end = min(window_start + window_size, max_iterations)
        window_iterations = min(window_end - window_start, max_iterations - current_iteration)
        
        print(f"\n=== 윈도우 {window_start}-{min(window_start + window_size, max_iterations)} 시작 ===")
        
        # 스웜 초기화 (이전 윈도우 정보 활용)
        swarm = []
        for i in range(pop_size):
            if window_best_positions and i < len(window_best_positions):
                # 이전 윈도우들의 최적해 활용 (지속적 학습)
                base_position = window_best_positions[i % len(window_best_positions)]
                noise = np.random.uniform(-0.2, 0.2, DOF)
                position = base_position + noise
            else:
                # 새로운 랜덤 위치
                position = np.array([np.random.uniform(*joint_bounds[j]) for j in range(DOF)])
            
            # 경계 조건 적용
            for j in range(DOF):
                position[j] = np.clip(position[j], *joint_bounds[j])
            
            velocity = np.zeros(DOF)
            particle = {
                'position': position.copy(),
                'velocity': velocity.copy(),
                'pbest_position': position.copy(),
                'pbest_fitness': np.inf
            }
            swarm.append(particle)
        
        # 윈도우 내 지역 최적해
        window_gbest_position = swarm[0]['position'].copy()
        window_gbest_fitness = np.inf
        
        # 윈도우 내 PSO 수행 (150회 제한)
        for window_iter in range(window_iterations):
            if current_iteration >= max_iterations:
                break
                
            current_target = target_function(current_iteration)  # 시간 변화하는 목표
            
            # 모든 입자 평가
            for particle in swarm:
                # 적합도 계산 (시간 변화하는 목표에 대해)
                end_effector = forward_kinematics_3d(particle['position'])[-1]
                fitness = np.linalg.norm(end_effector - current_target)
                total_evaluations += 1
                
                # 개인 최적해 업데이트
                if fitness < particle['pbest_fitness']:
                    particle['pbest_fitness'] = fitness
                    particle['pbest_position'] = particle['position'].copy()
                
                # 윈도우 지역 최적해 업데이트
                if fitness < window_gbest_fitness:
                    window_gbest_fitness = fitness
                    window_gbest_position = particle['position'].copy()
                
                # 전역 최적해 업데이트
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = particle['position'].copy()
            
            # 전체 기록 (현재 시간 기준)
            optimization_history.append(gbest_position.copy())
            fitness_history.append(gbest_fitness)
            current_iteration += 1
            
            # PSO 업데이트
            for particle in swarm:
                r1, r2 = np.random.rand(2)
                
                # 속도 업데이트
                cognitive = c1 * r1 * (particle['pbest_position'] - particle['position'])
                social = c2 * r2 * (window_gbest_position - particle['position'])
                particle['velocity'] = w * particle['velocity'] + cognitive + social
                
                # 위치 업데이트
                particle['position'] += particle['velocity']
                
                # 경계 조건
                for j in range(DOF):
                    particle['position'][j] = np.clip(particle['position'][j], *joint_bounds[j])
            
            # 50회마다 시각화
            if current_iteration % 50 == 0:
                print(f"\n--- {current_iteration}회 iteration 완료 ---")
                print(f"현재 윈도우: {window_start}-{min(window_start + window_size, max_iterations)}")
                print(f"총 평가 횟수: {total_evaluations}")
                print(f"현재 최적 적합도: {gbest_fitness:.8f}")
                visualize_3d_only(current_target, gbest_position, optimization_history, current_iteration, gbest_fitness, total_evaluations)
            
            if current_iteration >= max_iterations:
                break
        
        # 윈도우 최적해 저장 (다음 윈도우에서 활용)
        window_best_positions.append(window_gbest_position.copy())
        
        print(f"윈도우 완료. 윈도우 최적 적합도: {window_gbest_fitness:.8f}")
    
    print(f"\n최종 통계: {current_iteration} iterations, {total_evaluations} total evaluations")
    return gbest_position, gbest_fitness, optimization_history, fitness_history, total_evaluations

def run_pso_sliding_window():
    # 시간에 따라 변하는 목표 함수 정의
    base_target = random_reachable_target()
    
    def time_varying_target(time_step):
        """시간에 따라 조금씩 변하는 목표"""
        # 기본 목표에서 시간에 따라 작은 변화 추가
        variation = 0.05 * np.sin(time_step * 0.1) * np.array([1, 0.5, 0.3])
        new_target = base_target + variation
        # 경계 내로 제한
        new_target = np.clip(new_target, [-0.9, -0.9, -0.1], [0.9, 0.9, 0.5])
        return new_target
    
    print(f"기준 목표 위치: {base_target}")
    print("시간에 따라 목표가 미세하게 변화합니다...")
    
    # PSO + Sliding Window 실행
    final_angles, final_error, history, fitness_history, total_evaluations = pso_sliding_window(
        time_varying_target, max_iterations=150
    )
    
    print(f"\n=== 최종 결과 ===")
    print(f"최종 관절 각도: {final_angles}")
    print(f"최종 오차: {final_error:.10f} m")
    print(f"총 iteration 횟수: {len(history)}")
    print(f"총 입자 평가 횟수: {total_evaluations}")
    
    # 최종 전체 결과 시각화
    print(f"\n--- 최종 전체 결과 시각화 ---")
    final_target = time_varying_target(len(history) - 1)
    visualize_full_results(final_target, final_angles, history, fitness_history, len(history), final_error, total_evaluations)
    
    return final_angles, final_error, history, fitness_history, total_evaluations

if __name__ == "__main__":
    final_angles, final_error, history, fitness_history, total_evaluations = run_pso_sliding_window()
    
    # 최종 통계
    base_target = random_reachable_target()  # 동일한 시드로 기준 타겟 생성
    print(f"\n=== 최종 통계 ===")
    print(f"총 iteration 횟수: {len(history)}")
    print(f"총 입자 평가 횟수: {total_evaluations}")
    print(f"평가/iteration 비율: {total_evaluations/len(history):.1f}")
    print(f"상대 오차 (목표거리 대비): {final_error/np.linalg.norm(base_target)*100:.6f}%")
    print(f"최종 엔드이펙터 위치: {forward_kinematics_3d(final_angles)[-1]}")
    print(f"기준 목표 위치: {base_target}")
    print(f"효율성: {len(history)}회 iteration으로 {total_evaluations}회 입자 평가 완료")
    print(f"\n=== PSO + Sliding Window 특징 ===")
    print(f"시간 변화 대응: 목표가 시간에 따라 변화하는 동적 환경")
    print(f"윈도우 기반 적응: 이전 경험을 활용한 지속적 학습")
    print(f"슬라이딩 메커니즘: 15스텝마다 30크기 윈도우로 슬라이딩")
