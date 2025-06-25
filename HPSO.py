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

def hierarchical_pso(target, master_pop_size=15, sub_pop_size=8, max_iterations=200, w=0.5, c1=1.5, c2=1.5):
    master_swarm = []
    for i in range(master_pop_size):
        position = np.array([np.random.uniform(*joint_bounds[j]) for j in range(DOF)])
        velocity = np.zeros(DOF)
        sub_swarm = initialize_sub_swarm(position, sub_pop_size)
        particle = {
            'id': i,
            'position': position.copy(),
            'velocity': velocity.copy(),
            'pbest_position': position.copy(),
            'pbest_fitness': np.inf,
            'sub_swarm': sub_swarm
        }
        master_swarm.append(particle)
    gbest_position = master_swarm[0]['position'].copy()
    gbest_fitness = np.inf
    optimization_history = []
    fitness_history = []
    total_evaluations = 0
    print("=== 진정한 계층화된 PSO 시작 ===")
    print(f"마스터 스웜 크기: {master_pop_size}, 서브 스웜 크기: {sub_pop_size}")
    print(f"총 iteration 횟수: {max_iterations}")
    print(f"\n--- 초기 상태 (0회 iteration) ---")
    visualize_3d_only(target, gbest_position, optimization_history, 0, gbest_fitness, total_evaluations)
    for iteration in range(1, max_iterations + 1):
        iteration_evaluations = 0
        for master_idx, master_particle in enumerate(master_swarm):
            sub_best_position = None
            sub_best_fitness = np.inf
            for sub_particle in master_particle['sub_swarm']:
                end_effector = forward_kinematics_3d(sub_particle['position'])[-1]
                fitness = np.linalg.norm(end_effector - target)
                iteration_evaluations += 1
                total_evaluations += 1
                if fitness < sub_particle['pbest_fitness']:
                    sub_particle['pbest_fitness'] = fitness
                    sub_particle['pbest_position'] = sub_particle['position'].copy()
                if fitness < sub_best_fitness:
                    sub_best_fitness = fitness
                    sub_best_position = sub_particle['position'].copy()
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = sub_particle['position'].copy()
            if sub_best_position is not None and sub_best_fitness < master_particle['pbest_fitness']:
                master_particle['pbest_fitness'] = sub_best_fitness
                master_particle['pbest_position'] = sub_best_position.copy()
                master_particle['position'] = sub_best_position.copy()
        optimization_history.append(gbest_position.copy())
        fitness_history.append(gbest_fitness)
        for master_particle in master_swarm:
            r1, r2 = np.random.rand(2)
            cognitive = c1 * r1 * (master_particle['pbest_position'] - master_particle['position'])
            social = c2 * r2 * (gbest_position - master_particle['position'])
            master_particle['velocity'] = w * master_particle['velocity'] + cognitive + social
            master_particle['position'] += master_particle['velocity']
            for i in range(DOF):
                master_particle['position'][i] = np.clip(master_particle['position'][i], *joint_bounds[i])
        for master_particle in master_swarm:
            update_sub_swarm_positions(master_particle['sub_swarm'], gbest_position)
            update_sub_swarm_with_master_info(
                master_particle['sub_swarm'], 
                master_particle['position'],
                gbest_position
            )
        if iteration % 50 == 0:
            print(f"\n--- {iteration}회 iteration 완료 ---")
            print(f"이번 iteration 평가 횟수: {iteration_evaluations}")
            print(f"총 평가 횟수: {total_evaluations}")
            print(f"현재 최적 적합도: {gbest_fitness:.8f}")
            visualize_3d_only(target, gbest_position, optimization_history, iteration, gbest_fitness, total_evaluations)
   """조기종료조건 : 지금은 반복횟수 200으로 고정시키고 하는것이니 주석처리함
        if gbest_fitness < 1e-10: 
            print(f"목표 정확도 달성! (iteration {iteration}, 총 평가 {total_evaluations}회)")
            break
        """
	
	print(f"\n최종 통계: {iteration} iterations, {total_evaluations} total evaluations")
    return gbest_position, gbest_fitness, optimization_history, fitness_history, total_evaluations

def initialize_sub_swarm(master_position, sub_pop_size):
    sub_swarm = []
    noise_range = 0.3
    for _ in range(sub_pop_size):
        position = master_position + np.random.uniform(-noise_range, noise_range, DOF)
        for i in range(DOF):
            position[i] = np.clip(position[i], *joint_bounds[i])
        velocity = np.random.uniform(-0.1, 0.1, DOF)
        sub_particle = {
            'position': position.copy(),
            'velocity': velocity.copy(),
            'pbest_position': position.copy(),
            'pbest_fitness': np.inf
        }
        sub_swarm.append(sub_particle)
    return sub_swarm

def update_sub_swarm_positions(sub_swarm, global_best):
    for sub_particle in sub_swarm:
        r1, r2, r3 = np.random.rand(3)
        cognitive = 1.2 * r1 * (sub_particle['pbest_position'] - sub_particle['position'])
        local_social = 1.2 * r2 * (sub_particle['pbest_position'] - sub_particle['position'])
        global_social = 0.8 * r3 * (global_best - sub_particle['position'])
        sub_particle['velocity'] = (0.6 * sub_particle.get('velocity', np.zeros(DOF)) + 
                                  cognitive + local_social + global_social)
        sub_particle['position'] += sub_particle['velocity']
        for i in range(DOF):
            sub_particle['position'][i] = np.clip(sub_particle['position'][i], *joint_bounds[i])

def update_sub_swarm_with_master_info(sub_swarm, new_master_position, global_best):
    num_to_update = len(sub_swarm) // 3
    for i in range(num_to_update):
        noise = np.random.uniform(-0.2, 0.2, DOF)
        sub_swarm[i]['position'] = new_master_position + noise
        for j in range(DOF):
            sub_swarm[i]['position'][j] = np.clip(sub_swarm[i]['position'][j], *joint_bounds[j])
        sub_swarm[i]['velocity'] = np.random.uniform(-0.05, 0.05, DOF)

def run_hierarchical_pso():
    target = random_reachable_target()
    print(f"목표 위치: {target}")
    final_angles, final_error, history, fitness_history, total_evaluations = hierarchical_pso(target, max_iterations=200)
    print(f"\n=== 최종 결과 ===")
    print(f"최종 관절 각도: {final_angles}")
    print(f"최종 오차: {final_error:.10f} m")
    print(f"총 iteration 횟수: {len(history)}")
    print(f"총 입자 평가 횟수: {total_evaluations}")
    print(f"\n--- 최종 전체 결과 시각화 ---")
    visualize_full_results(target, final_angles, history, fitness_history, len(history), final_error, total_evaluations)
    return final_angles, final_error, history, fitness_history, total_evaluations

if __name__ == "__main__":
    final_angles, final_error, history, fitness_history, total_evaluations = run_hierarchical_pso()
    target = random_reachable_target()
    print(f"\n=== 최종 통계 ===")
    print(f"총 iteration 횟수: {len(history)}")
    print(f"총 입자 평가 횟수: {total_evaluations}")
    print(f"평가/iteration 비율: {total_evaluations/len(history):.1f}")
    print(f"상대 오차 (목표거리 대비): {final_error/np.linalg.norm(target)*100:.6f}%")
    print(f"최종 엔드이펙터 위치: {forward_kinematics_3d(final_angles)[-1]}")
    print(f"목표 위치: {target}")
    print(f"효율성: {len(history)}회 iteration으로 {total_evaluations}회 입자 평가 완료")
