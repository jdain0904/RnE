import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 기본 설정
DOF = 4
link_lengths = [0.3, 0.3, 0.2, 0.1]
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
np.random.seed(42)

def forward_kinematics_3d(joint_angles):
    coords = [(0, 0, 0)]
    x = y = z = 0
    theta_y = theta_z = 0
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        if i % 2 == 0:
            theta_y += angle
        else:
            theta_z += angle
        dx = length * np.cos(theta_y) * np.cos(theta_z)
        dy = length * np.sin(theta_y)
        dz = length * np.cos(theta_y) * np.sin(theta_z)
        x += dx; y += dy; z += dz
        coords.append((x, y, z))
    return np.array(coords)

def random_reachable_target(radius=0.9, z_min=-0.1, z_max=0.5):
    while True:
        p = np.random.uniform([0.0, -radius, z_min], [radius, radius, z_max])
        if np.linalg.norm(p) <= radius:
            return p

def initialize_sub_swarm(master_position, sub_pop_size):
    sub_swarm = []
    for _ in range(sub_pop_size):
        pos = master_position + np.random.uniform(-0.3, 0.3, DOF)
        pos = np.clip(pos, [b[0] for b in joint_bounds], [b[1] for b in joint_bounds])
        vel = np.random.uniform(-0.1, 0.1, DOF)
        sub_swarm.append({'position': pos.copy(), 'velocity': vel.copy(),
                          'pbest_position': pos.copy(), 'pbest_fitness': np.inf})
    return sub_swarm

def optimize_sub_swarm(sub_swarm, master_position, global_best, target, sub_iterations=10):
    sub_gbest = sub_swarm[0]['position'].copy()
    sub_gbest_fit = np.inf
    for _ in range(sub_iterations):
        for p in sub_swarm:
            eff = forward_kinematics_3d(p['position'])[-1]
            fit = np.linalg.norm(eff - target)
            if fit < p['pbest_fitness']:
                p['pbest_fitness'] = fit
                p['pbest_position'] = p['position'].copy()
            if fit < sub_gbest_fit:
                sub_gbest_fit = fit
                sub_gbest = p['position'].copy()
        for p in sub_swarm:
            r1, r2, r3 = np.random.rand(3)
            cog = 1.2 * r1 * (p['pbest_position'] - p['position'])
            loc = 1.2 * r2 * (sub_gbest - p['position'])
            glob = 0.8 * r3 * (global_best - p['position'])
            p['velocity'] = 0.6 * p['velocity'] + cog + loc + glob
            p['position'] += p['velocity']
            p['position'] = np.clip(p['position'], [b[0] for b in joint_bounds], [b[1] for b in joint_bounds])
    return sub_gbest

def update_sub_swarm_with_master_info(sub_swarm, new_master_position, global_best):
    for i in range(len(sub_swarm) // 3):
        noise = np.random.uniform(-0.2, 0.2, DOF)
        sub_swarm[i]['position'] = new_master_position + noise
        sub_swarm[i]['position'] = np.clip(sub_swarm[i]['position'], [b[0] for b in joint_bounds], [b[1] for b in joint_bounds])
        sub_swarm[i]['velocity'] = np.random.uniform(-0.05, 0.05, DOF)

def hierarchical_pso(target, master_pop_size=15, sub_pop_size=8, max_iter=200):
    master_swarm = []
    for _ in range(master_pop_size):
        pos = np.array([np.random.uniform(*joint_bounds[j]) for j in range(DOF)])
        vel = np.zeros(DOF)
        sub_swarm = initialize_sub_swarm(pos, sub_pop_size)
        master_swarm.append({'position': pos.copy(), 'velocity': vel.copy(),
                             'pbest_position': pos.copy(), 'pbest_fitness': np.inf,
                             'sub_swarm': sub_swarm})
    gbest = master_swarm[0]['position'].copy()
    gbest_fit = np.inf
    history = []

    for iter in range(max_iter):
        for master in master_swarm:
            sub_best = optimize_sub_swarm(master['sub_swarm'], master['position'], gbest, target)
            eff = forward_kinematics_3d(sub_best)[-1]
            fit = np.linalg.norm(eff - target)
            if fit < master['pbest_fitness']:
                master['pbest_fitness'] = fit
                master['pbest_position'] = sub_best.copy()
                master['position'] = sub_best.copy()
            if fit < gbest_fit:
                gbest_fit = fit
                gbest = sub_best.copy()

        for master in master_swarm:
            r1, r2 = np.random.rand(2)
            cog = 1.5 * r1 * (master['pbest_position'] - master['position'])
            soc = 1.5 * r2 * (gbest - master['position'])
            master['velocity'] = 0.5 * master['velocity'] + cog + soc
            master['position'] += master['velocity']
            master['position'] = np.clip(master['position'], [b[0] for b in joint_bounds], [b[1] for b in joint_bounds])

        for master in master_swarm:
            update_sub_swarm_with_master_info(master['sub_swarm'], master['position'], gbest)

        history.append(gbest.copy())  # ✅ 1개만 기록
        if (iter + 1) % 50 == 0 or (iter + 1) == max_iter:
            print(f"[Iter {iter+1}] 적합도: {gbest_fit:.8f}")

    return gbest, gbest_fit, history, target

def visualize_progress(target, current_best_angles, history, iteration, current_error):
    coords = forward_kinematics_3d(current_best_angles)
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    traj = np.array([forward_kinematics_3d(a)[-1] for a in history])
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--', color='gray', alpha=0.5)
    ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o', color='blue', linewidth=3)
    ax1.scatter(*target, color='red', s=100, marker='*')
    ax1.scatter(*coords[-1], color='green', s=80, marker='o')
    ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])
    ax1.set_title(f"Robot @ Iter {iteration}"); ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(132)
    norm = np.linalg.norm(target)
    errors = [np.linalg.norm(forward_kinematics_3d(a)[-1] - target)/norm for a in history]
    ax2.plot(range(len(errors)), errors, 'b-', linewidth=2)
    ax2.set_yscale('log'); ax2.set_title("Convergence"); ax2.grid(True)

    ax3 = fig.add_subplot(133)
    hist = np.array(history)
    for i in range(DOF):
        ax3.plot(hist[:, i], label=f'Joint {i+1}')
    ax3.set_title("Joint Angles"); ax3.legend(); ax3.grid(True)
    plt.tight_layout(); plt.show()

def animate_robot_arm_trajectory(history, target):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        angles = history[frame]
        coords = forward_kinematics_3d(angles)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o', color='blue')
        ax.scatter(*target, color='red', s=100, marker='*')
        ax.scatter(*coords[-1], color='green', s=80)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_title(f"Frame {frame+1} / {len(history)}")
    ani = FuncAnimation(fig, update, frames=len(history), interval=50)
    plt.show()

def run_hierarchical_pso():
    target = random_reachable_target()
    print(f"\n🎯 목표 위치: {target}")
    final_angles, final_error, history, target = hierarchical_pso(target, max_iter=200)
    print(f"\n✅ 최종 관절: {final_angles}")
    print(f"오차: {final_error:.10f}")
    visualize_progress(target, final_angles, history, len(history), final_error)
    return final_angles, final_error, history, target

if __name__ == "__main__":
    final_angles, final_error, history, target = run_hierarchical_pso()
    print(f"\n🎞️ 애니메이션 재생 중 (프레임 수: {len(history)})")
    animate_robot_arm_trajectory(history, target)
