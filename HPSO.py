# ==========================================
# 정식 HPSO 구조 기반 4자유도 로봇팔 경로 계획 및 애니메이션 시각화
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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

# ==========================================
# [2] 정식 HPSO 알고리즘 구현 (군집 + 계층)
# ==========================================

def hpso_inverse_kinematics_with_frames(target, pop_size=20, num_clusters=4, max_iter=200, w=0.5, c1=1.5, c2=1.5):
    gbest_history = []

    clusters = []
    for _ in range(num_clusters):
        cluster = []
        for _ in range(pop_size):
            position = np.array([np.random.uniform(low, high) for (low, high) in joint_bounds])
            velocity = np.zeros(DOF)
            particle = {
                "position": position,
                "velocity": velocity,
                "pbest_pos": position.copy(),
                "pbest_val": np.inf
            }
            cluster.append(particle)
        clusters.append(cluster)

    for iter in range(max_iter):
        for cluster in clusters:
            for p in cluster:
                end = forward_kinematics_3d(p["position"])[-1]
                error = np.linalg.norm(end - target)
                if error < p["pbest_val"]:
                    p["pbest_val"] = error
                    p["pbest_pos"] = p["position"].copy()

        cluster_best = min((p for c in clusters for p in c), key=lambda x: x['pbest_val'])
        gbest_history.append(cluster_best['pbest_pos'].copy())

        for cluster in clusters:
            parent_best = min(cluster, key=lambda x: x['pbest_val'])['pbest_pos']
            for p in cluster:
                r1, r2 = np.random.rand(DOF), np.random.rand(DOF)
                cognitive = c1 * r1 * (p["pbest_pos"] - p["position"])
                social = c2 * r2 * (parent_best - p["position"])
                p["velocity"] = w * p["velocity"] + cognitive + social
                p["position"] += p["velocity"]
                for i in range(DOF):
                    p["position"][i] = np.clip(p["position"][i], joint_bounds[i][0], joint_bounds[i][1])

    final_particle = min((p for c in clusters for p in c), key=lambda x: x['pbest_val'])
    final_angles = final_particle['pbest_pos']
    final_error = final_particle['pbest_val']
    return final_angles, final_error, gbest_history

# ==========================================
# [3] 실행 및 결과 준비 + 수렴곡선 시각화
# ==========================================

target = random_reachable_target()
pso_angles, pso_error, gbest_frames = hpso_inverse_kinematics_with_frames(target)

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
