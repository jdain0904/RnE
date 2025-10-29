import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

DOF = 6
link_lengths = [131.56e-3, 110.4e-3, 96e-3, 64.62e-3, 73.18e-3, 48.6e-3]
joint_bounds = [(-165, 165), (-165, 165), (-165, 165), (-165, 165), (-165, 165), (-175, 175)]
np.random.seed(42)

def forward_kinematics_3d(joint_angles):
    """
    6-DOF 로봇의 Forward Kinematics (DH convention 기반)
    joint_angles: 각도 (degree)
    """
    # degree를 radian으로 변환
    angles_rad = np.radians(joint_angles)
    
    # DH parameters for 6-DOF robot
    # [a, alpha, d, theta_offset]
    # a: link length, alpha: link twist, d: link offset, theta: joint angle
    dh_params = [
        [0, np.pi/2, link_lengths[0], 0],      # Joint 1
        [link_lengths[1], 0, 0, -np.pi/2],     # Joint 2
        [link_lengths[2], 0, 0, 0],            # Joint 3
        [0, np.pi/2, link_lengths[3], 0],      # Joint 4
        [0, -np.pi/2, link_lengths[4], 0],     # Joint 5
        [0, 0, link_lengths[5], 0]             # Joint 6
    ]
    
    # 초기 위치 (베이스)
    coords = [np.array([0, 0, 0])]
    
    # 누적 변환 행렬
    T = np.eye(4)
    
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = angles_rad[i] + theta_offset
        
        # DH 변환 행렬
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T_i = np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        
        # 누적 변환
        T = T @ T_i
        
        # 현재 관절 위치 추출
        coords.append(T[:3, 3].copy())
    
    return np.array(coords)

def random_reachable_target(radius=0.4, z_min=0.05, z_max=0.4):
    """6-DOF 로봇의 작업 공간 내에서 도달 가능한 랜덤 타겟 생성"""
    # 총 링크 길이: 약 0.524m
    max_reach = sum(link_lengths)  # 최대 도달 거리
    min_reach = max_reach * 0.1    # 최소 도달 거리 (너무 가까우면 특이점)
    
    while True:
        # 구형 좌표계로 샘플링
        r = np.random.uniform(min_reach, min(radius, max_reach * 0.8))
        theta = np.random.uniform(0, 2*np.pi)  # 방위각
        phi = np.random.uniform(0, np.pi/2)    # 고도각 (위쪽 반구)
        
        # 직교 좌표로 변환
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # z 범위 제한
        if z_min <= z <= z_max:
            # 원점으로부터의 거리 확인
            dist = np.sqrt(x**2 + y**2 + z**2)
            if min_reach <= dist <= min(radius, max_reach * 0.8):
                return np.array([x, y, z])

def animate_pso_optimization(target, gbest_positions_history, gbest_fitness_history, total_evaluations):
    """50fps 애니메이션으로 PSO 최적화 과정 시각화"""
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        
        # 현재까지의 궤적
        if frame > 0:
            traj=np.array([forward_kinematics_3d(a)[-1] for a in gbest_positions_history[:frame]])
            ax.plot(traj[:,0], traj[:,1], traj[:,2], '--', color='gray', alpha=0.6, linewidth=1.5, label='End-effector Path')
        
        # 현재 로봇 팔
        current_angles = gbest_positions_history[frame] if frame < len(gbest_positions_history) else gbest_positions_history[-1]
        arm = forward_kinematics_3d(current_angles)
        ax.plot(arm[:,0], arm[:,1], arm[:,2], '-o', linewidth=3.5, markersize=7, color='blue', label='Robot Arm')
        
        # 타겟과 엔드이펙터
        ax.scatter(*target, color='red', s=150, label='Target', marker='*')
        ax.scatter(*arm[-1], color='green', s=120, label='End Effector', marker='o')
        
        # 제목
        current_error = gbest_fitness_history[frame] if frame < len(gbest_fitness_history) else gbest_fitness_history[-1]
        current_evals = (frame + 1) * 20  # POPULATION=20
        ax.set_title(f"Iteration {frame}\nBest Error: {current_error:.8f} | Total Evals: {current_evals}", 
                     fontsize=15, fontweight='bold')
        
        # 축 설정 (6-DOF 로봇의 작업 공간에 맞게)
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0, 0.6])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # 애니메이션 생성 (50fps)
    frames = len(gbest_positions_history)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/50, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def visualize_full_results(final_target, final_angles, gbest_positions_history, gbest_fitness_history):
    final_coords=forward_kinematics_3d(final_angles)
    fig=plt.figure(figsize=(18,5))
    ax1=fig.add_subplot(131,projection='3d')
    traj=np.array([forward_kinematics_3d(a)[-1] for a in gbest_positions_history]) if len(gbest_positions_history)>0 else np.zeros((0,3))
    if len(traj)>0: ax1.plot(traj[:,0],traj[:,1],traj[:,2],'--',color='gray',alpha=0.6,linewidth=1.5,label='End-effector Path (gbest)')
    arm=forward_kinematics_3d(final_angles)
    ax1.plot(arm[:,0],arm[:,1],arm[:,2],'-o',linewidth=3,markersize=6,color='blue',label='Final Arm (gbest)')
    ax1.scatter(*final_target,color='red',s=100,label='Final Target',marker='*')
    ax1.scatter(*final_coords[-1],color='green',s=80,label='End Effector',marker='o')
    ax1.set_title(f"Final Configuration\n({len(gbest_positions_history)} Iterations)",fontsize=13,fontweight='bold')
    ax1.set_xlim([-0.6, 0.6]); ax1.set_ylim([-0.6, 0.6]); ax1.set_zlim([0, 0.6])
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)'); ax1.legend()

    ax2=fig.add_subplot(132)
    iters=np.arange(1,len(gbest_fitness_history)+1)
    if len(gbest_fitness_history)>0: ax2.plot(iters,gbest_fitness_history,'b-',linewidth=2,alpha=0.85)
    ax2.set_title(f'Convergence ({len(gbest_fitness_history)} iters)',fontsize=13,fontweight='bold')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Error (m)'); ax2.set_yscale('log')
    ax2.grid(True,alpha=0.3); ax2.set_xlim([1,max(1,len(gbest_fitness_history))])

    ax3=fig.add_subplot(133)
    if len(gbest_positions_history)>0:
        H=np.array(gbest_positions_history)
        cols=['red','blue','green','orange','purple','brown']
        for i in range(DOF):
            ax3.plot(iters,H[:,i],color=cols[i],linewidth=2,label=f'Joint {i+1}',alpha=0.9)
    ax3.set_title(f'Joint Angles (gbest, {len(gbest_positions_history)} iters)',fontsize=13,fontweight='bold')
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('Angle (deg)')
    ax3.grid(True,alpha=0.3); ax3.set_xlim([1,max(1,len(gbest_positions_history))]); ax3.legend(ncol=2)
    plt.tight_layout(); plt.show()


# ====== PSO 알고리즘 ======

DIMENSIONS=DOF
POPULATION=20
V_MAX=10.0  # 각도 단위에 맞게 증가
C1=2.0
C2=2.0
CONVERGENCE=1e-3

def initialize_pso_swarm():
    swarm=[]
    for _ in range(POPULATION):
        pos=np.array([np.random.uniform(*joint_bounds[j]) for j in range(DOF)])
        vel=np.random.uniform(-V_MAX,V_MAX,DOF)
        swarm.append({'pos':pos,'vel':vel,'pbest_pos':pos.copy(),'pbest_fit':np.inf})
    return swarm

def pso_step(swarm, target, inertia):
    best_pos=None
    best_fit=np.inf
    evals=0
    
    # 평가
    for p in swarm:
        ee=forward_kinematics_3d(p['pos'])[-1]  # forward_kinematics 내부에서 변환
        fit=np.linalg.norm(ee-target)
        evals+=1
        
        if fit<p['pbest_fit']:
            p['pbest_fit']=fit
            p['pbest_pos']=p['pos'].copy()
        
        if fit<best_fit:
            best_fit=fit
            best_pos=p['pos'].copy()
    
    # 갱신
    for p in swarm:
        r1,r2=np.random.rand(2)
        cog=C1*r1*(p['pbest_pos']-p['pos'])
        soc=C2*r2*(best_pos-p['pos'])
        p['vel']=inertia*p['vel']+cog+soc
        p['vel']=np.clip(p['vel'],-V_MAX,V_MAX)
        p['pos']=p['pos']+p['vel']
        
        for i in range(DOF):
            p['pos'][i]=np.clip(p['pos'][i],*joint_bounds[i])
    
    return best_pos, best_fit, evals

def pso_optimize(target, max_iterations=50, inertia_init=0.9, inertia_decay=0.99):
    gbest_positions_history=[]
    gbest_fitness_history=[]
    total_evaluations=0
    
    swarm=initialize_pso_swarm()
    inertia=inertia_init
    
    gbest_pos=swarm[0]['pos'].copy()
    gbest_fit=np.inf
    
    for iteration in range(max_iterations):
        best_pos, best_fit, evals = pso_step(swarm, target, inertia)
        total_evaluations+=evals
        
        if best_fit<gbest_fit:
            gbest_fit=best_fit
            gbest_pos=best_pos.copy()
        
        gbest_positions_history.append(gbest_pos.copy())
        gbest_fitness_history.append(gbest_fit)
        
        # 수렴 확인
        if gbest_fit<CONVERGENCE:
            print(f"Converged at iteration {iteration+1}")
            break
        
        # 관성 감소
        inertia*=inertia_decay
    
    return gbest_pos, gbest_fit, gbest_positions_history, gbest_fitness_history, total_evaluations

def run_pso():
    target=random_reachable_target()
    return pso_optimize(target, max_iterations=50), target

if __name__=="__main__":
    (final_angles, final_error, gbest_positions_history, gbest_fitness_history, total_evaluations), target = run_pso()
    
    print("\n=== 최종 결과 ===")
    print(f"타깃 위치: {target}")
    print(f"최종 관절 각도 (degree): {final_angles}")
    print(f"최종 오차: {final_error:.10f} m")
    print(f"총 iteration: {len(gbest_positions_history)}")
    print(f"총 평가 횟수: {total_evaluations}")
    
    # 50fps 애니메이션 표시
    print("\n=== 50fps 애니메이션 재생 중... ===")
    anim = animate_pso_optimization(target, gbest_positions_history, gbest_fitness_history, total_evaluations)
    
    # 최종 결과 그래프
    visualize_full_results(target, final_angles, gbest_positions_history, gbest_fitness_history)
    
    last_ee=forward_kinematics_3d(final_angles)[-1]
    print("\n=== 추가 통계 ===")
    print(f"최종 엔드이펙터: {last_ee}")
    print(f"최종 타깃: {target}")
    print(f"평가/iteration: {total_evaluations/max(1,len(gbest_positions_history)):.2f}")
    
    rel_err_pct=final_error/max(np.linalg.norm(target),1e-9)*100.0
    print(f"상대 오차(마지막 타깃 기준): {rel_err_pct:.6f}%")
