
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DOF = 4
link_lengths = [0.3, 0.3, 0.2, 0.1]
joint_bounds = [(-np.pi/2, np.pi/2) for _ in range(DOF)]
np.random.seed(42)

def forward_kinematics_3d(joint_angles):
    coords = [(0, 0, 0)]
    x=y=z=0.0
    theta_y=theta_z=0.0
    for i,(a,L) in enumerate(zip(joint_angles, link_lengths)):
        if i%2==0: theta_y+=a
        else: theta_z+=a
        dx=L*np.cos(theta_y)*np.cos(theta_z)
        dy=L*np.sin(theta_y)
        dz=L*np.cos(theta_y)*np.sin(theta_z)
        x+=dx; y+=dy; z+=dz
        coords.append((x,y,z))
    return np.array(coords)

def random_reachable_target(radius=0.9, z_min=-0.1, z_max=0.5):
    while True:
        p=np.random.uniform(low=[0.0,-radius,z_min], high=[radius,radius,z_max])
        if np.linalg.norm(p)<=radius: return p

def visualize_3d_only(target, joint_angles, gbest_positions_history, iteration_count, current_error, total_evaluations):
    arm=forward_kinematics_3d(joint_angles)
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111, projection='3d')
    if len(gbest_positions_history)>0:
        traj=np.array([forward_kinematics_3d(a)[-1] for a in gbest_positions_history])
        ax.plot(traj[:,0],traj[:,1],traj[:,2],'--',color='gray',alpha=0.6,linewidth=1.5,label='End-effector Path (gbest)')
    ax.plot(arm[:,0],arm[:,1],arm[:,2],'-o',linewidth=3.5,markersize=7,color='blue',label='Robot Arm (gbest)')
    ax.scatter(*target,color='red',s=150,label='Target',marker='*')
    ax.scatter(*arm[-1],color='green',s=120,label='End Effector',marker='o')
    ax.set_title(f"Iteration {iteration_count}\nBest Error: {current_error:.8f} | Total Evals: {total_evaluations}",fontsize=15,fontweight='bold')
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()

def visualize_full_results(final_target, final_angles, gbest_positions_history, gbest_fitness_history, targets_history):
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
    ax1.set_xlim([-1,1]); ax1.set_ylim([-1,1]); ax1.set_zlim([-1,1])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z'); ax1.legend()

    ax2=fig.add_subplot(132)
    rel=[]
    for e,tgt in zip(gbest_fitness_history, targets_history):
        denom=max(np.linalg.norm(tgt),1e-9); rel.append(100*e/denom)
    iters=np.arange(1,len(rel)+1)
    if len(rel)>0: ax2.plot(iters,rel,'b-',linewidth=2,alpha=0.85)
    ax2.set_title(f'Convergence (Relative Error, {len(rel)} iters)',fontsize=13,fontweight='bold')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Relative Error'); ax2.set_yscale('log')
    ax2.grid(True,alpha=0.3); ax2.set_xlim([1,max(1,len(rel))])

    ax3=fig.add_subplot(133)
    if len(gbest_positions_history)>0:
        H=np.array(gbest_positions_history)
        cols=['red','blue','green','orange']
        for i in range(DOF):
            ax3.plot(iters,H[:,i],color=cols[i],linewidth=2,label=f'Joint {i+1}',alpha=0.9)
    ax3.set_title(f'Joint Angles (gbest, {len(gbest_positions_history)} iters)',fontsize=13,fontweight='bold')
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('Angle (rad)')
    ax3.grid(True,alpha=0.3); ax3.set_xlim([1,max(1,len(gbest_positions_history))]); ax3.legend(ncol=2)
    plt.tight_layout(); plt.show()

# ====== 여기부터 HPSO 프레임을 그대로 쓰되 내부는 표준 PSO ======

DIMENSIONS=DOF
POPULATION=20
V_MAX=0.1
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

def pso_window_step(swarm, target, inertia):
    best_pos=None; best_fit=np.inf
    evals=0
    for p in swarm:  # 평가
        ee=forward_kinematics_3d(p['pos'])[-1]
        fit=np.linalg.norm(ee-target); evals+=1
        if fit<p['pbest_fit']:
            p['pbest_fit']=fit; p['pbest_pos']=p['pos'].copy()
        if fit<best_fit:
            best_fit=fit; best_pos=p['pos'].copy()
    for p in swarm:  # 갱신
        r1,r2=np.random.rand(2)
        cog=C1*r1*(p['pbest_pos']-p['pos'])
        soc=C2*r2*(best_pos-p['pos'])
        p['vel']=inertia*p['vel']+cog+soc
        p['vel']=np.clip(p['vel'],-V_MAX,V_MAX)
        p['pos']=p['pos']+p['vel']
        for i in range(DOF):
            p['pos'][i]=np.clip(p['pos'][i],*joint_bounds[i])
    return best_pos, best_fit, evals

def pso_in_hpso_frame(target_function,
                      max_iterations=150, window_size=30, slide_step=15,
                      visualize_every=50, inertia_init=None):
    gbest_positions_history=[]; gbest_fitness_history=[]; targets_history=[]
    total_evaluations=0; current_iteration=0
    swarm=initialize_pso_swarm()
    inertia = 0.5+0.5*np.random.rand() if inertia_init is None else float(inertia_init)
    gbest_pos=swarm[0]['pos'].copy(); gbest_fit=np.inf

    init_target=target_function(0)
    visualize_3d_only(init_target, gbest_pos, gbest_positions_history, 0, gbest_fit, total_evaluations)

    for window_start in range(0, max_iterations, slide_step):
        if current_iteration>=max_iterations: break
        window_end=min(window_start+window_size, max_iterations)
        window_iterations=min(window_end-window_start, max_iterations-current_iteration)

        for _ in range(window_iterations):
            if current_iteration>=max_iterations: break
            current_target=target_function(current_iteration)
            best_pos_win, best_fit_win, evals = pso_window_step(swarm, current_target, inertia)
            total_evaluations+=evals
            if best_fit_win<gbest_fit:
                gbest_fit=best_fit_win; gbest_pos=best_pos_win.copy()
            gbest_positions_history.append(gbest_pos.copy())
            gbest_fitness_history.append(gbest_fit)
            targets_history.append(current_target.copy())
            current_iteration+=1
            if visualize_every and (current_iteration%visualize_every==0):
                visualize_3d_only(current_target, gbest_pos, gbest_positions_history, current_iteration, gbest_fit, total_evaluations)
            if gbest_fit<CONVERGENCE: break
            inertia*=0.9
        if gbest_fit<CONVERGENCE: break

    return gbest_pos, gbest_fit, gbest_positions_history, gbest_fitness_history, targets_history, total_evaluations

def run_pso_in_hpso_frame():
    b_target=random_reachable_target()
    base_target=lambda t: b_target
    return pso_in_hpso_frame(base_target, max_iterations=150, visualize_every=50)

if __name__=="__main__":
    final_angles, final_error, gbest_positions_history, gbest_fitness_history, targets_history, total_evaluations = run_pso_in_hpso_frame()
    print("\n=== 최종 결과 ===")
    print(f"최종 관절 각도 (gbest): {final_angles}")
    print(f"최종 오차: {final_error:.10f} m")
    print(f"총 iteration: {len(gbest_positions_history)}")
    print(f"총 평가 횟수: {total_evaluations}")
    final_target=targets_history[-1]
    visualize_full_results(final_target, final_angles, gbest_positions_history, gbest_fitness_history, targets_history)
    last_ee=forward_kinematics_3d(final_angles)[-1]
    print("\n=== 추가 통계 ===")
    print(f"최종 엔드이펙터: {last_ee}")
    print(f"최종 타깃: {final_target}")
    print(f"평가/iteration: {total_evaluations/max(1,len(gbest_positions_history)):.2f}")
    rel_err_pct=final_error/max(np.linalg.norm(final_target),1e-9)*100.0
    print(f"상대 오차(마지막 타깃 기준): {rel_err_pct:.6f}%")
