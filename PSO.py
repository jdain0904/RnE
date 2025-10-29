import numpy as np
import time

DOF = 6
link_lengths = [131.56e-3, 110.4e-3, 96e-3, 64.62e-3, 73.18e-3, 48.6e-3]
joint_bounds = [(-165, 165), (-165, 165), (-165, 165), (-165, 165), (-165, 165), (-175, 175)]
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

# ====== PSO 파라미터 ======
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

def pso_step(swarm, target, inertia):
    best_pos=None
    best_fit=np.inf
    evals=0
    
    # 평가
    for p in swarm:
        ee=forward_kinematics_3d(p['pos'])[-1]
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

def pso_optimize(target, max_iterations=150, inertia_init=0.9, inertia_decay=0.99, record_trajectory=True):
    swarm=initialize_pso_swarm()
    inertia=inertia_init
    
    gbest_pos=swarm[0]['pos'].copy()
    gbest_fit=np.inf
    total_evaluations=0
    
    fitness_history=[]
    trajectory=[]  # 각 iteration의 best position 기록
    
    for iteration in range(max_iterations):
        best_pos, best_fit, evals = pso_step(swarm, target, inertia)
        total_evaluations+=evals
        
        if best_fit<gbest_fit:
            gbest_fit=best_fit
            gbest_pos=best_pos.copy()
        
        fitness_history.append(gbest_fit)
        
        # 궤적 기록
        if record_trajectory:
            trajectory.append(gbest_pos.copy())
       
        if gbest_fit<CONVERGENCE:
            break
        
        inertia*=inertia_decay
    
    return gbest_pos, gbest_fit, fitness_history, total_evaluations, trajectory

def send_trajectory_to_robot(trajectory, speed=50, delay=0.1, mc=None):
   
    for i, angles in enumerate(trajectory):
        # 각도를 리스트로 변환 (소수점 2자리)
        angle_list = [round(float(angles[i])*4096/330, 0) for i in range(5)]
        angle_list.append(round(float(angles[5])*4096/350))
        if mc is not None:
            # 실제 로봇에 전송
            mc.send_angles(angle_list, speed)
             print(f"Step {i+1}/{len(trajectory)}: Sent {angle_list}")
   
        # 다음 명령 전 대기
        if delay > 0:
            time.sleep(delay)
    

def run_pso(send_to_robot=False, mc=None, speed=50, delay=0.1):
    target=random_reachable_target()
    
    
    final_angles, final_error, fitness_history, total_evaluations, trajectory = pso_optimize(
        target, max_iterations=150, record_trajectory=True
    )
 
       # 로봇에 궤적 전송
    if send_to_robot:
        send_trajectory_to_robot(trajectory, speed=speed, delay=delay, mc=mc)
    
    return final_angles, final_error, fitness_history, total_evaluations, target, trajectory

if __name__=="__main__":
    # mc 객체 없이 실행 (명령만 출력)
    final_angles, final_error, fitness_history, total_evaluations, target, trajectory = run_pso(
        send_to_robot=True,  # True로 설정하면 명령 출력
        mc=None,  # 실제 로봇 객체를 여기에 전달
        speed=50,
        delay=0.1  # 실제 전송 시에는 0.1 이상 권장
    )
    print("system executed successfully")
