import numpy as np
import time

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
        angle_list=[round(float(a), 0) for a in angles]
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
