#1. Initiallization
 def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05, w: float = 0.4, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w = self.validator.check_float("w", w, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max
    #밑의 함수는 초기 입자를 생성하는 함수인데, 로봇 팔에 적용하는 경우에는 초기 입자 위치를 0,0으로 고정시키므로 후에 고려할 필요가 있음
    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent
for t_start in range(0, self.time_horizon, self.sliding_step):
    t_end=t_start+self.window_size
    window_time=(t_start,t_end)
    
    for epoch in range(self.epoch):
        for agent in self.population:
         
    #2. HPSO 이용 계층화

    #3. Velocity Calculation

    #4. Clamping
    #4-1 : 위치 Clamping

    #4-2 : Velocity Clamping
     v_new = np.where(v_new == 0, np.sign(0.5 - rand) * rand * self.v_max, v_new)
     v_new = np.sign(v_new) * np.minimum(np.abs(v_new), self.v_max)
     v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)

    #4-3 : 각도 Clamping

    #5. Position update

    #6. Evaluation
