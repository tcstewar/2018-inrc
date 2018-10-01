import nengo
import numpy as np

class CircularEnvironment(object):
    def __init__(self, seed=None, target_radius=0.3):
        self.rng = np.random.RandomState(seed=seed)
        self.dt = 0.001
        self.target_radius = target_radius
    
    def make_node(self):
        def step(t, x):
            if t == 0: 
                self.reset()
            speed = 2
            self.pos += x*self.dt * speed
            reward = 0
            
            dist = np.sqrt(np.sum((self.pos-self.target)**2))
            if dist < self.target_radius:
                reward = 1000
                self.reset_pos()
                
            r = np.sqrt(np.sum(self.pos**2))
            if r > 1:
                self.reset_pos()
            
            step._nengo_html_ = '''
            <svg width="100%" height="100%" viewbox="-100 -100 200 200">
                <circle cx=0 cy=0 r=99 style="stroke:blue" fill="white"/>
                <circle cx={tx} cy={ty} r={tr} fill="#bbb"/>
                <circle cx={x} cy={y} r=5 fill="black"/>
                
            </svg>
            '''.format(tx=self.target[0]*100, ty=-self.target[1]*100,
                       tr=self.target_radius*100,
                       x = self.pos[0]*100, y=-self.pos[1]*100)
            
            return [self.pos[0], self.pos[1], reward]
        return nengo.Node(step, size_in=2)
    
    def reset_pos(self):
        theta = self.rng.uniform(-np.pi, np.pi)
        r = self.rng.uniform(0.25, 0.75)
        self.pos = np.array([r*np.cos(theta), r*np.sin(theta)])
        
    def reset(self):
        target_theta = self.rng.uniform(-np.pi, np.pi)
        target_r = self.rng.uniform(0.25, 0.75)
        self.target = [target_r*np.cos(target_theta),
                  target_r*np.sin(target_theta)]
        self.reset_pos()


env = CircularEnvironment(seed=3)
model = nengo.Network(seed=5)
with model:
    env = env.make_node()
    
    ens = nengo.Ensemble(n_neurons=100, dimensions=2)

