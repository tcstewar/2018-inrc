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
    
    wander = nengo.Ensemble(10, 2)
    def wander_func(x):
        norm = np.linalg.norm(x)
        return -x[1]/norm, x[0]/norm

    nengo.Connection(env[:2], wander, function=wander_func)
    nengo.Connection(wander, env)

    search = nengo.Ensemble(30, 2)
    synapse = 0.1
    def oscillator(x):
        r = 10
        s = 20
        return [synapse * (-x[1] * s + x[0] * (r - x[0]**2 - x[1]**2)) + x[0],
                synapse * ( x[0] * s + x[1] * (r - x[0]**2 - x[1]**2)) + x[1]]

    nengo.Connection(search, search, function=oscillator, synapse=0.1)
    nengo.Connection(search, env, transform=0.5)    
    

    memory = nengo.Ensemble(n_neurons=200, dimensions=2, seed=1)
    nengo.Connection(memory, memory, synapse=0.1)
    
    
    go_to_target = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(memory, go_to_target)
    nengo.Connection(env[:2], go_to_target, transform=-1)

    mem_update = nengo.Ensemble(n_neurons=100, dimensions=2)
    nengo.Connection(env[:2], mem_update, transform=3)
    nengo.Connection(memory, mem_update, transform=-3)
    nengo.Connection(mem_update, memory, synapse=0.1, transform=1)

    nengo.Connection(go_to_target, env, transform=5, synapse=0.5)


    do_memory = nengo.Ensemble(n_neurons=100, dimensions=1)
    do_search = nengo.Ensemble(n_neurons=100, dimensions=1)
    bias = nengo.Node(1)
    nengo.Connection(bias, do_search)
    
    nengo.Connection(env[2], do_memory, synapse=0.1)
    nengo.Connection(env[2], do_search, transform=-1, synapse=0.1)
    
    nengo.Connection(do_memory, mem_update.neurons,
                     transform=-1*np.ones((mem_update.n_neurons, 1)))
    nengo.Connection(do_memory, wander.neurons,
                     transform=-1*np.ones((wander.n_neurons, 1)))
    nengo.Connection(do_memory, search.neurons,
                     transform=-1*np.ones((search.n_neurons, 1)))
    nengo.Connection(do_search, go_to_target.neurons,
                     transform=-1*np.ones((go_to_target.n_neurons, 1)))
    

    

