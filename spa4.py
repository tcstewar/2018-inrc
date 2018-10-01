import nengo
import nengo_spa as spa
s = spa.sym


model = spa.Network()
with model:
    D = 64
    vision = spa.State(D, label='vision')
    memory = spa.State(D, label='memory', feedback=1)
    
    with nengo.Network() as control:
        with spa.ActionSelection():
            spa.ifmax(memory@s.A, s.B >> memory)
            spa.ifmax(memory@s.B, s.C >> memory)
            spa.ifmax(memory@s.C, s.D >> memory)
            spa.ifmax(memory@s.D, s.E >> memory)
            spa.ifmax(memory@s.E, vision >> memory)
            spa.ifmax(0.5, vision >> memory)
    
print('number of neurons:', model.n_neurons)