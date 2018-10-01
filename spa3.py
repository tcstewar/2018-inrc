import nengo_spa as spa
s = spa.sym

model = spa.Network()
with model:
    D = 32
    memory = spa.State(D, label='memory', feedback=1)

    with spa.ActionSelection():
        spa.ifmax(memory@s.A, s.B >> memory)
        spa.ifmax(memory@s.B, s.C >> memory)
        spa.ifmax(memory@s.C, s.D >> memory)
        spa.ifmax(memory@s.D, s.E >> memory)
        spa.ifmax(memory@s.E, s.A >> memory)
        
print('number of neurons:', model.n_neurons)        

    
