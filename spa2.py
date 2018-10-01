import nengo_spa as spa
s = spa.sym

model = spa.Network()
with model:
    D = 32
    vision = spa.State(D, label='vision')
    speech = spa.State(D, label='speech')
    
    with spa.ActionSelection():
        spa.ifmax(vision@s.DOG, s.BARK >> speech)
        spa.ifmax(vision@s.CAT, s.MEOW >> speech)
        spa.ifmax(vision@s.RAT, s.SQUEAK >> speech)
        spa.ifmax(vision@s.COW, s.MOO >> speech)
        spa.ifmax(0.5, s.UNKNOWN >> speech)
        
print('number of neurons:', model.n_neurons)
