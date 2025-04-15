# Tiny Brain

## Neural Network

a basic cs231 neuron could be modelled by

`o = f ( sum (xi * wi ) + b )`


where:

`xi` is the input

`wi` is the synapse

`wi*xi` is the dendrite

`sum (xi * wi ) + b` is the cell body

`f` is the activation function

`o` is the output axon


#### Activation function
the activation should squash the values coming form the cell body function.

**hyperbolic tangent function**

`tanh(x) = (e^x - 1) / (e^x + 1)`



### Backpropagation to train the brain

#### Derivative
the definition of derivative

L =   lim     f(a + h) - f(a)  / h
h -> 0

giving a function
e = a*b
d = e + c
L = d * f

backpropagation means starting form L we are going to reverse and calculate the gradient in all intermediate steps
we can think L as the loss function of our neural network.

to compute the gradient starting from L we know that
dL / dd = f
dL / df = d

so we can think that
dL/dL = 1
dL/dd = f
dL/df = d

we know the derivative of the sum is always equal to 1
dd/dc = 1
dd/de = 1
now we have to compute dL/dc and to do this we have to consider the chain rule
dz/dx = dz/dy * dy/dx

so:
dL/dc = dL/dd * dd/dc

considering the neuron

n = sum (xi * wi ) + b
o = tanh ( n )

we know that
dtanh/dn = 1 - tanh(n)^2
but
tanh(n)=o
so the tanh local derivative is:
dtanh/dn = 1 - o^2



