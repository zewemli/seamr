SEmantic Activity Modeling and Recognition
==========================================

Determine location of sensor

Estimate activity profile through a day

----------------

Estimating resident location(s) can be difficult. Creating an estimator
for location requires modeling the receptive field of PIR sensors as well updating
locations for sensors where direct interactions are needed.

Instead of hand-crafting these models, which can vary between environments
we will use a population-based model which self-supervises and thus adapts to
not only new environments, but also adapts to changes in the environment.

We use Adaptive Probabalistic Markov Models to simulate the many hypotheses
regarding a resident's current intentions, which are conditional on the activity
which the resident is either currently engaged in or in which they are about to engage.

These APMMs form the DNA of smart agents which use a neural network to adapt to local information.
Overall we use an evolutionary scheme to promote effective agents and encourage exploration
into alterantive hypotheses.