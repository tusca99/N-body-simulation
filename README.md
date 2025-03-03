# N-body-simulation
N-body-simulation from a simple single-threaded brute force approach to more sofisticated distributed ones.


### First work
The first and simplest code is a code in C++ (simulation) and python (visualization) of the reconstruction of Voyager II spacecraft.
As the simulation did not account for the propelling properties of the Voyager II and it featured only a small amount of bodies (n~25) i focused on measuring the generated trajectory and the real one given the same initial (x,v) conditions of all bodies.

Later then I corrected the spacecraft trajectory each time it diverged by a given threshold and counted the number of corrections as well as the total energy gap for a wide number of cases changing both the time step and the correction threshold.

This is the inital work that I presented some years ago as a project for my bachelor course (i know, the code is very very *very* far from readable and organized, but it's pretty well memory optimized given my limited experience at the time).
All this initial work is under the **Voyager II folder**, keep in mind some variables or captions may be in Italian.


### Second revision of the first work
Now today, 3 years later I revised this code for another course project with the objective of expanding it further using different computational techniques as well leveraging these years of experience.
The first step has been re-organazing the original code in a tidier way, moving corresponding functions to each library and instantiating a class for the n-body system.

The code here uses a bit less memory as well, though it remains computationally inefficient at best (using the littlest amount of memory simply means saving each iteration to file, so of course we are pretty much limited by ofstream routines).
You can see all this material in the **first-revision** branch of this repository.


This is the base used for the second project mentioned above, here I started working on generalizing the simulator.
I do not longer need a specific initial condition of my particles but I can simply generate my initial system with a patternized initialization (so i can choose random, semi-random or maybe some astro patterns like galaxies).

Secondly I can start saving to RAM some intermediate computations so I can gradually remove the I/O limitations at the cost of RAM usage, though this has to be done carefully as I want to prepare it for a smooth transition to GPGPU memory management.

