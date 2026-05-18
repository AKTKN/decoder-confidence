Before run simulation, use need to generate stim circuits. Simulator load the circuit when execute sampling.
How to indicate the circuit information for simulation?

**Supported codes**  
    - `surface_code_Z`  
    - `superdense_color_code_Z`  
    - `bivariate_bicycle_code_Z`  

(Z means the basis of observable used in memory experiment)

Then, you need to specify detail parameters as folows:  
    - distance :code distance for code.  
    - rounds : the number of rounds of syndrome extraction.  
    - noise_model : noise model you want to insert.  
    - noise_strength : error rate of noise model.  

If you specify `bivariate_bicycle_code_Z`, you must specify additional parameters related to BB code like:  
    - l: 12  
    - m: 6  
    - a_x_pows: [3]  
    - a_y_pows: [1, 2]  
    - b_x_pows: [1, 2]  
    - b_y_pows: [3]  
(This is an example to generate $[[144, 12, 12]]$ BB code.)

You can generate specified stim circuit by running the following shell script(example): 
``` bash
./scripts/generate_circuits \
    -- code surface_code_Z \
    -- out_dir circuits/surface \
    -- noise_model uniform \
    -- rounds "d" \ 
    -- distance {1} \
    -- noise_strength {2} \
    ::: 3 5 7   # you can generate collectively
    ::: 0.001 0.002 0.005
```
- rounds are specified int or "{int}*d" where "d" is distance. For example, "4*d" means 4 * (distance) rounds. If you want to specify (distance) rounds, set "d" rather "1*d". 

**File format**
code={code},d={distance},rounds={rounds},noisemodel={noise_model},p={noise_strength}.stim


First, we implement wrapper, then build circuit generator module in src.