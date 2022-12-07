# Dynamic Time Warping

Implementing dynamic time warping in python.

## How to use

Clone the repository to a folder named `dtw`

```
$ git clone https://github.com/nikhilsrajan/dtw.git
```

Import into your python file and use the functions

```python
import dtw

...
```

## Examples

### Example 1: Same length timeseries

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import dtw


x = np.linspace(0,np.pi*2,25)
y1 = np.sin(x)
y2 = np.cos(x/1.5-0.4) + 0.4


# plot y1 and y2
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x, y1, 'o-', label='y1')
ax.plot(x, y2, 'o-', label='y2')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plotting y1 and y2')
```

![fig 1](./doc/fig1.png)

```python
# compute dtw distance
dtw_distance = dtw.distance(y1, y2) # ~= 8.2428

# get cost_matrix (not needed, only for visualisation)
cost_matrix = dtw.compute_accumulated_cost_matrix(y1, y2)

# get warp_indexes (not needed, only for visualisation)
warp_indexes = dtw.trace_warp_indexes(cost_matrix)


# plotting y1 and y2 with warp_indexes
fig, ax = plt.subplots(figsize=(10,5))
for idx_1, idx_2 in warp_indexes:
    ax.plot([x[idx_1], x[idx_2]], [y1[idx_1], y2[idx_2]], 'k--')
ax.plot(x, y1, 'o-', label='y1')
ax.plot(x, y2, 'o-', label='y2')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plotting y1 and y2 and the dtw warping order')
```

![fig 2](./doc/fig2.png)

```python
# plotting cost_matrix with warp path highlighting
g = sns.heatmap(cost_matrix.T)
_ = g.set_title('Cost matrix with warp path highlighting')
for idx_1, idx_2 in warp_indexes:
    g.add_patch(Rectangle((idx_1, idx_2), 1, 1, fill=True, color='yellow'))
```

![fig 3](./doc/fig3.png)


### Example 2: Different length timeseries

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import dtw


x = np.linspace(0,np.pi*2,50)
y_snippet = np.sin(x)
y_pulse = np.concatenate([np.full(75, 0), 1.25*y_snippet, np.full(25, 0)])


# plot y1 and y2
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(y_pulse, 'o-', label='y_pulse')
ax.plot(y_snippet, 'o-', label='y_snippet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plotting y_pulse and y_snippet')
```

![fig 4](./doc/fig4.png)

```python
# compute dtw distance
dtw_distance = dtw.distance(y_pulse, y_snippet) # ~= 4.7965

# get cost_matrix (not needed, only for visualisation)
cost_matrix = dtw.compute_accumulated_cost_matrix(y_pulse, y_snippet)

# get warp_indexes (not needed, only for visualisation)
warp_indexes = dtw.trace_warp_indexes(cost_matrix)


# plotting y1 and y2 with warp_indexes
fig, ax = plt.subplots(figsize=(15,5))
for idx_1, idx_2 in warp_indexes:
    ax.plot([idx_1, idx_2], [y_pulse[idx_1], y_snippet[idx_2]], 'k--')
ax.plot(y_pulse, 'o-', label='y_pulse')
ax.plot(y_snippet, 'o-', label='y_snippet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plotting y_pulse and y_snippet and the dtw warping order')
```

![fig 5](./doc/fig5.png)

```python
# plotting cost_matrix with warp path highlighting
scale = 5
M, N = cost_matrix.shape
fig, ax = plt.subplots(figsize=(scale*M/N,scale))
g = sns.heatmap(cost_matrix.T)
_ = g.set_title('Cost matrix with warp path highlighting')
for idx_1, idx_2 in warp_indexes:
    g.add_patch(Rectangle((idx_1, idx_2), 1, 1, fill=True, color='yellow'))
```

![fig 6](./doc/fig6.png)


### Example 3: Different length timeseries, longer one with noise

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import dtw


x = np.linspace(0,np.pi*2,50)
y_snippet = np.sin(x)
y_pulse = np.concatenate([np.full(75, 0), 1.25*y_snippet, np.full(25, 0)])
y_error = 0.1 * np.random.normal(0,1, y_pulse.shape)
y_pulse_e = y_pulse + y_error


# plot y1 and y2
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(y_pulse_e, 'o-', label='y_pulse_e')
ax.plot(y_snippet, 'o-', label='y_snippet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plotting y_pulse_e and y_snippet')
```

![fig 7](./doc/fig7.png)

```python
# compute dtw distance
dtw_distance = dtw.distance(y_pulse_e, y_snippet) # ~= 13.0214

# get cost_matrix (not needed, only for visualisation)
cost_matrix = dtw.compute_accumulated_cost_matrix(y_pulse_e, y_snippet)

# get warp_indexes (not needed, only for visualisation)
warp_indexes = dtw.trace_warp_indexes(cost_matrix)


# plotting y1 and y2 with warp_indexes
fig, ax = plt.subplots(figsize=(15,5))
for idx_1, idx_2 in warp_indexes:
    ax.plot([idx_1, idx_2], [y_pulse[idx_1], y_snippet[idx_2]], 'k--')
ax.plot(y_pulse_e, 'o-', label='y_pulse_e')
ax.plot(y_snippet, 'o-', label='y_snippet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plotting y_pulse_e and y_snippet and the dtw warping order')
```

![fig 8](./doc/fig8.png)

```python
# plotting cost_matrix with warp path highlighting
scale = 5
M, N = cost_matrix.shape
fig, ax = plt.subplots(figsize=(scale*M/N,scale))
g = sns.heatmap(cost_matrix.T)
_ = g.set_title('Cost matrix with warp path highlighting')
for idx_1, idx_2 in warp_indexes:
    g.add_patch(Rectangle((idx_1, idx_2), 1, 1, fill=True, color='yellow'))
```

![fig 9](./doc/fig9.png)
