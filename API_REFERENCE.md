# J-PARSE API Reference

## `jparse.JParseCore`

Pure J-PARSE algorithm. Only requires numpy.

```python
solver = jparse.JParseCore(gamma=0.1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 0.1 | Singularity threshold (0 < gamma < 1). Directions with σᵢ/σₘₐₓ < gamma are treated as singular. |

### `solver.compute(jacobian, ...)`

Compute the J-PARSE pseudo-inverse of a Jacobian matrix.

```python
J_parse = solver.compute(J)
J_parse, nullspace = solver.compute(J, return_nullspace=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jacobian` | ndarray | required | m × n Jacobian matrix |
| `singular_direction_gain_position` | float | 1.0 | Gain for position singular directions |
| `singular_direction_gain_angular` | float | 1.0 | Gain for angular singular directions |
| `position_dimensions` | int | None | Number of position rows (e.g., 3 for 3D) |
| `angular_dimensions` | int | None | Number of angular rows (e.g., 3 for 3D) |
| `return_nullspace` | bool | False | Also return nullspace projection matrix |

**Returns:**
- `J_parse` (ndarray): n × m J-PARSE pseudo-inverse matrix
- `nullspace` (ndarray, optional): n × n nullspace projection matrix

### `solver.pinv(jacobian)`

Standard Moore-Penrose pseudo-inverse (for comparison).

**Returns:** n × m pseudo-inverse matrix

### `solver.damped_least_squares(jacobian, damping=0.01)`

Damped least squares pseudo-inverse (for comparison).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jacobian` | ndarray | required | m × n Jacobian matrix |
| `damping` | float | 0.01 | Damping factor λ |

**Returns:** n × m DLS pseudo-inverse matrix

---

## `jparse.Robot`

High-level robot interface with URDF support (requires Pinocchio).

```python
robot = jparse.Robot.from_urdf("robot.urdf", "base_link", "ee_link", gamma=0.1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urdf` | str | required | Path to URDF file or XML string |
| `base_link` | str | required | Name of base link |
| `end_link` | str | required | Name of end-effector link |
| `gamma` | float | 0.1 | J-PARSE singularity threshold |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_joints` | int | Number of actuated joints |
| `gamma` | float | Current singularity threshold (settable) |

### `robot.jacobian(q)`

Compute the 6 × n geometric Jacobian.

**Returns:** 6 × n Jacobian matrix (rows 0-2: linear, rows 3-5: angular)

### `robot.jparse(q, ...)`

Compute J-PARSE pseudo-inverse at configuration q.

```python
J_parse = robot.jparse(q)
J_parse = robot.jparse(q, position_only=True)  # 3D position only
J_parse, nullspace = robot.jparse(q, return_nullspace=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | ndarray | required | Joint configuration |
| `position_only` | bool | False | Use only position rows (3×n) |
| `return_nullspace` | bool | False | Also return nullspace matrix |
| `singular_direction_gain_position` | float | 1.0 | Position gain |
| `singular_direction_gain_angular` | float | 1.0 | Angular gain |

**Returns:** J-PARSE pseudo-inverse (and optionally nullspace)

### `robot.forward_kinematics(q)`

Compute end-effector pose.

**Returns:** `(position, rotation)` - 3D position and 3×3 rotation matrix

### `robot.manipulability(q)`

Compute Yoshikawa's manipulability measure: √det(JJᵀ)

**Returns:** float (higher = better conditioned)

### `robot.inverse_condition_number(q)`

Compute σₘᵢₙ/σₘₐₓ of the Jacobian.

**Returns:** float in [0, 1] (0 = singular, 1 = isotropic)

---

## Utility Functions

```python
# Manipulability measure
m = jparse.manipulability_measure(J)  # √det(JJᵀ)

# Inverse condition number
icn = jparse.inverse_condition_number(J)  # σₘᵢₙ/σₘₐₓ
```

---

## ROS Integration

```python
from jparse_robotics.ros import ROSRobot

robot = ROSRobot.from_parameter_server("base_link", "ee_link", gamma=0.1)
robot.publish_ellipsoids(q, end_effector_pose)  # Visualize in RViz
```
