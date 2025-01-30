import jax.numpy as jp
from flax import struct


def outclip(val, amin, amax):
    is_fine = jp.logical_or(val < amin, amax < val)

    is_closer_to_amin = jp.abs(val - amin) < jp.abs(val - amax)

    return val * is_fine + (1-is_fine) * (
        amin * is_closer_to_amin + (1 - is_closer_to_amin) * amax
    )


def innerbound(oldval, newval, amin, amax):
    select_min = jp.abs(oldval - amin) < jp.abs(oldval - amax)

    direction = newval - oldval


    direction = newval - oldval

@struct.dataclass
class Control2DState:
    x: float = 1
    y: float = 0

    position_velocity: float = 0.0 # 1.0
    #position_acceleration: float = 0

    heading: float = 0 #jp.pi

    goal_pos: jp.ndarray = jp.array([180,0])
    goal_size: float = 10

    last_action: jp.ndarray = jp.array([0., 0.])

    #angular_velocity: float = 0
    #angular_acceleration: float = 0

    #extremum_positional_acceleration: float = 0.1
    #extremum_angular_acceleration: float = 0.1
    extremum_positional_velocity: float = 1.0
    #extremum_angular_velocity: float = 0.1
    margin_collision: float = 0.1

    collision_boxes_centers: jp.ndarray = jp.array([[1000,1000]]) #jp.array([[30,0]])#, [60,40], [60, -40]])
    collision_boxes_sizes: jp.ndarray = jp.array([[2, 2]]) #jp.array([[5,40]])#, [5,60], [5,60]])

    def observation_size(self):
        return self.array_state.shape[0]-2

    def action_size(self):
        return 2

    def is_inside_box(self, xy, box_center, box_size):
        #box_center = jp.array([50, 0])  # Center of the box
        box_half_size = box_size / 2 # jp.array([2.5, 10])  # Half-width and half-height
        box_min = box_center - box_half_size
        box_max = box_center + box_half_size

        # Check if inside the box
        inside_box = jp.logical_and(xy >= box_min, xy <= box_max)
        is_inside = jp.all(inside_box)
        return is_inside

    def is_inside_boxes(self, xy, boxes_center, boxes_size):
        xy = jp.atleast_2d(xy)
        boxes_center = jp.atleast_2d(boxes_center)
        boxes_size = jp.atleast_2d(boxes_size)

        def foreach_box(xy, box_center, box_size):
            return jax.vmap(functools.partial(self.is_inside_box, box_center=box_center, box_size=box_size))(xy=xy)

        ret = jax.vmap(functools.partial(foreach_box, xy))(boxes_center, boxes_size)
        return ret.any(axis=0)


    @property
    def _raycasting_worst_case_num_steps(self):
        # velocity-based
        return 30 #self.extremum_positional_velocity * 2 / self.margin_collision

    @property
    def pos(self):
        return jp.array([self.x, self.y])

    @property
    def array_state(self):
        return jp.array([self.x, self.y, self.position_velocity, self.heading, self.last_action[0], self.last_action[1]])

    @property
    def is_on_goal(self):
        distance_to_goal = jp.linalg.norm(self.pos - self.goal_pos)
        is_within_goal = distance_to_goal < self.goal_size
        return is_within_goal

    @property
    def reward(self):
        MAX_DISTANCE = jp.linalg.norm(jp.array([0, 50]) - self.goal_pos)
        distance_to_goal = jp.linalg.norm(self.pos - self.goal_pos)
        distance_reward = (MAX_DISTANCE - distance_to_goal) / MAX_DISTANCE

        goal_reward = 100 * self.is_on_goal

        return distance_reward + goal_reward

    def apply_topology(self, x, y, margin=0.1):
        current_pos = jp.array([x, y])
        old_pos = jp.array([self.x, self.y])

        direction = current_pos - old_pos
        direction = direction / jp.linalg.norm(direction)

        distance = jp.linalg.norm(current_pos - old_pos)
        distance_per_step = distance / self._raycasting_worst_case_num_steps
        steps = jp.arange(self._raycasting_worst_case_num_steps) * distance_per_step

        def mul(direction, step):
            return direction * step

        step_points = jax.vmap(functools.partial(mul, direction))(steps)  # * direction
        step_points = jp.nan_to_num(step_points, nan=0.0) + old_pos

        insideness = self.is_inside_boxes(step_points, self.collision_boxes_centers, self.collision_boxes_sizes)
        #insideness = jax.vmap(functools.partial(self.is_inside_box, box_center=jp.array([50,0]), box_size=jp.array([5, 40])))(xy=step_points)

        collided_pos =  step_points[jp.argmax(insideness) - 1]

        was_inside_box = self.is_inside_boxes(current_pos, self.collision_boxes_centers, self.collision_boxes_sizes).squeeze()


        new_pos = collided_pos * was_inside_box + (1 - was_inside_box) * current_pos

        x, y = new_pos[0], new_pos[1]

        is_inside_xbound = jp.logical_and(x >= 0, x <= 100)
        is_inside_ybound = jp.logical_and(y >= -50, y <= 50)

        was_bad_position = jp.logical_or(was_inside_box, jp.logical_not(is_inside_xbound))
        was_bad_position = jp.logical_or(was_bad_position, jp.logical_not(is_inside_ybound))

        return x.clip(0, 100), y.clip(-50, 50), was_bad_position

    def forward(self, action):
        # Unpack the action
        position_vel_delta, heading_delta = action

        # Update accelerations
        #position_acceleration = jp.clip(self.position_acceleration + position_accel_delta, -self.extremum_positional_acceleration, self.extremum_positional_acceleration)
        #angular_acceleration = jp.clip(self.angular_acceleration + heading_accel_delta, -self.extremum_angular_acceleration, self.extremum_angular_acceleration)

        # Update velocities
        position_velocity = jp.clip(self.position_velocity + position_vel_delta, 0, self.extremum_positional_velocity) #jp.clip(self.position_velocity + position_acceleration, 0, self.extremum_positional_velocity)
        #angular_velocity = jp.clip(self.angular_velocity + angular_acceleration, -self.extremum_angular_velocity, self.extremum_angular_velocity)

        # Update heading (angular displacement)
        heading = jp.mod(self.heading + heading_delta, 2*jp.pi)
        self = self.replace(heading=heading)

        # Update position based on current heading and velocity
        x = self.x + position_velocity * jp.cos(self.heading)
        y = self.y + position_velocity * jp.sin(self.heading)

        x = jnp.clip(x, 0, 200)
        y = jnp.clip(y, -50, 50)

        new_x = x * (1-self.is_on_goal) + self.goal_pos[0] * self.is_on_goal
        new_y = y * (1-self.is_on_goal) + self.goal_pos[1] * self.is_on_goal

        return self.replace(
            #position_acceleration=set_to_0_if_inside(position_acceleration),
#            angular_acceleration=set_to_0_if_inside(angular_acceleration),
            position_velocity=position_velocity, #set_to_0_if_inside(position_velocity),
 #           angular_velocity=set_to_0_if_inside(angular_velocity),
            heading=heading,
            x=new_x,
            y=new_y,
            last_action=action
        )


import functools

import flax.linen as nn
import jax
import jax.numpy as jnp


class Block(nn.Module):
  features: int


  @nn.compact
  def __call__(self, x, training: bool, activation):
    x = nn.Dense(self.features)(x)
    x = activation(x)
    #x = jax.nn.sigmoid(x)
    return x

class Model(nn.Module):
  dmid: int
  dout: int

  @nn.compact
  def __call__(self, x, training: bool):
    x = Block(self.dmid)(x, training, activation=nn.leaky_relu)
    x = Block(self.dmid)(x, training, activation=nn.leaky_relu)
    x = Block(self.dout)(x, training, activation=nn.tanh)

    def remap_vel(x):
        return x.at[0].set(x[0] / 2).at[1].set(x[1] / 4)

    if len(x.shape) == 2:
        x = jax.vmap(remap_vel)(x)
    else:
        x = remap_vel(x)
    return x

@functools.partial(jax.jit, static_argnums=(1,2))
def traverse_variables(variables, get=False, set_func=None, key=None):
    parameters = []
    i = 0
    ret_params = {}

    params = variables["params"]
    for block, v in params.items():
        for dense, v2 in v.items():
            #kernel = v2["kernel"]

            if get:
                parameters.append(v2["kernel"])
                parameters.append(v2["bias"])
            if set_func is not None:
                if key is not None:
                    key, rng = jax.random.split(key)
                else:
                    rng = key
                kernel = set_func(rng, i, v2["kernel"])
                i = i + 1

                if key is not None:
                    key, rng = jax.random.split(key)
                else:
                    rng = key
                bias = set_func(rng, i, v2["bias"])
                i = i + 1

                ret_params[block] = {dense: {"kernel": kernel, "bias": bias}}
    ret = {"params": ret_params}
    if get:
        ret = (ret, parameters)

    return ret

def gen_model(key, minvar, maxvar, hidden_size=16, outsize=2):
    model = Model(hidden_size, outsize)

    insize = Control2DState().array_state.shape[0]-2

    sample_x = jnp.ones((1, insize))
    key, rng = jax.random.split(key)

    variables = model.init(rng, sample_x, training=False)
    variables, kernels = traverse_variables(variables, get=True, set_func=(lambda k,i,kernel: jax.random.uniform(k, minval=minvar, maxval=maxvar, shape=kernel.shape)), key=key)

    kernel_shapes = tuple([x.shape for x in kernels])

    def size_as_flat(k):
        if len(k.shape) == 1:
            return k.shape[0]
        assert len(k.shape) == 2
        return k.shape[0] * k.shape[1]

    number_of_variables = sum([size_as_flat(x) for x in kernels])

    return model, variables, kernel_shapes, number_of_variables

@jax.jit
def recenter(particle, min_param, max_param):
    # recenter to [0,1] because it empirically gives nice RND values
    particle = (particle - min_param) / (max_param - min_param)
    return particle

def eval_mlp(model, vector, example_variables, kernel_shapes, num_steps=500):
    variables = coerce_vector_into_variables(vector, example_variables, kernel_shapes)

    def scannable(variables, carry, x):
        sim_state = carry
        array_state = sim_state.array_state[:-2]    # removes last_action

        action = model.apply(variables, array_state, training=False)

        #action = action.at[0].set(recenter(action[0], 0, 1))

        new_sim_state = sim_state.forward(action) #jp.zeros(2))#action)

        return new_sim_state, (new_sim_state.reward, new_sim_state.array_state)

    final_sim_state, (rewards, array_states) = jax.lax.scan(
        functools.partial(scannable, variables),
        Control2DState(),
        jp.arange(num_steps),
        unroll=5    # arbitrary
    )

    return rewards, array_states

def eval_actions(actions):
    maxvel = 5.0

    def scannable(carry, action):
        sim_state = carry
        new_sim_state = sim_state.forward(action) #jp.zeros(2))#action)
        return new_sim_state, (new_sim_state.reward, new_sim_state.array_state)

    final_sim_state, (rewards, array_states) = jax.lax.scan(
        scannable,
        Control2DState(extremum_positional_velocity=maxvel),
        actions,
        unroll=5    # arbitrary
    )

    return rewards, array_states

@functools.partial(jax.jit, static_argnums=(2,))
def coerce_vector_into_variables(vector, true_variables, true_kernel_shapes):
    def set_func(_, i, old_kernel, new_vars, true_kernel_shapes):
        def get_shape_for_i(j):
            return true_kernel_shapes[j]

        def get_num_vars_for_i(j):
            shape = get_shape_for_i(j)

            if len(shape) == 1:
                shape = (1,shape[0])

            return shape[0] * shape[1]

        def num_throwaway_before_i(j):
            throwaway = 0
            for i in range(j):
                throwaway += get_num_vars_for_i(i)
            return throwaway

        new_vars = new_vars[num_throwaway_before_i(i):num_throwaway_before_i(i) + get_num_vars_for_i(i)]
        new_vars = new_vars.reshape(get_shape_for_i(i))

        return new_vars

    variables, kernels = traverse_variables(true_variables,
                                            get=True,
                                            set_func=functools.partial(set_func, new_vars=vector, true_kernel_shapes=true_kernel_shapes),
                                            key=jax.random.PRNGKey(22331))
    return variables



if __name__ == "__main__":
    import pygame

    # Constants for the GUI
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    MAP_WIDTH = 100
    MAP_HEIGHT = 100
    #BOX_CENTER = (50, 0)
    #BOX_WIDTH = 5
    #BOX_HEIGHT = 40

    # Scaling factor to convert map coordinates to screen coordinates
    SCALE_X = SCREEN_WIDTH / MAP_WIDTH
    SCALE_Y = SCREEN_HEIGHT / MAP_HEIGHT


    # Convert map coordinates to screen coordinates
    def map_to_screen(x, y):
        screen_x = x * SCALE_X
        screen_y = SCREEN_HEIGHT / 2 - y * SCALE_Y
        return int(screen_x), int(screen_y)

    # Initialize Control2DState
    state = Control2DState()

    # Draw the map boundaries and the box
    def draw_map(screen):
        # Draw the map background
        screen.fill((220, 220, 220))  # Light gray background

        def draw_box(center, size):
            BOX_CENTER = center
            BOX_WIDTH, BOX_HEIGHT = size

            # Draw the box
            box_left, box_top = map_to_screen(BOX_CENTER[0] - BOX_WIDTH / 2, BOX_CENTER[1] + BOX_HEIGHT / 2)
            box_right, box_bottom = map_to_screen(BOX_CENTER[0] + BOX_WIDTH / 2, BOX_CENTER[1] - BOX_HEIGHT / 2)
            pygame.draw.rect(screen, (255, 0, 0), (box_left, box_top, box_right - box_left, box_bottom - box_top))
        for centers, sizes in zip(state.collision_boxes_centers, state.collision_boxes_sizes):
            draw_box(centers, sizes)

        # Draw the goal
        green_circle_x, green_circle_y = map_to_screen(90, 0)
        pygame.draw.circle(screen, (0, 255, 0), (green_circle_x, green_circle_y), 10 * SCALE_X)  # Green circle

        # Draw the map border
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 2)


    def draw_text(screen, text, position, font_size=24):
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, (0, 0, 0))  # Black color for the text
        screen.blit(text_surface, position)


    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Control2DState GUI")
    clock = pygame.time.Clock()

    running = True
    t = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get user input (arrow keys for movement)
        keys = pygame.key.get_pressed()
        action = [0.0, 0.0]
        if keys[pygame.K_UP]:
            action[0] += 0.02  # Increase position acceleration
        if keys[pygame.K_DOWN]:
            action[0] -= 0.02  # Decrease position acceleration
        if keys[pygame.K_LEFT]:
            action[1] -= 0.1  # Increase heading acceleration (turn left)
        if keys[pygame.K_RIGHT]:
            action[1] += 0.1  # Decrease heading acceleration (turn right)

        # Update the state
        state = state.forward(action)


        # Draw the map and the particle
        draw_map(screen)
        particle_x, particle_y = map_to_screen(state.x, state.y)
        pygame.draw.circle(screen, (0, 0, 255), (particle_x, particle_y), 5)  # Blue particle


        draw_text(screen, f"Position: ({state.x:.2f}, {state.y:.2f})", (20, 20), 24)
        draw_text(screen, f"Heading: {state.heading:.2f}", (20, 50), 24)
        draw_text(screen, f"Pos Vel: {state.position_velocity:.2f}", (20, 80), 24)
#        draw_text(screen, f"Pos Acc: {state.position_acceleration:.2f}", (20, 110), 24)
        draw_text(screen, f"Reward: {state.reward:.2f}", (20, 110), 24)
        draw_text(screen, f"T: {t}", (20, 140), 24)


        # Update the display
        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS
        t = t+1

    pygame.quit()