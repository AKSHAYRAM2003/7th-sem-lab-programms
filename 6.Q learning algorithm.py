import numpy as np
# Grid world settings
grid_size = 5
num_actions = 4 # up, down, left, right
num_states = grid_size * grid_size
# Q-learning parameters
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 1000
# Initialize Q-values
Q = np.zeros((num_states, num_actions))
# Obstacle and goal positions
obstacle_positions = [(1, 1), (2, 2), (3, 3)]
goal_position = (4, 4)
# Convert (row, column) coordinates to state index
def state_from_position(position):
 return position[0] * grid_size + position[1]
# Q-learning algorithm
for episode in range(num_episodes):
 current_position = (0, 0) # Starting position
 while current_position != goal_position:
  state = state_from_position(current_position)
 valid_actions = [action for action in range(num_actions) if current_position != obstacle_positions]
 action = np.random.choice(valid_actions) 
 next_row, next_col = current_position
 if action == 0: # Move up
  next_row = max(next_row - 1, 0)
 elif action == 1: # Move down
  next_row = min(next_row + 1, grid_size - 1)
 elif action == 2: # Move left
  next_col = max(next_col - 1, 0)
 else: # Move right
  next_col = min(next_col + 1, grid_size - 1)
 next_state = state_from_position((next_row, next_col))
 if next_state == state_from_position(goal_position):
  reward = 10
 elif next_state in [state_from_position(pos) for pos in obstacle_positions]:
  reward = -5
 else:
  reward = 0 
 # Update Q-value using Q-learning equation
 Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * 
np.max(Q[next_state, :])) 
 current_position = (next_row, next_col)
# Test the learned policy
current_position = (0, 0)
path = [(0, 0)]
while current_position != goal_position:
 state = state_from_position(current_position)
 action = np.argmax(Q[state, :])
 next_row, next_col = current_position
 if action == 0: # Move up
  next_row = max(next_row - 1, 0)
 elif action == 1: # Move down
  next_row = min(next_row + 1, grid_size - 1)
 elif action == 2: # Move left
  next_col = max(next_col - 1, 0)
 else: # Move right
  next_col = min(next_col + 1, grid_size - 1)
 path.append((next_row, next_col))
 current_position = (next_row, next_col)
print("Learned Policy Path:")
for row in range(grid_size):
 for col in range(grid_size):
  if (row, col) == goal_position:
   print(" G ", end='')
  elif (row, col) in obstacle_positions:
   print(" X ", end='')
  elif (row, col) in path:
   print(" * ", end='')
 else:
  print(" . ", end='')
  print()
