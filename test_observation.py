import sys
sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv
import os

print('Creating env...')
env = CppColonyEnv(map_size=10, curriculum_stage=0)
print('Created')

print('Setting step log...')
os.makedirs('logs', exist_ok=True)
env.set_step_log('logs/observation_debug.log')
print('Step log set')

print('Reset...')
obs, info = env.reset(seed=42)
print(f'Reset done, days={info["days"]}')

print('Step 1...')
action = env.action_space.sample()
obs, reward, done, trunc, info = env.step(action)
print(f'Step 1 done: action={info["action_name"]}, reward={reward:.2f}')
print('OK')
