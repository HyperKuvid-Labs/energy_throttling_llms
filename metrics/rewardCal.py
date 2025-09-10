#So here we get all the rewards frim dufferenet sources and combine them

#Lets start with teh cpu_metrics:



from cpu_reward import get_cpu_reward
from eagle_reward import get_eagle_reward
from gpu_reward import get_gpu_reward
from model_reward import get_model_reward
from runtime_reward import get_runtime_reward


cpu_reward = get_cpu_reward()

stats = {"branches": 12, "accepted": 90, "attempted": 100, "cascades": 1}
eagle_reward = get_eagle_reward(stats)

gpu_reward = get_gpu_reward()


metrics = runtime_tracker.finalize(avg_power_w=150)
runtime_reward = get_runtime_reward(metrics)

#constants 
alpha=0.0
beta=0.0
gamma=0.0
delta=0.0
eplison=0.0


#the final reward calcualtion where we can dyamically change the hyperparameters
final_reward = (alpha * cpu_reward +
                beta * eagle_reward +
                gamma * gpu_reward +
                delta * model_reward +
                eplison * runtime_reward)


