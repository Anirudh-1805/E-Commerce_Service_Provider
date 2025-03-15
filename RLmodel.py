"""
This code contains an assumptions for different aspects of vehicles that can be changed according to the company and includes penalty factors for time delay and more co2 emsissions and high avg aqicn factor 
along the route,the penalties again based on what we thiink is ideal, but again the values can be optimized and its the function it performs which is optimal, it gets the different paremters from previous code as
input and suggests the route(among 2 road routes since osrm gives at max2) and vehicle to be used and compares it with the user's choice."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        return states, actions, rewards, next_states, dones

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class RouteVehicleRL:
    def __init__(self, state_size, action_size, vehicles):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.vehicles = vehicles
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_target_every = 5
        
        # Neural Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer()
        self.training_step = 0
    
    def normalize_state(self, state):
        """Normalize state values to a reasonable range"""
        normalizers = np.array([1000, 24, 1000, 500, 1000])  # For distance, time, CO2, AQI, weight
        return state / normalizers
    
    def get_state(self, route, cargo_weight):
        """Convert route and cargo information into state vector"""
        state = np.array([
            route['distance'],
            route['travel_time'],
            route['co2_emissions'],
            route['avg_aqi'],
            cargo_weight
        ])
        return self.normalize_state(state)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def calculate_reward(self, cost, adjusted_co2):
        """Calculate reward based on cost and CO2 emissions"""
        if cost == float('inf'):
            return -1000.0
        
        # Normalize and combine cost and CO2 components
        cost_component = -cost / 1000.0
        co2_component = -adjusted_co2 / 100.0
        
        # Weight the components
        reward = 0.6 * cost_component + 0.4 * co2_component
        return float(reward)
    
    def train(self, state, action, reward, next_state, done):
        """Train the model using experience replay"""
        self.memory.add(state, action, reward, next_state, done)
        self.training_step += 1
        
        if len(self.memory.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if self.training_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def calculate_penalty(route, vehicle, cargo_weight):
    """Calculate penalties and total cost for a route."""
    try:
        if cargo_weight > vehicle['max_weight']:
            return float('inf'), float('inf')

        distance = route['distance']
        co2_emissions_given = route['co2_emissions']
        travel_time = route['travel_time']
        avg_aqi = route['avg_aqi']
        co2_per_km = vehicle['co2_per_km']
        avg_speed = vehicle['avg_speed']

        expected_co2 = co2_per_km * distance
        co2_penalty = 1.2 * max(0, co2_emissions_given - expected_co2)
        aqi_penalty = 0.45 * avg_aqi
        expected_travel_time = distance / avg_speed
        travel_time_penalty = 1.1 * max(0, travel_time - expected_travel_time)

        adjusted_co2 = co2_emissions_given + co2_penalty + aqi_penalty + travel_time_penalty
        total_cost = adjusted_co2 + vehicle['maintenance_cost'] + vehicle['service_cost']

        return float(total_cost), float(adjusted_co2)
    except Exception as e:
        print(f"Error in calculate_penalty: {str(e)}")
        return float('inf'), float('inf')

def optimize_route_and_vehicle(route1, route2, cargo_weight, vehicles):
    """Use RL to optimize route and vehicle selection"""
    try:
        state_size = 5
        action_size = len(vehicles)
        rl_agent = RouteVehicleRL(state_size, action_size, vehicles)
        
        n_episodes = 150
        best_reward = float('-inf')
        best_vehicle = None
        best_route = None
        
        print("\nTraining RL model...")
        for episode in range(n_episodes):
            if episode % 25 == 0:
                print(f"Episode {episode}/{n_episodes}")
            
            for route in [route1, route2]:
                state = rl_agent.get_state(route, cargo_weight)
                action = rl_agent.select_action(state)
                vehicle = vehicles[action]
                
                cost, adjusted_co2 = calculate_penalty(route, vehicle, cargo_weight)
                reward = rl_agent.calculate_reward(cost, adjusted_co2)
                
                if reward > best_reward and cost != float('inf'):
                    best_reward = reward
                    best_vehicle = vehicle
                    best_route = route
                
                next_state = state
                done = True
                rl_agent.train(state, action, reward, next_state, done)
        
        return best_vehicle, best_route
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        return None, None

def recommend_vehicle_rl(route1, route2, cargo_weight, vehicles):
    """Recommend the best vehicle and route using reinforcement learning."""
    try:
        best_vehicle, best_route = optimize_route_and_vehicle(route1, route2, cargo_weight, vehicles)
        if best_vehicle is None:
            return None, float('inf'), float('inf')
        
        cost, adjusted_co2 = calculate_penalty(best_route, best_vehicle, cargo_weight)
        return best_vehicle, cost, adjusted_co2
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return None, float('inf'), float('inf')

def get_float_input(prompt, min_value=0):
    """Helper function to get validated float input"""
    while True:
        try:
            value = float(input(prompt))
            if value < min_value:
                print(f"Please enter a value greater than {min_value}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

def save_model(rl_agent, filename="route_optimizer_model.pth"):
    """Save the trained model"""
    try:
        torch.save({
            'policy_net_state_dict': rl_agent.policy_net.state_dict(),
            'target_net_state_dict': rl_agent.target_net.state_dict(),
            'optimizer_state_dict': rl_agent.optimizer.state_dict(),
            'epsilon': rl_agent.epsilon
        }, filename)
        print(f"\nModel saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model(rl_agent, filename="route_optimizer_model.pth"):
    """Load a trained model"""
    try:
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            rl_agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            rl_agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rl_agent.epsilon = checkpoint['epsilon']
            print(f"\nModel loaded successfully from {filename}")
            return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    return False

def main():
    
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=== Route and Vehicle Optimization System ===\n")
        
        print("Enter details for Route 1:")
        distance1 = get_float_input("Distance (km): ")
        travel_time1 = get_float_input("Travel time (hours): ")
        co2_emissions1 = get_float_input("CO₂ emissions (kg): ")
        avg_aqi1 = get_float_input("Average AQI: ")

        print("\nEnter details for Route 2:")
        distance2 = get_float_input("Distance (km): ")
        travel_time2 = get_float_input("Travel time (hours): ")
        co2_emissions2 = get_float_input("CO₂ emissions (kg): ")
        avg_aqi2 = get_float_input("Average AQI: ")

        cargo_weight = get_float_input("\nEnter the cargo weight (kg): ")

        vehicles = [
            {'name': 'Motorcycle', 'max_weight': 10, 'co2_per_km': 0.1, 'maintenance_cost': 5, 'service_cost': 2, 'avg_speed': 40},
            {'name': 'Car', 'max_weight': 50, 'co2_per_km': 0.2, 'maintenance_cost': 10, 'service_cost': 3, 'avg_speed': 60},
            {'name': 'Minivan', 'max_weight': 100, 'co2_per_km': 0.3, 'maintenance_cost': 20, 'service_cost': 5, 'avg_speed': 70},
            {'name': 'Truck', 'max_weight': 500, 'co2_per_km': 0.5, 'maintenance_cost': 40, 'service_cost': 10, 'avg_speed': 50},
            {'name': '18-Wheeler', 'max_weight': 1000, 'co2_per_km': 0.8, 'maintenance_cost': 100, 'service_cost': 20, 'avg_speed': 40},
        ]

        print("\nAvailable vehicles:")
        for i, vehicle in enumerate(vehicles):
            print(f"{i}. {vehicle['name']} (Max weight: {vehicle['max_weight']} kg)")

        route1 = {'distance': distance1, 'travel_time': travel_time1, 'co2_emissions': co2_emissions1, 'avg_aqi': avg_aqi1}
        route2 = {'distance': distance2, 'travel_time': travel_time2, 'co2_emissions': co2_emissions2, 'avg_aqi': avg_aqi2}

        print("\nOptimizing route and vehicle selection using RL...")
        recommended_vehicle, best_cost, best_adjusted_co2 = recommend_vehicle_rl(route1, route2, cargo_weight, vehicles)

        if recommended_vehicle is None or best_cost == float('inf'):
            print("\nNo feasible solution found. Please check cargo weight and try again.")
            return

        cost1, adjusted_co2_1 = calculate_penalty(route1, recommended_vehicle, cargo_weight)
        cost2, adjusted_co2_2 = calculate_penalty(route2, recommended_vehicle, cargo_weight)
        
        print("\n=== Optimization Results ===")
        print(f"\nRecommended Vehicle: {recommended_vehicle['name']}")
        print(f"Recommended Route: {'Route 1' if cost1 < cost2 else 'Route 2'}")
        
        if cost1 != float('inf') and cost2 != float('inf'):
            print(f"\nRoute 1 Metrics:")
            print(f"- Cost: {cost1:.2f}")
            print(f"- Adjusted CO₂: {adjusted_co2_1:.2f}")
            
            print(f"\nRoute 2 Metrics:")
            print(f"- Cost: {cost2:.2f}")
            print(f"- Adjusted CO₂: {adjusted_co2_2:.2f}")
            
            cost_optimization = abs((cost1 - cost2) / max(cost1, cost2) * 100)
            co2_optimization = abs((adjusted_co2_1 - adjusted_co2_2) / max(adjusted_co2_1, adjusted_co2_2) * 100)
            
            print("\n=== Route Analysis ===")
            print("\nRoute 1:")
            print(f"- Distance: {route1['distance']} km")
            print(f"- Travel time: {route1['travel_time']} hours")
            print(f"- CO₂ emissions: {route1['co2_emissions']} kg")
            print(f"- Average AQI: {route1['avg_aqi']}")

            print("\nRoute 2:")
            print(f"- Distance: {route2['distance']} km")
            print(f"- Travel time: {route2['travel_time']} hours")
            print(f"- CO₂ emissions: {route2['co2_emissions']} kg")
            print(f"- Average AQI: {route2['avg_aqi']}")

            print(f"\nDifference between routes:")
            print(f"- Cost difference: {cost_optimization:.1f}%")
            print(f"- CO₂ difference: {co2_optimization:.1f}%")
        
            better_route = "Route 1" if cost1 < cost2 else "Route 2"
            print(f"\nBased on analysis, {better_route} appears more optimal")

        print("\nNote: Lower values indicate better performance")

        # Get user's vehicle choice
        while True:
            try:
                user_choice = int(input("\nConsidering these routes, which vehicle would you choose? (Enter the number): "))
                if 0 <= user_choice < len(vehicles):
                    user_vehicle = vehicles[user_choice]
                    break
                else:
                    print("Please enter a valid vehicle number from the list above.")
            except ValueError:
                print("Please enter a valid number.")

        print("\n=== Comparison: RL vs. Human Choice ===")
        print(f"\nYour Choice: {user_vehicle['name']}")
        if cargo_weight > user_vehicle['max_weight']:
            print("Your chosen vehicle cannot handle the cargo weight!")
        else:
            print(f"Maximum cargo capacity: {user_vehicle['max_weight']} kg")
            print(f"CO₂ emissions per km: {user_vehicle['co2_per_km']} kg/km")
            print(f"Average speed: {user_vehicle['avg_speed']} km/h")

        print(f"\nRL Recommended Vehicle: {recommended_vehicle['name']}")
        print(f"Maximum cargo capacity: {recommended_vehicle['max_weight']} kg")
        print(f"CO₂ emissions per km: {recommended_vehicle['co2_per_km']} kg/km")
        print(f"Average speed: {recommended_vehicle['avg_speed']} km/h")

        if user_vehicle['name'] == recommended_vehicle['name']:
            print("\nCongratulations! Your choice matches the RL recommendation!")
        else:
            print("\nYour choice differs from the RL recommendation.")
            if user_vehicle['co2_per_km'] > recommended_vehicle['co2_per_km']:
                print(f"The RL recommendation has lower CO₂ emissions per km.")
            if user_vehicle['max_weight'] < recommended_vehicle['max_weight']:
                print(f"The RL recommendation has higher cargo capacity.")
            if user_vehicle['avg_speed'] < recommended_vehicle['avg_speed']:
                print(f"The RL recommendation has higher average speed.")
        
        # Save the model
        save_choice = input("\nWould you like to save the trained model? (y/n): ").lower()
        if save_choice == 'y':
            state_size = 5
            action_size = len(vehicles)
            rl_agent = RouteVehicleRL(state_size, action_size, vehicles)
            save_model(rl_agent)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nThank you for using the Route and Vehicle Optimization System")
