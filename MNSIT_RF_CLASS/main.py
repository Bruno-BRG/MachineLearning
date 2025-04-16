# Import required libraries
import streamlit as st  # For creating the web interface
import matplotlib.pyplot as plt  # For displaying images
import numpy as np  # For numerical operations
import torch  # Deep learning framework
from stable_baselines3 import PPO  # Reinforcement Learning algorithm
import gymnasium as gym  # Environment framework
from gymnasium import spaces
from torchvision import datasets, transforms  # For loading MNIST dataset
import os

# Constants
MODEL_SAVE_PATH = "ppo_mnist_model"  # Where to save our trained model
TOTAL_TRAINING_STEPS = 50000  # How many times to train the model
TEST_SAMPLES = 1000  # Number of samples to use for accuracy testing

class MNISTDigitClassifier(gym.Env):
    """
    A custom environment for MNIST digit classification using reinforcement learning.
    This environment presents MNIST digits to the agent and rewards it for correct predictions.
    """
    def __init__(self, train=False):
        super(MNISTDigitClassifier, self).__init__()
        
        # Load MNIST dataset (download if not present)
        self.dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        
        # Define what the agent can observe (28x28 pixel images)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(28, 28),
            dtype=np.float32
        )
        
        # Define what actions the agent can take (predict digits 0-9)
        self.action_space = spaces.Discrete(10)
        
        # Keep track of current position in dataset
        self.current_index = 0

    def reset(self, seed=None, options=None):
        """Get a new random digit from the dataset"""
        # Choose a random digit from the dataset
        self.current_index = np.random.randint(0, len(self.dataset))
        image, self.correct_digit = self.dataset[self.current_index]
        
        # Convert image to numpy array for the environment
        observation = image.squeeze().numpy()
        return observation, {}

    def step(self, predicted_digit):
        """
        Process the agent's prediction and return the next state
        Args:
            predicted_digit: The digit (0-9) that the agent predicts
        """
        # Give reward of +1 for correct prediction, -1 for incorrect
        reward = 1 if predicted_digit == self.correct_digit else -1
        
        # Each prediction is a complete episode
        done = True
        
        # Get next observation
        next_observation, _ = self.reset()
        
        return next_observation, reward, done, False, {}


@st.cache_resource
def setup_and_load_model():
    """
    Sets up and loads the model, training it if necessary.
    Uses Streamlit caching to avoid retraining on every rerun.
    """
    if not os.path.exists(f"{MODEL_SAVE_PATH}.zip"):
        st.write("🤖 First time setup: Training a new model...")
        st.write("⏳ This may take a few minutes...")
        
        # Create training environment and model
        training_env = MNISTDigitClassifier(train=True)
        model = PPO("MlpPolicy", training_env, verbose=1)
        
        # Train the model
        model.learn(total_timesteps=TOTAL_TRAINING_STEPS)
        
        # Save the trained model
        model.save(MODEL_SAVE_PATH)
        st.write("✅ Training completed! Model saved.")
    else:
        st.write("✅ Loading pre-trained model...")
    
    return PPO.load(MODEL_SAVE_PATH)

# Main application code
def main():
    st.title("📱 MNIST Digit Recognition")
    st.write("This app uses Reinforcement Learning to recognize handwritten digits!")
    
    # Initialize model and environment
    try:
        model = setup_and_load_model()
        env = MNISTDigitClassifier(train=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

    # Sidebar with model testing
    with st.sidebar:
        st.title("🔍 Model Testing")
        if st.button("Test Model Accuracy"):
            run_accuracy_test(model, env)

    # Main area for individual predictions
    st.subheader("🎯 Try Individual Predictions")
    if st.button("Show Me a Prediction"):
        show_prediction(model, env)
    else:
        st.write("👆 Click the button above to see the model in action!")

def run_accuracy_test(model, env):
    """Run accuracy test on multiple samples"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    correct_predictions = 0
    
    for i in range(TEST_SAMPLES):
        # Get a digit and make prediction
        digit_image, _ = env.reset()
        predicted_digit, _ = model.predict(digit_image, deterministic=True)
        
        # Check if prediction was correct
        if predicted_digit == env.correct_digit:
            correct_predictions += 1
        
        # Update progress
        progress = (i + 1) / TEST_SAMPLES
        progress_bar.progress(progress)
        status_text.text(f"Testing: {i+1}/{TEST_SAMPLES}")
    
    # Calculate and display final accuracy
    accuracy = (correct_predictions / TEST_SAMPLES) * 100
    st.success(f"🎯 Model Accuracy: {accuracy:.2f}%")

def show_prediction(model, env):
    """Show a single prediction with visualization"""
    # Get a new digit and make prediction
    digit_image, _ = env.reset()
    predicted_digit, _ = model.predict(digit_image, deterministic=True)
    is_correct = (predicted_digit == env.correct_digit)

    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the digit image
        fig, ax = plt.subplots()
        ax.imshow(digit_image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        # Show prediction results
        st.markdown(f"### 🤖 Model's Guess: `{predicted_digit}`")
        st.markdown(f"### ✨ Actual Digit: `{env.correct_digit}`")
        st.markdown("### Result: " + ("✅ Correct!" if is_correct else "❌ Wrong!"))

# Run the application
if __name__ == "__main__":
    main()
